"""
EyeGlynt - Gaze Engine
======================
Runs the webcam + MediaPipe iris tracking in its own background thread,
so it never blocks Kivy's UI loop, and exposes a small, simple API:

    engine = GazeEngine()
    engine.start()
    engine.get_latest_feature()   -> (fx, fy) or None
    engine.add_calibration_point(fx, screen_x)
    engine.fit_calibration()
    engine.get_screen_x()         -> predicted screen X, or None

CURRENT LIMITATION (see project notes / testing results): plain webcam +
MediaPipe iris landmarks gave a strong, clean horizontal (fx) signal, but
an unusable vertical (fy) signal -- eyelid movement cancels out most of
the real vertical eye rotation. This is a known limitation of iris-only
2D tracking (vs. true PCCR with dedicated NIR illumination, which is
what the actual EyeGlynt hardware will use). Because of that, THIS
ENGINE ONLY CALIBRATES/PREDICTS THE HORIZONTAL (X) SCREEN POSITION.
Vertical is left to the mouse for now (see GazeManager in main.py) until
real PCCR is implemented once the IR camera + LED hardware exists.
"""

import os
import threading
import time
import urllib.request

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Both camera backends are optional: the laptop has OpenCV but no
# picamera2, the Pi has picamera2 (and OpenCV is unnecessary there).
# Importing them defensively means the same file runs on both machines
# with no edits -- see _open_camera() below for the selection logic.
try:
    import cv2
except ImportError:
    cv2 = None

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# --- Raspberry Pi camera settings -----------------------------------
# The Arducam OV9281 needs exposure set manually: libcamera's
# auto-exposure doesn't work well on this mono sensor, so frames come
# out nearly black at default settings. These values were found during
# hardware bring-up and gave 100% face detection in benchmarking.
# EXPECT TO LOWER THE GAIN once the IR LEDs are added -- they'll
# provide illumination, and high gain adds noise that hurts tracking.
PI_SHUTTER_US = 20000   # microseconds (20ms)
PI_ANALOGUE_GAIN = 8.0
# 640x400 measured ~14 fps end-to-end on a Pi 4 (MediaPipe inference is
# the bottleneck at ~68ms/frame, capture is only ~2ms). Raising this
# resolution will slow things down without helping tracking much.
CAPTURE_WIDTH, CAPTURE_HEIGHT = 640, 400

# Landmark indices, confirmed directly from MediaPipe's own connection
# graph -- see gaze_test.py for how these were verified.
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473
RIGHT_EYE_CONTOUR = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]
LEFT_EYE_CONTOUR = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]

# Exponential-moving-average smoothing applied to the raw gaze feature
# before it's used for prediction. Without this, tiny frame-to-frame
# landmark noise gets hugely amplified once mapped across the screen
# width (the calibrated eye-movement range is much smaller than the
# screen, so the regression's "gain" is high). Lower alpha = smoother
# but more lag; higher alpha = snappier but jumpier. Tune this once
# testing with real users -- worth documenting the chosen value and why,
# same as your dwell-time threshold.
SMOOTHING_ALPHA = 0.25


def _ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return
    print(f"[GazeEngine] Downloading face landmark model to '{MODEL_PATH}' ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[GazeEngine] Done.")


def _normalized_iris_position(landmarks, iris_idx, contour_indices, w, h):
    xs = [landmarks[i].x * w for i in contour_indices]
    ys = [landmarks[i].y * h for i in contour_indices]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    iris = landmarks[iris_idx]
    ix, iy = iris.x * w, iris.y * h
    fx = (ix - min_x) / (max_x - min_x) if max_x > min_x else 0.5
    fy = (iy - min_y) / (max_y - min_y) if max_y > min_y else 0.5
    return fx, fy


def _compute_gaze_feature(landmarks, w, h):
    rfx, rfy = _normalized_iris_position(landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_CONTOUR, w, h)
    lfx, lfy = _normalized_iris_position(landmarks, LEFT_IRIS_CENTER, LEFT_EYE_CONTOUR, w, h)
    return (rfx + lfx) / 2, (rfy + lfy) / 2


def _draw_dot(frame, cx, cy, radius=4, color=(0, 255, 0)):
    """Draws a small filled square marker on an RGB numpy frame.

    Uses plain numpy rather than cv2.circle so the live preview works
    on the Pi, where OpenCV isn't necessarily installed (picamera2
    handles capture there, so OpenCV would be dead weight).
    """
    h, w = frame.shape[:2]
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    if y0 < y1 and x0 < x1:
        frame[y0:y1, x0:x1] = color


# ---------------------------------------------------------------------
# Camera backends
# ---------------------------------------------------------------------
# Two interchangeable classes with the same three methods (read, close,
# and a constructor that raises if unavailable). The capture loop below
# doesn't know or care which one it got -- that's what lets the same
# code run on the laptop and the Pi.
#
# Both return frames as RGB numpy arrays, already mirrored, so anything
# downstream (MediaPipe, the preview) sees an identical format.

class _OpenCVCamera:
    """Laptop/USB webcam via OpenCV. Used when picamera2 isn't present."""

    name = "OpenCV webcam"

    def __init__(self):
        if cv2 is None:
            raise RuntimeError("OpenCV is not installed")
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise RuntimeError("Could not open webcam via OpenCV")

    def read(self):
        ok, frame = self._cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)  # mirror, so it feels like a mirror
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        self._cap.release()


class _PiCamera:
    """Raspberry Pi CSI camera (Arducam OV9281) via picamera2.

    OpenCV's VideoCapture cannot read libcamera-based CSI cameras at
    all, which is why this separate backend exists rather than just
    passing a different device index.
    """

    name = "Raspberry Pi camera (picamera2)"

    def __init__(self):
        if Picamera2 is None:
            raise RuntimeError("picamera2 is not installed")
        self._picam = Picamera2()
        config = self._picam.create_preview_configuration(
            main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"}
        )
        self._picam.configure(config)
        self._picam.set_controls({
            "ExposureTime": PI_SHUTTER_US,
            "AnalogueGain": PI_ANALOGUE_GAIN,
        })
        self._picam.start()
        time.sleep(2)  # let exposure settle before the first frames

    def read(self):
        frame = self._picam.capture_array()
        return frame[:, ::-1]  # mirror horizontally (numpy, no OpenCV needed)

    def close(self):
        self._picam.stop()


def _open_camera():
    """Returns a working camera backend, or None if neither is usable.

    Tries the Pi camera first: if picamera2 imported successfully we're
    almost certainly on a Pi, where it's the only option that works.
    """
    for backend in (_PiCamera, _OpenCVCamera):
        try:
            camera = backend()
            print(f"[GazeEngine] Using {backend.name}")
            return camera
        except Exception as e:
            print(f"[GazeEngine] {backend.name} unavailable: {e}")
    return None


class GazeEngine:
    """
    Owns the webcam + MediaPipe pipeline in a background thread.

    Thread safety: all state shared with the main (Kivy) thread is
    guarded by self._lock. Kivy code should only ever call the public
    methods below -- never touch the camera/MediaPipe objects directly,
    since those only exist inside the background thread.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_feature = None  # (fx, fy), or None if no face seen
        self._smoothed_feature = None  # internal EMA state, background-thread only
        self._latest_debug_frame = None  # RGB numpy frame with dots drawn, for a live preview
        self._running = False
        self._thread = None
        self._camera_ok = True  # flips False if the webcam can't be opened

        # Horizontal-only calibration: (fx, screen_x) pairs, and the
        # fitted regression coefficients once fit_calibration() runs.
        self._calibration_points = []
        self._coeffs = None  # (a, b) such that screen_x = a*fx + b

    # ---- lifecycle ---------------------------------------------------
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run_loop(self):
        try:
            _ensure_model_downloaded()
            base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"[GazeEngine] Could not set up MediaPipe: {e}")
            self._camera_ok = False
            self._running = False
            return

        camera = _open_camera()
        if camera is None:
            print("[GazeEngine] No usable camera found. Falling back to mouse-only.")
            self._camera_ok = False
            self._running = False
            return

        start_time = time.time()
        while self._running:
            rgb = camera.read()
            if rgb is None:
                continue
            h, w = rgb.shape[:2]
            # MediaPipe needs a contiguous array; mirroring with numpy
            # slicing produces a view, not a copy, so make it explicit.
            rgb = np.ascontiguousarray(rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            raw_feature = _compute_gaze_feature(result.face_landmarks[0], w, h) if result.face_landmarks else None

            if raw_feature is None:
                feature = None
                self._smoothed_feature = None  # reset so re-acquiring doesn't drag from a stale position
            elif self._smoothed_feature is None:
                feature = raw_feature  # first reading after (re)acquiring a face -- nothing to blend with yet
                self._smoothed_feature = raw_feature
            else:
                prev_fx, prev_fy = self._smoothed_feature
                raw_fx, raw_fy = raw_feature
                a = SMOOTHING_ALPHA
                feature = (a * raw_fx + (1 - a) * prev_fx, a * raw_fy + (1 - a) * prev_fy)
                self._smoothed_feature = feature

            debug_frame = rgb.copy()
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                for idx in (RIGHT_IRIS_CENTER, LEFT_IRIS_CENTER):
                    lm = landmarks[idx]
                    _draw_dot(debug_frame, int(lm.x * w), int(lm.y * h))

            with self._lock:
                self._latest_feature = feature
                self._latest_debug_frame = debug_frame

        camera.close()

    # ---- public API used by the Kivy app ------------------------------
    def is_camera_ok(self):
        return self._camera_ok

    def get_latest_feature(self):
        """Returns (fx, fy) or None if no face is currently detected."""
        with self._lock:
            return self._latest_feature

    def get_debug_frame(self):
        """Returns the latest camera frame (RGB numpy array, with iris dots
        drawn on it) for showing a live preview, or None if unavailable."""
        with self._lock:
            return self._latest_debug_frame

    def is_calibrated(self):
        return self._coeffs is not None

    def add_calibration_point(self, fx, screen_x):
        self._calibration_points.append((fx, screen_x))

    def clear_calibration(self):
        self._calibration_points = []
        self._coeffs = None

    def fit_calibration(self):
        """Fits screen_x = a*fx + b from the collected calibration points.
        Raises ValueError if fewer than 2 usable points were collected
        (e.g. no camera, or no face seen during calibration)."""
        if len(self._calibration_points) < 2:
            raise ValueError(f"Need at least 2 calibration points, got {len(self._calibration_points)}")
        fx_vals = np.array([p[0] for p in self._calibration_points])
        x_vals = np.array([p[1] for p in self._calibration_points])
        a_matrix = np.vstack([fx_vals, np.ones_like(fx_vals)]).T
        a, b = np.linalg.lstsq(a_matrix, x_vals, rcond=None)[0]
        self._coeffs = (float(a), float(b))

    def get_screen_x(self):
        """Returns the predicted screen X from the current gaze feature,
        or None if not calibrated or no face is currently detected."""
        if self._coeffs is None:
            return None
        feature = self.get_latest_feature()
        if feature is None:
            return None
        fx, _fy = feature
        a, b = self._coeffs
        return a * fx + b