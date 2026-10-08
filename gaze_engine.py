"""
EyeGlynt - Gaze Engine
======================
Runs the camera + MediaPipe tracking in its own background thread,
so it never blocks Kivy's UI loop, and exposes a small, simple API:

    engine = GazeEngine()
    engine.start()
    engine.get_latest_feature()   -> (fx, vy) or None; vy may be None
    engine.add_calibration_point(fx, screen_x, vy, screen_y)
    engine.fit_calibration()
    engine.get_screen_x()         -> predicted screen X, or None
    engine.get_screen_y()         -> predicted screen Y, or None

HOW GAZE IS MEASURED
--------------------
Left/right (fx): where the iris sits between the corners of the eye
(MediaPipe iris landmarks). This worked well in testing.

Up/down (vy): Pupil-Center Corneal Reflection (PCCR). The two IR LEDs
make small bright reflections ("glints") on the cornea. When the eye
rotates up or down, the PUPIL moves but the GLINTS stay almost still,
so the distance from the glints to the pupil measures where the eye is
looking. The old method (iris position between the eyelids) failed for
up/down because the eyelids move together with the eye; the glints do
not, which is why PCCR works where it failed.

On a laptop webcam there are no IR glints, so vy is usually None and
the app keeps using the mouse for up/down.
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
    from libcamera import controls as libcamera_controls  # names for camera settings (autofocus mode)
except ImportError:
    Picamera2 = None

MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# --- Raspberry Pi camera settings -----------------------------------
# Camera: Raspberry Pi Camera Module 3 NoIR (sensor imx708_noir).
# Unlike the earlier Arducam OV9281, Raspberry Pi ships a tuning file
# for this camera, so automatic exposure works and no manual shutter or
# gain values are needed. Exposure adapts on its own when the room
# lighting changes or the IR LEDs are on.
#
# Focus is fixed at the user's working distance instead of autofocus,
# so the lens never "hunts" in the middle of tracking. The lens position
# is in dioptres = 1 / distance in metres (0.5 m -> 2.0).
PI_FOCUS_DISTANCE_M = 0.5
# The Pi captures at 1536x864, the sensor's own full-view mode. The
# glints are only a few pixels wide, so the eyes need this detail.
# MediaPipe gets a half-size copy (768x432, every 2nd pixel), because it
# doesn't need the detail and a smaller picture keeps it fast.
CAPTURE_WIDTH, CAPTURE_HEIGHT = 1536, 864
PI_MEDIAPIPE_STEP = 2   # use every 2nd pixel for MediaPipe

# Landmark indices, confirmed directly from MediaPipe's own connection
# graph -- see gaze_test.py for how these were verified.
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_EDGE = [469, 470, 471, 472]   # 4 points on the edge of the iris
LEFT_IRIS_EDGE = [474, 475, 476, 477]
RIGHT_EYE_CORNERS = (33, 133)            # outer and inner corner
LEFT_EYE_CORNERS = (362, 263)
RIGHT_EYE_CONTOUR = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]
LEFT_EYE_CONTOUR = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]

# --- PCCR (up/down) settings -----------------------------------------
# A glint is the brightest spot near the iris. A pixel counts as part of
# a glint if it is within GLINT_TOLERANCE of the brightest pixel; and if
# even the brightest pixel is darker than GLINT_MIN_BRIGHTNESS, there is
# no glint at all (IR LEDs off, or a blink).
GLINT_MIN_BRIGHTNESS = 200   # 0 = black, 255 = white
GLINT_TOLERANCE = 25
# The pupil is the darkest part of the iris. The darkest
# PUPIL_DARKEST_PERCENT % of pixels inside the iris circle are taken
# as the pupil.
PUPIL_DARKEST_PERCENT = 15
# Glints are searched for slightly beyond the iris edge (they can sit
# near the edge when looking far up or down); the pupil only inside it.
GLINT_SEARCH_SCALE = 1.4
PUPIL_SEARCH_SCALE = 0.9
# If no glint is found for this many frames in a row (about 1 second),
# up/down is reported as unknown instead of keeping an old value.
# Shorter gaps, such as a blink, simply keep the last value.
MAX_MISSED_GLINT_FRAMES = 15

# Exponential-moving-average smoothing applied to the raw gaze feature
# before it's used for prediction. Without this, tiny frame-to-frame
# landmark noise gets hugely amplified once mapped across the screen
# width (the calibrated eye-movement range is much smaller than the
# screen, so the regression's "gain" is high). Lower alpha = smoother
# but more lag; higher alpha = snappier but jumpier. Tune this once
# testing with real users -- worth documenting the chosen value and why,
# same as your dwell-time threshold.
SMOOTHING_ALPHA = 0.25

# After calibration, each axis is only used if a straight line fits the
# calibration points well. R squared (0 to 1) measures that fit: 1 means
# the eye measurements line up perfectly with the dot positions, 0
# means no relationship at all. Below this value that axis falls back to
# the mouse, so a bad calibration can't make the pointer jump around.
MIN_FIT_R2 = 0.6


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


def _compute_horizontal_feature(landmarks, w, h):
    """fx: the iris position between the eye corners, averaged over both
    eyes. 0 = iris at one corner, 1 = at the other."""
    rfx, _ = _normalized_iris_position(landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_CONTOUR, w, h)
    lfx, _ = _normalized_iris_position(landmarks, LEFT_IRIS_CENTER, LEFT_EYE_CONTOUR, w, h)
    return (rfx + lfx) / 2


def _crop_square(image, cx, cy, radius):
    """Cuts a square of side 2*radius centred on (cx, cy) out of `image`,
    clipped at the image edges. Returns the crop and the position of its
    top-left corner in the full image."""
    h, w = image.shape[:2]
    x0, x1 = max(0, int(cx - radius)), min(w, int(cx + radius) + 1)
    y0, y1 = max(0, int(cy - radius)), min(h, int(cy + radius) + 1)
    return image[y0:y1, x0:x1], x0, y0


def find_glint(gray, cx, cy, radius):
    """Finds the IR glint(s) near (cx, cy) in a grayscale image.

    Returns the centre (x, y) of the bright pixels, in full-image
    coordinates, or None if nothing is bright enough. With two LEDs
    there are two glints per eye; both are within GLINT_TOLERANCE of the
    brightest pixel, so the centre found is the point between them.
    """
    crop, x0, y0 = _crop_square(gray, cx, cy, radius)
    if crop.size == 0:
        return None
    brightest = crop.max()
    if brightest < GLINT_MIN_BRIGHTNESS:
        return None
    ys, xs = np.nonzero(crop >= brightest - GLINT_TOLERANCE)
    return (x0 + xs.mean(), y0 + ys.mean())


def find_pupil(gray, cx, cy, radius):
    """Finds the pupil centre near (cx, cy) in a grayscale image.

    Only pixels inside a circle of `radius` are used, so dark eyelashes
    and eyelid shadows outside the iris are ignored. The darkest
    PUPIL_DARKEST_PERCENT % of those pixels are the pupil; their centre
    is returned in full-image coordinates, or None if the crop is empty.
    """
    crop, x0, y0 = _crop_square(gray, cx, cy, radius)
    if crop.size == 0:
        return None
    # Distance of every crop pixel from the circle's centre.
    rows, cols = np.ogrid[:crop.shape[0], :crop.shape[1]]
    inside = (cols + x0 - cx) ** 2 + (rows + y0 - cy) ** 2 <= radius ** 2
    if not inside.any():
        return None
    dark_limit = np.percentile(crop[inside], PUPIL_DARKEST_PERCENT)
    ys, xs = np.nonzero(inside & (crop <= dark_limit))
    return (x0 + xs.mean(), y0 + ys.mean())


def _measure_eye(frame, landmarks, iris_center, iris_edge, corners):
    """PCCR for one eye. Returns (vy, glint, pupil), positions in pixels
    of the full picture.

    vy is the vertical distance from the glint to the pupil, divided by
    the eye's width so it doesn't change when the user sits nearer or
    further away. vy is None if the glint or pupil wasn't found.
    """
    h, w = frame.shape[:2]

    def pixel(i):
        """Landmark i in pixels (MediaPipe gives 0-1 fractions)."""
        return landmarks[i].x * w, landmarks[i].y * h

    cx, cy = pixel(iris_center)
    # Iris radius: average distance from the iris centre to its 4 edge points.
    iris_radius = np.mean([np.hypot(pixel(i)[0] - cx, pixel(i)[1] - cy) for i in iris_edge])
    (ax, ay), (bx, by) = pixel(corners[0]), pixel(corners[1])
    eye_width = np.hypot(ax - bx, ay - by)
    if iris_radius < 2 or eye_width < 10:
        return None, None, None  # eye too small in the picture to measure

    # Grayscale (average of the colour channels) of only the small area
    # around this eye -- much faster than converting the whole picture.
    glint_radius = iris_radius * GLINT_SEARCH_SCALE
    box, x0, y0 = _crop_square(frame, cx, cy, glint_radius + 2)
    gray = box.mean(axis=2) if box.ndim == 3 else box

    # Search in the small box (positions relative to its corner x0, y0),
    # then add x0, y0 back to get positions in the full picture.
    glint = find_glint(gray, cx - x0, cy - y0, glint_radius)
    pupil = find_pupil(gray, cx - x0, cy - y0, iris_radius * PUPIL_SEARCH_SCALE)
    if glint is not None:
        glint = (glint[0] + x0, glint[1] + y0)
    if pupil is not None:
        pupil = (pupil[0] + x0, pupil[1] + y0)
    if glint is None or pupil is None:
        return None, glint, pupil
    vy = (pupil[1] - glint[1]) / eye_width
    return vy, glint, pupil


def _compute_vertical_feature(frame, landmarks):
    """vy: the PCCR up/down measurement, averaged over the eyes where it
    worked. Returns (vy or None, list of ("glint"/"pupil", x, y) points
    to draw on the preview)."""
    values, points = [], []
    for iris_center, iris_edge, corners in (
        (RIGHT_IRIS_CENTER, RIGHT_IRIS_EDGE, RIGHT_EYE_CORNERS),
        (LEFT_IRIS_CENTER, LEFT_IRIS_EDGE, LEFT_EYE_CORNERS),
    ):
        vy, glint, pupil = _measure_eye(frame, landmarks, iris_center, iris_edge, corners)
        if vy is not None:
            values.append(vy)
        if glint is not None:
            points.append(("glint", glint[0], glint[1]))
        if pupil is not None:
            points.append(("pupil", pupil[0], pupil[1]))
    vy = sum(values) / len(values) if values else None
    return vy, points


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


def _fit_line(inputs, outputs):
    """Least-squares straight line: output = a * input + b.
    Returns (a, b, r2), where r2 (R squared) says how well the line fits:
    1.0 = perfectly, 0 = not at all."""
    x = np.array(inputs, dtype=float)
    y = np.array(outputs, dtype=float)
    a_matrix = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(a_matrix, y, rcond=None)[0]
    predicted = a * x + b
    total = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum((y - predicted) ** 2) / total if total > 0 else 0.0
    return float(a), float(b), float(r2)


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
# `mediapipe_step` says how much to shrink the frame before MediaPipe.

class _OpenCVCamera:
    """Laptop/USB webcam via OpenCV. Used when picamera2 isn't present."""

    name = "OpenCV webcam"
    mediapipe_step = 1  # webcam pictures are already small

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
    """Raspberry Pi CSI camera (Camera Module 3 NoIR) via picamera2.

    OpenCV's VideoCapture cannot read libcamera-based CSI cameras at
    all, which is why this separate backend exists rather than just
    passing a different device index.
    """

    name = "Raspberry Pi camera (picamera2)"
    mediapipe_step = PI_MEDIAPIPE_STEP

    def __init__(self):
        if Picamera2 is None:
            raise RuntimeError("picamera2 is not installed")
        self._picam = Picamera2()
        # picamera2's format names are the reverse of the pixel order in
        # memory: "BGR888" gives pixels as [R, G, B], which is the RGB
        # order MediaPipe expects. (The old mono camera made every
        # channel the same, so the order didn't matter; with a colour
        # camera it does.)
        config = self._picam.create_preview_configuration(
            main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "BGR888"}
        )
        self._picam.configure(config)
        self._picam.start()
        # Fixed focus at the working distance (see PI_FOCUS_DISTANCE_M).
        # Exposure is left on automatic, its default.
        self._picam.set_controls({
            "AfMode": libcamera_controls.AfModeEnum.Manual,
            "LensPosition": 1.0 / PI_FOCUS_DISTANCE_M,
        })
        time.sleep(2)  # let exposure and focus settle before the first frames

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
    Owns the camera + MediaPipe pipeline in a background thread.

    Thread safety: all state shared with the main (Kivy) thread is
    guarded by self._lock. Kivy code should only ever call the public
    methods below -- never touch the camera/MediaPipe objects directly,
    since those only exist inside the background thread.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_feature = None  # (fx, vy), or None if no face seen
        self._smoothed_fx = None     # internal EMA state, background-thread only
        self._smoothed_vy = None
        self._missed_glint_frames = 0
        self._latest_debug_frame = None  # RGB numpy frame with dots drawn, for a live preview
        self._running = False
        self._thread = None
        self._camera_ok = True  # flips False if the camera can't be opened

        # Calibration points, and the fitted straight lines once
        # fit_calibration() runs. Each fit is (a, b, r2) such that
        # screen = a * feature + b, or None if that axis isn't usable.
        self._x_points = []  # (fx, screen_x)
        self._y_points = []  # (vy, screen_y)
        self._x_fit = None
        self._y_fit = None

    # ---- lifecycle ---------------------------------------------------
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _smooth(self, previous, new):
        """One step of the exponential moving average."""
        if previous is None:
            return new  # first reading -- nothing to blend with yet
        return SMOOTHING_ALPHA * new + (1 - SMOOTHING_ALPHA) * previous

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
            frame = camera.read()  # full-size RGB picture
            if frame is None:
                continue
            # MediaPipe gets a smaller copy (every Nth pixel). It needs a
            # contiguous array; slicing produces a view, so copy it.
            step = camera.mediapipe_step
            small = np.ascontiguousarray(frame[::step, ::step])
            sh, sw = small.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            debug_frame = small.copy()
            if not result.face_landmarks:
                feature = None
                # Reset so re-acquiring doesn't drag from a stale position.
                self._smoothed_fx = None
                self._smoothed_vy = None
            else:
                landmarks = result.face_landmarks[0]
                fx = _compute_horizontal_feature(landmarks, sw, sh)
                vy, points = _compute_vertical_feature(frame, landmarks)

                self._smoothed_fx = self._smooth(self._smoothed_fx, fx)
                if vy is not None:
                    self._missed_glint_frames = 0
                    self._smoothed_vy = self._smooth(self._smoothed_vy, vy)
                else:
                    # No glint this frame (e.g. a blink): keep the last
                    # value for a moment, then give up on it.
                    self._missed_glint_frames += 1
                    if self._missed_glint_frames > MAX_MISSED_GLINT_FRAMES:
                        self._smoothed_vy = None
                feature = (self._smoothed_fx, self._smoothed_vy)

                # Preview dots: green = iris (MediaPipe), white = glint,
                # yellow = pupil found in the picture.
                for idx in (RIGHT_IRIS_CENTER, LEFT_IRIS_CENTER):
                    lm = landmarks[idx]
                    _draw_dot(debug_frame, int(lm.x * sw), int(lm.y * sh))
                for kind, px, py in points:
                    color = (255, 255, 255) if kind == "glint" else (255, 220, 0)
                    _draw_dot(debug_frame, int(px / step), int(py / step), radius=2, color=color)

            with self._lock:
                self._latest_feature = feature
                self._latest_debug_frame = debug_frame

        camera.close()

    # ---- public API used by the Kivy app ------------------------------
    def is_camera_ok(self):
        return self._camera_ok

    def get_latest_feature(self):
        """Returns (fx, vy) or None if no face is currently detected.
        vy is None when no glint is visible (no IR, or a laptop webcam)."""
        with self._lock:
            return self._latest_feature

    def get_debug_frame(self):
        """Returns the latest camera frame (RGB numpy array, with iris,
        glint and pupil dots drawn on it) for showing a live preview, or
        None if unavailable."""
        with self._lock:
            return self._latest_debug_frame

    def is_calibrated(self):
        return self._x_fit is not None

    def is_vertical_calibrated(self):
        return self._y_fit is not None

    def add_calibration_point(self, fx, screen_x, vy=None, screen_y=None):
        """Stores one calibration dot's measurements. vy/screen_y are
        left out when no glint was seen at that dot."""
        self._x_points.append((fx, screen_x))
        if vy is not None and screen_y is not None:
            self._y_points.append((vy, screen_y))

    def clear_calibration(self):
        self._x_points = []
        self._y_points = []
        self._x_fit = None
        self._y_fit = None

    def fit_calibration(self):
        """Fits screen_x = a*fx + b, and screen_y = c*vy + d if there is
        up/down data from at least 2 different dot heights.

        Returns a short text summary for the user. Raises ValueError if
        the horizontal fit isn't possible or isn't good enough (e.g. no
        camera, or no face seen during calibration)."""
        if len(self._x_points) < 2:
            raise ValueError(f"Need at least 2 calibration points, got {len(self._x_points)}")
        a, b, r2_x = _fit_line([p[0] for p in self._x_points], [p[1] for p in self._x_points])
        print(f"[Calibration] left/right fit: R^2 = {r2_x:.2f} from {len(self._x_points)} points")
        if r2_x < MIN_FIT_R2:
            raise ValueError(f"left/right fit too weak (R^2 = {r2_x:.2f})")
        self._x_fit = (a, b, r2_x)

        heights = {round(p[1]) for p in self._y_points}
        if len(heights) < 2:
            print("[Calibration] up/down: no glints seen -- mouse will control up/down")
            self._y_fit = None
            return f"Left/right fit {r2_x:.2f}. Up/down: no IR glints seen, using the mouse."
        c, d, r2_y = _fit_line([p[0] for p in self._y_points], [p[1] for p in self._y_points])
        print(f"[Calibration] up/down fit: R^2 = {r2_y:.2f} from {len(self._y_points)} points")
        if r2_y < MIN_FIT_R2:
            self._y_fit = None
            return f"Left/right fit {r2_x:.2f}. Up/down fit too weak ({r2_y:.2f}), using the mouse."
        self._y_fit = (c, d, r2_y)
        return f"Left/right fit {r2_x:.2f}. Up/down fit {r2_y:.2f}."

    def get_screen_x(self):
        """Returns the predicted screen X from the current gaze feature,
        or None if not calibrated or no face is currently detected."""
        if self._x_fit is None:
            return None
        feature = self.get_latest_feature()
        if feature is None:
            return None
        a, b, _r2 = self._x_fit
        return a * feature[0] + b

    def get_screen_y(self):
        """Returns the predicted screen Y (Kivy: 0 = bottom) from the
        current PCCR measurement, or None if up/down isn't calibrated
        or no glint is currently visible."""
        if self._y_fit is None:
            return None
        feature = self.get_latest_feature()
        if feature is None or feature[1] is None:
            return None
        c, d, _r2 = self._y_fit
        return c * feature[1] + d
