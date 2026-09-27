"""
EyeGlynt - Gaze Tracking Diagnostic Script
===========================================
Goal: confirm your webcam + MediaPipe can find your face and your iris
position, before we build the full calibration + gaze-to-screen mapping
on top of it. This script does NOT compute a screen coordinate yet --
it just proves the raw signal is there and looks stable when you move
your eyes or your head.

IMPORTANT: MediaPipe changed its API in recent versions. Older
tutorials you'll find online use `mp.solutions.face_mesh` -- that no
longer exists in current MediaPipe. This script uses the current
"Tasks" API instead, which is what `pip install mediapipe` actually
gives you today.

SETUP (one-time):
    pip install mediapipe opencv-python numpy

The face landmark model file (~3.6 MB) is downloaded automatically the
first time you run this script, into the same folder as this file. If
that fails (e.g. blocked network), download it manually from the URL
in MODEL_URL below and place it next to this script.

RUN:
    python gaze_test.py
Press 'q' in the video window to quit.
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Landmark indices, confirmed directly from MediaPipe's own connection
# graph (mediapipe.tasks.vision.FaceLandmarksConnections) rather than
# copied from a tutorial -- index numbers have changed between
# MediaPipe versions before, and getting these wrong silently breaks
# everything built on top of them.
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473
RIGHT_EYE_OUTER, RIGHT_EYE_INNER = 33, 133
LEFT_EYE_INNER, LEFT_EYE_OUTER = 362, 263

# Full eyelid contour rings for each eye -- used to build a bounding box
# around each eye socket, so we can measure "how far toward the edge is
# the iris" as a percentage, instead of a raw, head-position-dependent
# pixel location.
RIGHT_EYE_CONTOUR = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]
LEFT_EYE_CONTOUR = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]


def normalized_iris_position(landmarks, iris_idx, contour_indices, w, h):
    """
    Returns (fx, fy), each roughly 0.0-1.0: how far across the eye socket
    (left edge to right edge, top edge to bottom edge) the iris center
    currently sits. 0.5, 0.5 is roughly "looking straight at the camera."
    This stays meaningful even if you move closer/further from the
    camera, unlike a raw pixel position.
    """
    xs = [landmarks[i].x * w for i in contour_indices]
    ys = [landmarks[i].y * h for i in contour_indices]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    iris = landmarks[iris_idx]
    ix, iy = iris.x * w, iris.y * h

    fx = (ix - min_x) / (max_x - min_x) if max_x > min_x else 0.5
    fy = (iy - min_y) / (max_y - min_y) if max_y > min_y else 0.5
    return fx, fy


def compute_gaze_feature(landmarks, w, h):
    """
    Combines both eyes into one (fx, fy) gaze feature by averaging --
    both eyes should be pointed the same direction, and averaging cancels
    out a bit of per-eye noise.
    """
    right_fx, right_fy = normalized_iris_position(landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_CONTOUR, w, h)
    left_fx, left_fy = normalized_iris_position(landmarks, LEFT_IRIS_CENTER, LEFT_EYE_CONTOUR, w, h)
    return (right_fx + left_fx) / 2, (right_fy + left_fy) / 2


def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return
    print(f"Downloading face landmark model to '{MODEL_PATH}' ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


def main():
    ensure_model_downloaded()

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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam. Is another app using it?")
        return

    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read a frame from the camera.")
            break

        frame = cv2.flip(frame, 1)  # mirror -- feels natural to look at
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode needs a monotonically increasing timestamp per frame
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]  # first (only) detected face

            def draw_point(idx, color):
                lm = landmarks[idx]
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, color, -1)

            # iris centers in green -- this is the signal we'll build on
            draw_point(RIGHT_IRIS_CENTER, (0, 255, 0))
            draw_point(LEFT_IRIS_CENTER, (0, 255, 0))
            # eye corners in yellow, just for reference right now
            for idx in (RIGHT_EYE_OUTER, RIGHT_EYE_INNER, LEFT_EYE_INNER, LEFT_EYE_OUTER):
                draw_point(idx, (0, 255, 255))

            cv2.putText(frame, "Face + iris detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            fx, fy = compute_gaze_feature(landmarks, w, h)
            cv2.putText(frame, f"gaze feature: ({fx:.2f}, {fy:.2f})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("EyeGlynt - Gaze Diagnostic (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()