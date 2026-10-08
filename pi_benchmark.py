"""
EyeGlynt - Raspberry Pi Performance Benchmark
==============================================
Measures how fast MediaPipe face-landmark detection actually runs on
this Pi, using the Camera Module 3 NoIR via picamera2.

This answers the key unknown before adapting the full gaze engine: if
this runs at 15+ fps, the existing design works as-is. If it's much
slower, we need to rethink (smaller capture resolution, processing
every Nth frame, etc.) before building on top of it.

It also verifies the whole Pi-side pipeline in one shot:
camera -> frame -> MediaPipe -> iris landmarks.

RUN (on the Pi, with the venv active):
    source ~/eyeglynt_env/bin/activate
    cd ~/eyeglynt
    python pi_benchmark.py

No preview window is opened, so this works fine over SSH.
"""

import os
import time
import urllib.request

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from picamera2 import Picamera2
from libcamera import controls  # names for camera settings (autofocus mode)

MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# The Camera Module 3 NoIR has automatic exposure that works out of the
# box, so no shutter or gain values are set here. Focus is fixed at the
# user's distance; lens position is in dioptres = 1 / distance in metres.
FOCUS_DISTANCE_M = 0.5

# Same as gaze_engine.py: capture at 1536x864 (detail for the IR
# glints), and give MediaPipe every 2nd pixel (768x432) to keep it fast.
CAPTURE_WIDTH, CAPTURE_HEIGHT = 1536, 864
MEDIAPIPE_STEP = 2

BENCHMARK_FRAMES = 100

RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473


def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return
    print(f"Downloading face landmark model to '{MODEL_PATH}' ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


def main():
    ensure_model_downloaded()

    print("Setting up MediaPipe ...")
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

    print("Setting up camera ...")
    picam2 = Picamera2()
    # picamera2's "BGR888" gives pixels in [R, G, B] order, which is
    # what MediaPipe expects (same as in gaze_engine.py).
    config = picam2.create_preview_configuration(
        main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()
    picam2.set_controls({
        "AfMode": controls.AfModeEnum.Manual,           # no autofocus hunting
        "LensPosition": 1.0 / FOCUS_DISTANCE_M,         # focus at the user's distance
    })
    time.sleep(2)  # let exposure and focus settle

    print(f"\nRunning {BENCHMARK_FRAMES} frames -- look at the camera!\n")

    capture_times = []
    inference_times = []
    frames_with_face = 0
    start_time = time.time()

    for i in range(BENCHMARK_FRAMES):
        t0 = time.time()
        frame = picam2.capture_array()
        t1 = time.time()

        small = np.ascontiguousarray(frame[::MEDIAPIPE_STEP, ::MEDIAPIPE_STEP])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        t2 = time.time()

        capture_times.append(t1 - t0)
        inference_times.append(t2 - t1)

        if result.face_landmarks:
            frames_with_face += 1
            if i % 20 == 0:
                lm = result.face_landmarks[0]
                r = lm[RIGHT_IRIS_CENTER]
                l = lm[LEFT_IRIS_CENTER]
                print(f"  frame {i:3d}: face found -- "
                      f"R iris ({r.x:.3f}, {r.y:.3f})  L iris ({l.x:.3f}, {l.y:.3f})")
        elif i % 20 == 0:
            print(f"  frame {i:3d}: no face detected")

    total_time = time.time() - start_time
    picam2.stop()

    avg_capture = np.mean(capture_times) * 1000
    avg_inference = np.mean(inference_times) * 1000
    fps = BENCHMARK_FRAMES / total_time

    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)
    print(f"Resolution:            {CAPTURE_WIDTH}x{CAPTURE_HEIGHT} "
          f"(MediaPipe: {CAPTURE_WIDTH // MEDIAPIPE_STEP}x{CAPTURE_HEIGHT // MEDIAPIPE_STEP})")
    print(f"Frames processed:      {BENCHMARK_FRAMES}")
    print(f"Face detected in:      {frames_with_face}/{BENCHMARK_FRAMES} frames")
    print(f"Avg capture time:      {avg_capture:.1f} ms")
    print(f"Avg MediaPipe time:    {avg_inference:.1f} ms")
    print(f"Overall throughput:    {fps:.1f} fps")
    print("=" * 55)

    if fps >= 15:
        print("\nPlenty fast -- the existing gaze engine design should work as-is.")
    elif fps >= 8:
        print("\nWorkable, but tight. Dwell selection will feel slightly less")
        print("responsive than on the laptop; worth optimizing later.")
    else:
        print("\nToo slow for comfortable real-time use. Worth trying a smaller")
        print("capture size, or processing every 2nd frame, before building on this.")

    if frames_with_face == 0:
        print("\nNOTE: no face was detected in ANY frame. Check that you were in")
        print("view of the camera, about FOCUS_DISTANCE_M away, and that the")
        print("picture isn't blurry (run rpicam-hello to look at it).")


if __name__ == "__main__":
    main()