"""
EyeGlynt - Raspberry Pi Performance Benchmark
==============================================
Measures how fast MediaPipe face-landmark detection actually runs on
this Pi, using the Arducam OV9281 via picamera2.

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

MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Camera exposure settings. The OV9281 needs these set manually --
# libcamera's auto-exposure doesn't work well on this sensor without a
# proper tuning file, so images come out nearly black by default.
# These are the values found to work during hardware bring-up; expect
# to lower the gain once the IR LEDs are added, since high gain adds
# noise that hurts tracking.
SHUTTER_US = 20000  # microseconds (20ms)
ANALOGUE_GAIN = 8.0

# Capture resolution. The sensor supports 640x400, 1280x720 and
# 1280x800. Starting at 640x400: fewer pixels means faster MediaPipe
# inference, and for eye tracking at close range it should be plenty.
CAPTURE_WIDTH, CAPTURE_HEIGHT = 640, 400

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
    # The OV9281 is monochrome, but MediaPipe expects 3-channel RGB, so
    # capture as RGB888 and let picamera2 handle the conversion.
    config = picam2.create_preview_configuration(
        main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.set_controls({
        "ExposureTime": SHUTTER_US,
        "AnalogueGain": ANALOGUE_GAIN,
    })
    picam2.start()
    time.sleep(2)  # let exposure settle

    print(f"\nRunning {BENCHMARK_FRAMES} frames -- look at the camera!\n")

    capture_times = []
    inference_times = []
    frames_with_face = 0
    start_time = time.time()

    for i in range(BENCHMARK_FRAMES):
        t0 = time.time()
        frame = picam2.capture_array()
        t1 = time.time()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
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
    print(f"Resolution:            {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}")
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
        print("\nNOTE: no face was detected in ANY frame. If you were in view of")
        print("the camera, the image is probably too dark -- try raising")
        print("SHUTTER_US or ANALOGUE_GAIN at the top of this file.")


if __name__ == "__main__":
    main()