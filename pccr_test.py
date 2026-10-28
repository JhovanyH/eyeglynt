"""
EyeGlynt - Up/Down (PCCR) Test
==============================
Checks that the up/down measurement (vy, from the IR glints) actually
changes when you look up and down, BEFORE relying on it in the app.

You look at the top, the middle and the bottom of the screen in turn.
For each, the script records vy for a few seconds and prints:
  - the average vy (where your eyes were looking),
  - the spread (how much it wobbled while you held still),
  - how often a glint was found.
Up/down tracking can work if the three averages are clearly different
and in order (top, middle, bottom), with gaps larger than the wobble.

It also saves a close-up of your eyes at each position (eyes_top.png,
eyes_middle.png, eyes_bottom.png): red marks = the glints it found,
blue marks = the pupil centres it found. These show whether it is
locking onto the right things.

RUN (on the Pi, IR LEDs on, sitting where a user would sit):
    source ~/eyeglynt_env/bin/activate
    cd ~/eyeglynt
    python pccr_test.py
"""

import struct
import time
import zlib

import numpy as np

from gaze_engine import GazeEngine

RECORD_SECONDS = 3.0   # how long to record at each position
SETTLE_SECONDS = 0.5   # ignore the first moment, while the eyes move there


def save_png(filename, image):
    """Saves an RGB numpy image as a PNG file using only Python's own
    libraries (no OpenCV or Pillow needed). A PNG file is a signature
    followed by three chunks: IHDR (size and colour type), IDAT (the
    compressed pixel rows, each starting with a 0 "no filter" byte) and
    IEND (the end)."""
    height, width = image.shape[:2]

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(b"\x00" + image[y].astype(np.uint8).tobytes() for y in range(height))
    with open(filename, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(rows)))
        f.write(chunk(b"IEND", b""))


def record(engine, seconds):
    """Reads vy from the engine for `seconds` seconds.
    Returns (list of vy values, number of readings with a face)."""
    values, face_readings = [], 0
    end = time.time() + seconds
    while time.time() < end:
        feature = engine.get_latest_feature()
        if feature is not None:
            face_readings += 1
            if feature[1] is not None:
                values.append(feature[1])
        time.sleep(1 / 15)  # about the camera's speed; no point reading faster
    return values, face_readings


def main():
    engine = GazeEngine()
    engine.start()

    print("Starting camera and MediaPipe ...")
    start = time.time()
    while engine.get_latest_feature() is None:
        if not engine.is_camera_ok():
            print("No camera found -- check the camera connection.")
            return
        if time.time() - start > 30:
            print("No face found after 30 seconds -- sit in front of the camera.")
            return
        time.sleep(0.2)
    print("Face found.\n")

    results = {}
    for position in ("TOP", "MIDDLE", "BOTTOM"):
        input(f"Look at the {position} of the screen (keep your head still), then press Enter...")
        time.sleep(SETTLE_SECONDS)
        values, face_readings = record(engine, RECORD_SECONDS)
        closeup = engine.get_eye_closeup()
        if closeup is not None:
            filename = f"eyes_{position.lower()}.png"
            save_png(filename, closeup)
            print(f"  saved {filename}")
        if not values:
            print(f"  {position}: no glints found. Are the IR LEDs on and aimed at your eyes?\n")
            return
        results[position] = (np.mean(values), np.std(values))
        print(f"  {position}: average vy = {np.mean(values):+.4f}, "
              f"wobble = {np.std(values):.4f}, "
              f"glint found in {len(values)}/{face_readings} readings\n")

    engine.stop()

    top, middle, bottom = (results[p][0] for p in ("TOP", "MIDDLE", "BOTTOM"))
    wobble = max(results[p][1] for p in results)
    in_order = (top < middle < bottom) or (top > middle > bottom)
    gaps_big_enough = min(abs(middle - top), abs(bottom - middle)) > 2 * wobble

    print("RESULT")
    print(f"  In order top -> middle -> bottom: {'yes' if in_order else 'NO'}")
    print(f"  Gaps larger than twice the wobble: {'yes' if gaps_big_enough else 'NO'}")
    if in_order and gaps_big_enough:
        print("  PASS: up/down can be tracked. Run main.py and calibrate.")
    else:
        print("  CHECK: up/down is too weak or noisy. Try moving the IR LEDs so the")
        print("  glints sit near the pupils, sitting a little closer, and keeping")
        print("  your head still. Send these numbers for help.")


if __name__ == "__main__":
    main()
