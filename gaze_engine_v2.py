"""
EyeGlynt - Gaze Engine, version 2 (jhova's version)
===================================================
The gaze engine for main_v2.py (jhova's version of the app). It
starts as an exact copy of gaze_engine.py, so it works from day one;
change anything inside it freely. main.py and gaze_engine.py (the
team's version) are not affected by changes made here.

RUN
---
    python main.py       the team's version (main.py + gaze_engine.py)
    python main_v2.py    this version (main_v2.py + gaze_engine_v2.py)

main_v2.py calls start, stop, is_camera_ok, get_latest_feature,
get_debug_frame, clear_calibration, add_calibration_point,
fit_calibration, get_screen_x and get_screen_y. If you rename one of
these or change what it takes or returns, update main_v2.py to match.

Below is the original description of how gaze_engine.py works.

EyeGlynt - Gaze Engine
======================
Runs the camera + MediaPipe tracking in its own background thread,
so it never blocks Kivy's UI loop, and exposes a small, simple API:

    engine = GazeEngine()
    engine.start()
    engine.get_latest_feature()   -> (fx, vy, iy, hx) or None; vy, hx may be None
    engine.add_calibration_point(fx, screen_x, vy, screen_y, iy, hx)
    engine.fit_calibration()
    engine.get_screen_x()         -> predicted screen X, or None
    engine.get_screen_y()         -> predicted screen Y, or None

HOW GAZE IS MEASURED
--------------------
Each direction has two measurements:

Iris (MediaPipe landmarks):
  fx (left/right): where the iris sits between the corners of the eye.
  iy (up/down): how far the iris centre sits above or below the line
  between the two eye corners. The eye corners don't move when the eye
  or the eyelids move, so both follow the eye.

Glints, Pupil-Center Corneal Reflection (PCCR):
  vy (up/down) and hx (left/right): the distance from the glints to the
  pupil. The two IR LEDs make small bright reflections ("glints") on the
  cornea. When the eye rotates, the PUPIL moves but the GLINTS stay
  almost still, so the glint-to-pupil distance measures where the eye is
  looking. The pupil and glints are found in the full-size picture, to a
  fraction of a pixel.

Calibration tries, for each direction, the glints, the iris and both
together, and uses whichever fits the calibration dots best. If the
chosen fit needs the glints and none is visible for a moment, the
iris-only fit takes over until they return.

On a laptop webcam there are no IR glints, so vy and hx are None; the
iris measurements still work, and up/down falls back to the mouse if
iy doesn't fit well enough.
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
# is in dioptres = 1 / distance in metres (0.4 m -> 2.5). The picture
# is only sharp within a few centimetres of this distance, so measure
# from the camera to the user's eyes and set it here.
PI_FOCUS_DISTANCE_M = 0.4
# Set this to True only if the camera is mounted upside down, i.e. if
# the app's preview shows your face upside down. read() then turns the
# picture the right way up first, because MediaPipe is made for upright
# faces.
PI_CAMERA_UPSIDE_DOWN = False
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
RIGHT_EYE_LIDS = (159, 145)              # middle of the upper and lower eyelid
LEFT_EYE_LIDS = (386, 374)

# --- PCCR (glint) settings -------------------------------------------
# A glint is a small bright reflection of an IR LED on the cornea. Pixels
# at least GLINT_MIN_BRIGHTNESS bright are grouped into spots of touching
# pixels, and a spot only counts as a glint if it is small: at most
# GLINT_MAX_AREA of the iris's area. Bigger bright patches are skin, the
# wet edge of the eyelid or glasses, not an LED reflection.
GLINT_MIN_BRIGHTNESS = 200   # 0 = black, 255 = white
GLINT_MAX_AREA = 0.1         # 10% of the iris
# The two IR boards make two glints per eye, always the same distance
# apart (the head rests on a support). Once that gap is known, two spots
# only count as the LED pair if their gap is within GLINT_GAP_TOLERANCE
# of it, so a stray reflection (a lamp, the wet eyelid edge) is not
# paired with an LED glint. When only one LED glint is found, the
# middle of the pair worked out from it must be within GLINT_JUMP_LIMIT
# of where the middle was last time. Both are fractions of the iris
# radius (about 2.5 and 5 pixels at 40 cm).
GLINT_GAP_TOLERANCE = 0.1
GLINT_JUMP_LIMIT = 0.2
# The pupil is the darkest round area of the iris. find_pupil() starts
# at the darkest point and keeps the pixels around it that are darker
# than PUPIL_THRESHOLD of the way from that point to the typical iris
# brightness (0.5 = halfway), within PUPIL_MAX_SIZE of the iris radius.
PUPIL_THRESHOLD = 0.5
PUPIL_MAX_SIZE = 0.6
# Search areas, as a fraction of the iris radius. The pupil's darkest
# point must lie in the middle half of the iris, so the dark eyelashes
# and eyelid shadow at the iris edge can't be taken for it. Glints are
# searched for only near the pupil: while the user looks at the screen,
# an LED near the camera always reflects close to the pupil, and the
# shiny edge of the eyelid is further out.
PUPIL_SEARCH_SCALE = 0.9
PUPIL_SEED_SCALE = 0.5
GLINT_SEARCH_SCALE = 0.75
# If a measurement is missing for this many frames in a row (about 1
# second), it is reported as unknown instead of keeping an old value.
# Shorter gaps, such as a blink or a glint lost for a moment, simply
# keep the last value.
MAX_MISSED_FRAMES = 15

# --- Blinks ----------------------------------------------------------
# During a blink the iris is hidden and MediaPipe only guesses where it
# is (usually too low), which would pull the pointer down. Each frame,
# the eye's openness (gap between the eyelids / eye width) is compared
# with its usual openness (the median of the last OPENNESS_HISTORY
# frames, about 2 seconds); below BLINK_OPENNESS of that, the frame is a
# blink and the last measurements are kept. Looking down at the bottom
# of the screen also narrows the eyes, so a blink must be a deep closing
# (BLINK_OPENNESS), and short: a blink lasts at most about 0.4 seconds
# (BLINK_MAX_FRAMES), so eyes that stay narrow longer than that are
# looking down, and are measured as usual.
BLINK_OPENNESS = 0.35
BLINK_MAX_FRAMES = 5
OPENNESS_HISTORY = 30

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

# --- Distance correction ---------------------------------------------
# Calibration is done at one distance from the screen. If the user later
# sits farther away, the screen looks smaller from where they are, so
# their eyes turn less to reach its edges and the pointer would be
# pulled toward the middle (closer: pushed outward). The face's size in
# the picture shows the distance -- twice as far looks half as big -- so
# the pointer's distance from the middle of the calibration dots is
# multiplied by
#     face size during calibration / face size now
# which equals  distance now / distance during calibration.
# The face size is the distance between the two OUTER eye corners
# (stable landmarks, about 9 cm apart on an adult face). The correction
# assumes the user faces the middle of the screen.
FACE_SIZE_CORNERS = (33, 263)
# The correction is kept within this range, so one bad face reading
# can't throw the pointer across the screen.
DISTANCE_SCALE_LIMITS = (0.7, 1.4)


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


def _face_size(landmarks, w, h):
    """Distance in pixels between the two outer eye corners: the face's
    size in the picture, used for the distance correction."""
    a, b = landmarks[FACE_SIZE_CORNERS[0]], landmarks[FACE_SIZE_CORNERS[1]]
    return float(np.hypot((a.x - b.x) * w, (a.y - b.y) * h))


def _compute_iris_height(landmarks, w, h):
    """iy: how far the iris centre sits below the line between the two
    eye corners, divided by the eye's width, averaged over both eyes
    (bigger = looking further down). The corners stay put when the eye
    or the eyelids move, so this follows the eye itself. (The old method
    measured the iris against the eyelids instead, and failed because
    the eyelids move up and down together with the eye.)"""
    total = 0.0
    for iris_idx, (c1, c2) in ((RIGHT_IRIS_CENTER, RIGHT_EYE_CORNERS),
                               (LEFT_IRIS_CENTER, LEFT_EYE_CORNERS)):
        a, b, iris = landmarks[c1], landmarks[c2], landmarks[iris_idx]
        eye_width = abs(b.x - a.x) * w
        corner_y = (a.y + b.y) / 2 * h          # height of the line between the corners
        if eye_width > 0:
            total += (iris.y * h - corner_y) / eye_width
    return total / 2


def _crop_square(image, cx, cy, radius):
    """Cuts a square of side 2*radius centred on (cx, cy) out of `image`,
    clipped at the image edges. Returns the crop and the position of its
    top-left corner in the full image."""
    h, w = image.shape[:2]
    x0, x1 = max(0, int(cx - radius)), min(w, int(cx + radius) + 1)
    y0, y1 = max(0, int(cy - radius)), min(h, int(cy + radius) + 1)
    return image[y0:y1, x0:x1], x0, y0


def find_glints(gray, cx, cy, radius, max_area):
    """Finds the IR glints inside the circle of `radius` around the pupil
    centre (cx, cy).

    1. Only pixels inside the circle are searched, so bright spots
       further out (eyelid skin, the wet edge of the lower eyelid) are
       not taken for glints.
    2. Pixels at least GLINT_MIN_BRIGHTNESS bright are grouped into
       spots of touching pixels.
    3. Spots bigger than `max_area` pixels are not LED reflections and
       are dropped.
    4. Each spot's centre is the average of its pixel positions, weighted
       by how bright each pixel is. This places the centre to a fraction
       of a pixel, which matters because looking from the top to the
       bottom of the screen moves the pupil only a few pixels.

    Returns a list of glint centres (x, y), one per glint, in the
    coordinates of `gray`. An empty list means no glint (IR LEDs off).
    """
    crop, x0, y0 = _crop_square(gray, cx, cy, radius)
    if crop.size == 0:
        return []
    rows, cols = np.ogrid[:crop.shape[0], :crop.shape[1]]
    inside = (cols + x0 - cx) ** 2 + (rows + y0 - cy) ** 2 <= radius ** 2
    bright = inside & (crop >= GLINT_MIN_BRIGHTNESS)
    glints = []
    for spot in _group_touching(bright):
        if len(spot) > max_area:
            continue  # too big to be an LED reflection
        ys = np.array([pixel[0] for pixel in spot])
        xs = np.array([pixel[1] for pixel in spot])
        # Brighter pixels count more; the dimmest allowed pixel counts 1.
        weights = crop[ys, xs].astype(float) - GLINT_MIN_BRIGHTNESS + 1
        glints.append((x0 + np.average(xs, weights=weights),
                       y0 + np.average(ys, weights=weights)))
    return glints


def _group_touching(mask):
    """Groups the True pixels of `mask` into spots of touching pixels
    (diagonal neighbours count as touching). Returns a list of spots,
    each a list of (row, column) pixels."""
    remaining = set(zip(*np.nonzero(mask)))
    spots = []
    while remaining:
        to_visit = [remaining.pop()]  # start a new spot from any leftover pixel
        spot = []
        while to_visit:
            row, col = to_visit.pop()
            spot.append((row, col))
            # Add its bright neighbours to the same spot.
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    neighbour = (row + d_row, col + d_col)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        to_visit.append(neighbour)
        spots.append(spot)
    return spots


def _neighbourhood(image, size, combine):
    """Combines each pixel with the (2*size+1) x (2*size+1) square around
    it. combine="mean" blurs (removes single-pixel noise), "min" takes the
    darkest neighbour, "max" the brightest."""
    padded = np.pad(image.astype(float), size, mode="edge")
    h, w = image.shape
    shifted = [padded[dy:dy + h, dx:dx + w]
               for dy in range(2 * size + 1) for dx in range(2 * size + 1)]
    if combine == "mean":
        return np.mean(shifted, axis=0)
    if combine == "min":
        return np.min(shifted, axis=0)
    return np.max(shifted, axis=0)


def _remove_glints(crop):
    """Paints over the glints, working inwards from their edges.

    A glint often sits ON the pupil. Left alone, it would punch a bright
    hole in the pupil and pull the pupil centre away. Each round, every
    glint pixel that touches a non-glint pixel takes the darkest of those
    neighbours, so a glint on the pupil becomes pupil-dark and a glint on
    the iris becomes iris-coloured.
    """
    # Glint pixels, plus 1 pixel of glow around them.
    glint = _neighbourhood(crop >= GLINT_MIN_BRIGHTNESS, 1, "max") > 0
    result = crop.astype(float)
    for _ in range(10):  # 10 rounds fills glints up to 20 pixels across
        if not glint.any():
            break
        # Darkest NON-glint neighbour of every pixel (glint pixels count as white).
        darkest_neighbour = _neighbourhood(np.where(glint, 255, result), 1, "min")
        edge = glint & (darkest_neighbour < 255)  # glint pixels next to a non-glint pixel
        result[edge] = darkest_neighbour[edge]
        glint = glint & ~edge
    return result


def find_pupil(gray, cx, cy, iris_radius):
    """Finds the pupil centre near the iris centre (cx, cy) in a
    grayscale image.

    1. Paint over the glints, blur the area slightly (to remove noise)
       and find its darkest point in the middle of the iris (within
       PUPIL_SEED_SCALE of the iris radius): that point is inside the
       pupil.
    2. Keep the pixels near that point, inside PUPIL_SEARCH_SCALE of the
       iris radius, that are clearly darker than the iris (see
       PUPIL_THRESHOLD); together they are the pupil.
    3. Return their centre, in the coordinates of `gray`, or None.

    Starting from the darkest point in the middle of the iris (instead of
    using every dark pixel in the iris) means eyelashes or eyelid shadows
    at the iris edge are not mixed in with the pupil.
    """
    radius = iris_radius * PUPIL_SEARCH_SCALE
    crop, x0, y0 = _crop_square(gray, cx, cy, radius)
    if crop.size == 0:
        return None
    smooth = _neighbourhood(_remove_glints(crop), 2, "mean")
    rows, cols = np.ogrid[:crop.shape[0], :crop.shape[1]]
    distance_sq = (cols + x0 - cx) ** 2 + (rows + y0 - cy) ** 2
    inside = distance_sq <= radius ** 2
    middle = distance_sq <= (iris_radius * PUPIL_SEED_SCALE) ** 2
    if not middle.any():
        return None

    # Step 1: darkest point in the middle of the iris (other pixels count as white).
    in_circle = np.where(middle, smooth, 255)
    seed_y, seed_x = np.unravel_index(np.argmin(in_circle), in_circle.shape)
    darkest = in_circle[seed_y, seed_x]
    typical = np.median(smooth[inside])  # typical brightness of the iris

    # Step 2: pupil = pixels near the seed that are darker than the limit.
    limit = darkest + PUPIL_THRESHOLD * (typical - darkest)
    near_seed = (cols - seed_x) ** 2 + (rows - seed_y) ** 2 <= (radius * PUPIL_MAX_SIZE) ** 2
    ys, xs = np.nonzero(inside & near_seed & (smooth <= limit))
    if len(xs) == 0:
        return None
    return (x0 + xs.mean(), y0 + ys.mean())


def _eye_openness(landmarks, w, h):
    """How open the eyes are: the gap between the upper and lower eyelid
    divided by the eye's width, averaged over both eyes."""
    total = 0.0
    for (upper, lower), (c1, c2) in ((RIGHT_EYE_LIDS, RIGHT_EYE_CORNERS),
                                     (LEFT_EYE_LIDS, LEFT_EYE_CORNERS)):
        gap = abs(landmarks[lower].y - landmarks[upper].y) * h
        eye_width = abs(landmarks[c2].x - landmarks[c1].x) * w
        if eye_width > 0:
            total += gap / eye_width
    return total / 2


class _Smoother:
    """Cleans up one measurement, frame by frame.

    1. Spike filter: the middle value (median) of the last 3 readings,
       so a single bad reading, such as a wrong glint, is ignored.
    2. Exponential moving average (SMOOTHING_ALPHA) to calm the jitter.
    3. When a reading is missing (None: no glint, or a blink), the last
       value is kept for up to MAX_MISSED_FRAMES frames, then forgotten.

    `value` is the cleaned-up measurement, or None if there is none.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = None
        self._recent = []   # the last 3 readings
        self._missed = 0    # missing readings in a row

    def update(self, reading):
        if reading is None:
            self._missed += 1
            if self._missed > MAX_MISSED_FRAMES:
                self.reset()  # gone for too long: forget the old value
            return
        self._missed = 0
        self._recent = (self._recent + [reading])[-3:]
        spike_free = float(np.median(self._recent))
        if self.value is None:
            self.value = spike_free  # first reading: nothing to blend with yet
        else:
            self.value = SMOOTHING_ALPHA * spike_free + (1 - SMOOTHING_ALPHA) * self.value


class _Eye:
    """PCCR for one eye: finds its pupil and glints in each picture and
    keeps its own smoothed measurements:

        vy = (pupil y - glint y) / eye width     (up/down)
        hx = (pupil x - glint x) / eye width     (left/right)

    Dividing by the eye's width means they don't change when the user
    sits nearer or further away. "glint" is the point halfway between
    the two glints (see _glint_middle).
    """

    def __init__(self, iris_center, iris_edge, corners):
        self.iris_center = iris_center
        self.iris_edge = iris_edge
        self.corners = corners
        self.vy = _Smoother()
        self.hx = _Smoother()
        # Remembered about the glints (see _glint_middle):
        self.half_gap = None          # from the middle of the pair to the right-hand glint
        self.middle = None            # where the middle of the pair was last time
        self.frames_without_pair = 0  # pictures in a row in which the pair wasn't found

    def measure(self, frame, landmarks):
        """Measures this eye in the full-size picture and updates vy and
        hx. Returns the glints and pupil found, as ("glint"/"pupil", x, y)
        points (full-picture pixels) to draw on the preview."""
        h, w = frame.shape[:2]

        def pixel(i):
            """Landmark i in pixels (MediaPipe gives 0-1 fractions)."""
            return landmarks[i].x * w, landmarks[i].y * h

        cx, cy = pixel(self.iris_center)
        # Iris radius: average distance from the iris centre to its 4 edge points.
        iris_radius = np.mean([np.hypot(pixel(i)[0] - cx, pixel(i)[1] - cy) for i in self.iris_edge])
        (ax, ay), (bx, by) = pixel(self.corners[0]), pixel(self.corners[1])
        eye_width = np.hypot(ax - bx, ay - by)
        if iris_radius < 2 or eye_width < 10:
            self.skip()  # eye too small in the picture to measure
            return []

        # Grayscale (average of the colour channels) of only the small
        # area around this eye -- much faster than converting the whole
        # picture. It is big enough for the glint search around the
        # pupil, which can be near the iris edge. Positions found in it
        # are relative to its corner (x0, y0); adding x0, y0 gives
        # positions in the full picture.
        search_radius = iris_radius * (PUPIL_SEARCH_SCALE + GLINT_SEARCH_SCALE)
        box, x0, y0 = _crop_square(frame, cx, cy, search_radius + 2)
        gray = box.mean(axis=2) if box.ndim == 3 else box
        pupil = find_pupil(gray, cx - x0, cy - y0, iris_radius)
        if pupil is None:
            self.skip()
            return []
        glint_area = GLINT_MAX_AREA * np.pi * iris_radius ** 2
        glints = find_glints(gray, pupil[0], pupil[1], iris_radius * GLINT_SEARCH_SCALE, glint_area)
        pupil = (pupil[0] + x0, pupil[1] + y0)
        glints = [(gx + x0, gy + y0) for gx, gy in glints]
        points = [("glint", gx, gy) for gx, gy in glints] + [("pupil", pupil[0], pupil[1])]

        middle = self._glint_middle(glints, pupil, iris_radius)
        if middle is None:
            self.skip()
            return points
        self.vy.update((pupil[1] - middle[1]) / eye_width)
        self.hx.update((pupil[0] - middle[0]) / eye_width)
        return points

    def skip(self):
        """No measurement this frame (blink, no glint): keep the last one."""
        self.vy.update(None)
        self.hx.update(None)

    def _glint_middle(self, glints, pupil, iris_radius):
        """The point halfway between the two LED glints, or None.

        If one glint is missing (hidden by the eyelid, or it slipped off
        the cornea), the middle is worked out from the other glint and
        the remembered gap between the two. Simply using the glint that
        is left would make the measurement jump by half that gap, which
        is more than its whole top-to-bottom range.
        """
        if not glints:
            return None
        pair = self._find_pair(glints, pupil, iris_radius)
        if pair is not None:
            left, right = pair
            middle = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
            gap = ((right[0] - left[0]) / 2, (right[1] - left[1]) / 2)
            if self.half_gap is None or self.frames_without_pair > MAX_MISSED_FRAMES:
                self.half_gap = gap  # first pair, or the old one is out of date
            else:
                # Averaged over recent pictures, so one noisy picture doesn't matter.
                self.half_gap = tuple(SMOOTHING_ALPHA * new + (1 - SMOOTHING_ALPHA) * old
                                      for new, old in zip(gap, self.half_gap))
            self.frames_without_pair = 0
        else:
            self.frames_without_pair += 1
            if self.half_gap is None or self.middle is None:
                # Only one LED in use, or no pair seen yet: use the glint nearest the pupil.
                middle = min(glints, key=lambda g: np.hypot(g[0] - pupil[0], g[1] - pupil[1]))
            else:
                # Each glint is either the left one of the pair (the middle
                # is half a gap to its right), the right one (half a gap
                # to its left), or both glints merged into one spot (the
                # middle is the spot itself). Use the guess nearest to the
                # last middle.
                dx, dy = self.half_gap
                guesses = [(gx + dx, gy + dy) for gx, gy in glints] + \
                          [(gx - dx, gy - dy) for gx, gy in glints] + list(glints)
                last_x, last_y = self.middle
                middle = min(guesses, key=lambda m: np.hypot(m[0] - last_x, m[1] - last_y))
                if np.hypot(middle[0] - last_x, middle[1] - last_y) > GLINT_JUMP_LIMIT * iris_radius:
                    return None  # no glint where one is expected: can't tell
        self.middle = middle
        return middle

    def _find_pair(self, glints, pupil, iris_radius):
        """The two LED glints as (left, right), or None.

        Until the gap between them is known (or if the pair hasn't been
        found for MAX_MISSED_FRAMES pictures), it is learned from a
        picture with exactly two glints, which are then almost certainly
        the two LEDs. After that, a pair must be the same distance apart
        as before (within GLINT_GAP_TOLERANCE), checking the glints
        nearest the pupil first.
        """
        if self.half_gap is None or self.frames_without_pair > MAX_MISSED_FRAMES:
            if len(glints) != 2:
                return None
            return tuple(sorted(glints))  # sorted left to right (by x)
        nearest = sorted(glints, key=lambda g: np.hypot(g[0] - pupil[0], g[1] - pupil[1]))
        for i, first in enumerate(nearest):
            for second in nearest[i + 1:]:
                left, right = sorted((first, second))
                gap_change = np.hypot((right[0] - left[0]) / 2 - self.half_gap[0],
                                      (right[1] - left[1]) / 2 - self.half_gap[1]) * 2
                if gap_change <= GLINT_GAP_TOLERANCE * iris_radius:
                    return left, right
        return None


def _average_known(values):
    """Average of the values that are not None, or None if there are none."""
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None


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


def _fit_plane(inputs_a, inputs_b, outputs):
    """Least squares with two inputs: output = a1 * input_a + a2 * input_b + b.
    Returns (a1, a2, b, r2), with r2 as in _fit_line."""
    p = np.array(inputs_a, dtype=float)
    q = np.array(inputs_b, dtype=float)
    y = np.array(outputs, dtype=float)
    a_matrix = np.vstack([p, q, np.ones_like(p)]).T
    a1, a2, b = np.linalg.lstsq(a_matrix, y, rcond=None)[0]
    predicted = a1 * p + a2 * q + b
    total = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum((y - predicted) ** 2) / total if total > 0 else 0.0
    return float(a1), float(a2), float(b), float(r2)


def _adjusted_r2(r2, points, inputs):
    """R squared lowered a little for every input a fit uses. A fit with
    more inputs always matches the calibration dots a bit better, even
    when the extra input is just noise; this makes the comparison fair.
    (The standard "adjusted R squared".)"""
    if points - inputs - 1 <= 0:
        return 0.0
    return 1.0 - (1.0 - r2) * (points - 1) / (points - inputs - 1)


def _fit_axis(points):
    """Fits one direction (left/right or up/down) from the calibration
    points, each (glint value or None, iris value, screen position).

    Tries three straight-line fits -- glints only, iris only, and both
    together -- and returns (best fit, iris-only fit). Each fit is a dict
    with "name", the slope for each input it uses ("glint" and/or
    "iris"), "b" and "r2" (adjusted, see _adjusted_r2), or None if that
    fit wasn't possible (dots at fewer than 2 different positions). The
    iris-only fit is the backup for moments when no glint is seen.
    """
    fits = []
    iris_fit = None
    glint_points = [p for p in points if p[0] is not None]
    # The glint fits are only tried if glints were seen at 2 out of 3 dots
    # or more; fitted to fewer dots, a line can fit well just by chance.
    if (len(glint_points) >= 2 * len(points) / 3
            and len({round(screen) for g, i, screen in glint_points}) >= 2):
        c, d, r2 = _fit_line([g for g, i, screen in glint_points],
                             [screen for g, i, screen in glint_points])
        fits.append({"name": "glints", "glint": c, "b": d,
                     "r2": _adjusted_r2(r2, len(glint_points), 1)})
        c1, c2, d, r2 = _fit_plane([g for g, i, screen in glint_points],
                                   [i for g, i, screen in glint_points],
                                   [screen for g, i, screen in glint_points])
        fits.append({"name": "glints + iris", "glint": c1, "iris": c2, "b": d,
                     "r2": _adjusted_r2(r2, len(glint_points), 2)})
    if len({round(screen) for g, i, screen in points}) >= 2:
        c, d, r2 = _fit_line([i for g, i, screen in points], [screen for g, i, screen in points])
        iris_fit = {"name": "iris", "iris": c, "b": d,
                    "r2": _adjusted_r2(r2, len(points), 1)}
        fits.append(iris_fit)
    if not fits:
        return None, None
    print("[Calibration]   fits: " + ", ".join(f"{f['name']} {f['r2']:.2f}" for f in fits))
    return max(fits, key=lambda f: f["r2"]), iris_fit


def _predict(fit, backup, glint_value, iris_value):
    """Screen position from a fit made by _fit_axis. If the fit needs the
    glints and none is seen right now, the backup (iris-only) fit is
    used instead. None if neither can be used."""
    if "glint" in fit and glint_value is None:
        fit = backup
        if fit is None:
            return None
    position = fit["b"]
    if "glint" in fit:
        position += fit["glint"] * glint_value
    if "iris" in fit:
        position += fit["iris"] * iris_value
    return position


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
        if PI_CAMERA_UPSIDE_DOWN:
            frame = frame[::-1, ::-1]  # rotate 180 degrees: flip top-bottom and left-right
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
        self._latest_feature = None  # (fx, vy, iy, hx), or None if no face seen
        # Cleaned-up measurements (see _Smoother), background-thread only.
        self._fx = _Smoother()
        self._iy = _Smoother()
        self._face_size = _Smoother()
        self._eyes = [_Eye(RIGHT_IRIS_CENTER, RIGHT_IRIS_EDGE, RIGHT_EYE_CORNERS),
                      _Eye(LEFT_IRIS_CENTER, LEFT_IRIS_EDGE, LEFT_EYE_CORNERS)]
        self._openness_history = []  # eye openness in recent frames, for the blink check
        self._closed_frames = 0      # frames in a row with the eyes nearly closed
        self._latest_face_size = None  # smoothed face size, shared with the Kivy thread
        self._latest_debug_frame = None  # RGB numpy frame with dots drawn, for a live preview
        self._latest_eyes = None  # (full frame, glint/pupil points, iris centres) for close-ups
        self._running = False
        self._thread = None
        self._camera_ok = True  # flips False if the camera can't be opened

        # Calibration points, and the fitted straight lines once
        # fit_calibration() runs (see _fit_axis). A fit is None if that
        # direction isn't usable; the backups are the iris-only fits.
        self._x_points = []  # (hx, fx, screen_x); hx is None if no glint was seen
        self._y_points = []  # (vy, iy, screen_y); vy is None if no glint was seen
        self._x_fit = None
        self._y_fit = None
        self._x_backup = None
        self._y_backup = None

        # Distance correction (see the settings at the top): the face size
        # at each calibration dot, their average once fitted, and the
        # middle of the calibration dots on screen.
        self._calib_face_sizes = []
        self._calib_face_size = None
        self._centre_x = None
        self._centre_y = None

    # ---- lifecycle ---------------------------------------------------
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _is_blink(self, landmarks, w, h):
        """True if the eyes are much less open than usual, for no longer
        than a blink lasts (see the blink settings at the top)."""
        openness = _eye_openness(landmarks, w, h)
        self._openness_history = (self._openness_history + [openness])[-OPENNESS_HISTORY:]
        nearly_closed = openness < BLINK_OPENNESS * float(np.median(self._openness_history))
        self._closed_frames = self._closed_frames + 1 if nearly_closed else 0
        return 0 < self._closed_frames <= BLINK_MAX_FRAMES

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
            eyes = None
            if not result.face_landmarks:
                feature = None
                # Reset so re-acquiring doesn't drag from a stale position.
                for smoother in [self._fx, self._iy, self._face_size] + \
                        [eye.vy for eye in self._eyes] + [eye.hx for eye in self._eyes]:
                    smoother.reset()
            else:
                landmarks = result.face_landmarks[0]
                self._face_size.update(_face_size(landmarks, sw, sh))
                points = []
                if self._is_blink(landmarks, sw, sh):
                    # Keep the last measurements until the eyes open again.
                    self._fx.update(None)
                    self._iy.update(None)
                    for eye in self._eyes:
                        eye.skip()
                else:
                    self._fx.update(_compute_horizontal_feature(landmarks, sw, sh))
                    self._iy.update(_compute_iris_height(landmarks, sw, sh))
                    for eye in self._eyes:
                        points += eye.measure(frame, landmarks)
                fh, fw = frame.shape[:2]
                iris_centres = [(landmarks[i].x * fw, landmarks[i].y * fh)
                                for i in (RIGHT_IRIS_CENTER, LEFT_IRIS_CENTER)]
                eyes = (frame, points, iris_centres)

                # Each eye's glint measurements are smoothed separately and
                # then averaged, so the result doesn't jump when one eye's
                # glints are lost for a moment.
                vy = _average_known([eye.vy.value for eye in self._eyes])
                hx = _average_known([eye.hx.value for eye in self._eyes])
                if self._fx.value is None:
                    feature = None  # eyes closed for over a second
                else:
                    feature = (self._fx.value, vy, self._iy.value, hx)

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
                self._latest_face_size = self._face_size.value
                self._latest_debug_frame = debug_frame
                self._latest_eyes = eyes

        camera.close()

    # ---- public API used by the Kivy app ------------------------------
    def is_camera_ok(self):
        return self._camera_ok

    def get_latest_feature(self):
        """Returns (fx, vy, iy, hx) or None if no face is currently
        detected. vy and hx are None when no glint is visible (no IR, or
        a laptop webcam)."""
        with self._lock:
            return self._latest_feature

    def get_debug_frame(self):
        """Returns the latest camera frame (RGB numpy array, with iris,
        glint and pupil dots drawn on it) for showing a live preview, or
        None if unavailable."""
        with self._lock:
            return self._latest_debug_frame

    def get_eye_closeup(self, half_size=40, zoom=4):
        """Returns a picture of both eyes side by side, enlarged `zoom`
        times, with the glints marked in red and the pupils in blue, for
        checking what the PCCR measurement is locking onto. None if no
        face is currently detected."""
        with self._lock:
            eyes = self._latest_eyes
        if eyes is None:
            return None
        frame, points, iris_centres = eyes
        size = 2 * half_size
        crops = []
        for cx, cy in iris_centres:
            crop = np.zeros((size, size, 3), dtype=np.uint8)
            part, x0, y0 = _crop_square(frame, cx, cy, half_size)
            crop[:part.shape[0], :part.shape[1]] = part[:size, :size]
            for kind, px, py in points:
                lx, ly = int(round(px - x0)), int(round(py - y0))
                if 0 <= lx < size and 0 <= ly < size:
                    crop[ly, lx] = (255, 0, 0) if kind == "glint" else (0, 80, 255)
            crops.append(crop)
        both = np.concatenate(crops, axis=1)
        # Enlarge: repeat every pixel `zoom` times down and across.
        return np.repeat(np.repeat(both, zoom, axis=0), zoom, axis=1)

    def is_calibrated(self):
        return self._x_fit is not None

    def is_vertical_calibrated(self):
        return self._y_fit is not None

    def add_calibration_point(self, fx, screen_x, vy=None, screen_y=None, iy=None, hx=None):
        """Stores one calibration dot's measurements. vy and hx are None
        when no glint was seen at that dot. The current face size is
        stored too, as the reference for the distance correction."""
        self._x_points.append((hx, fx, screen_x))
        with self._lock:
            face_size = self._latest_face_size
        if face_size is not None:
            self._calib_face_sizes.append(face_size)
        if screen_y is not None and iy is not None:
            self._y_points.append((vy, iy, screen_y))

    def clear_calibration(self):
        self._x_points = []
        self._y_points = []
        self._x_fit = None
        self._y_fit = None
        self._x_backup = None
        self._y_backup = None
        self._calib_face_sizes = []
        self._calib_face_size = None

    def fit_calibration(self):
        """Fits left/right and up/down (see _fit_axis): each uses the
        glints, the iris or both, whichever fits the calibration dots best.

        Returns a short text summary for the user, one line per
        direction. Raises ValueError if the left/right fit isn't possible
        or isn't good enough (e.g. no camera, or no face seen during
        calibration)."""
        if len(self._x_points) < 2:
            raise ValueError(f"Need at least 2 calibration points, got {len(self._x_points)}")
        print(f"[Calibration] left/right, {len(self._x_points)} points:")
        self._x_fit, self._x_backup = _fit_axis(self._x_points)
        if self._x_fit is None:
            raise ValueError("left/right: dots seen at fewer than 2 positions")
        if self._x_fit["r2"] < MIN_FIT_R2:
            weak = self._x_fit["r2"]
            self._x_fit = None
            raise ValueError(f"left/right fit too weak (R^2 = {weak:.2f})")
        summary = f"Left/right fit {self._x_fit['r2']:.2f} ({self._x_fit['name']})"

        # References for the distance correction.
        self._centre_x = float(np.mean([p[2] for p in self._x_points]))
        if self._y_points:
            self._centre_y = float(np.mean([p[2] for p in self._y_points]))
        if self._calib_face_sizes:
            self._calib_face_size = float(np.mean(self._calib_face_sizes))
            print(f"[Calibration] face size {self._calib_face_size:.0f} px (distance reference)")

        print(f"[Calibration] up/down, {len(self._y_points)} points:")
        self._y_fit, self._y_backup = _fit_axis(self._y_points)
        if self._y_fit is None:
            return summary + "\nUp/down: not enough data, using the mouse"
        if self._y_fit["r2"] < MIN_FIT_R2:
            weak = self._y_fit["r2"]
            self._y_fit = None
            return summary + f"\nUp/down fit {weak:.2f} (too weak, using the mouse)"
        return summary + f"\nUp/down fit {self._y_fit['r2']:.2f} ({self._y_fit['name']})"

    def _usable_backup(self, backup):
        """The iris-only backup fit, if it is good enough to use."""
        return backup if backup is not None and backup["r2"] >= MIN_FIT_R2 else None

    def _distance_scale(self):
        """distance now / distance during calibration, from the face size
        (see the settings at the top). 1.0 if either size is unknown."""
        with self._lock:
            face_size = self._latest_face_size
        if self._calib_face_size is None or not face_size:
            return 1.0
        low, high = DISTANCE_SCALE_LIMITS
        return min(high, max(low, self._calib_face_size / face_size))

    def get_screen_x(self):
        """Returns the predicted screen X from the current gaze feature,
        or None if not calibrated or no face is currently detected."""
        if self._x_fit is None:
            return None
        feature = self.get_latest_feature()
        if feature is None:
            return None
        fx, _vy, _iy, hx = feature
        x = _predict(self._x_fit, self._usable_backup(self._x_backup), hx, fx)
        if x is None:
            return None
        # Distance correction: stretch or shrink around the middle.
        return self._centre_x + (x - self._centre_x) * self._distance_scale()

    def get_screen_y(self):
        """Returns the predicted screen Y (Kivy: 0 = bottom) from the
        current up/down measurements, or None if up/down isn't
        calibrated, no face is detected, or the fit uses the glints, no
        glint is currently visible and the iris alone isn't good enough."""
        if self._y_fit is None:
            return None
        feature = self.get_latest_feature()
        if feature is None:
            return None
        _fx, vy, iy, _hx = feature
        y = _predict(self._y_fit, self._usable_backup(self._y_backup), vy, iy)
        if y is None:
            return None
        # Distance correction: stretch or shrink around the middle.
        return self._centre_y + (y - self._centre_y) * self._distance_scale()
