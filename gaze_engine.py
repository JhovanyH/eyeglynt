"""EyeGlynt gaze engine: camera + MediaPipe tracking in a background thread.

Measurements (averaged over both eyes):
  fx      left/right: the iris (or pupil) between the eye corners
  iy      up/down: the iris (or pupil) height against the eye corners
  vy, hx  up/down and left/right by PCCR: pupil centre minus the IR glints
Calibration fits each direction with the glints, the iris or both, whichever
fits best; while no glint is seen, the iris-only fit is used.

Main calls: start(), add_calibration_point(), fit_calibration(),
get_screen_x(), get_screen_y().
"""

import logging
import os
import subprocess
import threading
import time
import urllib.request
import warnings

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Camera libraries: OpenCV on a laptop, picamera2 on the Pi (see _open_camera).
try:
    import cv2
except ImportError:
    cv2 = None

try:
    from picamera2 import Picamera2
    from libcamera import controls as libcamera_controls  # autofocus mode names
except ImportError:
    Picamera2 = None

MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# --- Raspberry Pi camera (Camera Module 3 NoIR) ---
# Automatic exposure. The lens stays fixed while tracking; FOCUS_ON_EYES finds
# the position where the eyes are sharpest (in dioptres) when calibration starts.
# The camera's own autofocus was not used: it focused on the wrong area, and
# IR light focuses at a different position than visible light.
PI_LENS_POSITION = 4.25
FOCUS_ON_EYES = True
FOCUS_LENS_POSITIONS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
FOCUS_FINE_STEP = 0.125
# Pictures to skip while the lens moves, before measuring sharpness.
FOCUS_MAX_WAIT_PICTURES = 10
FOCUS_SETTLE_PICTURES = 3
# Lens focal length in mm, for the face distance (Module 3 Wide: 2.75, set automatically).
PI_LENS_FOCAL_MM = 4.74
PI_SENSOR_PIXEL_MM = 0.0014  # 1.4 micrometres
IRIS_DIAMETER_MM = 11.7  # about the same in all adults
# True only if the camera is mounted upside down (the preview shows the face upside down).
PI_CAMERA_UPSIDE_DOWN = False
# Full-view mode; the eyes need the detail. MediaPipe gets a half-size copy.
CAPTURE_WIDTH, CAPTURE_HEIGHT = 1536, 864
PI_MEDIAPIPE_STEP = 2  # MediaPipe uses every 2nd pixel

# MediaPipe landmark numbers (checked in gaze_test.py).
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_EDGE = [469, 470, 471, 472]  # 4 points on the iris edge
LEFT_IRIS_EDGE = [474, 475, 476, 477]
RIGHT_EYE_CORNERS = (33, 133)  # outer and inner corner
LEFT_EYE_CORNERS = (362, 263)
RIGHT_EYE_CONTOUR = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]
LEFT_EYE_CONTOUR = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]
RIGHT_EYE_LIDS = (159, 145)  # middle of the upper and lower eyelid
LEFT_EYE_LIDS = (386, 374)

# --- PCCR (glints) ---
# A glint is the IR LED's reflection on the cornea: a small spot at least
# GLINT_MIN_BRIGHTNESS bright and at most GLINT_MAX_AREA of the iris's area.
GLINT_MIN_BRIGHTNESS = 200   # 0 = black, 255 = white
GLINT_MAX_AREA = 0.1  # share of the iris's area
# A glint must also be this much brighter than the pixels around it (eyelid skin is not).
GLINT_MIN_CONTRAST = 25
# The two LED glints keep the same gap; a pair must match it within
# GLINT_GAP_TOLERANCE, and a single glint must be within GLINT_JUMP_LIMIT of the
# last middle (both as shares of the iris radius).
GLINT_GAP_TOLERANCE = 0.1
GLINT_JUMP_LIMIT = 0.2
# Pupil (the dark disc in the iris), found by find_pupil(). Sizes are shares of
# the iris radius.
PUPIL_MIDDLE_SCALE = 0.5
PUPIL_RING = (0.75, 0.95)
PUPIL_MAX_SIZE = 0.65
PUPIL_MIN_CONTRAST = 15
# Glints brighter than the iris by this much are painted over before finding the pupil.
PUPIL_GLINT_CONTRAST = 20
# Use the pupil centre found in the picture for fx and iy, instead of MediaPipe's
# iris centre (more accurate on sharp IR pictures). Falls back to MediaPipe if
# the pupil isn't found.
USE_PUPIL = True
# Glints are searched for only this close to the pupil (share of the iris radius).
GLINT_SEARCH_SCALE = 0.75
# A missing measurement keeps its last value this long (rides out a blink),
# then counts as unknown. Glints are dropped sooner so the iris fit takes over.
IRIS_HOLD_TIME = 1.0    # seconds
GLINT_HOLD_TIME = 0.3   # seconds
# Pictures without the glint pair before its gap is forgotten.
MAX_MISSED_FRAMES = 15
# Pictures in a row with rejected glints before the glints' last place is forgotten.
GLINT_RESTART_FRAMES = 3

# --- Blinks ---
# A blink: the eyes much less open than usual (BLINK_OPENNESS), for at most
# BLINK_MAX_FRAMES. The last measurements are kept during a blink.
BLINK_OPENNESS = 0.35
BLINK_MAX_FRAMES = 5
OPENNESS_HISTORY = 30

# Smoothing of the gaze measurements, in seconds (same at any frame rate).
SMOOTHING_TIME = 0.4    # seconds
# Per-picture averaging for what the eyes remember (pupil offset, glint gap).
SMOOTHING_ALPHA = 0.25

# Resting screen: process only this many pictures a second.
LOW_POWER_FPS = 3.0  # pictures a second

# A direction is used only if its calibration fit has at least this R squared.
MIN_FIT_R2 = 0.6

# --- Distance correction ---
# If the user sits nearer or farther than during calibration, the pointer is
# scaled around the middle by the face size (outer eye corners) then and now.
FACE_SIZE_CORNERS = (33, 263)
# Limits of that scale.
DISTANCE_SCALE_LIMITS = (0.7, 1.4)

# --- Head movement correction ---
# Moves the pointer by the head's turn and shift since calibration. Off: on the
# real tests (2-3 November) it made the accuracy worse. The head pose is still
# logged.
HEAD_CORRECTION = False
VIEWING_DISTANCE_CM = 40  # eyes to screen during calibration
EYE_CORNER_DISTANCE_CM = 9.0  # outer eye corners, adult
# The Pi sends 1920 x 1080 pixels to a 21.5 x 13.5 cm picture area, so
# left/right and up/down have different pixels per cm.
SCREEN_RESOLUTION = (1920, 1080)
SCREEN_WIDTH_CM, SCREEN_HEIGHT_CM = 21.5, 13.5  # measured with a ruler
SCREEN_PX_PER_CM_X = SCREEN_RESOLUTION[0] / SCREEN_WIDTH_CM    # left/right
SCREEN_PX_PER_CM_Y = SCREEN_RESOLUTION[1] / SCREEN_HEIGHT_CM   # up/down
# Point pairs on opposite sides of the face, for the head's turn and tilt.
TURN_PAIRS = [(33, 263), (234, 454), (61, 291)]  # eye corners, face sides, mouth corners
TILT_PAIRS = [(10, 0), (168, 2), (9, 164)]  # top and bottom face points
# The head pose is smoothed more (MediaPipe's depth is noisy).
HEAD_SMOOTHING_TIME = 1.0  # seconds


# Hide picamera2's debug lines.
logging.getLogger("picamera2").setLevel(logging.INFO)
# Hide a harmless protobuf warning MediaPipe causes on every picture.
warnings.filterwarnings("ignore", message=r"SymbolDatabase\.GetPrototype\(\) is deprecated")


def _pi_health():
    """The Pi's temperature and why it is slowing down (under-voltage or heat), as (degrees C or
    None, reason or None)."""
    temperature = None
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temperature = int(f.read().strip()) / 1000
    except (OSError, ValueError):
        pass
    flags = None
    try:
        with open("/sys/devices/platform/soc/soc:firmware/get_throttled") as f:
            flags = int(f.read().strip(), 16)
    except (OSError, ValueError):
        try:  # older systems: vcgencmd
            answer = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                                    text=True, timeout=2).stdout  # "throttled=0x50005"
            flags = int(answer.strip().split("=")[1], 16)
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            pass
    reason = None
    if flags is not None:
        if flags & 0x1:
            reason = "power supply too weak (under-voltage)"
        elif flags & 0xE:
            reason = "too hot"
        elif flags & 0x10000:
            reason = "power supply too weak at moments (under-voltage since start-up)"
    return temperature, reason


def _ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return
    print(f"[GazeEngine] Downloading face landmark model to '{MODEL_PATH}' ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[GazeEngine] Done.")


def _normalized_iris_position(landmarks, iris_idx, contour_indices, w, h, centre=None):
    """Iris centre (or centre) within the eye's outline: (0, 0) top left, (1, 1) bottom right."""
    xs = [landmarks[i].x * w for i in contour_indices]
    ys = [landmarks[i].y * h for i in contour_indices]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if centre is None:
        iris = landmarks[iris_idx]
        ix, iy = iris.x * w, iris.y * h
    else:
        ix, iy = centre
    fx = (ix - min_x) / (max_x - min_x) if max_x > min_x else 0.5
    fy = (iy - min_y) / (max_y - min_y) if max_y > min_y else 0.5
    return fx, fy


def _compute_horizontal_feature(landmarks, w, h, centres=None):
    """fx: the iris (or pupil) position between the eye corners, averaged over both eyes."""
    right, left = centres if centres is not None else (None, None)
    rfx, _ = _normalized_iris_position(landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_CONTOUR, w, h, right)
    lfx, _ = _normalized_iris_position(landmarks, LEFT_IRIS_CENTER, LEFT_EYE_CONTOUR, w, h, left)
    return (rfx + lfx) / 2


def _face_size(landmarks, w, h):
    """Distance in pixels between the outer eye corners (the face's size)."""
    a, b = landmarks[FACE_SIZE_CORNERS[0]], landmarks[FACE_SIZE_CORNERS[1]]
    return float(np.hypot((a.x - b.x) * w, (a.y - b.y) * h))


def _iris_diameter(landmarks, w, h):
    """Iris diameter in pixels, averaged over both eyes."""
    total = 0.0
    for centre, edge in ((RIGHT_IRIS_CENTER, RIGHT_IRIS_EDGE), (LEFT_IRIS_CENTER, LEFT_IRIS_EDGE)):
        c = landmarks[centre]
        total += 2 * np.mean([np.hypot((landmarks[i].x - c.x) * w, (landmarks[i].y - c.y) * h)
                              for i in edge])
    return total / 2


def _eye_boxes(landmarks, w, h):
    """Boxes (x0, y0, x1, y1) around the two eyes."""
    boxes = []
    for contour in (RIGHT_EYE_CONTOUR, LEFT_EYE_CONTOUR):
        xs = [landmarks[i].x * w for i in contour]
        ys = [landmarks[i].y * h for i in contour]
        boxes.append((max(0, int(min(xs)) - 5), max(0, int(min(ys)) - 5),
                      min(w, int(max(xs)) + 6), min(h, int(max(ys)) + 6)))
    return boxes


def _eye_sharpness(frame, boxes):
    """How sharp the eyes are: brightness differences between neighbouring pixels, independent of
    exposure."""
    if frame is None:
        return 0.0
    total = 0.0
    for x0, y0, x1, y1 in boxes:
        box = frame[y0:y1, x0:x1]
        if box.shape[0] < 2 or box.shape[1] < 2:
            continue
        gray = box.mean(axis=2) if box.ndim == 3 else box.astype(float)
        across = np.diff(gray, axis=1)[:-1, :]
        down = np.diff(gray, axis=0)[:, :-1]
        total += 1000 * (across ** 2 + down ** 2).mean() / max(gray.mean(), 1.0) ** 2
    return total


def _compute_iris_height(landmarks, w, h, centres=None):
    """iy: how far the iris (or pupil) sits below the line between the eye corners, divided by the
    eye's width, averaged over both eyes."""
    total = 0.0
    for iris_idx, (c1, c2), centre in zip((RIGHT_IRIS_CENTER, LEFT_IRIS_CENTER),
                                          (RIGHT_EYE_CORNERS, LEFT_EYE_CORNERS),
                                          centres if centres is not None else (None, None)):
        a, b = landmarks[c1], landmarks[c2]
        iris_y = landmarks[iris_idx].y * h if centre is None else centre[1]
        eye_width = abs(b.x - a.x) * w
        corner_y = (a.y + b.y) / 2 * h  # height of the line between the corners
        if eye_width > 0:
            total += (iris_y - corner_y) / eye_width
    return total / 2


def _head_pose(landmarks, w, h):
    """Head pose (turn, tilt, x, y): turn and tilt in radians from the depth difference of point
    pairs, x, y the middle of the outer eye corners in pixels."""
    def point(i):
        return landmarks[i].x * w, landmarks[i].y * h, landmarks[i].z * w

    turns = []
    for a, b in TURN_PAIRS:
        (ax, _, az), (bx, _, bz) = sorted([point(a), point(b)])  # left point first
        turns.append(np.arctan2(bz - az, bx - ax))
    tilts = []
    for a, b in TILT_PAIRS:
        (_, ay, az), (_, by, bz) = sorted([point(a), point(b)], key=lambda p: p[1])  # top point first
        tilts.append(np.arctan2(bz - az, by - ay))
    (ax, ay, _), (bx, by, _) = point(FACE_SIZE_CORNERS[0]), point(FACE_SIZE_CORNERS[1])
    return float(np.mean(turns)), float(np.mean(tilts)), (ax + bx) / 2, (ay + by) / 2


def _head_shift(pose, reference, face_size, distance_cm):
    """Screen shift (x, y) of the gaze caused by the head's change from reference to pose."""
    turn, tilt, x, y = pose
    turn0, tilt0, x0, y0 = reference
    cm_per_pixel = EYE_CORNER_DISTANCE_CM / face_size
    right_cm = distance_cm * np.tan(turn - turn0) + (x - x0) * cm_per_pixel
    down_cm = distance_cm * np.tan(tilt - tilt0) + (y - y0) * cm_per_pixel
    return right_cm * SCREEN_PX_PER_CM_X, -down_cm * SCREEN_PX_PER_CM_Y


def _crop_square(image, cx, cy, radius):
    """Square crop of side 2*radius around (cx, cy); returns the crop and its corner."""
    h, w = image.shape[:2]
    x0, x1 = max(0, int(cx - radius)), min(w, int(cx + radius) + 1)
    y0, y1 = max(0, int(cy - radius)), min(h, int(cy + radius) + 1)
    return image[y0:y1, x0:x1], x0, y0


def find_glints(gray, cx, cy, radius, max_area):
    """Finds the IR glints within radius of the pupil centre: small, bright spots that stand out
    from their surroundings. Returns their centres (x, y), brightness-weighted."""
    # 3 extra pixels so spots at the edge have surroundings.
    crop, x0, y0 = _crop_square(gray, cx, cy, radius + 3)
    if crop.size == 0:
        return []
    rows, cols = np.ogrid[:crop.shape[0], :crop.shape[1]]
    inside = (cols + x0 - cx) ** 2 + (rows + y0 - cy) ** 2 <= radius ** 2
    bright = inside & (crop >= GLINT_MIN_BRIGHTNESS)
    glints = []
    for spot in _group_touching(bright):
        if len(spot) > max_area:
            continue  # too big for an LED reflection
        ys = np.array([pixel[0] for pixel in spot])
        xs = np.array([pixel[1] for pixel in spot])
        # Pixels up to 2 away from the spot.
        in_spot = np.zeros(crop.shape, dtype=bool)
        in_spot[ys, xs] = True
        around = (_neighbourhood(in_spot, 2, "max") > 0) & ~in_spot
        if crop[ys, xs].max() - crop[around].mean() < GLINT_MIN_CONTRAST:
            continue  # not brighter than its surroundings: skin
        # Brighter pixels count more.
        weights = crop[ys, xs].astype(float) - GLINT_MIN_BRIGHTNESS + 1
        glints.append((x0 + np.average(xs, weights=weights),
                       y0 + np.average(ys, weights=weights)))
    return glints


def _group_touching(mask):
    """Groups True pixels into spots of touching pixels (diagonals count)."""
    remaining = set(zip(*np.nonzero(mask)))
    spots = []
    while remaining:
        to_visit = [remaining.pop()]  # start a new spot
        spot = []
        while to_visit:
            row, col = to_visit.pop()
            spot.append((row, col))
            # Add bright neighbours to the spot.
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    neighbour = (row + d_row, col + d_col)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        to_visit.append(neighbour)
        spots.append(spot)
    return spots


def _neighbourhood(image, size, combine):
    """Combines each pixel with the square around it: "mean", "min" or "max" (OpenCV if
    available)."""
    if cv2 is not None:
        image = image.astype(np.float32)
        square = (2 * size + 1, 2 * size + 1)
        if combine == "mean":
            return cv2.blur(image, square, borderType=cv2.BORDER_REPLICATE)
        kernel = np.ones(square, np.uint8)
        if combine == "min":
            return cv2.erode(image, kernel, borderType=cv2.BORDER_REPLICATE)
        return cv2.dilate(image, kernel, borderType=cv2.BORDER_REPLICATE)
    padded = np.pad(image.astype(float), size, mode="edge")
    h, w = image.shape
    shifted = [padded[dy:dy + h, dx:dx + w]
               for dy in range(2 * size + 1) for dx in range(2 * size + 1)]
    if combine == "mean":
        return np.mean(shifted, axis=0)
    if combine == "min":
        return np.min(shifted, axis=0)
    return np.max(shifted, axis=0)


def _remove_glints(crop, limit, where=None):
    """Paints over glints (pixels brighter than limit) with their darkest neighbours, from the
    edges in."""
    bright = crop > limit
    if where is not None:
        bright &= where
    # Glint pixels plus 1 pixel around them.
    glint = _neighbourhood(bright, 1, "max") > 0
    result = crop.astype(float)
    for _ in range(10):  # fills glints up to 20 pixels across
        if not glint.any():
            break
        # Darkest non-glint neighbour of every pixel.
        darkest_neighbour = _neighbourhood(np.where(glint, 255, result), 1, "min")
        edge = glint & (darkest_neighbour < 255)  # glint pixels next to a non-glint pixel
        result[edge] = darkest_neighbour[edge]
        glint = glint & ~edge
    return result


def find_pupil(gray, cx, cy, iris_radius):
    """Finds the pupil centre near the iris centre (cx, cy): pixels as dark as the pupil count
    fully, pixels on its edge partly, and the centre is their weighted average. None if no dark
    pupil is seen."""
    radius = iris_radius
    crop, x0, y0 = _crop_square(gray, cx, cy, radius + 2)
    if crop.shape[0] < 2 * radius or crop.shape[1] < 2 * radius:
        return None  # eye partly outside the picture
    rows, cols = np.ogrid[:crop.shape[0], :crop.shape[1]]
    distance_sq = (cols + x0 - cx) ** 2 + (rows + y0 - cy) ** 2
    ring = ((distance_sq >= (PUPIL_RING[0] * radius) ** 2)
            & (distance_sq <= (PUPIL_RING[1] * radius) ** 2))
    middle = distance_sq <= (PUPIL_MIDDLE_SCALE * radius) ** 2
    if not ring.any() or not middle.any():
        return None

    # Steps 1-3: iris brightness, paint over glints, pupil darkness.
    glint_limit = np.median(crop[ring]) + PUPIL_GLINT_CONTRAST
    near_pupil = distance_sq <= (PUPIL_MAX_SIZE * radius + 2) ** 2
    smooth = _neighbourhood(_remove_glints(crop, glint_limit, near_pupil), 1, "mean")
    iris = np.median(smooth[ring])
    pupil = np.percentile(smooth[middle], 5)
    if iris - pupil < PUPIL_MIN_CONTRAST:
        return None

    # Step 4: weight 1 near the pupil's darkness, 0 near the iris's brightness.
    weight = np.clip(2 * (iris - smooth) / (iris - pupil) - 0.5, 0, 1)
    weight[distance_sq > (PUPIL_MAX_SIZE * radius) ** 2] = 0
    total = weight.sum()
    if total < 5:
        return None  # too few dark pixels
    # Step 5: weighted centre.
    return (x0 + (weight * cols).sum() / total, y0 + (weight * rows).sum() / total)


def _eye_openness(landmarks, w, h):
    """Gap between the eyelids divided by the eye's width, averaged over both eyes."""
    total = 0.0
    for (upper, lower), (c1, c2) in ((RIGHT_EYE_LIDS, RIGHT_EYE_CORNERS),
                                     (LEFT_EYE_LIDS, LEFT_EYE_CORNERS)):
        gap = abs(landmarks[lower].y - landmarks[upper].y) * h
        eye_width = abs(landmarks[c2].x - landmarks[c1].x) * w
        if eye_width > 0:
            total += gap / eye_width
    return total / 2


class _Smoother:
    """Cleans one measurement: median of the last 3 readings (spike filter), then a moving average
    over smoothing_time; a missing reading keeps the last value for hold_time. latest is the
    spike-filtered reading only."""

    def __init__(self, hold_time, smoothing_time=SMOOTHING_TIME):
        self.hold_time = hold_time
        self.smoothing_time = smoothing_time
        self.reset()

    def reset(self):
        self.value = None
        self.latest = None
        self._recent = []  # last 3 readings
        self._last_reading_time = None

    def update(self, reading):
        now = time.time()
        self.latest = None
        if reading is None:
            if self._last_reading_time is not None and now - self._last_reading_time > self.hold_time:
                self.reset()  # gone too long: forget
            return
        since_last = None if self._last_reading_time is None else now - self._last_reading_time
        self._last_reading_time = now
        self._recent = (self._recent + [reading])[-3:]
        spike_free = float(np.median(self._recent))
        self.latest = spike_free
        if self.value is None:
            self.value = spike_free  # first reading
        else:
            # The new reading's share grows with the time since the last one.
            alpha = 1.0 - np.exp(-since_last / self.smoothing_time)
            self.value = alpha * spike_free + (1 - alpha) * self.value


class _Eye:
    """PCCR for one eye: vy and hx = (pupil - glint middle) / eye width, smoothed. Also gives the
    eye's centre for fx and iy (see USE_PUPIL)."""

    def __init__(self, iris_center, iris_edge, corners):
        self.iris_center = iris_center
        self.iris_edge = iris_edge
        self.corners = corners
        self.vy = _Smoother(GLINT_HOLD_TIME)
        self.hx = _Smoother(GLINT_HOLD_TIME)
        # Glint memory (see _glint_middle):
        self.half_gap = None  # from the pair's middle to the right glint
        self.middle = None  # the pair's middle last time
        self.frames_without_pair = 0  # pictures without the pair
        self.frames_rejected = 0  # pictures with rejected glints
        # Latest picture (full-picture pixels, None if not found):
        self.iris_point = None   # MediaPipe's iris centre
        self.pupil_point = None  # found by find_pupil
        self.eye_width = None
        # Usual pupil position relative to MediaPipe's iris centre, for when the pupil isn't found.
        self.pupil_offset = None

    def forget(self):
        """Forgets the glints and pupil (face lost)."""
        self.half_gap = None
        self.middle = None
        self.frames_without_pair = 0
        self.frames_rejected = 0
        self.pupil_offset = None

    def centre(self):
        """The eye's centre for fx and iy: the pupil, else MediaPipe's iris centre plus the usual
        offset, or None."""
        if self.pupil_point is not None:
            return self.pupil_point
        if self.iris_point is None:
            return None
        if self.pupil_offset is None:
            return self.iris_point
        dx, dy = self.pupil_offset
        return (self.iris_point[0] + dx * self.eye_width, self.iris_point[1] + dy * self.eye_width)

    def measure(self, frame, landmarks):
        """Measures this eye and updates vy and hx; returns the glint and pupil points for the
        preview."""
        h, w = frame.shape[:2]
        self.iris_point = None
        self.pupil_point = None

        def pixel(i):
            """Landmark i in pixels (MediaPipe gives 0-1 fractions)."""
            return landmarks[i].x * w, landmarks[i].y * h

        cx, cy = pixel(self.iris_center)
        # Iris radius: average distance to its 4 edge points.
        iris_radius = np.mean([np.hypot(pixel(i)[0] - cx, pixel(i)[1] - cy) for i in self.iris_edge])
        (ax, ay), (bx, by) = pixel(self.corners[0]), pixel(self.corners[1])
        eye_width = np.hypot(ax - bx, ay - by)
        if iris_radius < 2 or eye_width < 10:
            self.skip()  # eye too small to measure
            return []
        self.iris_point = (cx, cy)
        self.eye_width = eye_width

        # Grayscale crop around this eye, big enough for the glint search.
        search_radius = iris_radius * (PUPIL_MAX_SIZE + GLINT_SEARCH_SCALE) + 3
        box, x0, y0 = _crop_square(frame, cx, cy, search_radius)
        gray = box.mean(axis=2) if box.ndim == 3 else box
        pupil = find_pupil(gray, cx - x0, cy - y0, iris_radius)
        if pupil is None:
            self.skip()
            return []
        glint_area = GLINT_MAX_AREA * np.pi * iris_radius ** 2
        glints = find_glints(gray, pupil[0], pupil[1], iris_radius * GLINT_SEARCH_SCALE, glint_area)
        pupil = (pupil[0] + x0, pupil[1] + y0)
        self.pupil_point = pupil
        offset = ((pupil[0] - cx) / eye_width, (pupil[1] - cy) / eye_width)
        if self.pupil_offset is None:
            self.pupil_offset = offset
        else:  # averaged over recent pictures
            self.pupil_offset = tuple(SMOOTHING_ALPHA * new + (1 - SMOOTHING_ALPHA) * old
                                      for new, old in zip(offset, self.pupil_offset))
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
        """No measurement this picture: keep the last one."""
        self.vy.update(None)
        self.hx.update(None)

    def _glint_middle(self, glints, pupil, iris_radius):
        """Middle of the two LED glints, or None. With one glint missing, it is worked out from the
        other and the remembered gap."""
        if not glints:
            return None
        pair = self._find_pair(glints, pupil, iris_radius)
        if pair is not None:
            left, right = pair
            middle = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
            gap = ((right[0] - left[0]) / 2, (right[1] - left[1]) / 2)
            if self.half_gap is None or self.frames_without_pair > MAX_MISSED_FRAMES:
                self.half_gap = gap  # first pair, or the old gap is out of date
            else:
                # Averaged over recent pictures.
                self.half_gap = tuple(SMOOTHING_ALPHA * new + (1 - SMOOTHING_ALPHA) * old
                                      for new, old in zip(gap, self.half_gap))
            self.frames_without_pair = 0
        else:
            self.frames_without_pair += 1
            if self.half_gap is None or self.middle is None:
                # No pair known yet: use the glint nearest the pupil.
                middle = min(glints, key=lambda g: np.hypot(g[0] - pupil[0], g[1] - pupil[1]))
            else:
                # The glint is the left one, the right one, or both merged: use the guess nearest
                # the last middle.
                dx, dy = self.half_gap
                guesses = [(gx + dx, gy + dy) for gx, gy in glints] + \
                          [(gx - dx, gy - dy) for gx, gy in glints] + list(glints)
                last_x, last_y = self.middle
                middle = min(guesses, key=lambda m: np.hypot(m[0] - last_x, m[1] - last_y))
                if np.hypot(middle[0] - last_x, middle[1] - last_y) > GLINT_JUMP_LIMIT * iris_radius:
                    # Can't tell which glint it is.
                    self.frames_rejected += 1
                    if self.frames_rejected >= GLINT_RESTART_FRAMES:
                        # The glints have moved (GLINT_RESTART_FRAMES): start again next time.
                        self.middle = None
                        self.frames_rejected = 0
                    return None
        self.frames_rejected = 0
        self.middle = middle
        return middle

    def _find_pair(self, glints, pupil, iris_radius):
        """The two LED glints as (left, right), or None; the gap is learned from a picture with
        exactly two glints."""
        if self.half_gap is None or self.frames_without_pair > MAX_MISSED_FRAMES:
            if len(glints) != 2:
                return None
            return tuple(sorted(glints))  # left to right
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
    """Draws a small square marker on an RGB frame (numpy, no OpenCV needed)."""
    h, w = frame.shape[:2]
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    if y0 < y1 and x0 < x1:
        frame[y0:y1, x0:x1] = color


def _least_squares(columns, outputs):
    """Least squares: output = c1 * column1 + ... + b. Returns ([c1, ...], b, r2)."""
    y = np.array(outputs, dtype=float)
    a_matrix = np.column_stack([np.array(column, dtype=float) for column in columns] + [np.ones_like(y)])
    solution = np.linalg.lstsq(a_matrix, y, rcond=None)[0]
    predicted = a_matrix @ solution
    total = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum((y - predicted) ** 2) / total if total > 0 else 0.0
    return [float(c) for c in solution[:-1]], float(solution[-1]), float(r2)


def _left_out_error(columns, outputs):
    """Leave-one-out error (RMS, screen pixels): how well the fit predicts dots it hasn't seen."""
    errors = []
    for left_out in range(len(outputs)):
        keep = [k for k in range(len(outputs)) if k != left_out]
        coefs, b, _ = _least_squares([[column[k] for k in keep] for column in columns],
                                     [outputs[k] for k in keep])
        predicted = b + sum(c * column[left_out] for c, column in zip(coefs, columns))
        errors.append(predicted - outputs[left_out])
    return float(np.sqrt(np.mean(np.square(errors))))


def _adjusted_r2(r2, points, inputs):
    """Adjusted R squared: lowered a little for every input used."""
    if points - inputs - 1 <= 0:
        return 0.0
    return 1.0 - (1.0 - r2) * (points - 1) / (points - inputs - 1)


def _fit_axis(points, shakes):
    """Fits one direction from points (glint value or None, iris value, screen position): glints
    only, iris only and both, and returns (best fit, iris-only fit). The best fit has the lowest
    expected error: leave-one-out error plus the shaking it would cause (shakes = glint and iris
    shake)."""
    shake_of = {"glint": shakes[0], "iris": shakes[1]}
    candidates = []  # (name, points, inputs)
    glint_points = [p for p in points if p[0] is not None]
    # Glint fits only if glints were seen at every dot; otherwise the pointer would
    # switch between the glint and iris fits and jump.
    if (len(glint_points) == len(points)
            and len({round(screen) for g, i, screen in glint_points}) >= 2):
        candidates.append(("glints", glint_points, ["glint"]))
        candidates.append(("glints + iris", glint_points, ["glint", "iris"]))
    if len({round(screen) for g, i, screen in points}) >= 2:
        candidates.append(("iris", points, ["iris"]))

    fits = []
    for name, used, inputs in candidates:
        values = {"glint": [g for g, i, screen in used], "iris": [i for g, i, screen in used]}
        columns = [values[k] for k in inputs]
        screens = [screen for g, i, screen in used]
        coefs, b, r2 = _least_squares(columns, screens)
        fit = {"name": name, "b": b, "r2": _adjusted_r2(r2, len(used), len(inputs))}
        fit.update(zip(inputs, coefs))
        shake = np.sqrt(sum((fit[k] * shake_of[k]) ** 2 for k in inputs))
        fit["error"] = float(np.hypot(_left_out_error(columns, screens), shake))
        fits.append(fit)
    if not fits:
        return None, None
    print("[Calibration]   fits: " + ", ".join(
        f"{f['name']} (R2 {f['r2']:.2f}, expected error {f['error']:.0f} px)" for f in fits))
    iris_fit = next((f for f in fits if f["name"] == "iris"), None)
    return min(fits, key=lambda f: f["error"]), iris_fit


def _predict(fit, backup, glint_value, iris_value):
    """Screen position from a fit; uses the iris-only backup when no glint is seen. None if neither
    works."""
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
# Camera backends: read() returns a mirrored RGB picture, the same on both.
# ---------------------------------------------------------------------

class _OpenCVCamera:
    """Laptop/USB webcam via OpenCV."""

    name = "OpenCV webcam"
    mediapipe_step = 1  # webcam pictures are already small
    infrared = False  # normal webcam: no clear pupil
    focal_length_px = None  # unknown: no distance estimate

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
        frame = cv2.flip(frame, 1)  # mirror
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        self._cap.release()


class _PiCamera:
    """Raspberry Pi camera (Camera Module 3 NoIR) via picamera2."""

    name = "Raspberry Pi camera (picamera2)"
    mediapipe_step = PI_MEDIAPIPE_STEP
    infrared = True  # NoIR + IR LEDs: dark pupil

    def __init__(self):
        if Picamera2 is None:
            raise RuntimeError("picamera2 is not installed")
        self._picam = Picamera2()
        # "BGR888" in picamera2 gives pixels in RGB order, as MediaPipe expects.
        config = self._picam.create_preview_configuration(
            main={"size": (CAPTURE_WIDTH, CAPTURE_HEIGHT), "format": "BGR888"}
        )
        self._picam.configure(config)
        self._picam.start()
        # Fixed focus (see FOCUS_ON_EYES); automatic exposure.
        self._picam.set_controls({
            "AfMode": libcamera_controls.AfModeEnum.Manual,
            "LensPosition": PI_LENS_POSITION,
        })
        time.sleep(2)  # let exposure and focus settle
        self.focal_length_px = self._focal_length_px()

    def _focal_length_px(self):
        """Lens focal length in picture pixels, for the face distance, or None."""
        try:
            focal_mm = PI_LENS_FOCAL_MM
            if "wide" in str(self._picam.camera_properties.get("Model", "")).lower():
                focal_mm = 2.75   # Camera Module 3 Wide
            crop = self._picam.capture_metadata().get("ScalerCrop")
            return focal_mm / (PI_SENSOR_PIXEL_MM * crop[2] / CAPTURE_WIDTH)
        except Exception as e:
            print(f"[GazeEngine] Could not work out the focal length: {e}")
            return None

    def set_lens(self, dioptres):
        """Moves the lens to dioptres and returns a picture taken once it got there."""
        self._picam.set_controls({"AfMode": libcamera_controls.AfModeEnum.Manual,
                                  "LensPosition": float(dioptres)})
        for waited in range(1, FOCUS_MAX_WAIT_PICTURES + 1):
            # Each picture reports its lens position.
            lens = self._picam.capture_metadata().get("LensPosition")
            if lens is None and waited >= FOCUS_SETTLE_PICTURES:
                break  # not reported: fixed number of pictures
            if lens is not None and abs(lens - dioptres) < 0.05:
                break  # the lens is there
        self._picam.capture_array()  # one more while the lens settles
        return self.read()

    def describe(self):
        """The camera's mode and sensor area, as text."""
        try:
            sensor = self._picam.camera_configuration().get("sensor", {})
            crop = self._picam.capture_metadata().get("ScalerCrop")
            return f"sensor mode {sensor.get('output_size')}, sensor area used {crop}"
        except Exception as e:
            return f"unknown ({e})"

    def read(self):
        frame = self._picam.capture_array()
        if PI_CAMERA_UPSIDE_DOWN:
            frame = frame[::-1, ::-1]  # rotate 180 degrees
        return frame[:, ::-1]  # mirror

    def close(self):
        self._picam.stop()


def _open_camera():
    """Returns a working camera (the Pi camera first), or None."""
    for backend in (_PiCamera, _OpenCVCamera):
        try:
            camera = backend()
            print(f"[GazeEngine] Using {backend.name}")
            return camera
        except Exception as e:
            print(f"[GazeEngine] {backend.name} unavailable: {e}")
    return None


class GazeEngine:
    """Camera + MediaPipe in a background thread. Shared state is guarded by self._lock; the app
    only calls the public methods."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_feature = None  # (fx, vy, iy, hx), None if no face
        self._latest_raw_feature = None  # the same, not averaged over time
        # Smoothed measurements (background thread only).
        self._fx = _Smoother(IRIS_HOLD_TIME)
        self._iy = _Smoother(IRIS_HOLD_TIME)
        self._face_size = _Smoother(IRIS_HOLD_TIME)
        self._head = [_Smoother(IRIS_HOLD_TIME, HEAD_SMOOTHING_TIME) for _ in range(4)]  # see _head_pose
        self._eyes = [_Eye(RIGHT_IRIS_CENTER, RIGHT_IRIS_EDGE, RIGHT_EYE_CORNERS),
                      _Eye(LEFT_IRIS_CENTER, LEFT_IRIS_EDGE, LEFT_EYE_CORNERS)]
        self._openness_history = []  # for the blink check
        self._closed_frames = 0  # frames in a row with the eyes nearly closed
        self._eyes_closed = False  # see eyes_closed
        self._latest_face_size = None  # smoothed face size
        self._latest_head = None  # smoothed head pose
        self._latest_debug_frame = None  # preview frame with dots drawn
        self._latest_eyes = None  # for eye close-ups and eye pictures
        self._fps = None  # pictures processed per second
        self._slowdown = None  # see _pi_health
        self._focus_requested = FOCUS_ON_EYES  # focus the lens on the eyes
        self._focusing = False  # True while focusing
        self._low_power = False  # see LOW_POWER_FPS
        self._lens = None  # chosen lens position (dioptres)
        self._eye_boxes = None  # boxes around the eyes
        self._use_pupil = False  # see USE_PUPIL
        self._focal_px = None  # focal length in pixels
        self._iris_size = _Smoother(IRIS_HOLD_TIME)  # iris diameter, smoothed
        self._latest_distance = None  # cm, see get_face_distance_cm
        self.camera_mode = ""  # see _PiCamera.describe
        self._running = False
        self._thread = None
        self._camera_ok = True  # False if the camera can't be opened

        # Calibration points, and the fits from fit_calibration() (None = unusable).
        self._x_points = []  # (hx, fx, screen_x); hx None without a glint
        self._y_points = []  # (vy, iy, screen_y); vy None without a glint
        self._shakes = []  # shake of (fx, vy, iy, hx) at each dot
        self._x_heads = []  # head pose at each dot
        self._y_heads = []  # the same for _y_points
        self._calib_head = None  # average head pose during calibration
        self._x_fit = None
        self._y_fit = None
        self._x_backup = None
        self._y_backup = None

        # Distance correction: face sizes at the dots, their average, the dots' middle.
        self._calib_face_sizes = []
        self._calib_face_size = None
        self._centre_x = None
        self._centre_y = None

        # Re-centring shift (x, y) in screen pixels (see set_pointer_shift).
        self._pointer_shift = (0.0, 0.0)
        # Up/down error of the accuracy test (cm), for Row mode in main.py.
        self.updown_error_cm = None

    # ---- lifecycle ----
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _is_blink(self, landmarks, w, h):
        """True if the eyes are much less open than usual, for no longer than a blink."""
        openness = _eye_openness(landmarks, w, h)
        self._openness_history = (self._openness_history + [openness])[-OPENNESS_HISTORY:]
        nearly_closed = openness < BLINK_OPENNESS * float(np.median(self._openness_history))
        self._closed_frames = self._closed_frames + 1 if nearly_closed else 0
        self._eyes_closed = nearly_closed
        return 0 < self._closed_frames <= BLINK_MAX_FRAMES

    def eyes_closed(self):
        """True if the eyes were nearly closed in the latest picture, however long (used by
        calibration)."""
        return self._eyes_closed

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

        describe = getattr(camera, "describe", None)
        if describe is not None:
            self.camera_mode = describe()
            print(f"[GazeEngine] Camera: {self.camera_mode}")
        self._use_pupil = USE_PUPIL and getattr(camera, "infrared", False)
        self._focal_px = getattr(camera, "focal_length_px", None)
        print("[GazeEngine] Eye position from " + ("the pupil found in the picture (USE_PUPIL)"
                                                   if self._use_pupil else "MediaPipe's iris centre"))
        start_time = time.time()
        # fps and time per step, printed every 5 seconds.
        fps_count, fps_start = 0, time.time()
        spent = {"camera": 0.0, "MediaPipe": 0.0, "eyes": 0.0}
        picture_number = 0
        last_picture_at = 0.0
        while self._running:
            if self._low_power:
                # Resting: wait for the next picture, a little at a time.
                wait = last_picture_at + 1 / LOW_POWER_FPS - time.time()
                if wait > 0:
                    time.sleep(min(wait, 0.1))
                    continue
            last_picture_at = time.time()
            if self._focus_requested:
                # Focus requested: done here (the camera belongs to this thread) once the eyes are
                # seen.
                if getattr(camera, "set_lens", None) is None:
                    self._focus_requested = False  # laptop webcam can't focus
                elif self._eye_boxes is not None:
                    self._focus_requested = False
                    self._focus_on_eyes(camera)
                    self._reset_measurements()  # start afresh with sharp pictures
                    fps_count, fps_start = 0, time.time()  # don't count the pause
            step_start = time.time()
            frame = camera.read()  # full-size RGB
            if frame is None:
                continue
            spent["camera"] += time.time() - step_start
            fps_count += 1
            picture_number += 1
            if time.time() - fps_start >= 5:
                self._fps = fps_count / (time.time() - fps_start)
                times = ", ".join(f"{name} {seconds / fps_count * 1000:.0f} ms" for name, seconds in spent.items())
                temperature, self._slowdown = _pi_health()
                health = f"; processor {temperature:.0f} °C" if temperature is not None else ""
                if self._slowdown:
                    health += f"; THE PI IS SLOWED DOWN: {self._slowdown}"
                resting = " (resting: low power)" if self._low_power else ""
                print(f"[GazeEngine] {self._fps:.1f} pictures a second{resting}; per picture: {times}{health}")
                fps_count, fps_start = 0, time.time()
                spent = dict.fromkeys(spent, 0.0)
            # Smaller contiguous copy for MediaPipe.
            step = camera.mediapipe_step
            small = np.ascontiguousarray(frame[::step, ::step])
            sh, sw = small.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
            timestamp_ms = int((time.time() - start_time) * 1000)
            step_start = time.time()
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            spent["MediaPipe"] += time.time() - step_start

            debug_frame = small.copy()
            eyes = None
            fh, fw = frame.shape[:2]
            if not result.face_landmarks:
                feature = raw_feature = None
                # Start afresh when the face comes back.
                self._reset_measurements()
            else:
                landmarks = result.face_landmarks[0]
                self._eye_boxes = _eye_boxes(landmarks, fw, fh)
                self._face_size.update(_face_size(landmarks, sw, sh))
                self._iris_size.update(_iris_diameter(landmarks, fw, fh))
                for smoother, value in zip(self._head, _head_pose(landmarks, sw, sh)):
                    smoother.update(value)
                points = []
                if self._is_blink(landmarks, sw, sh):
                    # Blink: keep the last measurements.
                    self._fx.update(None)
                    self._iy.update(None)
                    for eye in self._eyes:
                        eye.skip()
                else:
                    step_start = time.time()
                    for eye in self._eyes:
                        points += eye.measure(frame, landmarks)
                    spent["eyes"] += time.time() - step_start
                    # Eye centres: the pupils (USE_PUPIL), or MediaPipe's iris centres (None).
                    centres = None
                    if self._use_pupil:
                        centres = [eye.centre() for eye in self._eyes]
                        if any(centre is None for centre in centres):
                            centres = None  # an eye couldn't be measured
                    self._fx.update(_compute_horizontal_feature(landmarks, fw, fh, centres))
                    self._iy.update(_compute_iris_height(landmarks, fw, fh, centres))
                iris_centres = [(landmarks[i].x * fw, landmarks[i].y * fh)
                                for i in (RIGHT_IRIS_CENTER, LEFT_IRIS_CENTER)]
                eyes = (frame, points, iris_centres, landmarks, picture_number)

                # Each eye's glint measurements are smoothed, then averaged.
                vy = _average_known([eye.vy.value for eye in self._eyes])
                hx = _average_known([eye.hx.value for eye in self._eyes])
                if self._fx.value is None:
                    feature = None  # eyes closed over a second
                else:
                    feature = (self._fx.value, vy, self._iy.value, hx)
                # This picture's own readings.
                raw_feature = None
                if self._fx.latest is not None and self._iy.latest is not None:
                    raw_feature = (self._fx.latest, _average_known([eye.vy.latest for eye in self._eyes]),
                                   self._iy.latest, _average_known([eye.hx.latest for eye in self._eyes]))

                # Preview dots: green = iris, white = glint, yellow = pupil.
                for idx in (RIGHT_IRIS_CENTER, LEFT_IRIS_CENTER):
                    lm = landmarks[idx]
                    _draw_dot(debug_frame, int(lm.x * sw), int(lm.y * sh))
                for kind, px, py in points:
                    color = (255, 255, 255) if kind == "glint" else (255, 220, 0)
                    _draw_dot(debug_frame, int(px / step), int(py / step), radius=2, color=color)

            with self._lock:
                self._latest_feature = feature
                self._latest_raw_feature = raw_feature
                head = [smoother.value for smoother in self._head]
                self._latest_head = None if None in head else tuple(head)
                self._latest_face_size = self._face_size.value
                iris = self._iris_size.value
                self._latest_distance = (self._focal_px * IRIS_DIAMETER_MM / iris / 10
                                         if self._focal_px and iris else None)
                self._latest_debug_frame = debug_frame
                self._latest_eyes = eyes
            self._focusing = False  # see _focus_on_eyes

        camera.close()

    def _reset_measurements(self):
        """Forgets the smoothed measurements and what the eyes remember."""
        for smoother in [self._fx, self._iy, self._face_size, self._iris_size] + self._head + \
                [eye.vy for eye in self._eyes] + [eye.hx for eye in self._eyes]:
            smoother.reset()
        for eye in self._eyes:
            eye.forget()

    def _focus_on_eyes(self, camera):
        """Moves the lens to where the eyes are sharpest (coarse steps, then fine). Takes a few
        seconds without gaze readings."""
        self._focusing = True
        with self._lock:
            self._latest_feature = None  # no readings meanwhile
            self._latest_raw_feature = None
        started = time.time()
        sharpness = {}  # lens position -> sharpness

        def try_lens(lens):
            lens = round(float(lens), 3)
            if lens >= 0 and lens not in sharpness:
                sharpness[lens] = _eye_sharpness(camera.set_lens(lens), self._eye_boxes)

        try:
            for lens in FOCUS_LENS_POSITIONS:
                try_lens(lens)
            best = max(sharpness, key=sharpness.get)
            for steps in (-2, -1, 1, 2):  # finer steps either side
                try_lens(best + steps * FOCUS_FINE_STEP)
            best = max(sharpness, key=sharpness.get)
            camera.set_lens(best)
            self._lens = best
            tried = ", ".join(f"{lens:g}: {value:.1f}" for lens, value in sorted(sharpness.items()))
            print(f"[GazeEngine] Focused on the eyes: lens {best:g} in {time.time() - started:.1f} s "
                  f"(sharpness at each lens position: {tried})")
        except Exception as e:
            print(f"[GazeEngine] Focusing on the eyes failed: {e}")
        # _focusing turns False only after the next picture is measured.

    # ---- public API used by the app ----
    def is_camera_ok(self):
        return self._camera_ok

    def get_latest_feature(self):
        """(fx, vy, iy, hx), or None if no face; vy and hx are None without a glint."""
        with self._lock:
            return self._latest_feature

    def get_latest_raw_feature(self):
        """Like get_latest_feature, but this picture's readings, not averaged over time."""
        with self._lock:
            return self._latest_raw_feature

    def get_debug_frame(self):
        """Latest camera frame with the iris, glint and pupil dots drawn, or None."""
        with self._lock:
            return self._latest_debug_frame

    def get_head(self):
        """Smoothed head pose (turn, tilt, x, y), or None."""
        with self._lock:
            return self._latest_head

    def get_slowdown(self):
        """Why the Pi is slowing itself down (see _pi_health), or None."""
        return self._slowdown

    def get_fps(self):
        """Pictures processed per second, or None until it's measured."""
        return self._fps

    def set_low_power(self, on):
        """True: process only LOW_POWER_FPS pictures a second; False: as many as possible."""
        self._low_power = bool(on)

    def request_focus(self):
        """Asks for the lens to be focused on the eyes (in the background)."""
        if FOCUS_ON_EYES:
            self._focus_requested = True

    def is_focusing(self):
        """True while focusing on the eyes is waiting or going on."""
        return self._focus_requested or self._focusing

    def get_lens_position(self):
        """Lens position from the last focusing (dioptres), or None."""
        return self._lens

    def get_face_distance_cm(self):
        """Eyes-to-camera distance in cm from the iris size, or None."""
        with self._lock:
            return self._latest_distance

    def get_eye_picture(self):
        """Both eyes at full detail for the calibration's eye pictures: (picture number, grayscale
        picture, its corner (x, y), 478 landmarks), or None."""
        with self._lock:
            eyes = self._latest_eyes
        if eyes is None:
            return None
        frame, _points, _centres, landmarks, number = eyes
        h, w = frame.shape[:2]
        pixels = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
        around_eyes = pixels[RIGHT_EYE_CONTOUR + LEFT_EYE_CONTOUR]
        # Both eyes plus a margin.
        margin = 0.25 * np.hypot(*(pixels[FACE_SIZE_CORNERS[0]] - pixels[FACE_SIZE_CORNERS[1]]))
        x0 = int(max(0, around_eyes[:, 0].min() - margin))
        x1 = int(min(w, around_eyes[:, 0].max() + margin))
        y0 = int(max(0, around_eyes[:, 1].min() - margin))
        y1 = int(min(h, around_eyes[:, 1].max() + margin))
        part = frame[y0:y1, x0:x1]
        gray = part.mean(axis=2) if part.ndim == 3 else part
        return number, np.round(gray).astype(np.uint8), (x0, y0), pixels

    def get_eye_closeup(self, half_size=40, zoom=4):
        """Both eyes side by side, enlarged, glints in red and pupils in blue, or None."""
        with self._lock:
            eyes = self._latest_eyes
        if eyes is None:
            return None
        frame, points, iris_centres = eyes[:3]
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
        # Enlarge by repeating pixels.
        return np.repeat(np.repeat(both, zoom, axis=0), zoom, axis=1)

    def is_calibrated(self):
        return self._x_fit is not None

    def is_vertical_calibrated(self):
        return self._y_fit is not None

    def add_calibration_point(self, fx, screen_x, vy=None, screen_y=None, iy=None, hx=None,
                              shakes=None, head_and_size=None):
        """Stores one calibration dot: the measurements, their shakes, and the face size and head
        pose (now, or head_and_size)."""
        head, face_size = head_and_size if head_and_size is not None else self.head_and_face_size()
        self._x_points.append((hx, fx, screen_x))
        self._x_heads.append(head)
        self._shakes.append(shakes or (None, None, None, None))
        if face_size is not None:
            self._calib_face_sizes.append(face_size)
        if screen_y is not None and iy is not None:
            self._y_points.append((vy, iy, screen_y))
            self._y_heads.append(head)

    def head_and_face_size(self):
        """(head pose, face size) now, as stored with a dot."""
        with self._lock:
            face_size = self._latest_face_size
        return self.get_head(), face_size

    def clear_calibration(self):
        self._x_points = []
        self._y_points = []
        self._shakes = []
        self._x_fit = None
        self._y_fit = None
        self._x_backup = None
        self._y_backup = None
        self._calib_face_sizes = []
        self._calib_face_size = None
        self._x_heads = []
        self._y_heads = []
        self._calib_head = None
        self._pointer_shift = (0.0, 0.0)
        self.updown_error_cm = None

    def get_pointer_shift(self):
        """Re-centring shift (x, y) in screen pixels."""
        return self._pointer_shift

    def set_pointer_shift(self, x, y):
        """Moves the gaze pointer by (x, y) screen pixels on top of the fits (re-centring in
        main.py)."""
        self._pointer_shift = (float(x), float(y))

    def fit_calibration(self):
        """Fits left/right and up/down; returns a short summary. Raises ValueError if left/right
        can't be fitted."""
        if len(self._x_points) < 2:
            raise ValueError(f"Need at least 2 calibration points, got {len(self._x_points)}")
        self._pointer_shift = (0.0, 0.0)  # new fits: no re-centring

        # References for the distance and head corrections.
        self._centre_x = float(np.mean([p[2] for p in self._x_points]))
        if self._y_points:
            self._centre_y = float(np.mean([p[2] for p in self._y_points]))
        if self._calib_face_sizes:
            self._calib_face_size = float(np.mean(self._calib_face_sizes))
            print(f"[Calibration] face size {self._calib_face_size:.0f} px (distance reference)")
        heads = [head for head in self._x_heads if head is not None]
        if heads:
            self._calib_head = tuple(float(v) for v in np.mean(heads, axis=0))

        print(f"[Calibration] left/right, {len(self._x_points)} points:")
        self._x_fit, self._x_backup = _fit_axis(self._without_head_shift(self._x_points, self._x_heads, 0),
                                                (self._typical_shake(3), self._typical_shake(0)))
        if self._x_fit is None:
            raise ValueError("left/right: dots seen at fewer than 2 positions")
        if self._x_fit["r2"] < MIN_FIT_R2:
            weak = self._x_fit["r2"]
            self._x_fit = None
            raise ValueError(f"left/right fit too weak (R^2 = {weak:.2f})")
        summary = f"Left/right fit {self._x_fit['r2']:.2f} ({self._x_fit['name']})"

        print(f"[Calibration] up/down, {len(self._y_points)} points:")
        self._y_fit, self._y_backup = _fit_axis(self._without_head_shift(self._y_points, self._y_heads, 1),
                                                (self._typical_shake(1), self._typical_shake(2)))
        if self._y_fit is None:
            return summary + "\nUp/down: not enough data, using the mouse"
        if self._y_fit["r2"] < MIN_FIT_R2:
            weak = self._y_fit["r2"]
            self._y_fit = None
            return summary + f"\nUp/down fit {weak:.2f} (too weak, using the mouse)"
        return summary + f"\nUp/down fit {self._y_fit['r2']:.2f} ({self._y_fit['name']})"

    def _without_head_shift(self, points, heads, axis):
        """Calibration points with each target moved back by the head's shift at that dot."""
        if not HEAD_CORRECTION or self._calib_head is None or not self._calib_face_size:
            return points
        corrected = []
        for (glint, iris, screen), head in zip(points, heads):
            shift = 0.0
            if head is not None:
                shift = _head_shift(head, self._calib_head, self._calib_face_size, VIEWING_DISTANCE_CM)[axis]
            corrected.append((glint, iris, screen - shift))
        return corrected

    def _head_correction(self):
        """Head shift since calibration (x, y) in screen pixels, or (0, 0) if off."""
        head = self.get_head()
        with self._lock:
            face_size = self._latest_face_size
        if not HEAD_CORRECTION or head is None or self._calib_head is None or not face_size:
            return 0.0, 0.0
        return _head_shift(head, self._calib_head, face_size, VIEWING_DISTANCE_CM * self._distance_scale())

    def _typical_shake(self, index):
        """Median shake of measurement index over the dots, or 0."""
        values = [shakes[index] for shakes in self._shakes if shakes[index] is not None]
        return float(np.median(values)) if values else 0.0

    def _usable_backup(self, backup):
        """The iris-only backup fit, if it is good enough to use."""
        return backup if backup is not None and backup["r2"] >= MIN_FIT_R2 else None

    def _distance_scale(self):
        """Distance now / distance during calibration, from the face size; 1.0 if unknown."""
        with self._lock:
            face_size = self._latest_face_size
        if self._calib_face_size is None or not face_size:
            return 1.0
        low, high = DISTANCE_SCALE_LIMITS
        return min(high, max(low, self._calib_face_size / face_size))

    def get_screen_x(self):
        """Predicted screen x, or None if not calibrated or no face."""
        if self._x_fit is None:
            return None
        feature = self.get_latest_feature()
        if feature is None:
            return None
        fx, _vy, _iy, hx = feature
        x = _predict(self._x_fit, self._usable_backup(self._x_backup), hx, fx)
        if x is None:
            return None
        # Distance correction.
        x = self._centre_x + (x - self._centre_x) * self._distance_scale()
        # Head correction and re-centring.
        return x + self._head_correction()[0] + self._pointer_shift[0]

    def get_screen_y(self):
        """Predicted screen y (0 = bottom), or None if up/down isn't calibrated or can't be
        measured now."""
        if self._y_fit is None:
            return None
        feature = self.get_latest_feature()
        if feature is None:
            return None
        _fx, vy, iy, _hx = feature
        y = _predict(self._y_fit, self._usable_backup(self._y_backup), vy, iy)
        if y is None:
            return None
        # Distance correction.
        y = self._centre_y + (y - self._centre_y) * self._distance_scale()
        # Head correction and re-centring.
        return y + self._head_correction()[1] + self._pointer_shift[1]
