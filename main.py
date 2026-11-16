"""EyeGlynt on-screen interface (Kivy): calibration, Keyboard and Quick Fires.

Every button is picked by dwell: looking at it for DWELL_TIME seconds.
Run: python main.py
"""

import csv
import importlib.util
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import zlib

import numpy as np

import pyttsx3
# Redraw the screen at most SCREEN_FPS times a second (set before Kivy loads).
from kivy.config import Config
SCREEN_FPS = 30
Config.set("graphics", "maxfps", str(SCREEN_FPS))
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.properties import NumericProperty
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle, Ellipse, RoundedRectangle, Line, InstructionGroup
from kivy.uix.widget import Widget
from kivy.core.image import Image as CoreImage
from kivy.utils import escape_markup

from gaze_engine import GazeEngine, SCREEN_PX_PER_CM_X, SCREEN_PX_PER_CM_Y
from icons import draw_icon
from dashboard_link import dashboard, slug  # caregiver dashboard


def rgb(r, g, b, a=1.0):
    """Converts 0-255 colour values to Kivy's 0-1 range."""
    return (r / 255, g / 255, b / 255, a)


# Colours sampled directly from the pixels of Figure 29.
FRAME_BG = rgb(58, 59, 64)       # device frame and screen background
PANEL_BG = rgb(70, 71, 80)       # keyboard and tile panels
PURPLE = rgb(106, 31, 173)       # KEYBOARD tab, Speak, Backspace
ORANGE = rgb(226, 112, 42)       # QUICK FIRES tab
KEY_FACE = rgb(99, 98, 98)       # letter keys
KEY_BORDER = rgb(154, 154, 154)  # thin outline around keys and panels
PINK = rgb(226, 169, 241)        # Quick Access words; Environment tiles
GREEN = rgb(191, 236, 172)       # "Yes"; Social/Emotional tiles
YELLOW = rgb(255, 235, 153)      # Basic Needs tiles
TILE_RED = rgb(250, 155, 155)    # "No"; Emergency tiles (Figures 30-31)
TILE_ORANGE = rgb(252, 187, 117) # Pain tiles
TILE_BLUE = rgb(163, 203, 250)   # People/Family tiles
WHITE = rgb(255, 255, 255)
DARK_TEXT = rgb(30, 30, 30)
REST_BG = rgb(22, 22, 28)  # resting screen
REST_TEXT = rgb(165, 165, 180)
PLACEHOLDER = rgb(140, 140, 140)

# Quick Fire pictures (Mulberry Symbols, CC BY-SA 4.0; see assets/symbols/CREDITS.md).
SYMBOL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "symbols")


def _load_symbol(name):
    """Loads assets/symbols/<name>.png as a texture, or None if it is missing."""
    path = os.path.join(SYMBOL_DIR, f"{name}.png")
    if not os.path.exists(path):
        print(f"[Symbols] Missing {path}; showing text only")
        return None
    return CoreImage(path).texture

# ---------------------------------------------------------------------------
# 0. SETTINGS
# ---------------------------------------------------------------------------
# Touchscreen: 10.1 inch, 1280 x 800.
SCREEN_W, SCREEN_H = 1280, 800
Window.size = (SCREEN_W, SCREEN_H)

# How long the pointer must stay on a button to select it.
DWELL_TIME = 4.0  # seconds (a dashboard session can pick 2-6 s)

# Sticky buttons: gaze jitter and blinks don't restart a dwell.
STICKY_MARGIN = 60  # pixels (about 1 cm) around the button that still count as on it
LEAVE_DELAY = 0.3  # seconds away before the dwell is cancelled
# After a selection, a different key that appears under the eyes starts its
# dwell after this pause, without the eyes leaving it (once in a row).
NEW_KEY_PAUSE = 1.0  # seconds

# How often the screen code runs (less often leaves more for eye tracking).
DWELL_FILL_RATE = 10   # times a second the dwell progress bar moves
GAZE_CHECK_RATE = 15   # times a second the pointer is checked against the buttons
REPEAT_TIME = DWELL_TIME  # held keys repeat at the dwell time

# A pointer past the screen's edge is pulled back this far inside it.
EDGE_INSET = 20  # pixels

# AUTOMATIC RE-CENTRING: the pointer drifts as the head moves, so after each
# selection it is moved part of the way back to the selected key's middle.
# Back or Backspace soon after takes that step back (the key was likely wrong).
DRIFT_CORRECTION = True
DRIFT_LEARN_RATE = 0.3  # share of each miss that is corrected
DRIFT_STEP_LIMIT_CM = 1.5  # cm, the most one key's miss can count for
DRIFT_MAX_CM = 5.0  # cm, the most the total correction can be
DRIFT_SETTLE_TIME = 1.0  # seconds ignored at the start of each dwell
UNDO_BUTTONS = ("Back", "Backspace")
DRIFT_UNDO_SECONDS = 20  # seconds within which Back/Backspace undoes a correction
DRIFT_PAGE_OPENERS = ("group", "category")  # buttons whose step Back undoes

# ROW MODE, for users whose up/down does not calibrate: one row is highlighted
# and buttons are picked by looking left/right. The Row up / Row down strips at
# the screen's edges move the highlight. "auto" = on when up/down failed or is
# more than ROW_MODE_UPDOWN_CM off, "on" = always, "off" = never.
ROW_MODE = "auto"
ROW_MODE_UPDOWN_CM = 2.0
ROW_STRIP_WIDTH = 0.1  # share of the screen's width for each strip
ROW_STRIP_ROLES = ("row_up", "row_down")
ROW_HIGHLIGHT = rgb(255, 214, 0)  # outline of the row in use
ROW_DIM = (0, 0, 0, 0.45)  # over the other rows

# Tapping a button picks it at once (a backup to gaze). Taps are marked "(tap)"
# on the dashboard. Set to False on test days so only gaze can pick.
TOUCH_SELECT = True
TAP_MARK = " (tap)"

# While the eyes are not seen, nothing fills up or gets picked.
FREEZE_WHEN_GAZE_LOST = True

# For testing only: a yellow circle where the gaze is (people tend to chase it).
SHOW_GAZE_POINTER = False

# Screen shown after calibration: "quickfire" or "keyboard".
HOME_SCREEN = "quickfire"

# RESTING: after REST_AFTER seconds without a selection (0 = never), or with the
# Rest button, a dark screen with one Start button is shown and the camera
# processes fewer pictures.
REST_AFTER = 180  # seconds

# ---------------------------------------------------------------------------
# 1. TEXT-TO-SPEECH
# ---------------------------------------------------------------------------
# Piper voice (works offline), set up once with: bash setup_voice.sh
# Backups: espeak-ng, then pyttsx3 on a laptop.

# Piper voice, in the "voices" folder (see setup_voice.sh).
VOICE_NAME = "en_US-amy-medium"
# How fast Piper talks: 1.0 is its normal speed, bigger numbers are slower.
VOICE_SPEED = 1.15
# How loud Piper is: 1.0 is the loudest. Lower it (e.g. 0.7) if it crackles.
VOICE_VOLUME = 1.0

VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
VOICE_FILE = os.path.join(VOICES_DIR, VOICE_NAME + ".onnx")
APLAY = shutil.which("aplay")  # None when not on Linux
# Piper is used when it, its voice and aplay are installed (the Pi).
USE_PIPER = (importlib.util.find_spec("piper") is not None
             and os.path.exists(VOICE_FILE) and APLAY is not None)
# Hide Piper's debug lines.
logging.getLogger("piper").setLevel(logging.INFO)

# Backup voice, espeak-ng: 100 is its normal volume and 200 the loudest.
ESPEAK_VOLUME = 200
# How fast espeak-ng talks, in words per minute (its normal is 175).
ESPEAK_SPEED = 140
ESPEAK = shutil.which("espeak-ng")   # None when espeak-ng isn't installed

_messages = queue.Queue()  # messages waiting for Piper
_player = None  # the program speaking right now
_piper_ok = USE_PIPER  # False if the Piper voice fails to load


def speak(text: str):
    """Says the text out loud; a new message cuts off the one being spoken."""
    global _player
    print(f"[TTS] Speaking: {text}")
    stop_speaking()
    if _piper_ok:
        # Only the newest message is said.
        while not _messages.empty():
            try:
                _messages.get_nowait()
            except queue.Empty:
                break
        _messages.put(text)
    elif ESPEAK:
        try:
            # "--": text starting with "-" is read, not taken as an option.
            _player = subprocess.Popen(
                [ESPEAK, "-a", str(ESPEAK_VOLUME), "-s", str(ESPEAK_SPEED), "--", text])
        except Exception as e:
            print(f"[TTS] Error: {e}")
    else:
        # Laptop: a new pyttsx3 engine each time (reusing one goes silent on Windows).
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[TTS] Error: {e}")


def stop_speaking():
    """Cuts off the message being spoken, if there is one."""
    if _player is not None and _player.poll() is None:
        _player.terminate()


def _piper_voice():
    """Background thread: loads the Piper voice once, then says each message."""
    global _player, _piper_ok
    try:
        from piper import PiperVoice, SynthesisConfig
        voice = PiperVoice.load(VOICE_FILE)
        settings = SynthesisConfig(length_scale=VOICE_SPEED, volume=VOICE_VOLUME)
        # Warm-up so the first message comes out quicker.
        list(voice.synthesize("Ready.", settings))
        print(f"[TTS] Piper voice ready: {VOICE_NAME}")
    except Exception as e:
        print(f"[TTS] The Piper voice could not start ({e}), using espeak-ng instead")
        _piper_ok = False
        try:
            speak(_messages.get_nowait())  # say a waiting message with espeak-ng
        except queue.Empty:
            pass
        return

    while True:
        text = _messages.get()  # waits for a message from speak()
        try:
            # Piper makes the sound one sentence at a time.
            pieces = list(voice.synthesize(text, settings))
        except Exception as e:
            print(f"[TTS] Error: {e}")
            continue
        if not pieces or not _messages.empty():
            continue  # nothing to say, or a newer message is waiting
        sound = b"".join(piece.audio_int16_bytes for piece in pieces)
        try:
            # aplay plays the raw 16-bit mono sound.
            _player = subprocess.Popen(
                [APLAY, "-q", "-t", "raw", "-f", "S16_LE", "-c", "1",
                 "-r", str(pieces[0].sample_rate), "-"],
                stdin=subprocess.PIPE)
            _player.stdin.write(sound)
            _player.stdin.close()
        except (BrokenPipeError, OSError):
            pass  # cut off by a newer message


def start_voice():
    """Starts the Piper voice in the background, when it is set up."""
    if _piper_ok:
        threading.Thread(target=_piper_voice, daemon=True).start()
    elif ESPEAK:
        print("[TTS] Piper voice not set up, using espeak-ng")


# Play silence so the amplifier stays on (no "tick" before each message).
KEEP_SPEAKER_AWAKE = True
_awake = None  # the program playing silence


def keep_speaker_awake():
    """Pi only: plays silence in the background until the app closes."""
    global _awake
    if not KEEP_SPEAKER_AWAKE or not (USE_PIPER or ESPEAK) or not APLAY:
        return
    if _awake is not None:
        return   # already playing
    try:
        _awake = subprocess.Popen(
            ["sh", "-c",
             "aplay -q -t raw -f S16_LE -r 48000 -c 2 /dev/zero & read x; kill $!"],
            stdin=subprocess.PIPE)
    except Exception as e:
        print(f"[TTS] Could not keep the speaker awake: {e}")


# ---------------------------------------------------------------------------
# 2. DWELL BUTTON
# ---------------------------------------------------------------------------
class DwellButton(Button):
    """A button that fills up while the pointer rests on it and calls on_selected after dwell_time.
    With repeatable=True it keeps firing while the pointer stays on it, like a held key."""

    progress = NumericProperty(0.0)  # 0.0 -> 1.0

    # Called at every dwell selection (GazeManager, for re-centring).
    selection_listener = None
    # Called when the button is tapped (GazeManager, see TOUCH_SELECT).
    tap_listener = None

    def __init__(self, on_selected=None, dwell_time=None,
                 repeatable=False, repeat_time=None,
                 bg_color=None, border_color=None, text_color=None,
                 radius=10, icon=None, icon_color=None, icon_layout="top",
                 selectable=True, image=None, **kwargs):
        super().__init__(**kwargs)
        self.on_selected = on_selected
        # None = the current DWELL_TIME / REPEAT_TIME.
        self.dwell_time = dwell_time if dwell_time is not None else DWELL_TIME
        self.repeatable = repeatable
        self.repeat_time = repeat_time if repeat_time is not None else REPEAT_TIME
        self._active_dwell_time = self.dwell_time
        self._event = None
        self.tapped = False  # True while on_selected runs for a tap

        self.selectable = selectable  # False: shown, but gaze ignores it
        self.bg_color = bg_color
        self.border_color = border_color
        self.radius = radius
        self.icon = icon
        self.image_texture = _load_symbol(image) if image else None
        self.icon_layout = icon_layout
        self.icon_color = icon_color or text_color or WHITE
        if text_color is not None:
            self.color = text_color

        if bg_color is not None:
            # Hide Kivy's default button image.
            self.background_normal = ""
            self.background_down = ""
            self.background_disabled_normal = ""
            self.background_color = (0, 0, 0, 0)

        # Layers: background, icon, dwell progress on top.
        self._bg_group = InstructionGroup()
        self.canvas.before.add(self._bg_group)
        self._icon_group = InstructionGroup()
        self.canvas.add(self._icon_group)
        self._progress_group = InstructionGroup()
        self.canvas.after.add(self._progress_group)

        self.bind(pos=self._redraw, size=self._redraw, disabled=self._redraw)
        self._redraw()

    # -- drawing --
    def set_bg_color(self, color):
        """Changes the fill colour, e.g. to highlight an active Shift key."""
        self.bg_color = color
        self._redraw()

    def _draw_image(self, box):
        """Draws the picture as large as fits in box, keeping its proportions."""
        bx, by, bw, bh = box
        tw, th = self.image_texture.size
        scale = min(bw / tw, bh / th)
        dw, dh = tw * scale, th * scale
        self._icon_group.add(Color(1, 1, 1, 1))  # white = the picture's own colours
        self._icon_group.add(Rectangle(texture=self.image_texture,
                                       pos=(bx + (bw - dw) / 2, by + (bh - dh) / 2),
                                       size=(dw, dh)))

    def _progress_color(self):
        """Light fill on dark buttons, dark fill on light ones."""
        if self.bg_color is None:
            return (0.2, 0.8, 0.4, 0.55)
        r, g, b = self.bg_color[:3]
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        return (0, 0, 0, 0.28) if brightness > 0.6 else (1, 1, 1, 0.32)

    def _redraw(self, *args):
        x, y = self.pos
        w, h = self.size
        r = min(self.radius, w / 2, h / 2)
        self.opacity = 0.35 if self.disabled else 1.0

        self._bg_group.clear()
        if self.bg_color is not None:
            self._bg_group.add(Color(*self.bg_color))
            self._bg_group.add(RoundedRectangle(pos=self.pos, size=self.size, radius=[r]))
        if self.border_color is not None:
            self._bg_group.add(Color(*self.border_color))
            self._bg_group.add(Line(rounded_rectangle=(x, y, w, h, r), width=1.1))

        self._icon_group.clear()
        if self.icon or self.image_texture is not None:
            bg = self.bg_color or (0, 0, 0, 1)
            if self.icon_layout == "top":
                # Picture on top, text at the bottom.
                self.text_size = (w, h)
                self.halign, self.valign = "center", "bottom"
                self.padding = [6, 4, 6, h * 0.08]
                box = (x, y + h * 0.34, w, h * 0.58)
            elif self.icon_layout == "left":
                # Picture on the left, text beside it.
                icon_w = h * 0.5
                self.text_size = (w, h)
                self.halign, self.valign = "center", "middle"
                self.padding = [icon_w + 8, 0, 4, 0]
                box = (x + 8, y + h * 0.25, icon_w, h * 0.5)
            else:  # "center": picture only
                box = (x + w * 0.2, y + h * 0.2, w * 0.6, h * 0.6)

            if self.image_texture is not None:
                self._draw_image(box)
            else:
                draw_icon(self._icon_group, self.icon, *box, self.icon_color, bg)

        self._progress_group.clear()
        if self.progress > 0:
            self._progress_group.add(Color(*self._progress_color()))
            pw = w * self.progress
            self._progress_group.add(
                RoundedRectangle(pos=self.pos, size=(pw, h), radius=[min(r, pw / 2)])
            )

    # -- tapping (see TOUCH_SELECT) --
    def on_release(self):
        """A tap or click picks the button at once (see TOUCH_SELECT)."""
        if not TOUCH_SELECT or not self.selectable or self.disabled or self.on_selected is None:
            return
        self.cancel_dwell()
        if DwellButton.tap_listener is not None:
            DwellButton.tap_listener(self)
        self.tapped = True
        try:
            self.on_selected(self)
        finally:
            self.tapped = False

    # -- dwell timing --
    @property
    def dwelling(self):
        """True while the progress bar is filling."""
        return self._event is not None

    def start_dwell(self, dwell_time=None):
        if self._event is not None:
            return  # already dwelling
        self.progress = 0.0
        self._active_dwell_time = dwell_time if dwell_time is not None else self.dwell_time
        self._event = Clock.schedule_interval(self._tick, 1 / DWELL_FILL_RATE)

    def cancel_dwell(self):
        if self._event is not None:
            self._event.cancel()
            self._event = None
        self.progress = 0.0
        self._redraw()

    def _tick(self, dt):
        self.progress = min(1.0, self.progress + dt / self._active_dwell_time)
        self._redraw()
        if self.progress >= 1.0:
            self.cancel_dwell()
            # Tell GazeManager first, while the button still has its text.
            if DwellButton.selection_listener is not None:
                DwellButton.selection_listener(self)
            if self.on_selected:
                self.on_selected(self)
            if self.repeatable:
                # Repeatable: re-arm until the pointer leaves (GazeManager then cancels it).
                self.start_dwell(dwell_time=self.repeat_time)
            return False  # stop this Clock event


class Panel(BoxLayout):
    """A BoxLayout with a rounded background and an optional outline."""

    def __init__(self, bg_color=PANEL_BG, border_color=None, radius=10, **kwargs):
        super().__init__(**kwargs)
        self._bg_color = bg_color
        self._border_color = border_color
        self._radius = radius
        self._group = InstructionGroup()
        self.canvas.before.add(self._group)
        self.bind(pos=self._redraw, size=self._redraw)

    def _redraw(self, *args):
        x, y = self.pos
        w, h = self.size
        r = min(self._radius, w / 2, h / 2)
        self._group.clear()
        self._group.add(Color(*self._bg_color))
        self._group.add(RoundedRectangle(pos=self.pos, size=self.size, radius=[r]))
        if self._border_color is not None:
            self._group.add(Color(*self._border_color))
            self._group.add(Line(rounded_rectangle=(x, y, w, h, r), width=1.1))


def _find_dwell_buttons(widget):
    found = []
    if isinstance(widget, DwellButton):
        found.append(widget)
    for child in widget.children:
        found.extend(_find_dwell_buttons(child))
    return found


def row_mode_wanted(gaze_engine):
    """True if Row mode should be on (see ROW_MODE)."""
    if ROW_MODE == "on":
        return True
    if ROW_MODE != "auto" or gaze_engine is None or not gaze_engine.is_calibrated():
        return False  # "off", or no gaze (mouse)
    if not gaze_engine.is_vertical_calibrated():
        return True
    error = getattr(gaze_engine, "updown_error_cm", None)
    return error is not None and error > ROW_MODE_UPDOWN_CM


def _group_rows(buttons):
    """Groups buttons into rows [left, bottom, right, top], top row first."""
    boxes = []
    for btn in buttons:
        left, bottom = btn.to_window(btn.x, btn.y)
        right, top = btn.to_window(btn.right, btn.top)
        boxes.append((left, bottom, right, top))
    boxes.sort(key=lambda box: -(box[1] + box[3]) / 2)
    rows = []
    for left, bottom, right, top in boxes:
        middle = (bottom + top) / 2
        if rows and rows[-1][1] <= middle <= rows[-1][3]:
            row = rows[-1]
            rows[-1] = [min(row[0], left), min(row[1], bottom), max(row[2], right), max(row[3], top)]
        else:
            rows.append([left, bottom, right, top])
    return rows


# ---------------------------------------------------------------------------
# 2b. CAREGIVER DASHBOARD (see dashboard_link.py)
# ---------------------------------------------------------------------------
def tell_dashboard(btn, interface, kind, element_id, text, category=None, navigation=False):
    """Records the selection of btn in the dashboard's Activity Log."""
    if getattr(btn, "tapped", False):  # picked by a tap, not by gaze
        dashboard.selection(interface, kind, element_id, text + TAP_MARK, category=category,
                            dwell_s=0.0, pointer=None, navigation=navigation)
        return
    pointer = None
    gaze_manager = getattr(App.get_running_app(), "gaze_manager", None)
    if gaze_manager is not None:
        x, y = gaze_manager.get_pointer_pos()
        pointer = (x, Window.height - y)  # pixels from the top left
    dashboard.selection(interface, kind, element_id, text, category=category,
                        dwell_s=btn._active_dwell_time, pointer=pointer, navigation=navigation)


DEVICE_DWELL_TIME = DWELL_TIME  # restored when a dashboard session ends


def set_dwell_time(seconds):
    """Changes every button's dwell time (dashboard session, or back to the default)."""
    global DWELL_TIME, REPEAT_TIME, CLEAR_CONFIRM_TIME
    if seconds == DWELL_TIME:
        return
    DWELL_TIME = REPEAT_TIME = seconds  # for buttons made from now on
    CLEAR_CONFIRM_TIME = REPEAT_TIME + 2.0
    for screen in App.get_running_app().root.screens:
        for btn in _find_dwell_buttons(screen):  # and the ones already there
            btn.dwell_time = btn.repeat_time = seconds
    print(f"[Dashboard] Dwell time is now {seconds:.1f} seconds")


# ---------------------------------------------------------------------------
# 3. GAZE MANAGER
# ---------------------------------------------------------------------------
class GazeManager:
    """Moves the pointer with the gaze (the mouse when not calibrated) and starts or cancels the
    dwell of the button under it."""

    def __init__(self, screen_manager: ScreenManager, gaze_engine: GazeEngine = None):
        self.sm = screen_manager
        self.gaze_engine = gaze_engine
        self._current_target = None
        self._away_since = None  # when the pointer left the current button
        self._blocked = None  # a new button the gaze must leave first
        self._after_tap = False  # the last selection was a tap
        self._frozen = False  # the eyes are not seen
        # Last selection, when, and whether NEW_KEY_PAUSE started its dwell.
        self._selected, self._selected_at = None, 0.0
        self._paused_start, self._after_paused_start = None, False
        self.pointer = None  # test circle (SHOW_GAZE_POINTER)
        # Re-centring: gaze points during this dwell, and the last step (for undo).
        self._dwell_points = []
        self._dwell_start = None
        self._last_step = None
        # Row mode: the screen's rows, the one in use, and the page they belong to.
        self.row_mode = False
        self.on_row_mode = None  # called with True/False
        self.row_overlay = None  # RowOverlay, added in on_start
        self._rows, self._row_index, self._rows_page = [], 0, None
        DwellButton.selection_listener = self._on_selection
        DwellButton.tap_listener = self._on_tap
        Clock.schedule_interval(self._update, 1 / GAZE_CHECK_RATE)

    def _gaze_pos(self):
        """Gaze pointer (x, y) in window pixels, or None if gaze doesn't give both."""
        if self.gaze_engine is None:
            return None
        x, y = self.gaze_engine.get_screen_x(), self.gaze_engine.get_screen_y()
        if self.row_mode:
            y = self._row_y()  # Row mode: the highlighted row
        return None if x is None or y is None else (x, y)

    def gaze_lost(self):
        """True when calibrated but the eyes are not seen right now."""
        engine = self.gaze_engine
        return (FREEZE_WHEN_GAZE_LOST and engine is not None and engine.is_calibrated()
                and engine.get_screen_x() is None)

    def _row_y(self):
        """Middle of the highlighted row (Row mode), or None."""
        if not self._rows:
            return None
        row = self._rows[self._row_index]
        return (row[1] + row[3]) / 2

    def move_row(self, step):
        """Row up (step -1) or Row down (+1), wrapping at the ends."""
        if self._rows:
            self._row_index = (self._row_index + step) % len(self._rows)
            self._draw_rows()

    def _update_rows(self, screen, buttons):
        """Finds the rows of the screen on display (Row mode)."""
        if not self.row_mode or screen.name == "calibration":
            self._rows = []
        else:
            in_rows = [btn for btn in buttons if getattr(btn, "role", None) not in ROW_STRIP_ROLES]
            rows = _group_rows(in_rows)
            page = (screen.name, tuple(sorted(id(btn) for btn in in_rows)))
            if page != self._rows_page:
                # New page: start on the first row under the top bar.
                self._rows_page = page
                self._row_index = 1 if len(rows) > 1 else 0
            self._row_index = min(self._row_index, max(len(rows) - 1, 0))
            self._rows = rows
        self._draw_rows()

    def _draw_rows(self):
        if self.row_overlay is not None:
            show = self._rows if len(self._rows) > 1 else []
            self.row_overlay.show(show, self._row_index)

    def get_pointer_pos(self):
        x, y = Window.mouse_pos  # start from the mouse...
        if self.gaze_engine is not None:
            gaze_x = self.gaze_engine.get_screen_x()  # None if not available
            gaze_y = self.gaze_engine.get_screen_y()
            if gaze_x is not None:
                x = gaze_x  # ...and use gaze where it works
            if gaze_y is not None:
                y = gaze_y
        if self.row_mode and self._rows:
            y = self._row_y()  # Row mode: the highlighted row
        # Keep the pointer on the screen.
        x = min(max(x, EDGE_INSET), Window.width - EDGE_INSET)
        y = min(max(y, EDGE_INSET), Window.height - EDGE_INSET)
        return (x, y)

    def _on_tap(self, btn):
        """A tap counts as a selection, without re-centring."""
        if self._current_target is not None:
            self._current_target.cancel_dwell()
        # Keep the tapped button as the target so dwell doesn't pick it again.
        self._current_target, self._away_since, self._blocked = btn, None, None
        self._dwell_points, self._dwell_start = [], None
        self._selected, self._selected_at = self._key_of(btn), time.time()
        self._after_paused_start = False
        self._paused_start = None
        self._after_tap = True

    def _on_selection(self, btn):
        """At every dwell selection: re-centre the pointer towards the button's middle."""
        self._selected, self._selected_at = self._key_of(btn), time.time()
        self._after_paused_start = btn is self._paused_start
        self._paused_start = None
        points = [(x, y) for t, x, y in self._dwell_points
                  if self._dwell_start is not None and t - self._dwell_start >= DRIFT_SETTLE_TIME]
        self._dwell_points = []
        self._dwell_start = time.time()  # a repeating key starts its next dwell now
        if (not DRIFT_CORRECTION or self.gaze_engine is None or btn is not self._current_target
                or self.sm.current == "calibration"
                or getattr(btn, "role", None) in ROW_STRIP_ROLES):  # strips: no middle
            return
        shift_x, shift_y = self.gaze_engine.get_pointer_shift()
        if btn.text in UNDO_BUTTONS:
            # Back/Backspace right after: undo the last step (that key was likely wrong).
            last = self._last_step
            if (last is not None and time.time() - last[2] <= DRIFT_UNDO_SECONDS
                    and (btn.text != "Back" or last[3])):
                step_x, step_y = last[0], last[1]
                self.gaze_engine.set_pointer_shift(shift_x - step_x, shift_y - step_y)
                print(f"[Re-centring] {btn.text}: last step taken back")
            self._last_step = None
            return
        # Too few gaze readings: no step.
        if len(points) < 10:
            return
        middle_x, middle_y = btn.to_window(*btn.center)
        limit_x = DRIFT_STEP_LIMIT_CM * SCREEN_PX_PER_CM_X
        limit_y = DRIFT_STEP_LIMIT_CM * SCREEN_PX_PER_CM_Y
        miss_x = min(max(np.mean([x for x, y in points]) - middle_x, -limit_x), limit_x)
        miss_y = min(max(np.mean([y for x, y in points]) - middle_y, -limit_y), limit_y)
        if self.row_mode:
            miss_y = 0  # not measured in Row mode
        max_x, max_y = DRIFT_MAX_CM * SCREEN_PX_PER_CM_X, DRIFT_MAX_CM * SCREEN_PX_PER_CM_Y
        new_x = min(max(shift_x - DRIFT_LEARN_RATE * miss_x, -max_x), max_x)
        new_y = min(max(shift_y - DRIFT_LEARN_RATE * miss_y, -max_y), max_y)
        self.gaze_engine.set_pointer_shift(new_x, new_y)
        self._last_step = (new_x - shift_x, new_y - shift_y, time.time(),
                           getattr(btn, "role", None) in DRIFT_PAGE_OPENERS)
        # Kivy's y grows upwards: positive miss_y = pointer too high.
        print(f"[Re-centring] {btn.text or getattr(btn, 'icon', '') or 'button'}: pointer was "
              f"{miss_x / SCREEN_PX_PER_CM_X:+.1f} cm right, {miss_y / SCREEN_PX_PER_CM_Y:+.1f} cm up; "
              f"correction now {new_x / SCREEN_PX_PER_CM_X:+.1f} cm right, "
              f"{new_y / SCREEN_PX_PER_CM_Y:+.1f} cm up")

    def _update(self, dt):
        active_screen = self.sm.current_screen
        if active_screen is None:
            return
        wanted = row_mode_wanted(self.gaze_engine)
        if wanted != self.row_mode:
            self.row_mode = wanted
            self._rows_page = None
            print(f"[Row mode] {'on: up/down is not used, pick rows with Row up/Row down' if wanted else 'off'}")
            if self.on_row_mode is not None:
                self.on_row_mode(wanted)
        buttons = [btn for btn in _find_dwell_buttons(active_screen)
                   if not btn.disabled and btn.selectable]  # disabled or display-only
        if active_screen.name == "calibration" and self._gaze_pos() is None:
            buttons = []  # Start: eyes or a tap only, never a resting mouse
        self._update_rows(active_screen, buttons)
        pos = self.get_pointer_pos()
        if self.gaze_lost():
            # Eyes not seen: cancel any dwell and pick nothing until they are back.
            if self._current_target is not None:
                self._current_target.cancel_dwell()
            self._current_target, self._away_since, self._blocked = None, None, None
            self._dwell_points, self._dwell_start = [], None
            if self.pointer is not None:
                self.pointer.move_to(pos, visible=False)
            if not self._frozen:
                self._frozen = True
                print("[Gaze] eyes not seen: nothing is selected until they are seen again")
            return
        if self._frozen:
            self._frozen = False
            print("[Gaze] eyes seen again")
        if self._current_target is not None:
            gaze = self._gaze_pos()
            if gaze is not None:
                self._dwell_points.append((time.time(), *gaze))
        if self.pointer is not None:
            # Hidden during calibration.
            self.pointer.move_to(pos, visible=active_screen.name != "calibration")
        current = self._current_target

        # A new page replaced the selected button: ignore the button now under the
        # gaze until the gaze leaves it.
        if current is not None and current not in buttons:
            current.cancel_dwell()
            self._current_target = current = None
            self._away_since = None
            self._dwell_points, self._dwell_start = [], None
            self._blocked = self._button_at(buttons, pos)
        # Unless it is a different key (see NEW_KEY_PAUSE).
        if self._blocked is not None:
            if self._blocked in buttons and self._is_near(self._blocked, pos, STICKY_MARGIN):
                if self._new_key_ready(self._blocked):
                    self._current_target, self._blocked = self._blocked, None
                    self._start_paused_dwell(self._current_target)
                return
            self._blocked = None
            self._after_tap = False  # the pointer left

        # Sticky: keep dwelling while the pointer is near the current button...
        if current in buttons and self._is_near(current, pos, STICKY_MARGIN):
            self._away_since = None
            # A key that changed in place (group -> letter) starts again.
            if not current.dwelling and self._new_key_ready(current):
                self._start_paused_dwell(current)
            return
        # ...and give it up only after LEAVE_DELAY away.
        if current in buttons:
            now = time.time()
            if self._away_since is None:
                self._away_since = now
            if now - self._away_since < LEAVE_DELAY:
                return
        self._away_since = None

        target = self._button_at(buttons, pos)
        if target is not current:
            if current is not None:
                current.cancel_dwell()
            self._after_tap = False  # the pointer moved on
            self._current_target = target
            self._dwell_points, self._dwell_start = [], time.time()  # see _on_selection
            if target is not None:
                target.start_dwell()

    @staticmethod
    def _key_of(btn):
        """What a button stands for: its keyboard role and value, else its text."""
        role = getattr(btn, "role", None)
        if role is not None:
            return (role, getattr(btn, "key_value", ""))
        return (None, getattr(btn, "key_value", None) or btn.text)

    def _new_key_ready(self, btn):
        """True if btn is a new key and NEW_KEY_PAUSE has passed."""
        return (not self._after_paused_start and not self._after_tap
                and self._key_of(btn) != self._selected
                and time.time() - self._selected_at >= NEW_KEY_PAUSE)

    def _start_paused_dwell(self, btn):
        """Starts a dwell without the eyes leaving the key (NEW_KEY_PAUSE)."""
        self._dwell_points, self._dwell_start = [], time.time()  # see _on_selection
        self._paused_start = btn
        btn.start_dwell()

    def _button_at(self, buttons, pos):
        """The button directly under `pos`, or None."""
        for btn in buttons:
            if self._is_near(btn, pos, 0):
                return btn
        return None

    @staticmethod
    def _is_near(btn, pos, margin):
        """True if pos is on the button or within margin pixels of it."""
        x, y = btn.to_widget(*pos)
        return (btn.x - margin <= x <= btn.right + margin
                and btn.y - margin <= y <= btn.top + margin)


class GazePointer(Widget):
    """Yellow test circle (SHOW_GAZE_POINTER)."""

    RADIUS = 18  # pixels

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            self._color = Color(1, 0.85, 0, 1)  # yellow
            self._circle = Line(circle=(0, 0, self.RADIUS), width=2)

    def move_to(self, pos, visible=True):
        self._color.a = 1 if visible else 0
        self._circle.circle = (pos[0], pos[1], self.RADIUS)


class RowOverlay(Widget):
    """Row mode: outlines the row in use and dims the others."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._group = InstructionGroup()
        self.canvas.add(self._group)
        self._shown = None

    def show(self, rows, index):
        key = (tuple(tuple(round(v) for v in row) for row in rows), index)
        if key == self._shown:
            return
        self._shown = key
        self._group.clear()
        if rows:
            # Dim the area between the strips, except the row in use.
            area_left, area_right = min(r[0] for r in rows), max(r[2] for r in rows)
            area_bottom, area_top = min(r[1] for r in rows), max(r[3] for r in rows)
            left, bottom, right, top = rows[index]
            self._group.add(Color(*ROW_DIM))
            self._group.add(Rectangle(pos=(area_left, top + 6),
                                      size=(area_right - area_left, max(area_top - top - 6, 0))))
            self._group.add(Rectangle(pos=(area_left, area_bottom),
                                      size=(area_right - area_left, max(bottom - 6 - area_bottom, 0))))
            self._group.add(Color(*ROW_HIGHLIGHT))
            self._group.add(Line(rectangle=(left - 6, bottom - 6, right - left + 12, top - bottom + 12),
                                 width=4))


# ---------------------------------------------------------------------------
# 3b. NAVIGATION (KEYBOARD and QUICK FIRES tabs)
# ---------------------------------------------------------------------------
class Navigator:
    """Switches between the Keyboard and Quick Fires screens."""

    def __init__(self, screen_manager):
        self.sm = screen_manager

    def go(self, name):
        if name == "rest" and self.sm.current != "rest":
            # Start on the resting screen returns here.
            self.sm.get_screen("rest").return_to = self.sm.current
        self.sm.current = name


# ---------------------------------------------------------------------------
# 4. KEYBOARD (Figure 29, revised)
# ---------------------------------------------------------------------------
# Each screen: a top bar (the two tabs and the screen's controls), then big
# buttons. Every target is at least about 3 x 3 cm.
TOP_BAR_HEIGHT = 0.24  # share of the screen's height
TAB_WIDTH = 0.15  # share of the screen's width per tab

# Quick Access replies along the bottom: spoken at once.
QUICK_ACCESS_WORDS = [("Yes", GREEN), ("No", TILE_RED), ("Please", PINK), ("Thank you", PINK)]

# Clear needs a second selection within CLEAR_CONFIRM_TIME seconds.
CLEAR_CONFIRM_TIME = REPEAT_TIME + 2.0

# LETTER GROUPS instead of QWERTY keys (too small for the tracker): pick a group,
# then a letter. 123 switches to numbers and punctuation.
LETTER_GROUPS = ["ABCDE", "FGHIJ", "KLMNO", "PQRST", "UVWXYZ"]
SYMBOL_GROUPS = ["12345", "67890", ".,?!'", "-/:;()", "&@\"#%"]

# Positions on the 2 x 4 grid: 0 1 2 3 / 4 5 6 7.
KEY_GRID_COLUMNS = 4
GROUP_CELLS = (0, 1, 2, 4, 5)  # the 5 groups
SPACE_CELL = 3
MODE_CELL = 6                   # 123 / ABC
SHIFT_CELL = 7
BACK_CELL = 7  # back to the groups

SENTENCE_END = (".", "?", "!")
ATTACHING_PUNCTUATION = ".,?!"  # punctuation that attaches to the word before

# Look of the keyboard keys.
KEY_STYLE = dict(bg_color=KEY_FACE, border_color=KEY_BORDER, text_color=WHITE, radius=8)


class AppScreen(Screen):
    """Shared frame of the Keyboard and Quick Fires screens: the top bar with the tabs and
    controls, and the content below."""

    TAB = ""  # "keyboard" or "quickfire"

    def __init__(self, navigator, **kwargs):
        super().__init__(**kwargs)
        self.nav = navigator

    def build_frame(self, controls, content):
        # [Row up strip] frame [Row down strip]; strips only in Row mode.
        self._outer = Panel(bg_color=FRAME_BG, radius=0, orientation="horizontal")
        self._strips = [self._row_strip("Row up", "arrow_up", "row_up", -1),
                        self._row_strip("Row down", "arrow_down", "row_down", +1)]
        frame = Panel(bg_color=FRAME_BG, radius=0, orientation="vertical", padding=10, spacing=10)
        top_bar = BoxLayout(orientation="horizontal", spacing=10, size_hint_y=TOP_BAR_HEIGHT)
        top_bar.add_widget(self._tab("KEYBOARD", "keyboard", "keyboard", PURPLE))
        top_bar.add_widget(self._tab("QUICK FIRES", "flame", "quickfire", ORANGE))
        controls.size_hint_x = 1 - 2 * TAB_WIDTH
        top_bar.add_widget(controls)
        frame.add_widget(top_bar)
        content.size_hint_y = 1 - TOP_BAR_HEIGHT
        frame.add_widget(content)
        self._outer.add_widget(frame)
        self.add_widget(self._outer)

    def _row_strip(self, text, icon, role, step):
        """A tall Row up / Row down strip (Row mode)."""
        box = BoxLayout(size_hint_x=ROW_STRIP_WIDTH, padding=[10, 10, 0, 10] if step < 0 else [0, 10, 10, 10])
        btn = DwellButton(text=text, bold=True, font_size=26, icon=icon, icon_layout="top",
                          bg_color=KEY_FACE, text_color=WHITE, radius=12, repeatable=True,
                          on_selected=lambda b: self._on_row_step(b, step))
        btn.role = role
        box.add_widget(btn)
        return box

    def set_row_mode(self, on):
        """Shows or hides the Row up / Row down strips."""
        up, down = self._strips
        if on and up.parent is None:
            for box in self._strips:
                # The dwell time may have changed while hidden.
                btn = box.children[0]
                btn.dwell_time = btn.repeat_time = REPEAT_TIME
            self._outer.add_widget(up, index=len(self._outer.children))  # leftmost
            self._outer.add_widget(down)  # rightmost
        elif not on and up.parent is not None:
            self._outer.remove_widget(up)
            self._outer.remove_widget(down)

    def _on_row_step(self, btn, step):
        tell_dashboard(btn, self.TAB, "control", btn.role.replace("_", ":"), btn.text,
                       navigation=True)
        App.get_running_app().gaze_manager.move_row(step)

    def _tab(self, text, icon, screen_name, color):
        """A tab; the current screen's tab is outlined and not selectable."""
        own = screen_name == self.TAB
        return DwellButton(text=text, bold=True, font_size=26, icon=icon, icon_layout="top",
                           bg_color=color, text_color=WHITE, radius=12,
                           border_color=WHITE if own else None, selectable=not own,
                           size_hint_x=TAB_WIDTH,
                           on_selected=lambda b: self._on_tab(b, screen_name))

    def _on_tab(self, btn, screen_name):
        tell_dashboard(btn, self.TAB, "control", f"tab:{screen_name}", f"{btn.text} tab",
                       navigation=True)
        self.nav.go(screen_name)


class KeyboardScreen(AppScreen):
    """Figure 29, revised: type a message with the letter groups, then Speak it."""

    TAB = "keyboard"

    def __init__(self, navigator, **kwargs):
        super().__init__(navigator, **kwargs)
        self.composed_text = ""
        self.shift_on = True  # a message starts with a capital
        self.symbols_on = False  # True: numbers and punctuation
        self.open_group = None  # None: groups shown; else the open group's characters
        self.cells = []  # the 8 big buttons, row by row
        self.spoken = False  # True after Speak: the next character starts a new message
        self._clear_timer = None  # set while Clear waits for its second selection

        content = BoxLayout(orientation="vertical", spacing=10)
        content.add_widget(self._build_keys_panel())
        content.add_widget(self._build_quick_access())
        self.build_frame(self._build_top_controls(), content)

        self._refresh_keys()
        self._refresh_text()

    # -- layout --
    def _build_top_controls(self):
        """The message box, Speak, Backspace and Clear."""
        row = BoxLayout(orientation="horizontal", spacing=10)

        box = Panel(bg_color=WHITE, radius=12, padding=[14, 10], size_hint_x=0.4)
        # The message scrolls to show the newest text.
        self.text_scroll = ScrollView(do_scroll_x=False, bar_width=4)
        self.text_label = Label(font_size=44, halign="left", valign="top", size_hint_y=None,
                                markup=True)  # markup: for the cursor
        # Wrap at the box width.
        self.text_label.bind(width=lambda w, v: setattr(w, "text_size", (v, None)))
        self.text_label.bind(texture_size=self._fit_text_height)
        self.text_scroll.add_widget(self.text_label)
        box.add_widget(self.text_scroll)
        row.add_widget(box)

        action = dict(bg_color=PURPLE, text_color=WHITE, radius=12, bold=True, font_size=26,
                      icon_layout="top", size_hint_x=0.2)
        row.add_widget(DwellButton(text="Speak", icon="speak",
                                   on_selected=self._speak_text, **action))
        row.add_widget(DwellButton(text="Backspace", icon="erase", repeatable=True,
                                   on_selected=self._backspace, **action))
        # Repeatable: keeping the eyes on it gives the second selection.
        self.clear_btn = DwellButton(text="Clear", repeatable=True, on_selected=self._on_clear,
                                     **{**action, "font_size": 34})
        row.add_widget(self.clear_btn)
        return row

    def _build_keys_panel(self):
        """The 8 big buttons; _refresh_keys() gives each its job."""
        panel = Panel(bg_color=PANEL_BG, border_color=KEY_BORDER, orientation="vertical",
                      size_hint_y=0.67, padding=8)
        grid = GridLayout(cols=KEY_GRID_COLUMNS, spacing=8)
        for _ in range(2 * KEY_GRID_COLUMNS):
            btn = DwellButton(on_selected=self._on_cell, **KEY_STYLE)
            self.cells.append(btn)
            grid.add_widget(btn)
        panel.add_widget(grid)
        return panel

    def _build_quick_access(self):
        """The Quick Access replies (QUICK_ACCESS_WORDS)."""
        row = GridLayout(cols=len(QUICK_ACCESS_WORDS), spacing=10, size_hint_y=0.33)
        for word, color in QUICK_ACCESS_WORDS:
            btn = DwellButton(text=word, bold=True, font_size=40, bg_color=color,
                              text_color=DARK_TEXT, radius=12,
                              on_selected=self._on_quick_access)
            row.add_widget(btn)
        return row

    # -- refreshing --
    def _fit_text_height(self, *args):
        """The label is exactly as tall as its text."""
        self.text_label.height = self.text_label.texture_size[1]

    def _scroll_to_end(self, dt):
        """Scrolls a long message to its end (scroll_y: 1 = top, 0 = bottom)."""
        overflowing = self.text_label.height > self.text_scroll.height
        self.text_scroll.scroll_y = 0 if overflowing else 1

    def _refresh_text(self):
        # escape_markup: show [ and & as typed.
        if self.composed_text and self.spoken:
            # Grey after Speak.
            self.text_label.text = escape_markup(self.composed_text)
            self.text_label.color = PLACEHOLDER
        elif self.composed_text:
            # Purple cursor (also shows a trailing Space).
            self.text_label.text = escape_markup(self.composed_text) + "[color=6a1fad]|[/color]"
            self.text_label.color = DARK_TEXT
        else:
            self.text_label.text = "Type, here..."
            self.text_label.color = PLACEHOLDER
        # Scroll on the next frame, once Kivy has measured the text.
        Clock.schedule_once(self._scroll_to_end, 0)

    def _refresh_keys(self):
        """Shows either the 5 groups or the characters of the open group."""
        if self.open_group is None:
            groups = SYMBOL_GROUPS if self.symbols_on else LETTER_GROUPS
            for cell, group in zip(GROUP_CELLS, groups):
                self._set_cell(cell, "group", text=" ".join(self._shown(group)), value=group)
            self._set_cell(SPACE_CELL, "space", text="Space")
            self._set_cell(MODE_CELL, "mode", text="ABC" if self.symbols_on else "123")
            self._set_cell(SHIFT_CELL, "shift", icon="shift")
        else:
            for cell in range(len(self.cells)):
                if cell == BACK_CELL:
                    self._set_cell(cell, "back", text="Back", icon="arrow_left")
                elif cell < len(self.open_group):
                    ch = self.open_group[cell]
                    self._set_cell(cell, "char", text=self._shown(ch), value=ch)
                else:
                    self._set_cell(cell, "empty")

    def _shown(self, chars):
        """Letters are shown in the case they will be typed in."""
        return chars.upper() if self.shift_on else chars.lower()

    def _set_cell(self, index, role, text="", value="", icon=None):
        """Gives one big button its job: group, char, space, mode, shift, back or empty."""
        btn = self.cells[index]
        btn.role = role
        btn.key_value = value
        btn.text = text
        btn.font_size = 64 if role == "char" else 40
        btn.icon = icon
        btn.icon_layout = "left" if text else "center"
        # Undo text placement an icon may have set.
        btn.text_size, btn.halign, btn.valign, btn.padding = (None, None), "center", "middle", [0, 0, 0, 0]
        # No key repeats: the eyes must leave and come back.
        btn.repeatable = False
        btn.disabled = role == "empty" or (role == "shift" and self.symbols_on)
        highlight = role == "shift" and self.shift_on and not self.symbols_on
        btn.set_bg_color(PURPLE if highlight else KEY_FACE)  # also redraws

    def _after_edit(self):
        """Runs after every change to the message."""
        self._update_auto_shift()
        self._refresh_text()
        self._refresh_keys()

    def _update_auto_shift(self):
        """Capitalises the next letter at the start of a message or sentence."""
        text = self.composed_text
        trimmed = text.rstrip(" ")
        self.shift_on = (
            trimmed == ""
            or trimmed.endswith("\n")
            or (text.endswith(" ") and trimmed.endswith(SENTENCE_END))
        )

    # -- actions --
    def _on_cell(self, btn):
        """One of the 8 big buttons was selected."""
        self._tell_dashboard_cell(btn)
        if btn.role == "group":
            self.open_group = btn.key_value
        elif btn.role == "char":
            self.open_group = None  # back to the groups
            self._type_char(btn.key_value)
        elif btn.role == "back":
            self.open_group = None
        elif btn.role == "space":
            # A message never starts with a space.
            if self.composed_text and not self.spoken:
                self._insert_text(" ")
        elif btn.role == "mode":
            self.symbols_on = not self.symbols_on
        elif btn.role == "shift":
            self.shift_on = not self.shift_on
        self._refresh_keys()

    def _tell_dashboard_cell(self, btn):
        """Records a big-button selection on the dashboard."""
        if btn.role == "char":
            tell_dashboard(btn, "keyboard", "key", f"kb:{btn.key_value.upper()}", btn.text)
        elif btn.role == "space":
            tell_dashboard(btn, "keyboard", "key", "kb:space", "Space")
        elif btn.role == "group":
            tell_dashboard(btn, "keyboard", "control", f"kb:group:{btn.key_value}",
                           f"Group {btn.text}", navigation=True)
        else:
            names = {"back": "Back to the groups", "mode": f"{btn.text} (switch characters)",
                     "shift": "Shift"}
            tell_dashboard(btn, "keyboard", "control", f"kb:{btn.role}",
                           names.get(btn.role, btn.role), navigation=True)

    def _start_new_if_spoken(self):
        """After Speak, the first new character starts a new message."""
        if self.spoken:
            self.composed_text = ""
            self.spoken = False

    def _type_char(self, ch):
        self._start_new_if_spoken()
        if ch.isalpha():
            ch = ch.upper() if self.shift_on else ch.lower()
        if ch in ATTACHING_PUNCTUATION and self.composed_text.endswith(" "):
            # Punctuation after a Space joins the word before: "more ." -> "more. "
            self.composed_text = self.composed_text[:-1] + ch + " "
            self._after_edit()
            return
        self._insert_text(ch)

    def _insert_text(self, text):
        self._start_new_if_spoken()
        self.composed_text += text
        self._after_edit()

    def _backspace(self, btn):
        tell_dashboard(btn, "keyboard", "key", "kb:backspace", "Backspace")
        self.spoken = False  # Backspace after Speak edits the spoken message
        self.composed_text = self.composed_text[:-1]
        self._after_edit()

    def _on_clear(self, btn):
        """First selection: ask (see CLEAR_CONFIRM_TIME). Second: erase."""
        if self._clear_timer is None:
            tell_dashboard(btn, "keyboard", "control", "kb:clear?", "Clear? (asks to confirm)",
                           navigation=True)
            btn.text = "Clear?"
            self._clear_timer = Clock.schedule_once(lambda dt: self._cancel_clear(),
                                                    CLEAR_CONFIRM_TIME)
            return
        tell_dashboard(btn, "keyboard", "control", "kb:clear", "Clear (message erased)")
        self._cancel_clear()
        self.composed_text = ""
        self.spoken = False
        self._after_edit()

    def _cancel_clear(self):
        """Clear goes back to waiting for a first selection."""
        if self._clear_timer is not None:
            self._clear_timer.cancel()
            self._clear_timer = None
        self.clear_btn.text = "Clear"

    def say(self, text):
        """Speaks a Quick Access reply or Quick Fire phrase and shows it in the message box."""
        self._cancel_clear()
        self.composed_text = text
        self.spoken = True
        self.shift_on = True  # next message starts with a capital
        self._refresh_text()
        self._refresh_keys()
        # Laptop speech blocks the screen, so start it once the text is shown.
        Clock.schedule_once(lambda dt: speak(text), 0.1)

    def _on_quick_access(self, btn):
        tell_dashboard(btn, "keyboard", "shorthand", f"kb:sh:{slug(btn.text)}", btn.text)
        self.say(btn.text)

    def _speak_text(self, btn):
        if not self.composed_text.strip():
            # Nothing to say.
            tell_dashboard(btn, "keyboard", "control", "kb:speak", "Speak (nothing typed)",
                           navigation=True)
            return
        tell_dashboard(btn, "keyboard", "speak", "kb:speak", self.composed_text)
        speak(self.composed_text)
        self.spoken = True
        self.shift_on = True  # next message starts with a capital
        self._refresh_text()
        self._refresh_keys()

    def on_pre_enter(self, *args):
        self.open_group = None   # always start from the groups
        self._cancel_clear()
        self._refresh_keys()


# ---------------------------------------------------------------------------
# 5. QUICK FIRES (Figures 30-31)
# ---------------------------------------------------------------------------
# Each category: tile colour, then (phrase, picture) pairs. The phrases are
# preliminary, to be confirmed by caregiver and therapist interviews.
QUICKFIRE_CATEGORIES = {
    "Emergency": (TILE_RED, [
        ("Help me, please!", "help"),
        ("Call the nurse", "nurse"),
        ("Call the doctor", "doctor"),
        ("I can't breathe", "oxygen_mask"),
        ("Call an ambulance", "ambulance"),
        ("Call my family", "mobile_phone"),
        ("I feel sick", "vomit"),
        ("Please listen to me", "hear"),
    ]),
    "Pain": (TILE_ORANGE, [
        ("I am in pain", "back_ache"),
        ("My head hurts", "headache"),
        ("My stomach hurts", "stomach_ache"),
        ("My chest hurts", "chest"),
        ("My throat hurts", "throat"),
        ("It hurts here", "point"),
        ("I need my medicine", "tablets"),
        ("Please move me", "move"),
    ]),
    "Basic Needs": (YELLOW, [
        ("I'm hungry", "hungry"),
        ("I'm thirsty", "thirsty"),
        ("I need to use the comfort room", "need_toilet"),
        ("I want to sleep", "sleep"),
        ("I want to sit up", "sit"),
        ("I want to lie down", "lie_down"),
        ("I need a bath", "bath"),
        ("I need my glasses", "glasses"),
    ]),
    "Social/Emotional": (GREEN, [
        ("I am happy", "happy"),
        ("I am sad", "sad"),
        ("I am tired", "tired"),
        ("I am okay", "okay"),
        ("I am scared", "scared"),
        ("I am worried", "worried"),
        ("I am confused", "confused"),
        ("I love you", "love"),
    ]),
    "People/Family": (TILE_BLUE, [
        ("I want to see my family", "family"),
        ("Who is here?", "who"),
        ("Talk to me", "talk"),
        ("I want visitors", "visitor"),
        ("Hello", "hello"),
        ("Please pray with me", "pray"),
        ("Where is my family?", "where"),
        ("I want a hug", "hug"),
    ]),
    "Environment": (PINK, [
        ("It's too hot", "hot"),
        ("It's too cold", "cold"),
        ("Turn on the light", "light_on"),
        ("Turn off the light", "light_off"),
        ("Turn on the TV", "tv_on"),
        ("Open the window", "window"),
        ("It's too noisy", "noisy"),
        ("I want to go outside", "outside"),
    ]),
}
# Two steps: 6 category tiles (3 x 2), then 8 phrases and Back (3 x 3).
QUICKFIRE_COLUMNS = 3
# The picture shown on each category's tile.
CATEGORY_PICTURES = {
    "Emergency": "help",
    "Pain": "back_ache",
    "Basic Needs": "hungry",
    "Social/Emotional": "happy",
    "People/Family": "family",
    "Environment": "light_on",
}


class QuickFireScreen(AppScreen):
    """Figures 30-31 in two steps: pick a category, then a phrase, which is spoken at once."""

    TAB = "quickfire"

    def __init__(self, navigator, **kwargs):
        super().__init__(navigator, **kwargs)
        self.current_category = None  # None: categories shown
        self.step_label = Label(bold=True, font_size=38, color=WHITE)
        step_box = Panel(bg_color=PANEL_BG, border_color=KEY_BORDER, radius=12, size_hint_x=0.8)
        step_box.add_widget(self.step_label)
        # Rest button (see REST_AFTER).
        controls = BoxLayout(orientation="horizontal", spacing=10)
        controls.add_widget(step_box)
        controls.add_widget(DwellButton(text="Rest", icon="moon", bg_color=KEY_FACE, text_color=WHITE,
                                        radius=12, bold=True, font_size=26, icon_layout="top",
                                        size_hint_x=0.2, on_selected=self._on_rest))
        self.tiles_panel = Panel(bg_color=PANEL_BG, border_color=KEY_BORDER, padding=10)
        self.build_frame(controls, self.tiles_panel)
        self._show_categories()

    def _show_categories(self):
        self.current_category = None
        self.step_label.text = "Pick a category"
        grid = GridLayout(cols=QUICKFIRE_COLUMNS, spacing=10)
        for name, (color, _phrases) in QUICKFIRE_CATEGORIES.items():
            tile = DwellButton(text=name, bold=True, font_size=34, bg_color=color,
                               text_color=DARK_TEXT, radius=12,
                               image=CATEGORY_PICTURES.get(name), icon_layout="top",
                               on_selected=self._on_category)
            tile.key_value = name
            tile.role = "category"   # opens a page (see DRIFT_PAGE_OPENERS)
            grid.add_widget(tile)
        self._show_grid(grid)

    def _show_phrases(self, name):
        self.current_category = name
        self.step_label.text = f"{name}: pick a phrase"
        color, phrases = QUICKFIRE_CATEGORIES[name]
        grid = GridLayout(cols=QUICKFIRE_COLUMNS, spacing=10)
        for phrase, picture in phrases:
            tile = DwellButton(text=phrase, font_size=30, bg_color=color, text_color=DARK_TEXT,
                               radius=12, image=picture, icon_layout="top",
                               on_selected=self._on_phrase)
            tile.key_value = phrase
            grid.add_widget(tile)
        grid.add_widget(DwellButton(text="Back", bold=True, font_size=34, icon="arrow_left",
                                    icon_layout="left", on_selected=self._on_back,
                                    **{**KEY_STYLE, "radius": 12}))
        self._show_grid(grid)

    def _show_grid(self, grid):
        self.tiles_panel.clear_widgets()
        self.tiles_panel.add_widget(grid)

    def _on_category(self, btn):
        tell_dashboard(btn, "quickfire", "control", f"qf:cat:{slug(btn.key_value)}",
                       f"Category: {btn.key_value}", navigation=True)
        self._show_phrases(btn.key_value)

    def _on_back(self, btn):
        tell_dashboard(btn, "quickfire", "control", "qf:back", "Back to the categories",
                       navigation=True)
        self._show_categories()

    def _on_phrase(self, btn):
        tell_dashboard(btn, "quickfire", "phrase",
                       f"qf:{slug(self.current_category)}:{slug(btn.key_value)}", btn.key_value,
                       category=self.current_category)
        # Shown here and in the keyboard's message box, and spoken.
        self.step_label.text = f"Said: {btn.key_value}"
        self.nav.sm.get_screen("keyboard").say(btn.key_value)

    def _on_rest(self, btn):
        tell_dashboard(btn, "quickfire", "control", "qf:rest", "Rest (resting screen)",
                       navigation=True)
        self.nav.go("rest")

    def on_pre_enter(self, *args):
        self._show_categories()  # always start from the categories


class RestScreen(Screen):
    """Resting screen (see REST_AFTER): one Start button, camera slowed down."""

    def __init__(self, navigator, gaze_engine=None, **kwargs):
        super().__init__(**kwargs)
        self.nav = navigator
        self.gaze_engine = gaze_engine
        self.return_to = HOME_SCREEN  # set by Navigator.go("rest")
        frame = Panel(bg_color=REST_BG, radius=0, orientation="vertical", padding=[40, 30], spacing=10)
        frame.add_widget(Label(text="EyeGlynt is resting", bold=True, font_size=56,
                               color=REST_TEXT, size_hint_y=0.2))
        frame.add_widget(Label(text="Look at Start to use it again", font_size=36,
                               color=REST_TEXT, size_hint_y=0.13))
        row = BoxLayout(orientation="horizontal", size_hint_y=0.52)
        row.add_widget(Widget(size_hint_x=0.3))
        self.start_btn = DwellButton(text="Start", bold=True, font_size=48, bg_color=TILE_BLUE,
                                     text_color=DARK_TEXT, radius=20, size_hint_x=0.4,
                                     on_selected=self._on_start)
        row.add_widget(self.start_btn)
        row.add_widget(Widget(size_hint_x=0.3))
        frame.add_widget(row)
        frame.add_widget(Widget(size_hint_y=0.15))
        self.add_widget(frame)

    def _on_start(self, btn):
        back_to = self.return_to if self.return_to in ("keyboard", "quickfire") else HOME_SCREEN
        tell_dashboard(btn, back_to, "control", "rest:start", "Start (after resting)",
                       navigation=True)
        print(f"[Rest] Start: back to {back_to}")
        self.nav.go(back_to)

    def on_enter(self, *args):
        print("[Rest] Resting: low power until Start is selected")
        if self.gaze_engine is not None:
            self.gaze_engine.set_low_power(True)

    def on_leave(self, *args):
        if self.gaze_engine is not None:
            self.gaze_engine.set_low_power(False)


# ---------------------------------------------------------------------------
# 6. CALIBRATION SCREEN
# ---------------------------------------------------------------------------
# Stages: READY (camera preview, Start), COUNTDOWN (3, 2, 1), DOTS (9 points),
# then the accuracy test.

# Dot positions as screen fractions (x, y), y = 0 at the bottom: the centre
# first, then clockwise around the edge.
CALIBRATION_POINTS = [
    (0.50, 0.47),                               # centre
    (0.08, 0.82), (0.50, 0.82), (0.92, 0.82),  # top row
    (0.92, 0.47),                               # middle right
    (0.92, 0.12), (0.50, 0.12), (0.08, 0.12),  # bottom row
    (0.08, 0.47),                               # middle left
]
CALIBRATION_CAPTURE_TIME = 4.0  # seconds per dot
CALIBRATION_SETTLE_TIME = 1.5  # seconds ignored while the eyes move onto the dot
# The dot starts big and shrinks during the settle time, leading the eyes to its
# centre. Sizes in pixels (about 85 per cm).
DOT_START_SIZE = 120
DOT_SIZE = 44
DOT_CENTRE_SIZE = 12
# Up to REDO_MAX_DOTS dots where the eyes were closed, not seen or wandering are
# shown again at the end (0 = off).
REDO_MAX_DOTS = 3
REDO_MIN_SEEN = 0.5
REDO_MAX_SPREAD = 0.02
COUNTDOWN_SECONDS = 3  # 3, 2, 1
# Wait up to this long for the camera to focus after the countdown.
FOCUS_WAIT_SECONDS = 10
# Best distance from the camera; the READY stage asks the user to move if more
# than DISTANCE_TOLERANCE_CM off.
BEST_DISTANCE_CM = 43
DISTANCE_TOLERANCE_CM = 3  # advice below 40 cm or above 46 cm
RESULT_SECONDS = 10  # seconds the result stays on screen
# A Poor or failed result restarts calibration, up to CALIBRATION_TRIES times;
# then the app goes on and alerts the caregiver.
CALIBRATION_TRIES = 3
RETRY_SECONDS = 6
# Result line when up/down did not calibrate.
UPDOWN_FAILED_LINE = ("\nUp/down did not work and uses the mouse -- please calibrate again"
                      if ROW_MODE == "off" else
                      "\nUp/down did not work (Row mode will be used) -- please calibrate again")

# ACCURACY TEST: 6 more dots measure how far off the pointer is and how much
# it shakes (cm). Every reading is saved to a CSV file in logs/.
RUN_ACCURACY_TEST = True

# Refit the calibration from all 15 dots after the test.
TEST_DOTS_IN_CALIBRATION = True

# Engine calibration saved when the screen opens (for Stop calibration).
CALIBRATION_STATE = ("_x_points", "_y_points", "_shakes", "_x_fit", "_y_fit",
                     "_x_backup", "_y_backup", "_calib_face_sizes", "_calib_face_size",
                     "_x_heads", "_y_heads", "_calib_head", "_centre_x", "_centre_y",
                     "_pointer_shift", "updown_error_cm")
ACCURACY_TEST_POINTS = [
    (0.25, 0.65), (0.50, 0.65), (0.75, 0.65),  # upper row
    (0.75, 0.30), (0.50, 0.30), (0.25, 0.30),  # lower row
]
# Result: "Good" up to GOOD_CM off and shaking, "OK" up to OK_CM, else "Poor".
GOOD_CM = 1.5
OK_CM = 2.5
# For testing: save eye pictures at each dot in logs/. Set to False for real
# sessions unless the participant agreed (Data Privacy Act).
SAVE_EYE_PICTURES = True
EYE_PICTURES_PER_DOT = 6
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

GREEN_TEXT = rgb(120, 220, 120)  # "Face detected"
RED_TEXT = rgb(255, 120, 120)  # "Face not detected"
AMBER_TEXT = rgb(255, 200, 90)  # "OK" result

# Calibration screen sizes in pixels (about 85 per cm). The preview fills
# PREVIEW_WIDTH of the screen's width.
CALIBRATION_TEXT_SIZE = 32  # instructions above the preview
FACE_TEXT_SIZE = 34  # under the preview
START_TEXT_SIZE = 38  # Start Calibration button
COUNTDOWN_TEXT_SIZE = 72     # "Calibration starts in 3"
DOT_TEXT_SIZE = 30           # "Look at the red dot (1 of 9)" at the top edge
RESULT_TITLE_SIZE = 64       # "Calibration done: Good"
RESULT_TEXT_SIZE = 42  # "1.2 cm off, shaking 0.9 cm"
RESULT_DETAILS_SIZE = 30  # fps, distance, log file
PREVIEW_WIDTH = 0.5


def _hex(color):
    """A colour as Kivy markup hex, e.g. "78dc78"."""
    return "".join(f"{round(c * 255):02x}" for c in color[:3])


def save_gray_png(filename, image):
    """Saves a grayscale numpy picture (0-255) as a PNG using only zlib."""
    height, width = image.shape

    def part(name, data):
        return (len(data).to_bytes(4, "big") + name + data
                + zlib.crc32(name + data).to_bytes(4, "big"))

    rows = b"".join(b"\x00" + image[y].tobytes() for y in range(height))
    header = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes([8, 0, 0, 0, 0])
    with open(filename, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + part(b"IHDR", header)
                + part(b"IDAT", zlib.compress(rows)) + part(b"IEND", b""))


class CalibrationScreen(Screen):
    def __init__(self, gaze_engine: GazeEngine, on_done, **kwargs):
        super().__init__(**kwargs)
        self.gaze_engine = gaze_engine
        self.on_done = on_done
        self.current_target_px = None  # (x, y) of the dot on display
        self.target_index = 0
        self.samples = []
        self.stage = "ready"  # "ready", "countdown", "dots", "redo" or "test"
        self.summary = ""  # the result
        self.log_rows = []  # every reading, saved to CSV
        self.test_results = []  # per test dot: target x, y, pointer positions
        self._countdown_left = 0
        self._sample_event = None
        self._preview_event = None
        self._countdown_event = None
        self._retry_event = None  # calibrating again (see CALIBRATION_TRIES)
        self._tries = 0  # results so far that said "calibrate again"
        self._leaving = False  # True once leaving
        self.eye_pictures = []  # see SAVE_EYE_PICTURES
        self._last_picture_number = None

        # Instructions at the top.
        self.instruction_label = Label(
            font_size=CALIBRATION_TEXT_SIZE, size_hint=(1, 0.19), pos_hint={"top": 0.99, "x": 0},
            halign="center", valign="top", line_height=1.1,
        )
        self.instruction_label.bind(size=lambda w, s: setattr(w, "text_size", s))
        self.add_widget(self.instruction_label)

        # Live camera preview with the iris dots, only in the READY stage.
        self.preview_image = Image(
            size_hint=(PREVIEW_WIDTH, 0.53), fit_mode="contain",
            pos_hint={"center_x": 0.5, "top": 0.8},
        )
        self.add_widget(self.preview_image)

        # What the camera sees, then what to do.
        self.face_label = Label(
            font_size=FACE_TEXT_SIZE, bold=True, size_hint=(1, 0.1),
            pos_hint={"center_x": 0.5, "y": 0.165}, halign="center", valign="middle",
        )
        self.face_label.bind(size=lambda w, s: setattr(w, "text_size", s))
        self.add_widget(self.face_label)

        # Start: works by tap or by dwell.
        self.start_btn = DwellButton(
            text="Start Calibration", bold=True, font_size=START_TEXT_SIZE,
            bg_color=PURPLE, text_color=WHITE, radius=16,
            size_hint=(None, None), size=(480, 115),
            pos_hint={"center_x": 0.5, "y": 0.03},
            on_selected=lambda b: self._start_countdown(),
        )
        self.start_btn.bind(on_release=lambda b: self._start_countdown())

        with self.canvas:  # the dot, off-screen until shown
            Color(0.9, 0.2, 0.2, 1)
            self.dot = Ellipse(pos=(-200, -200), size=(DOT_SIZE, DOT_SIZE))
            Color(1, 1, 1, 1)
            self.dot_centre = Ellipse(pos=(-200, -200), size=(DOT_CENTRE_SIZE, DOT_CENTRE_SIZE))

    # ---- stage 1: READY ----
    def on_pre_enter(self, *args):
        # Kept so Stop calibration can put it back.
        self._previous_calibration = {name: getattr(self.gaze_engine, name)
                                      for name in CALIBRATION_STATE if hasattr(self.gaze_engine, name)}
        self._tries = 0
        self._get_ready()

    def _get_ready(self):
        """READY stage: live preview and Start."""
        self._cancel_retry()
        self.target_index = 0
        self.log_rows = []
        self.test_results = []
        # The old calibration stays on until Start, so Start can be picked with the eyes.
        self.stage = "ready"
        self._leaving = False
        self._hide_dot()
        self.eye_pictures = []
        self._dot_results = {}  # dot -> its readings
        self._redo_queue = []  # dots to show again
        self._redo_total = 0
        # Focus the camera on the user's eyes.
        self.gaze_engine.request_focus()
        self._place_label("top", CALIBRATION_TEXT_SIZE, 0.19)
        self.instruction_label.text = (
            ("Get ready for calibration\n" if self._tries == 0 else
             f"Let's calibrate again (try {self._tries + 1} of {CALIBRATION_TRIES})\n") +
            "Sit about 40 cm from the camera, with your whole face in the camera view.\n"
            "When calibration starts, keep your head still and follow the red dot with your eyes.\n"
            "Preview: green = iris, white = IR glints, yellow = pupil."
        )
        self.face_label.text = "Starting camera..."
        self.face_label.color = WHITE
        self.preview_image.opacity = 0  # shown once the first picture arrives
        # Added once the first camera picture arrives.
        if self.start_btn.parent is not None:
            self.remove_widget(self.start_btn)
        self._shown_frame = None  # picture on show
        self._preview_event = Clock.schedule_interval(self._update_preview, 1 / 15)

    def _update_preview(self, dt):
        """15 times a second in READY: shows the camera picture and face status."""
        if not self.gaze_engine.is_camera_ok():
            # No camera: go on with mouse control.
            self._stop_preview()
            self.face_label.text = "No camera detected\nUsing mouse control instead"
            self.face_label.color = RED_TEXT
            self._leave_after(2.0)
            return

        frame = self.gaze_engine.get_debug_frame()
        if frame is None:
            return  # camera still starting

        # Draw only new pictures, into the same texture.
        if frame is not self._shown_frame:
            self._shown_frame = frame
            h, w = frame.shape[:2]
            texture = self.preview_image.texture
            if texture is None or tuple(texture.size) != (w, h):
                texture = Texture.create(size=(w, h), colorfmt="rgb")
                texture.flip_vertical()  # Kivy textures are bottom-up
                self.preview_image.texture = texture
            texture.blit_buffer(frame.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
            self.preview_image.canvas.ask_update()
            self.preview_image.opacity = 1

        feature = self.gaze_engine.get_latest_feature()
        distance = self.gaze_engine.get_face_distance_cm()
        if self.gaze_engine.is_focusing() and distance is not None:
            # Face seen; the camera is focusing.
            self.face_label.text = "Face detected\nFocusing the camera on your eyes..."
            self.face_label.color = WHITE
        elif feature is None:
            self.face_label.text = "Face not detected\nMove into the camera view"
            self.face_label.color = RED_TEXT
        else:
            # Glints are optional: without them the iris is used.
            glints = "IR glints seen" if feature[1] is not None else "no IR glints (the iris will be used)"
            advice, color = "Look at Start or tap it when you are ready", GREEN_TEXT
            if distance is not None:
                glints += f", {distance:.0f} cm from the camera"
                # Advice only; Start still works.
                if distance > BEST_DISTANCE_CM + DISTANCE_TOLERANCE_CM:
                    advice, color = f"Move closer, to about {BEST_DISTANCE_CM} cm", WHITE
                elif distance < BEST_DISTANCE_CM - DISTANCE_TOLERANCE_CM:
                    advice, color = f"Move back, to about {BEST_DISTANCE_CM} cm", WHITE
            self.face_label.text = f"Face detected: {glints}\n{advice}"
            self.face_label.color = color

        # The camera works: show Start.
        if self.start_btn.parent is None:
            self.add_widget(self.start_btn)

    def _stop_preview(self):
        if self._preview_event is not None:
            self._preview_event.cancel()
            self._preview_event = None

    def _place_label(self, where, font_size, height, markup=False):
        """Moves the instruction text to "top" (a strip) or "center" (large)."""
        self.instruction_label.markup = markup
        self.instruction_label.font_size = font_size
        self.instruction_label.size_hint_y = height
        if where == "top":
            self.instruction_label.pos_hint = {"x": 0, "top": 0.99}
            self.instruction_label.valign = "top"
        else:
            self.instruction_label.pos_hint = {"x": 0, "center_y": 0.5}
            self.instruction_label.valign = "middle"

    # ---- stage 2: COUNTDOWN ----
    def _start_countdown(self):
        if self.stage != "ready":
            return  # already started
        self.stage = "countdown"
        self.gaze_engine.clear_calibration()
        # Focus once more now that the user is in place.
        self.gaze_engine.request_focus()
        self._stop_preview()
        self.remove_widget(self.start_btn)
        self.preview_image.opacity = 0  # hide the preview from now on
        self.face_label.text = ""
        self._place_label("center", COUNTDOWN_TEXT_SIZE, 0.35)
        self._countdown_left = COUNTDOWN_SECONDS
        self._show_countdown()
        self._countdown_event = Clock.schedule_interval(self._countdown_tick, 1.0)

    def _show_countdown(self):
        self.instruction_label.text = (
            f"Calibration starts in {self._countdown_left}\n"
            "Look at the red dot when it appears"
        )

    def _countdown_tick(self, dt):
        self._countdown_left -= 1
        if self._countdown_left > 0:
            self._show_countdown()
            return
        # Wait for the camera to finish focusing (up to FOCUS_WAIT_SECONDS).
        if self.gaze_engine.is_focusing() and self._countdown_left > -FOCUS_WAIT_SECONDS:
            self.instruction_label.text = "Focusing the camera on your eyes...\nKeep your head still"
            return
        self._countdown_event.cancel()
        self._countdown_event = None
        # Small text at the top, clear of the dots.
        self._place_label("top", DOT_TEXT_SIZE, 0.1)
        self.stage = "dots"
        self._show_current_target()

    # ---- stage 3: DOTS (and the accuracy test) ----
    def _points(self):
        return ACCURACY_TEST_POINTS if self.stage == "test" else CALIBRATION_POINTS

    def _show_current_target(self):
        # Use the window's size now (it is still growing at start-up).
        fx, fy = self._points()[self.target_index]
        x, y = self.width * fx, self.height * fy
        self.current_target_px = (x, y)
        self._draw_dot(0.0)
        self._ticks_after_settle = 0  # checks after the settle time
        self.samples = []  # fx readings (left/right)
        self.samples_vy = []  # vy readings (up/down, glints)
        self.samples_iy = []  # iy readings (up/down, iris)
        self.samples_hx = []  # hx readings (left/right, glints)
        self.pointer_samples = []  # accuracy test: pointer positions
        self.smoothed_samples = []  # averaged (fx, vy, iy, hx)
        self._pictures_at_dot = 0  # eye pictures kept at this dot
        self._target_start_time = time.time()
        self._sample_event = Clock.schedule_interval(self._collect_sample, 1 / 30)
        Clock.schedule_once(self._finish_current_target, CALIBRATION_CAPTURE_TIME)

    def _draw_dot(self, elapsed):
        """Draws the dot, shrinking from DOT_START_SIZE to DOT_SIZE."""
        x, y = self.current_target_px
        shrunk = min(1.0, elapsed / CALIBRATION_SETTLE_TIME) if CALIBRATION_SETTLE_TIME > 0 else 1.0
        size = DOT_START_SIZE + (DOT_SIZE - DOT_START_SIZE) * shrunk
        self.dot.size = (size, size)
        self.dot.pos = (x - size / 2, y - size / 2)
        self.dot_centre.pos = (x - DOT_CENTRE_SIZE / 2, y - DOT_CENTRE_SIZE / 2)

    def _hide_dot(self):
        self.dot.pos = self.dot_centre.pos = (-200, -200)

    def _collect_sample(self, dt):
        feature = self.gaze_engine.get_latest_feature()
        # This picture's own readings (averaged ones lag behind the eyes).
        raw = self.gaze_engine.get_latest_raw_feature()
        elapsed = time.time() - self._target_start_time
        if elapsed <= CALIBRATION_SETTLE_TIME + 0.1:
            self._draw_dot(elapsed)   # still shrinking
        pointer = (self.gaze_engine.get_screen_x(), self.gaze_engine.get_screen_y())
        self._log(elapsed, feature, pointer, raw)
        if elapsed >= CALIBRATION_SETTLE_TIME:
            self._ticks_after_settle += 1
            self._keep_eye_picture(elapsed)
        # Readings with closed eyes stay in the CSV but not in the dot.
        if raw is not None and self.gaze_engine.eyes_closed():
            raw = None
        if self.stage == "test" and elapsed >= CALIBRATION_SETTLE_TIME:
            self.pointer_samples.append(pointer)
        # Skip the settle time at each dot.
        if raw is not None and elapsed >= CALIBRATION_SETTLE_TIME:
            fx, vy, iy, hx = raw
            self.samples.append(fx)
            self.samples_iy.append(iy)
            if vy is not None:
                self.samples_vy.append(vy)
            if hx is not None:
                self.samples_hx.append(hx)
        # Averaged readings, only to measure shaking.
        if feature is not None and elapsed >= CALIBRATION_SETTLE_TIME:
            self.smoothed_samples.append(feature)

        remaining = max(0.0, CALIBRATION_CAPTURE_TIME - elapsed)
        point_num = self.target_index + 1
        total_points = len(self._points())
        if self.stage == "redo":
            point_num = self._redo_total - len(self._redo_queue)
            total_points = self._redo_total
        face_status = "face detected" if feature is not None else "FACE NOT DETECTED -- move into camera view"
        what = {"test": "Accuracy test: look", "redo": "Once more: look"}.get(self.stage, "Look")
        self.instruction_label.text = (
            f"{what} at the red dot  ({point_num} of {total_points})  --  "
            f"{remaining:.1f}s  --  [{face_status}]"
        )

    def _keep_eye_picture(self, elapsed):
        """Keeps a picture of the eyes at this dot (see SAVE_EYE_PICTURES)."""
        if not SAVE_EYE_PICTURES or self._pictures_at_dot >= EYE_PICTURES_PER_DOT:
            return
        # Spread the pictures over the measured part of the dot.
        spacing = (CALIBRATION_CAPTURE_TIME - CALIBRATION_SETTLE_TIME) / EYE_PICTURES_PER_DOT
        if elapsed < CALIBRATION_SETTLE_TIME + self._pictures_at_dot * spacing:
            return
        picture = self.gaze_engine.get_eye_picture()
        if picture is None or picture[0] == self._last_picture_number:
            return  # no face, or the same picture
        self._last_picture_number = picture[0]
        self._pictures_at_dot += 1
        target_x, target_y = self.current_target_px
        self.eye_pictures.append((self.stage, self.target_index + 1, round(target_x), round(target_y),
                                  round(elapsed, 3), picture))

    def _log(self, elapsed, feature, pointer, raw=None):
        """Keeps one reading for the CSV file."""
        def rounded(value, digits):
            return None if value is None else round(value, digits)

        fx, vy, iy, hx = feature if feature is not None else (None, None, None, None)
        # Head pose, angles in degrees.
        turn, tilt, head_x, head_y = self.gaze_engine.get_head() or (None, None, None, None)
        turn, tilt = [None if a is None else float(np.degrees(a)) for a in (turn, tilt)]
        target_x, target_y = self.current_target_px
        self.log_rows.append([self.stage, self.target_index + 1, round(target_x), round(target_y),
                              round(elapsed, 3)]
                             + [rounded(v, 5) for v in (fx, vy, iy, hx)]
                             + [rounded(pointer[0], 1), rounded(pointer[1], 1),
                                rounded(self.gaze_engine.get_fps(), 1)]
                             + [rounded(v, 2) for v in (turn, tilt, head_x, head_y)]
                             + [rounded(self.gaze_engine.get_lens_position(), 3),
                                rounded(self.gaze_engine.get_face_distance_cm(), 1)]
                             + [rounded(v, 5) for v in (raw if raw is not None else (None,) * 4)])

    def _finish_current_target(self, dt):
        if self._sample_event is not None:
            self._sample_event.cancel()
            self._sample_event = None
        if self.manager is None or self.manager.current != self.name:
            return  # calibration was stopped

        if self.stage == "test":
            self.test_results.append((*self.current_target_px, self.pointer_samples))
            # Used for the fits only after the test result (TEST_DOTS_IN_CALIBRATION).
            if TEST_DOTS_IN_CALIBRATION:
                self._add_dot_to_calibration()
            self.target_index += 1
            if self.target_index < len(ACCURACY_TEST_POINTS):
                self._show_current_target()
            else:
                self._finish_test()
            return

        self._keep_dot_result()
        if self.stage == "dots":
            self.target_index += 1
            if self.target_index < len(CALIBRATION_POINTS):
                self._show_current_target()
                return
            self._redo_queue = self._dots_to_redo()
            self._redo_total = len(self._redo_queue)
            if self._redo_queue:
                print("[Calibration] showing dot(s) "
                      + ", ".join(str(index + 1) for index in self._redo_queue)
                      + " again: the eyes did not stay on them")
                self.stage = "redo"
        if self._redo_queue:
            self.target_index = self._redo_queue.pop(0)
            self._show_current_target()
        else:
            self._finish_calibration()

    def _keep_dot_result(self):
        """Keeps this dot's readings; they go to the engine after all dots."""
        seen = len(self.samples) / self._ticks_after_settle if self._ticks_after_settle else 0.0
        spread = None
        if len(self.samples) >= 2:
            spread = float(max(np.std(self.samples), np.std(self.samples_iy)))
        self._dot_results[self.target_index] = (self._dot_reading(), self.gaze_engine.head_and_face_size(),
                                                seen, spread)
        print(f"[Calibration] dot {self.target_index + 1}: eyes seen {seen:.0%} of the time, "
              + ("no readings" if spread is None else f"spread {spread:.3f}"))

    def _dots_to_redo(self):
        """Indexes of dots to show again, worst first (see REDO_MAX_DOTS)."""
        missed = []
        for index, (reading, head_and_size, seen, spread) in self._dot_results.items():
            if seen < REDO_MIN_SEEN or spread is None:
                missed.append((float("inf"), index))  # eyes closed or not seen
            elif spread > REDO_MAX_SPREAD:
                missed.append((spread, index))  # eyes wandered
        missed.sort(key=lambda m: (-m[0], m[1]))
        return [index for _, index in missed[:REDO_MAX_DOTS]]

    def _add_dot_to_calibration(self):
        """Gives the engine this dot's average readings (accuracy test dots)."""
        reading = self._dot_reading()
        if reading is not None:
            self.gaze_engine.add_calibration_point(**reading)

    def _dot_reading(self):
        """This dot's average readings as add_calibration_point arguments, or None."""
        if not self.samples:
            return None
        avg_fx = sum(self.samples) / len(self.samples)
        target_x, target_y = self.current_target_px
        # Glint readings count only if a glint was seen in at least half of them.
        avg_vy = avg_hx = None
        if len(self.samples_vy) >= len(self.samples) / 2:
            avg_vy = sum(self.samples_vy) / len(self.samples_vy)
        if len(self.samples_hx) >= len(self.samples) / 2:
            avg_hx = sum(self.samples_hx) / len(self.samples_hx)
        avg_iy = sum(self.samples_iy) / len(self.samples_iy)
        # How much each measurement wobbled at this dot (for choosing the fits).
        shakes = []
        for index in range(4):
            values = [sample[index] for sample in self.smoothed_samples if sample[index] is not None]
            shakes.append(float(np.std(values)) if len(values) >= 2 else None)
        return dict(fx=avg_fx, screen_x=target_x, vy=avg_vy, screen_y=target_y, iy=avg_iy, hx=avg_hx,
                    shakes=tuple(shakes))

    def _finish_calibration(self):
        for index in sorted(self._dot_results):
            reading, head_and_size, seen, spread = self._dot_results[index]
            if reading is not None:
                self.gaze_engine.add_calibration_point(**reading, head_and_size=head_and_size)
        try:
            fits = self.gaze_engine.fit_calibration()
        except ValueError as e:
            # Not enough readings: use the mouse.
            print(f"[Calibration] {e} -- falling back to mouse control.")
            self.summary = "Calibration did not work, using the mouse -- please calibrate again"
            self._show_result()
            return
        print("[Calibration] " + fits.replace("\n", "; "))  # for testing
        self.summary = "Calibration done"
        if not self.gaze_engine.is_vertical_calibrated():
            self.summary += UPDOWN_FAILED_LINE
        if RUN_ACCURACY_TEST:
            self.stage = "test"
            self.target_index = 0
            self._show_current_target()
        else:
            self._show_result()

    def _finish_test(self):
        """Works out the accuracy test result, overall and per dot."""
        offsets_x, offsets_y, shakes, missing, per_dot = [], [], [], 0, []
        for (target_x, target_y, pointers), (fx, fy) in zip(self.test_results, ACCURACY_TEST_POINTS):
            seen = [(x, y) for x, y in pointers if x is not None and y is not None]
            missing += len(pointers) - len(seen)
            if not seen:
                continue
            xs = np.array([x for x, y in seen]) / SCREEN_PX_PER_CM_X  # cm
            ys = np.array([y for x, y in seen]) / SCREEN_PX_PER_CM_Y
            # Accuracy: distance of the average pointer from the dot.
            dx = xs.mean() - target_x / SCREEN_PX_PER_CM_X
            dy = ys.mean() - target_y / SCREEN_PX_PER_CM_Y
            offsets_x.append(abs(dx))
            offsets_y.append(abs(dy))
            # Precision: spread of the pointer around its average.
            shake = np.sqrt(np.mean((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))
            shakes.append(shake)
            # Kivy's y grows upwards: positive dy = too high.
            where = ("upper" if fy > 0.5 else "lower") + " " + \
                    ("left" if fx < 0.4 else "right" if fx > 0.6 else "middle")
            per_dot.append(f"{where}: {abs(dx):.1f} cm too far {'right' if dx > 0 else 'left'}, "
                           f"{abs(dy):.1f} cm too {'high' if dy > 0 else 'low'}, shaking {shake:.1f} cm")
        if not shakes:
            self._learn_test_dots()
            self._show_result()  # no gaze pointer; the summary says why
            return
        distance = float(np.mean(np.hypot(offsets_x, offsets_y)))
        # For Row mode (ROW_MODE_UPDOWN_CM).
        self.gaze_engine.updown_error_cm = float(np.mean(offsets_y))
        shaking = float(np.mean(shakes))
        # Details in the terminal.
        print("[Accuracy test] " + "\n[Accuracy test] ".join(per_dot))
        if missing:
            print(f"[Accuracy test] {missing} readings had no gaze pointer")
        worst = max(distance, shaking)
        rating = ("Good" if worst <= GOOD_CM else "OK" if worst <= OK_CM
                  else "Poor, please calibrate again")
        self.summary = self.summary.replace("Calibration done", f"Calibration done: {rating}", 1)
        self.summary += (f"\n{distance:.1f} cm off (left/right {np.mean(offsets_x):.1f}, "
                         f"up/down {np.mean(offsets_y):.1f}), shaking {shaking:.1f} cm")
        self._learn_test_dots()
        self._show_result()

    def _learn_test_dots(self):
        """Refits from all 15 dots; keeps the 9-dot fits if that fails or loses up/down."""
        if not TEST_DOTS_IN_CALIBRATION:
            return
        engine = self.gaze_engine
        before = {name: getattr(engine, name) for name in CALIBRATION_STATE if hasattr(engine, name)}
        had_vertical = engine.is_vertical_calibrated()
        try:
            fits = engine.fit_calibration()
        except ValueError as e:
            fits = None
            print(f"[Calibration] with the test dots: {e}")
        if fits is None or (had_vertical and not engine.is_vertical_calibrated()):
            for name, value in before.items():
                setattr(engine, name, value)
            print("[Calibration] kept the fits from the 9 calibration dots")
            return
        print("[Calibration] with the 6 test dots too: " + fits.replace("\n", "; "))
        if not had_vertical and engine.is_vertical_calibrated():
            # The test dots fixed up/down.
            self.summary = self.summary.replace(UPDOWN_FAILED_LINE, "")

    def _show_result(self):
        self._hide_dot()
        self.stage = "result"
        self._place_label("center", RESULT_TEXT_SIZE, 0.8, markup=True)
        fps = self.gaze_engine.get_fps()
        saved = self._save_log()
        distance = self.gaze_engine.get_face_distance_cm()
        details = [f"camera {fps:.0f} pictures a second" if fps is not None else "",
                   f"face {distance:.0f} cm from the camera" if distance is not None else "",
                   f"saved to {saved}" if saved else ""]
        details = ", ".join(d for d in details if d)
        # First line (the rating) biggest, coloured by the result.
        title, *rest = self.summary.split("\n")
        color = (GREEN_TEXT if title.endswith(": Good") else AMBER_TEXT if title.endswith(": OK")
                 else RED_TEXT if ("Poor" in title or "did not work" in title) else WHITE)
        lines = [f"[size={RESULT_TITLE_SIZE}][b][color={_hex(color)}]{escape_markup(title)}"
                 f"[/color][/b][/size]"]
        lines += [escape_markup(line) for line in rest]
        # A throttled Pi gives fewer pictures a second.
        slowdown = self.gaze_engine.get_slowdown()
        if slowdown:
            lines.append(f"[color={_hex(RED_TEXT)}]Warning: the Pi is slowed down: "
                         f"{escape_markup(slowdown)}[/color]")
        # Poor: calibrate again, up to CALIBRATION_TRIES times.
        poor = "please calibrate again" in self.summary
        if poor:
            self._tries += 1
        again = poor and self._tries < CALIBRATION_TRIES
        if again:
            lines.append(f"[b]Calibrating again in {RETRY_SECONDS} seconds "
                         f"(try {self._tries + 1} of {CALIBRATION_TRIES})[/b]")
        else:
            if "Up/down did not work" not in self.summary and row_mode_wanted(self.gaze_engine):
                lines.append(f"Up/down is more than {ROW_MODE_UPDOWN_CM:g} cm off, so Row mode will be used")
            if poor and CALIBRATION_TRIES > 1:
                lines.append(f"[b][color={_hex(AMBER_TEXT)}]Poor {self._tries} times in a row, so EyeGlynt "
                             f"goes on anyway. Recalibrate from the dashboard.[/color][/b]")
        if details:
            lines.append(f"[size={RESULT_DETAILS_SIZE}]({escape_markup(details)})[/size]")
        self.instruction_label.text = "\n".join(lines)
        if again:
            print(f"[Calibration] not good enough: calibrating again (try {self._tries + 1} "
                  f"of {CALIBRATION_TRIES})")
            self._retry_event = Clock.schedule_once(lambda dt: self._calibrate_again(), RETRY_SECONDS)
            return
        if poor and CALIBRATION_TRIES > 1:
            print(f"[Calibration] poor {self._tries} times in a row: going on anyway, caregiver alerted")
            dashboard.calibration_gave_up(self._tries)
        self._leave_after(RESULT_SECONDS)  # time to read the result

    def _calibrate_again(self):
        """Back to READY after a poor result, unless calibration was stopped."""
        self._retry_event = None
        if self.manager is not None and self.manager.current == self.name and self.stage == "result":
            self._get_ready()

    def _cancel_retry(self):
        if self._retry_event is not None:
            self._retry_event.cancel()
            self._retry_event = None

    def _save_log(self):
        """Saves every reading to logs/gaze_<date>_<time>.csv; returns its name or ""."""
        if not self.log_rows:
            return ""
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            name = time.strftime("gaze_%Y%m%d_%H%M%S.csv")
            with open(os.path.join(LOG_DIR, name), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["stage", "dot", "dot_x", "dot_y", "seconds",
                                 "fx", "vy", "iy", "hx", "pointer_x", "pointer_y", "camera_fps",
                                 "head_turn_deg", "head_tilt_deg", "head_x", "head_y",
                                 "lens", "distance_cm", "fx_raw", "vy_raw", "iy_raw", "hx_raw"])
                writer.writerows(self.log_rows)
            self._save_eye_pictures(os.path.join(LOG_DIR, name[:-4] + "_eyes"))
            return f"logs/{name}"
        except OSError as e:
            print(f"[Calibration] Could not save the readings: {e}")
            return ""

    def _save_eye_pictures(self, folder):
        """Writes the eye pictures to folder as PNGs plus pictures.jsonl (stage, dot, crop
        position, face landmarks)."""
        if not self.eye_pictures:
            return
        try:
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "pictures.jsonl"), "w") as index:
                index.write(json.dumps({"camera": self.gaze_engine.camera_mode,
                                        "lens": self.gaze_engine.get_lens_position(),
                                        "distance_cm": self.gaze_engine.get_face_distance_cm(),
                                        "screen": [round(self.width), round(self.height)]}) + "\n")
                for k, (stage, dot, dot_x, dot_y, seconds, picture) in enumerate(self.eye_pictures):
                    number, gray, (x0, y0), landmarks = picture
                    filename = f"{stage}_{dot}_{k:03d}.png"
                    save_gray_png(os.path.join(folder, filename), gray)
                    index.write(json.dumps({"file": filename, "stage": stage, "dot": dot,
                                            "dot_x": dot_x, "dot_y": dot_y, "seconds": seconds,
                                            "picture_number": number, "x0": x0, "y0": y0,
                                            "landmarks": np.round(landmarks, 1).tolist()}) + "\n")
            print(f"[Calibration] {len(self.eye_pictures)} eye pictures saved in {folder}")
        except OSError as e:
            print(f"[Calibration] Could not save the eye pictures: {e}")

    # ---- leaving ----
    def _leave_after(self, seconds):
        if self._leaving:
            return  # already leaving
        self._leaving = True
        Clock.schedule_once(lambda dt: self._done(), seconds)

    def cancel(self):
        """Stop calibration (dashboard): stop the dots and restore the old calibration."""
        self._leaving = True
        self._cancel_retry()
        if self._sample_event is not None:
            self._sample_event.cancel()
            self._sample_event = None
        self._hide_dot()
        for name, value in getattr(self, "_previous_calibration", {}).items():
            setattr(self.gaze_engine, name, value)
        print("[Calibration] Stopped from the dashboard"
              + ("" if self.gaze_engine.is_calibrated() else " (no earlier calibration: using the mouse)"))

    def on_leave(self, *args):
        self._stop_preview()
        self._cancel_retry()
        if self._countdown_event is not None:
            self._countdown_event.cancel()
            self._countdown_event = None

    def _done(self):
        """Leaves calibration, only if it is still the screen on display."""
        if self.manager is not None and self.manager.current == self.name:
            self.on_done()


# ---------------------------------------------------------------------------
# 7. APP
# ---------------------------------------------------------------------------
class EyeGlyntApp(App):
    def build(self):
        self.title = "EyeGlynt"
        keep_speaker_awake()
        start_voice()
        dashboard.start()  # if installed
        self.gaze_engine = GazeEngine()
        self.gaze_engine.start()

        sm = ScreenManager()
        self.sm = sm
        self.navigator = Navigator(sm)
        sm.add_widget(KeyboardScreen(self.navigator, name="keyboard"))
        sm.add_widget(QuickFireScreen(self.navigator, name="quickfire"))
        sm.add_widget(RestScreen(self.navigator, gaze_engine=self.gaze_engine, name="rest"))
        sm.add_widget(
            CalibrationScreen(
                gaze_engine=self.gaze_engine,
                on_done=self._calibration_done,
                name="calibration",
            )
        )
        sm.current = "calibration"

        # Dwell selection by gaze, or the mouse when not calibrated.
        self.gaze_manager = GazeManager(sm, gaze_engine=self.gaze_engine)
        self.gaze_manager.on_row_mode = self._on_row_mode
        Clock.schedule_interval(self._check_dashboard, 0.1)
        # Resting: counts from the last selection or when the screen came up.
        self._screen_since = time.time()
        sm.bind(current=lambda *_: setattr(self, "_screen_since", time.time()))
        Clock.schedule_interval(self._check_rest, 1.0)
        return sm

    def _check_rest(self, dt):
        """Every second: rest after REST_AFTER seconds without a selection."""
        if not REST_AFTER or self.sm.current not in ("keyboard", "quickfire"):
            return
        target = self.gaze_manager._current_target
        if target is not None and target.dwelling:
            return  # a selection is on its way
        idle = time.time() - max(self.gaze_manager._selected_at, self._screen_since)
        if idle >= REST_AFTER:
            print(f"[Rest] No selection for {REST_AFTER:.0f} s: resting")
            self.navigator.go("rest")

    def _on_row_mode(self, on):
        """Row mode turned on or off: show or hide the strips."""
        for name in ("keyboard", "quickfire"):
            self.sm.get_screen(name).set_row_mode(on)

    def _calibration_done(self):
        # Recalibration counts as successful if the engine is calibrated again.
        dashboard.recalibration_finished(success=self.gaze_engine.is_calibrated())
        self.sm.current = HOME_SCREEN

    def _check_dashboard(self, dt):
        """10 times a second: report the device state and carry out dashboard buttons."""
        calibrating = self.sm.current == "calibration"
        if not self.gaze_engine.is_camera_ok() or self.sm.current == "rest":
            state = "idle"  # no camera, or resting
        else:
            state = "calibrating" if calibrating else "tracking"
        dashboard.update(state, fps=self.gaze_engine.get_fps(),
                         eyes_seen=self.gaze_engine.get_latest_feature() is not None)
        for command, details in dashboard.take_commands():
            if command == "recalibrate":
                # Recalibrate pressed on the dashboard.
                dashboard.recalibration_started("manual_caregiver")
                if not calibrating:
                    self.sm.current = "calibration"
            elif command == "cancel_recalibration" and self.sm.current == "calibration":
                # Stop calibration pressed on the dashboard.
                self.sm.get_screen("calibration").cancel()
                dashboard.recalibration_cancelled()
                self.sm.current = HOME_SCREEN
            elif command in ("start_session", "set_dwell_time") and details.get("dwell_time_ms"):
                # Dwell time picked for this session.
                set_dwell_time(details["dwell_time_ms"] / 1000)
            elif command == "end_session":
                set_dwell_time(DEVICE_DWELL_TIME)

    def on_start(self):
        # Open maximized (title bar kept).
        Window.maximize()
        # Row mode highlight, over the screens.
        self.gaze_manager.row_overlay = RowOverlay()
        Window.add_widget(self.gaze_manager.row_overlay)
        if SHOW_GAZE_POINTER:
            # Added last so it is drawn on top.
            self.gaze_manager.pointer = GazePointer()
            Window.add_widget(self.gaze_manager.pointer)

    def on_stop(self):
        self.gaze_engine.stop()
        dashboard.stop()


if __name__ == "__main__":
    EyeGlyntApp().run()
