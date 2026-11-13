"""
EyeGlynt - On-Screen Interface
==============================
The Calibration, Keyboard (Figure 29) and Quick-Fire (Figures 30-31)
screens of EyeGlynt, built with Kivy.

Every selectable element is a DwellButton: it selects only after the
pointer stays on it for DWELL_TIME seconds (the Midas Touch solution).
GazeManager decides where the pointer is:
  - after a successful calibration, left/right comes from the iris
    position and up/down from PCCR (the IR glints), see gaze_engine.py;
  - whichever direction isn't calibrated, or can't be measured at the
    moment (no face, no glint), uses the mouse instead, so the device
    is never left unusable.

RUN
---
    python main.py      (laptop: inside .venv; Pi: inside eyeglynt_env,
                         started from the Pi's own screen)
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
# The screen is redrawn at most SCREEN_FPS times a second (Kivy's default
# is 60). Must be set before the rest of Kivy is loaded. See SCREEN_FPS.
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
from dashboard_link import dashboard, slug   # the caregiver dashboard (see dashboard_link.py)


def rgb(r, g, b, a=1.0):
    """Converts 0-255 colour values (as read from the design) to Kivy's 0-1 range."""
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
REST_BG = rgb(22, 22, 28)          # the resting screen (see REST_AFTER): dark, calm
REST_TEXT = rgb(165, 165, 180)
PLACEHOLDER = rgb(140, 140, 140)

# Quick-Fire pictures (Mulberry Symbols, CC BY-SA 4.0; see
# assets/symbols/CREDITS.md). The path is built from this file's own
# location, so it works no matter which folder the app is started from.
SYMBOL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "symbols")


def _load_symbol(name):
    """Loads assets/symbols/<name>.png as a Kivy texture. Returns None,
    with a warning, if the file is missing: the tile then shows text only
    instead of crashing the app."""
    path = os.path.join(SYMBOL_DIR, f"{name}.png")
    if not os.path.exists(path):
        print(f"[Symbols] Missing {path}; showing text only")
        return None
    return CoreImage(path).texture

# ---------------------------------------------------------------------------
# 0. GLOBAL SETTINGS
# ---------------------------------------------------------------------------
# Your ACTUAL touchscreen (per components_bought.pdf) is a 10.1" 1280x800
# IPS capacitive screen -- not the 1024x600 resistive screen written in the
# thesis draft (Plan A). Build and test against the real resolution.
SCREEN_W, SCREEN_H = 1280, 800
Window.size = (SCREEN_W, SCREEN_H)

# Dwell time: how long the pointer must stay on a button to select it.
# 4.0 seconds instead of the flowchart's 1.5-second example: in testing
# (29 October), 1.5 s and then 2 s were too quick. With the pointer
# still off by a few centimetres, keys were selected before the user
# could settle on the right one.
DWELL_TIME = 4.0  # seconds (default; a caregiver dashboard session can pick 2-6 s)

# Gaze is never perfectly steady: the pointer jitters by a few millimetres
# even when the eyes are still, and a blink can throw it off for a moment.
# Without help, every jitter across a button's edge restarts the dwell, so
# a selection may never finish. Two settings make buttons "sticky":
STICKY_MARGIN = 60   # pixels (about 1 cm): the button being dwelled on still
                     # counts as looked at while the pointer is this close to it
LEAVE_DELAY = 0.3    # seconds: the pointer must stay away from the button this
                     # long before its dwell is cancelled (rides out blinks)
# After a selection, a DIFFERENT key may appear right under the eyes
# (opening "a b c d e" puts "a" where the group was; a Quick Fires
# category puts a phrase there). That key starts its dwell after this
# pause, without the eyes having to leave it and come back. The pause
# lets the person see the new key first. The SAME key (a phrase that was
# just spoken, Speak, Shift, 123, a tab) still waits until the eyes
# leave it, so it is never selected twice by accident. This works once
# in a row: after a key selected this way, the next one needs the eyes to
# leave and come back, so resting the eyes on one spot can't keep
# selecting (group, letter, group, letter...).
NEW_KEY_PAUSE = 1.0  # seconds

# How often the screen code runs. The screen and the eye tracking share
# the Pi's processor, and Python lets only one of them run Python code at a
# time. On 12 November the tracking ran at 8-9 pictures a second during
# calibration but only about 6 while typing, when a button was nearly
# always filling up: each step of the fill redraws the whole screen. The
# fill now moves 10 times a second (smooth enough over a 4-second dwell),
# and the gaze is checked 15 times a second (the camera gives fewer
# pictures than that anyway). SCREEN_FPS (above) caps all redrawing.
DWELL_FILL_RATE = 10   # times a second the dwell progress bar moves
GAZE_CHECK_RATE = 15   # times a second the pointer is checked against the buttons
REPEAT_TIME = DWELL_TIME  # held keys repeat at the SAME rate as the initial dwell,
                          # so there's only one dwell-time value to document/defend.
                          # Set this to a different (smaller) number later if you
                          # decide, with testing data, that repeats should be faster.

# Calibration is least accurate at the edges of the screen, and the
# pointer often overshoots there, past the edge. A pointer outside the
# screen is pulled back to EDGE_INSET pixels inside it, so an overshoot
# still lands on the button at the edge (Clear, Space, Thank you, ...).
EDGE_INSET = 20  # pixels

# AUTOMATIC RE-CENTRING (drift correction). Right after calibrating, the
# pointer lands about 1 cm from where the eyes look, but then it slowly
# slides off as the head sinks or turns a little. In the saved logs (22
# pairs of calibrations a few minutes apart, 11 November), the earlier
# calibration missed the later test dots by 4.1 cm on average, against
# 1.7 cm for a fresh one. That slide is mostly the same everywhere on
# the screen: one shift for the whole screen took it down to 2.0 cm.
# So the app keeps re-centring itself while it is used. Every selected
# key tells where the pointer was while the user looked at it (its
# average during the dwell, after DRIFT_SETTLE_TIME) and where the eyes
# were (the key's middle). After each selection, the pointer is moved
# DRIFT_LEARN_RATE of the way back towards the key's middle, so a slow
# slide is undone, key after key, before it gets big enough to pick the
# wrong key. Selecting Back or Backspace (UNDO_BUTTONS) shortly after
# takes that last step back, since the key was probably the wrong one.
# Back does this only right after a letter group or a Quick Fire
# category was opened (DRIFT_PAGE_OPENERS): after a phrase is spoken,
# Back just returns to the categories, and the phrase was the right one.
# In a simulation of 80 selections with the pointer sliding 2 to 4 cm,
# the right key was picked 53-79% of the time without this and about
# 94% with it; without any slide it costs about 1% (94.8 -> 94.1%).
DRIFT_CORRECTION = True
DRIFT_LEARN_RATE = 0.3      # share of each key's miss that is corrected
DRIFT_STEP_LIMIT_CM = 1.5   # one key's miss counts as at most this, each direction
DRIFT_MAX_CM = 5.0          # the total correction never goes beyond this
DRIFT_SETTLE_TIME = 1.0     # seconds: the start of each dwell is ignored (eyes still landing)
UNDO_BUTTONS = ("Back", "Backspace")
DRIFT_UNDO_SECONDS = 20     # Back/Backspace takes back a correction this recent
DRIFT_PAGE_OPENERS = ("group", "category")   # button roles whose step Back takes back

# ROW MODE, for users whose up/down does not calibrate (two stroke
# patients on 13 November 2026 ended with Poor calibrations). Without it,
# up/down would then follow the mouse, which such a user can't move. In
# Row mode only left/right is used. One row of buttons is highlighted at
# a time and the eyes are taken to be on that row, so a button is picked
# by looking left or right along it. Two tall strips at the screen's
# edges move the highlight: "Row up" on the left, "Row down" on the
# right. The edges are the easiest places to look at, because the
# pointer stops there (see EDGE_INSET). After a page change (a letter
# group, a category), the first row under the top bar is highlighted.
# ROW_MODE: "auto" = on when up/down did not calibrate, or when the
# accuracy test found up/down more than ROW_MODE_UPDOWN_CM off; "on" =
# always (for testing); "off" = never (up/down follows the mouse).
ROW_MODE = "auto"
ROW_MODE_UPDOWN_CM = 2.0
ROW_STRIP_WIDTH = 0.1             # each strip's share of the screen's width
ROW_STRIP_ROLES = ("row_up", "row_down")
ROW_HIGHLIGHT = rgb(255, 214, 0)  # the outline around the row in use
ROW_DIM = (0, 0, 0, 0.45)         # drawn over the other rows

# For testing only: draws a small yellow circle where the app thinks the
# person is looking, so you can see how far off the tracking is. It is
# off, also while testing: people tend to look at the circle instead of
# the button, and when the circle is a little off, the eyes follow it and
# push it further away, so the pointer seems to get worse after a few
# seconds (reported on 5 November). Without it, the button being looked
# at still shows where the gaze is: it fills up during the dwell. Set it
# to True only to check where the pointer is.
SHOW_GAZE_POINTER = False

# HOME SCREEN: the screen shown after every calibration (and after Stop
# calibration on the dashboard). "quickfire" opens the Quick Fires phrases
# (asked for on 12 November), "keyboard" the keyboard. The tabs at the top
# still switch between the two at any time.
HOME_SCREEN = "quickfire"

# RESTING (idle or sleep mode), asked for by the engineers who evaluated
# EyeGlynt on 12 November. When the user is not using EyeGlynt, it shows a
# dark resting screen with one big Start button. Nothing else can be
# selected there, so the user can look anywhere without speaking or
# typing by accident, and the camera pictures are processed only
# LOW_POWER_FPS times a second (gaze_engine.py) to save power and heat.
#   - The user can rest it with the Rest button on Quick Fires.
#   - It rests by itself after REST_AFTER seconds without any selection
#     on the keyboard or Quick Fires (0 = never). 3 minutes, so the
#     2-minute free-look test is never cut short.
#   - Looking at Start for the dwell time goes back to the same screen.
# The caregiver dashboard shows the device as "idle" while it rests and
# sends no "gaze lost" alerts, so a user who closes their eyes to sleep
# doesn't set them off. Recalibrate on the dashboard still works.
REST_AFTER = 180  # seconds

# ---------------------------------------------------------------------------
# 1. TEXT-TO-SPEECH
# ---------------------------------------------------------------------------
# The voice is Piper: a free voice that sounds like a real person and works
# on the Pi without internet. One-time setup on the Pi, in the Terminal
# (it installs Piper and downloads the voice, see setup_voice.sh):
#     cd ~/eyeglynt
#     bash setup_voice.sh
# The sound plays through the speaker picked in the Pi's volume menu
# (right-click the speaker icon at the top right, choose MAX98357A, the I2S
# amplifier).
#
# Backups, so the app always talks: without Piper or its voice file, the
# robot-like espeak-ng voice is used (sudo apt install -y espeak-ng), and on
# a laptop without either, pyttsx3.

# Which Piper voice to use. Its two files (.onnx and .onnx.json) are in the
# "voices" folder next to this file. Other voices can be heard at
# https://rhasspy.github.io/piper-samples/ and downloaded with, e.g.,
# "bash setup_voice.sh en_US-ryan-medium".
VOICE_NAME = "en_US-amy-medium"
# How fast Piper talks: 1.0 is its normal speed, bigger numbers are slower.
VOICE_SPEED = 1.15
# How loud Piper is: 1.0 is the loudest. Lower it (e.g. 0.7) if it crackles.
VOICE_VOLUME = 1.0

VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
VOICE_FILE = os.path.join(VOICES_DIR, VOICE_NAME + ".onnx")
APLAY = shutil.which("aplay")   # plays the sound; None when not on Linux
# Piper is used when it is installed, its voice file is there and aplay can
# play the sound (that is, on the Pi).
USE_PIPER = (importlib.util.find_spec("piper") is not None
             and os.path.exists(VOICE_FILE) and APLAY is not None)
# The Piper voice writes a DEBUG line with the sounds of every message;
# they cluttered the terminal (12 November).
logging.getLogger("piper").setLevel(logging.INFO)

# Backup voice, espeak-ng: 100 is its normal volume and 200 the loudest.
ESPEAK_VOLUME = 200
# How fast espeak-ng talks, in words per minute (its normal is 175).
ESPEAK_SPEED = 140
ESPEAK = shutil.which("espeak-ng")   # None when espeak-ng isn't installed

_messages = queue.Queue()   # messages waiting for the Piper voice
_player = None   # the program saying a message right now (aplay or espeak-ng)
_piper_ok = USE_PIPER   # turns False if the Piper voice fails to load


def speak(text: str):
    """
    Says the text out loud.

    With Piper or espeak-ng (the Pi), the voice talks in the background
    and speak() returns at once, so the screen and the gaze pointer keep
    moving while it talks. A new message cuts off one that is still being
    spoken, so two messages never talk over each other.
    """
    global _player
    print(f"[TTS] Speaking: {text}")
    stop_speaking()
    if _piper_ok:
        # Forget any message still waiting to be said, then hand this one
        # to the Piper voice (_piper_voice, below).
        while not _messages.empty():
            try:
                _messages.get_nowait()
            except queue.Empty:
                break
        _messages.put(text)
    elif ESPEAK:
        try:
            # "--" makes sure text that starts with "-" is read out, not
            # taken as an option.
            _player = subprocess.Popen(
                [ESPEAK, "-a", str(ESPEAK_VOLUME), "-s", str(ESPEAK_SPEED), "--", text])
        except Exception as e:
            print(f"[TTS] Error: {e}")
    else:
        # Laptop: a brand-new pyttsx3 engine is made for every call,
        # instead of reusing one. pyttsx3's Windows voice (SAPI5) has a
        # known quirk where reusing one engine across multiple
        # say()/runAndWait() calls in the same run often goes silent after
        # the first utterance -- a fresh engine each time avoids that
        # reliably. This one does hold up the screen until it has finished.
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
    """
    Runs in the background for as long as the app is open (started by
    start_voice). Loads the Piper voice once, which takes a few seconds,
    then turns each message from speak() into sound and plays it.
    """
    global _player, _piper_ok
    try:
        from piper import PiperVoice, SynthesisConfig
        voice = PiperVoice.load(VOICE_FILE)
        settings = SynthesisConfig(length_scale=VOICE_SPEED, volume=VOICE_VOLUME)
        # A first practice run (not played), so the first real message
        # comes out quicker.
        list(voice.synthesize("Ready.", settings))
        print(f"[TTS] Piper voice ready: {VOICE_NAME}")
    except Exception as e:
        print(f"[TTS] The Piper voice could not start ({e}), using espeak-ng instead")
        _piper_ok = False
        try:
            speak(_messages.get_nowait())   # say a waiting message with espeak-ng
        except queue.Empty:
            pass
        return

    while True:
        text = _messages.get()   # waits here until speak() hands over a message
        try:
            # Piper makes the sound one sentence at a time.
            pieces = list(voice.synthesize(text, settings))
        except Exception as e:
            print(f"[TTS] Error: {e}")
            continue
        if not pieces or not _messages.empty():
            continue   # nothing to say, or a newer message is waiting: say that one
        sound = b"".join(piece.audio_int16_bytes for piece in pieces)
        try:
            # aplay plays the raw sound (16-bit, one channel, at the voice's
            # sample rate), which is fed to it through a pipe.
            _player = subprocess.Popen(
                [APLAY, "-q", "-t", "raw", "-f", "S16_LE", "-c", "1",
                 "-r", str(pieces[0].sample_rate), "-"],
                stdin=subprocess.PIPE)
            _player.stdin.write(sound)
            _player.stdin.close()
        except (BrokenPipeError, OSError):
            pass   # cut off by a newer message (stop_speaking)


def start_voice():
    """Starts the Piper voice in the background, when it is set up."""
    if _piper_ok:
        threading.Thread(target=_piper_voice, daemon=True).start()
    elif ESPEAK:
        print("[TTS] Piper voice not set up, using espeak-ng")


# The amplifier switches itself off when nothing has played for a few
# seconds, and makes a small "tick" each time it switches back on. While
# the app is open, silence plays all the time to keep it awake, so every
# message starts cleanly (the tick then only happens once, as the app
# opens). Set to False to switch this off.
KEEP_SPEAKER_AWAKE = True
_awake = None   # the program playing the silence


def keep_speaker_awake():
    """
    Pi only: starts playing silence in the background. It stops by
    itself when the app closes, however the app closes: the little shell
    command waits for this app's pipe ("read x"), which closes together
    with the app, and then stops the silence ("kill").
    """
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
# 2. DWELL BUTTON -- the core reusable "gaze-selectable" widget
# ---------------------------------------------------------------------------
class DwellButton(Button):
    """
    A button that fills up like a progress bar while the pointer rests on
    it, and fires `on_selected` once `dwell_time` has passed. This is the
    dwell-time selection method from the Methodology (the fix for the
    Midas Touch problem), built once here so every screen reuses it.

    Set `repeatable=True` for buttons where holding the gaze in place
    should keep re-firing (typing the same letter twice, like "food"):
    after each selection it re-arms itself with `repeat_time`, which is
    equal to DWELL_TIME, for as long as the pointer stays on it, like
    holding down a physical key. Leave it False (the default) wherever an
    accidental repeat would be unsafe or annoying: phrases, Speak, Clear,
    navigation.

    Optional styling (used by the Figure 29 keyboard):
      bg_color      flat rounded fill instead of Kivy's default grey button
      border_color  thin outline around the button
      text_color    label colour
      radius        corner rounding in pixels
      icon          name of an icon from icons.py
      icon_layout   "top" (icon above the text), "left" (icon before the
                    text) or "center" (icon only, no text)
      selectable    False shows the button but lets gaze pass over it, for
                    example the tab of the screen already on display
      image         file name of a picture in assets/symbols (Quick-Fire
                    tiles), drawn where an icon would go
    """

    progress = NumericProperty(0.0)  # 0.0 -> 1.0

    # Called with the button at every dwell selection, before on_selected
    # (set by GazeManager, for DRIFT_CORRECTION).
    selection_listener = None

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

        self.selectable = selectable  # False: shown normally, but gaze ignores it
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
            # Hide Kivy's default button image; we draw our own shape.
            self.background_normal = ""
            self.background_down = ""
            self.background_disabled_normal = ""
            self.background_color = (0, 0, 0, 0)

        # Three drawing layers: our background (under the text), the icon
        # (over the text's layer), and the dwell progress (on top of all).
        self._bg_group = InstructionGroup()
        self.canvas.before.add(self._bg_group)
        self._icon_group = InstructionGroup()
        self.canvas.add(self._icon_group)
        self._progress_group = InstructionGroup()
        self.canvas.after.add(self._progress_group)

        self.bind(pos=self._redraw, size=self._redraw, disabled=self._redraw)
        self._redraw()

    # -- drawing ---------------------------------------------------------
    def set_bg_color(self, color):
        """Changes the fill colour, e.g. to highlight an active Shift key."""
        self.bg_color = color
        self._redraw()

    def _draw_image(self, box):
        """Draws the picture as large as fits inside `box`, keeping its
        proportions and centring it."""
        bx, by, bw, bh = box
        tw, th = self.image_texture.size
        scale = min(bw / tw, bh / th)
        dw, dh = tw * scale, th * scale
        self._icon_group.add(Color(1, 1, 1, 1))  # white = draw the picture's own colours
        self._icon_group.add(Rectangle(texture=self.image_texture,
                                       pos=(bx + (bw - dw) / 2, by + (bh - dh) / 2),
                                       size=(dw, dh)))

    def _progress_color(self):
        """White-ish fill on dark buttons, dark fill on light ones, so the
        dwell progress is visible on every colour in the design."""
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
                # Picture in the upper part; text pinned to the bottom (it
                # wraps onto a second line if it is too long for one).
                self.text_size = (w, h)
                self.halign, self.valign = "center", "bottom"
                self.padding = [6, 4, 6, h * 0.08]
                box = (x, y + h * 0.34, w, h * 0.58)
            elif self.icon_layout == "left":
                # Picture on the left; text in the remaining space.
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

    # -- dwell timing (unchanged logic) -----------------------------------
    @property
    def dwelling(self):
        """True while the progress bar is filling."""
        return self._event is not None

    def start_dwell(self, dwell_time=None):
        if self._event is not None:
            return  # already dwelling on this button
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
            # Tell the GazeManager first, while the button still has the
            # text it was selected with (see DRIFT_CORRECTION).
            if DwellButton.selection_listener is not None:
                DwellButton.selection_listener(self)
            if self.on_selected:
                self.on_selected(self)
            if self.repeatable:
                # Re-arm straight away with repeat_time. GazeManager has not
                # touched this button (its target hasn't changed), so it keeps
                # firing until the pointer leaves, at which point GazeManager
                # calls cancel_dwell() and the repeating stops.
                self.start_dwell(dwell_time=self.repeat_time)
            return False  # stop this particular Clock event


class Panel(BoxLayout):
    """A BoxLayout with a rounded, filled background and an optional outline.
    Used for the device frame, the keyboard panel and the word panel."""

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
        return False   # "off", or no gaze at all (the mouse is used)
    if not gaze_engine.is_vertical_calibrated():
        return True
    error = getattr(gaze_engine, "updown_error_cm", None)
    return error is not None and error > ROW_MODE_UPDOWN_CM


def _group_rows(buttons):
    """Groups buttons into rows: lists [left, bottom, right, top] in
    window pixels, the top row first. A button whose middle lies within
    the height of the row above it belongs to that row."""
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
# 2b. CAREGIVER DASHBOARD -- see dashboard_link.py
# ---------------------------------------------------------------------------
def tell_dashboard(btn, interface, kind, element_id, text, category=None, navigation=False):
    """Records the selection of `btn` in the caregiver dashboard's Activity
    Log (only while a caregiver has a session running there).
    interface: "keyboard" or "quickfire"; kind: "key", "shorthand",
    "phrase", "speak" or "control"; navigation=True for a selection that
    only moves around the screens (see NAVIGATION_ONLY_IN_FREE_LOOK in
    dashboard_link.py)."""
    pointer = None
    gaze_manager = getattr(App.get_running_app(), "gaze_manager", None)
    if gaze_manager is not None:
        x, y = gaze_manager.get_pointer_pos()
        pointer = (x, Window.height - y)   # pixels from the TOP left of the screen
    dashboard.selection(interface, kind, element_id, text, category=category,
                        dwell_s=btn._active_dwell_time, pointer=pointer, navigation=navigation)


DEVICE_DWELL_TIME = DWELL_TIME   # the setting at the top, used again when a dashboard session ends


def set_dwell_time(seconds):
    """Changes the dwell time of every button: the dwell time the caregiver
    picked for the session on the dashboard (2-6 s), and DWELL_TIME again
    when the session ends."""
    global DWELL_TIME, REPEAT_TIME, CLEAR_CONFIRM_TIME
    if seconds == DWELL_TIME:
        return
    DWELL_TIME = REPEAT_TIME = seconds           # for the buttons made from now on
    CLEAR_CONFIRM_TIME = REPEAT_TIME + 2.0
    for screen in App.get_running_app().root.screens:
        for btn in _find_dwell_buttons(screen):  # and for the ones already there
            btn.dwell_time = btn.repeat_time = seconds
    print(f"[Dashboard] Dwell time is now {seconds:.1f} seconds")


# ---------------------------------------------------------------------------
# 3. GAZE MANAGER -- drives dwell selection from a pointer position
# ---------------------------------------------------------------------------
class GazeManager:
    """
    Polls a pointer position 30x/sec and starts/cancels dwell timers on
    whichever DwellButton is currently under it.

    Pointer X comes from gaze once `gaze_engine` is calibrated, and
    pointer Y too if the up/down (PCCR) calibration succeeded. Each one
    falls back to the mouse on its own whenever gaze can't give it (not
    calibrated, no face, no glint), so the app is never left unusable.

    Buttons are "sticky" (see STICKY_MARGIN and LEAVE_DELAY), so a
    jittery pointer or a blink doesn't restart a dwell that is under way.

    It also re-centres the gaze pointer after every selection (see
    DRIFT_CORRECTION).
    """

    def __init__(self, screen_manager: ScreenManager, gaze_engine: GazeEngine = None):
        self.sm = screen_manager
        self.gaze_engine = gaze_engine
        self._current_target = None
        self._away_since = None  # when the pointer left the current button, or None
        self._blocked = None     # a new button the gaze must leave before it can be used
        # What was selected last and when, and whether its dwell was
        # started by NEW_KEY_PAUSE (then the next key can't be).
        self._selected, self._selected_at = None, 0.0
        self._paused_start, self._after_paused_start = None, False
        self.pointer = None      # the test circle (SHOW_GAZE_POINTER), added in on_start
        # Re-centring (see DRIFT_CORRECTION): the gaze pointer's positions
        # (time, x, y) while dwelling on the current button, and the last
        # step taken (x, y, time, whether that button opened a page), so
        # Back/Backspace can take it back.
        self._dwell_points = []
        self._dwell_start = None
        self._last_step = None
        # Row mode (see ROW_MODE): the rows of the screen on display, the
        # one in use, and what told the page apart (to notice a new page).
        self.row_mode = False
        self.on_row_mode = None     # called with True/False when Row mode turns on/off
        self.row_overlay = None     # the highlight (RowOverlay), added in on_start
        self._rows, self._row_index, self._rows_page = [], 0, None
        DwellButton.selection_listener = self._on_selection
        Clock.schedule_interval(self._update, 1 / GAZE_CHECK_RATE)

    def _gaze_pos(self):
        """Where the gaze pointer is, (x, y) in window pixels, before it is
        kept on the screen; None unless gaze gives both directions (not
        the mouse)."""
        if self.gaze_engine is None:
            return None
        x, y = self.gaze_engine.get_screen_x(), self.gaze_engine.get_screen_y()
        if self.row_mode:
            y = self._row_y()   # Row mode: up/down is the highlighted row
        return None if x is None or y is None else (x, y)

    def _row_y(self):
        """The middle of the highlighted row (Row mode), or None."""
        if not self._rows:
            return None
        row = self._rows[self._row_index]
        return (row[1] + row[3]) / 2

    def move_row(self, step):
        """Row up (step -1) or Row down (+1), going round at the ends."""
        if self._rows:
            self._row_index = (self._row_index + step) % len(self._rows)
            self._draw_rows()

    def _update_rows(self, screen, buttons):
        """Finds the rows of the screen on display (Row mode only)."""
        if not self.row_mode or screen.name == "calibration":
            self._rows = []
        else:
            in_rows = [btn for btn in buttons if getattr(btn, "role", None) not in ROW_STRIP_ROLES]
            rows = _group_rows(in_rows)
            page = (screen.name, tuple(sorted(id(btn) for btn in in_rows)))
            if page != self._rows_page:
                # A new screen or page: start on the first row under the top bar.
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
                x = gaze_x  # ...and replace each direction gaze can provide
            if gaze_y is not None:
                y = gaze_y
        if self.row_mode and self._rows:
            y = self._row_y()   # Row mode: up/down is the highlighted row (see ROW_MODE)
        # Keep the pointer on the screen (see EDGE_INSET).
        x = min(max(x, EDGE_INSET), Window.width - EDGE_INSET)
        y = min(max(y, EDGE_INSET), Window.height - EDGE_INSET)
        return (x, y)

    def _on_selection(self, btn):
        """Runs at every dwell selection: re-centres the gaze pointer (see
        DRIFT_CORRECTION). The pointer's average while the user looked at
        the button, against the button's middle, is how far off it is."""
        self._selected, self._selected_at = self._key_of(btn), time.time()
        self._after_paused_start = btn is self._paused_start
        self._paused_start = None
        points = [(x, y) for t, x, y in self._dwell_points
                  if self._dwell_start is not None and t - self._dwell_start >= DRIFT_SETTLE_TIME]
        self._dwell_points = []
        self._dwell_start = time.time()   # a repeating key starts its next dwell now
        if (not DRIFT_CORRECTION or self.gaze_engine is None or btn is not self._current_target
                or self.sm.current == "calibration"
                or getattr(btn, "role", None) in ROW_STRIP_ROLES):   # at the edge: no middle to aim at
            return
        shift_x, shift_y = self.gaze_engine.get_pointer_shift()
        if btn.text in UNDO_BUTTONS:
            # The key before was probably the wrong one: take its step back.
            # Back means that only when the key before opened the page Back
            # leaves (a letter group or a category), not after a phrase.
            last = self._last_step
            if (last is not None and time.time() - last[2] <= DRIFT_UNDO_SECONDS
                    and (btn.text != "Back" or last[3])):
                step_x, step_y = last[0], last[1]
                self.gaze_engine.set_pointer_shift(shift_x - step_x, shift_y - step_y)
                print(f"[Re-centring] {btn.text}: last step taken back")
            self._last_step = None
            return
        # Too few gaze readings (no face, or the mouse was used): no step.
        if len(points) < 10:
            return
        middle_x, middle_y = btn.to_window(*btn.center)
        limit_x = DRIFT_STEP_LIMIT_CM * SCREEN_PX_PER_CM_X
        limit_y = DRIFT_STEP_LIMIT_CM * SCREEN_PX_PER_CM_Y
        miss_x = min(max(np.mean([x for x, y in points]) - middle_x, -limit_x), limit_x)
        miss_y = min(max(np.mean([y for x, y in points]) - middle_y, -limit_y), limit_y)
        if self.row_mode:
            miss_y = 0   # up/down is not measured in Row mode
        max_x, max_y = DRIFT_MAX_CM * SCREEN_PX_PER_CM_X, DRIFT_MAX_CM * SCREEN_PX_PER_CM_Y
        new_x = min(max(shift_x - DRIFT_LEARN_RATE * miss_x, -max_x), max_x)
        new_y = min(max(shift_y - DRIFT_LEARN_RATE * miss_y, -max_y), max_y)
        self.gaze_engine.set_pointer_shift(new_x, new_y)
        self._last_step = (new_x - shift_x, new_y - shift_y, time.time(),
                           getattr(btn, "role", None) in DRIFT_PAGE_OPENERS)
        # Kivy's y grows upwards: a positive miss_y means the pointer was too high.
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
                   if not btn.disabled and btn.selectable]  # greyed-out/display-only can't be selected
        self._update_rows(active_screen, buttons)
        pos = self.get_pointer_pos()
        if self._current_target is not None:
            gaze = self._gaze_pos()
            if gaze is not None:
                self._dwell_points.append((time.time(), *gaze))
        if self.pointer is not None:
            # Hidden during calibration, where it would distract from the dots.
            self.pointer.move_to(pos, visible=active_screen.name != "calibration")
        current = self._current_target

        # A new page has just replaced the button that was selected (for
        # example, picking a Quick Fires category shows its phrases). The
        # new button that now sits under the gaze is ignored until the gaze
        # leaves it, so a phrase is never spoken just because it appeared
        # where the person was already looking.
        if current is not None and current not in buttons:
            current.cancel_dwell()
            self._current_target = current = None
            self._away_since = None
            self._dwell_points, self._dwell_start = [], None
            self._blocked = self._button_at(buttons, pos)
        # It may still be used without leaving it if it is a different key
        # from the one just selected (see NEW_KEY_PAUSE).
        if self._blocked is not None:
            if self._blocked in buttons and self._is_near(self._blocked, pos, STICKY_MARGIN):
                if self._new_key_ready(self._blocked):
                    self._current_target, self._blocked = self._blocked, None
                    self._start_paused_dwell(self._current_target)
                return
            self._blocked = None

        # Sticky buttons (see STICKY_MARGIN and LEAVE_DELAY): keep dwelling
        # on the current button while the pointer is near it...
        if current in buttons and self._is_near(current, pos, STICKY_MARGIN):
            self._away_since = None
            # A keyboard key changes in place (the group becomes its first
            # letter): once it shows a different key, it starts again.
            if not current.dwelling and self._new_key_ready(current):
                self._start_paused_dwell(current)
            return
        # ...and give it up only once the pointer has been away for a moment.
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
            self._current_target = target
            self._dwell_points, self._dwell_start = [], time.time()   # see _on_selection
            if target is not None:
                target.start_dwell()

    @staticmethod
    def _key_of(btn):
        """What a button stands for: its job and value on the keyboard
        (a group, a letter, Space, 123/ABC...), else its phrase or text."""
        role = getattr(btn, "role", None)
        if role is not None:
            return (role, getattr(btn, "key_value", ""))
        return (None, getattr(btn, "key_value", None) or btn.text)

    def _new_key_ready(self, btn):
        """True if `btn` is a different key from the one selected last,
        NEW_KEY_PAUSE has passed since, and that key's own dwell wasn't
        started this way (see NEW_KEY_PAUSE)."""
        return (not self._after_paused_start and self._key_of(btn) != self._selected
                and time.time() - self._selected_at >= NEW_KEY_PAUSE)

    def _start_paused_dwell(self, btn):
        """Starts the dwell of a key the eyes didn't have to leave (see
        NEW_KEY_PAUSE)."""
        self._dwell_points, self._dwell_start = [], time.time()   # see _on_selection
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
        """True if `pos` (window coordinates) is on the button or within
        `margin` pixels of its edges."""
        x, y = btn.to_widget(*pos)
        return (btn.x - margin <= x <= btn.right + margin
                and btn.y - margin <= y <= btn.top + margin)


class GazePointer(Widget):
    """The yellow test circle (see SHOW_GAZE_POINTER)."""

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
    """Row mode (see ROW_MODE): a yellow outline around the row in use,
    and the other rows dimmed. Redrawn only when something changes."""

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
            # Dim everything between the strips above and below the row in
            # use (also what can't be selected, such as the message box).
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
# 3b. NAVIGATION -- the KEYBOARD and QUICK FIRES tabs
# ---------------------------------------------------------------------------
class Navigator:
    """Switches between the Keyboard and Quick Fires screens, for the two
    tabs in the top bar of each screen."""

    def __init__(self, screen_manager):
        self.sm = screen_manager

    def go(self, name):
        if name == "rest" and self.sm.current != "rest":
            # Start on the resting screen comes back here.
            self.sm.get_screen("rest").return_to = self.sm.current
        self.sm.current = name


# ---------------------------------------------------------------------------
# 4. KEYBOARD INTERFACE (Figure 29, revised)
# ---------------------------------------------------------------------------
# REVISED LAYOUT (the team's suggestion, 29 October 2026). It replaces the
# side tabs, the Home/Back title bar and the footer of Figures 29-31, so
# the big buttons get the whole width of the screen. Both screens have
# three bands:
#   top bar:  the KEYBOARD and QUICK FIRES tabs, then the screen's own
#             controls (keyboard: message box, Speak, Backspace)
#   middle:   the screen's big buttons
#   bottom:   (keyboard only) the Quick Access replies
# Home and Back are gone: the two tabs reach every screen, and Quick
# Fires has its own Back tile. Every target is now at least about
# 3 x 3 cm, as big as the 2-3 cm accuracy of camera eye tracking allows.
TOP_BAR_HEIGHT = 0.24   # share of the screen's height
TAB_WIDTH = 0.15        # share of the screen's width, for each tab

# Quick Access replies, shown along the bottom of the keyboard. Selecting
# one speaks it at once, without typing it, because these are answers
# the user often needs quickly; it also appears in the message box, so
# the person listening can read it too. Preliminary: to be confirmed by the
# caregiver and therapist interviews (Data Gathering Procedure).
QUICK_ACCESS_WORDS = [("Yes", GREEN), ("No", TILE_RED), ("Please", PINK), ("Thank you", PINK)]

# Clear erases the whole message, so it needs two selections in a row:
# the first turns it into "Clear?", and only a second one (keep looking
# at it) erases. If the second doesn't come within CLEAR_CONFIRM_TIME
# seconds, it turns back into "Clear". A glance can't erase a message.
# The second selection takes REPEAT_TIME, so the wait is always 2
# seconds longer than that, whatever the dwell time is set to.
CLEAR_CONFIRM_TIME = REPEAT_TIME + 2.0

# LETTER GROUPS instead of the QWERTY keys of Figure 29. A QWERTY key on
# this screen is about 1.6 cm wide, but camera eye tracking is usually
# off by 2 to 3 cm, so single keys were missed. Instead, the keys panel
# is 8 big buttons (2 rows x 4). First the user picks a GROUP of letters,
# then the panel shows that group's letters as big buttons and the user
# picks one; it is typed and the panel goes back to the groups. That is
# two selections per letter, but every target is about four times as
# big as a QWERTY key.
#
# The 123 button swaps the letters for numbers and punctuation, in the
# same layout: 5 groups, 26 characters in total.
LETTER_GROUPS = ["ABCDE", "FGHIJ", "KLMNO", "PQRST", "UVWXYZ"]
SYMBOL_GROUPS = ["12345", "67890", ".,?!'", "-/:;()", "&@\"#%"]

# Where things go on the 2 x 4 grid of big buttons, numbered row by row
# from the top left (0 1 2 3 / 4 5 6 7).
KEY_GRID_COLUMNS = 4
GROUP_CELLS = (0, 1, 2, 4, 5)   # the 5 groups
SPACE_CELL = 3
MODE_CELL = 6                   # 123 / ABC
SHIFT_CELL = 7
BACK_CELL = 7                   # when a group is open: back to the groups

SENTENCE_END = (".", "?", "!")
ATTACHING_PUNCTUATION = ".,?!"  # punctuation that belongs right after a word

# Shared look of the dark keyboard keys.
KEY_STYLE = dict(bg_color=KEY_FACE, border_color=KEY_BORDER, text_color=WHITE, radius=8)


class AppScreen(Screen):
    """The frame shared by the Keyboard and Quick Fires screens (see
    REVISED LAYOUT above): a top bar with the KEYBOARD and QUICK FIRES
    tabs followed by the screen's own controls, and the screen's content
    below it.

    Subclasses set TAB, build their controls and content, then call
    build_frame(controls, content). Keeping this in one place means both
    screens always look and behave the same.
    """

    TAB = ""  # "keyboard" or "quickfire": which tab is this screen's own

    def __init__(self, navigator, **kwargs):
        super().__init__(**kwargs)
        self.nav = navigator

    def build_frame(self, controls, content):
        # The whole screen: [Row up strip] frame [Row down strip]. The
        # strips are only there in Row mode (see set_row_mode).
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
        """A tall Row up / Row down strip (see ROW_MODE). Keeping the eyes
        on it keeps moving the highlight, one row per dwell."""
        box = BoxLayout(size_hint_x=ROW_STRIP_WIDTH, padding=[10, 10, 0, 10] if step < 0 else [0, 10, 10, 10])
        btn = DwellButton(text=text, bold=True, font_size=26, icon=icon, icon_layout="top",
                          bg_color=KEY_FACE, text_color=WHITE, radius=12, repeatable=True,
                          on_selected=lambda b: self._on_row_step(b, step))
        btn.role = role
        box.add_widget(btn)
        return box

    def set_row_mode(self, on):
        """Shows the Row up / Row down strips (Row mode) or hides them."""
        up, down = self._strips
        if on and up.parent is None:
            for box in self._strips:
                # The dwell time may have changed while they were hidden (see set_dwell_time).
                btn = box.children[0]
                btn.dwell_time = btn.repeat_time = REPEAT_TIME
            self._outer.add_widget(up, index=len(self._outer.children))   # leftmost
            self._outer.add_widget(down)                                   # rightmost
        elif not on and up.parent is not None:
            self._outer.remove_widget(up)
            self._outer.remove_widget(down)

    def _on_row_step(self, btn, step):
        tell_dashboard(btn, self.TAB, "control", btn.role.replace("_", ":"), btn.text,
                       navigation=True)
        App.get_running_app().gaze_manager.move_row(step)

    def _tab(self, text, icon, screen_name, color):
        """One of the two tabs. The tab of the screen on display gets a
        white border and is not selectable; the other one switches to its
        screen."""
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
    """Figure 29, revised: compose a message with the letter groups, then
    Speak it.

    Layout, top to bottom (see REVISED LAYOUT above): the tabs with the
    message box, Speak and Backspace; the 8 big buttons of the letter
    groups (see LETTER_GROUPS); the Quick Access replies.

    After Speak, the message turns grey, and the next character typed
    starts a new message, so Clear isn't needed between messages. Clear
    (top right) is for starting over before speaking; it asks for a
    second look first (see CLEAR_CONFIRM_TIME).
    """

    TAB = "keyboard"

    def __init__(self, navigator, **kwargs):
        super().__init__(navigator, **kwargs)
        self.composed_text = ""
        self.shift_on = True      # a message starts with a capital letter
        self.symbols_on = False   # False: letters shown; True: numbers and punctuation
        self.open_group = None    # None: the groups are shown; else the open group's characters
        self.cells = []           # the 8 big buttons of the keys panel, row by row
        self.spoken = False       # True after Speak: the next character starts a new message
        self._clear_timer = None  # set while Clear waits for its second selection

        content = BoxLayout(orientation="vertical", spacing=10)
        content.add_widget(self._build_keys_panel())
        content.add_widget(self._build_quick_access())
        self.build_frame(self._build_top_controls(), content)

        self._refresh_keys()
        self._refresh_text()

    # -- building the layout -------------------------------------------------
    def _build_top_controls(self):
        """The message box, Speak, Backspace and Clear, to the right of the tabs."""
        row = BoxLayout(orientation="horizontal", spacing=10)

        box = Panel(bg_color=WHITE, radius=12, padding=[14, 10], size_hint_x=0.4)
        # The message sits in a ScrollView: when it grows taller than the box,
        # the view scrolls to the end so the newest text is always visible.
        self.text_scroll = ScrollView(do_scroll_x=False, bar_width=4)
        self.text_label = Label(font_size=44, halign="left", valign="top", size_hint_y=None,
                                markup=True)  # markup: for the coloured cursor (see _refresh_text)
        # Wrap at the box width; let the height grow with the text.
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
        # Repeatable, so that simply keeping the eyes on it gives the second selection.
        self.clear_btn = DwellButton(text="Clear", repeatable=True, on_selected=self._on_clear,
                                     **{**action, "font_size": 34})
        row.add_widget(self.clear_btn)
        return row

    def _build_keys_panel(self):
        """The 8 big buttons. They are created once; _refresh_keys() gives
        each one its job (a group, a character, Space, ...) for the view
        on display."""
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
        """The Quick Access replies along the bottom (QUICK_ACCESS_WORDS)."""
        row = GridLayout(cols=len(QUICK_ACCESS_WORDS), spacing=10, size_hint_y=0.33)
        for word, color in QUICK_ACCESS_WORDS:
            btn = DwellButton(text=word, bold=True, font_size=40, bg_color=color,
                              text_color=DARK_TEXT, radius=12,
                              on_selected=self._on_quick_access)
            row.add_widget(btn)
        return row

    # -- refreshing what is shown ---------------------------------------------
    def _fit_text_height(self, *args):
        """The label is exactly as tall as its text."""
        self.text_label.height = self.text_label.texture_size[1]

    def _scroll_to_end(self, dt):
        """Kivy's scroll_y runs from 1 (top) to 0 (bottom). A long message is
        scrolled to its end, so the newest words show; a message that fits
        stays at the top of the box."""
        overflowing = self.text_label.height > self.text_scroll.height
        self.text_scroll.scroll_y = 0 if overflowing else 1

    def _refresh_text(self):
        # escape_markup: characters such as [ or & in the message are shown
        # as they are, not read as markup.
        if self.composed_text and self.spoken:
            # Grey after Speak: the next character starts a new message.
            self.text_label.text = escape_markup(self.composed_text)
            self.text_label.color = PLACEHOLDER
        elif self.composed_text:
            # A purple bar marks where the next character goes. Without it,
            # a Space at the end of the message can't be seen at all.
            self.text_label.text = escape_markup(self.composed_text) + "[color=6a1fad]|[/color]"
            self.text_label.color = DARK_TEXT
        else:
            self.text_label.text = "Type, here..."
            self.text_label.color = PLACEHOLDER
        # Scroll once Kivy has re-measured the text, on the next frame.
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
        """Gives one of the 8 big buttons its job. role is "group",
        "char", "space", "mode", "shift", "back" or "empty"."""
        btn = self.cells[index]
        btn.role = role
        btn.key_value = value
        btn.text = text
        btn.font_size = 64 if role == "char" else 40
        btn.icon = icon
        btn.icon_layout = "left" if text else "center"
        # Undo the text placement an icon may have set on this button before.
        btn.text_size, btn.halign, btn.valign, btn.padding = (None, None), "center", "middle", [0, 0, 0, 0]
        # No key selects again while the eyes stay on it: after a selection,
        # the eyes must leave the key and come back. Otherwise, opening a
        # group and then pausing to find the letter would type whatever
        # letter appeared under the eyes.
        btn.repeatable = False
        btn.disabled = role == "empty" or (role == "shift" and self.symbols_on)
        highlight = role == "shift" and self.shift_on and not self.symbols_on
        btn.set_bg_color(PURPLE if highlight else KEY_FACE)   # also redraws the button

    def _after_edit(self):
        """Runs after every change to the message."""
        self._update_auto_shift()
        self._refresh_text()
        self._refresh_keys()

    def _update_auto_shift(self):
        """Capitalises the next letter at the start of the message, of a new
        line, or after a sentence ends (". ", "? ", "! "). This saves the
        user a Shift selection for every sentence."""
        text = self.composed_text
        trimmed = text.rstrip(" ")
        self.shift_on = (
            trimmed == ""
            or trimmed.endswith("\n")
            or (text.endswith(" ") and trimmed.endswith(SENTENCE_END))
        )

    # -- actions --------------------------------------------------------------
    def _on_cell(self, btn):
        """Runs when one of the 8 big buttons is selected."""
        self._tell_dashboard_cell(btn)
        if btn.role == "group":
            self.open_group = btn.key_value
        elif btn.role == "char":
            self.open_group = None   # back to the groups for the next character
            self._type_char(btn.key_value)
        elif btn.role == "back":
            self.open_group = None
        elif btn.role == "space":
            # A message never starts with a space (nor does a new one after Speak).
            if self.composed_text and not self.spoken:
                self._insert_text(" ")
        elif btn.role == "mode":
            self.symbols_on = not self.symbols_on
        elif btn.role == "shift":
            self.shift_on = not self.shift_on
        self._refresh_keys()

    def _tell_dashboard_cell(self, btn):
        """Records the selection of one of the 8 big buttons on the caregiver
        dashboard (see tell_dashboard). A typed character or Space is a key;
        the others (a group, Back, 123/ABC, Shift) only move around the
        keyboard."""
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
            # Punctuation typed after a Space belongs to the word before it:
            # "more ." becomes "more. ", keeping a space after it, so the user
            # doesn't spend a Backspace and a Space on it.
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
        self.spoken = False   # Backspace after Speak edits the spoken message
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
        """Speaks a ready-made reply (Quick Access) or Quick Fire phrase and
        shows it in the message box, in place of the message there, as if
        it had been typed and spoken."""
        self._cancel_clear()
        self.composed_text = text
        self.spoken = True
        self.shift_on = True   # the next message starts with a capital letter
        self._refresh_text()
        self._refresh_keys()
        # On a laptop, speaking holds up the screen until the voice has
        # finished, so it starts a moment later, once the message box
        # shows the text.
        Clock.schedule_once(lambda dt: speak(text), 0.1)

    def _on_quick_access(self, btn):
        tell_dashboard(btn, "keyboard", "shorthand", f"kb:sh:{slug(btn.text)}", btn.text)
        self.say(btn.text)

    def _speak_text(self, btn):
        if not self.composed_text.strip():
            # Nothing to say, so it only counts in a free-look test.
            tell_dashboard(btn, "keyboard", "control", "kb:speak", "Speak (nothing typed)",
                           navigation=True)
            return
        tell_dashboard(btn, "keyboard", "speak", "kb:speak", self.composed_text)
        speak(self.composed_text)
        self.spoken = True
        self.shift_on = True   # the next message starts with a capital letter
        self._refresh_text()
        self._refresh_keys()

    def on_pre_enter(self, *args):
        self.open_group = None   # always start from the groups
        self._cancel_clear()
        self._refresh_keys()


# ---------------------------------------------------------------------------
# 5. QUICK-FIRE INTERFACE (Figures 30-31, with the manuscript's six tabs)
# ---------------------------------------------------------------------------
# Each category: its tile colour, then (phrase spoken, picture file) pairs.
# The phrases are PRELIMINARY: they are to be replaced with the set
# confirmed by the caregiver and therapist interviews (Data Gathering
# Procedure). Pictures are in assets/symbols; see CREDITS.md there.
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
# Quick Fires work in two steps, like the letter groups: first the 6
# categories as big tiles (3 x 2), then the chosen category's 8 phrases
# plus a Back tile (3 x 3). Every tile is then at least about 5.5 x 3.5 cm,
# bigger than the 2-3 cm the eye tracker can be off by. (The design in
# Figures 30-31 had a row of category tabs only about 1.4 cm tall,
# which the eye tracker kept missing.)
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
    """Figures 30-31, reorganised for eye tracking into two steps: pick
    one of the manuscript's six categories, then one of its phrases.

    Selecting a phrase speaks it at once, with no confirmation step, as
    the manuscript specifies for urgent messages. It is also shown in the
    top bar and in the keyboard's message box. The Back tile returns
    to the categories. The top bar (see REVISED LAYOUT) says which step
    the user is on.
    """

    TAB = "quickfire"

    def __init__(self, navigator, **kwargs):
        super().__init__(navigator, **kwargs)
        self.current_category = None   # None: the categories are shown
        self.step_label = Label(bold=True, font_size=38, color=WHITE)
        step_box = Panel(bg_color=PANEL_BG, border_color=KEY_BORDER, radius=12, size_hint_x=0.8)
        step_box.add_widget(self.step_label)
        # Rest (see REST_AFTER), the same size as Speak on the keyboard.
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
        # Shown in the top bar here and in the keyboard's message box, and spoken.
        self.step_label.text = f"Said: {btn.key_value}"
        self.nav.sm.get_screen("keyboard").say(btn.key_value)

    def _on_rest(self, btn):
        tell_dashboard(btn, "quickfire", "control", "qf:rest", "Rest (resting screen)",
                       navigation=True)
        self.nav.go("rest")

    def on_pre_enter(self, *args):
        self._show_categories()   # always start from the categories


class RestScreen(Screen):
    """The resting screen (idle or sleep mode, see REST_AFTER): dark, with
    one big Start button and nothing else to select. While it is shown,
    the gaze engine processes fewer pictures (LOW_POWER_FPS). Looking at
    Start for the dwell time goes back to the screen the user was on."""

    def __init__(self, navigator, gaze_engine=None, **kwargs):
        super().__init__(**kwargs)
        self.nav = navigator
        self.gaze_engine = gaze_engine
        self.return_to = HOME_SCREEN   # set by Navigator.go("rest")
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
# 6. CALIBRATION SCREEN -- get ready, then a 9-point calibration
# ---------------------------------------------------------------------------
# The screen has three stages (and the accuracy test, see below):
#   1. READY:     a large camera preview and a Start Calibration button, so
#                 the user can sit correctly and see their face is detected
#                 before anything starts. The button only appears once the
#                 camera is actually sending pictures, so a slow start-up
#                 can never make calibration begin before the user is ready.
#   2. COUNTDOWN: 3, 2, 1 after Start is pressed, so the user has time to
#                 put their hands down and look at the screen.
#   3. DOTS:      the red dot visits 9 points (a 3 x 3 grid) and the gaze
#                 is measured at each one.
#
# Each of the 3 columns is measured 3 times and each of the 3 rows 3
# times, so one bad reading has much less effect on the left/right and
# up/down fits.

# Where the dots appear, as fractions of the screen: (x, y), with y = 0 at
# the BOTTOM of the screen (Kivy's convention). The order starts at the
# centre and then goes clockwise around the edge, so the eyes never have
# to jump across the whole screen between two dots.
CALIBRATION_POINTS = [
    (0.50, 0.47),                               # centre
    (0.08, 0.82), (0.50, 0.82), (0.92, 0.82),   # top row: left, middle, right
    (0.92, 0.47),                               # middle right
    (0.92, 0.12), (0.50, 0.12), (0.08, 0.12),   # bottom row: right, middle, left
    (0.08, 0.47),                               # middle left
]
CALIBRATION_CAPTURE_TIME = 4.0  # seconds the dot stays at each point
CALIBRATION_SETTLE_TIME = 1.5   # first part of those seconds is ignored: the
                                # eyes are still moving onto the new dot. Users
                                # with motor impairments may need this time.
# THE DOT appears big (DOT_START_SIZE) and shrinks to DOT_SIZE during the
# settle time: the movement catches the eye and leads it to the dot's
# middle, which has a small white centre (DOT_CENTRE_SIZE) to aim at.
# Sizes in pixels (about 85 make 1 cm). On 13 November 2026 two stroke
# patients' eyes did not stay on the old dot (40 pixels, 3 seconds per
# dot, 1 second ignored), so it is now bigger and stays a second longer.
DOT_START_SIZE = 120
DOT_SIZE = 44
DOT_CENTRE_SIZE = 12
# DOTS SHOWN AGAIN. After the 9 dots, up to REDO_MAX_DOTS dots at which
# the eyes did not stay on the dot are shown once more (the worst first),
# and the new readings replace the old ones. A dot counts as missed if,
# after the settle time, the eyes were closed or not seen for more than
# half of the time (REDO_MIN_SEEN), or the readings moved around more
# than REDO_MAX_SPREAD (standard deviation of fx or iy: the eyes
# wandered). Checked on 13 November: 29 of 34 of the developer's own
# saved calibrations would have shown no dot again (0.3 dots on average),
# while the two patients' eye pictures had 1 to 4 such dots per
# calibration. REDO_MAX_DOTS = 0 turns this off.
REDO_MAX_DOTS = 3
REDO_MIN_SEEN = 0.5
REDO_MAX_SPREAD = 0.02
COUNTDOWN_SECONDS = 3           # 3, 2, 1 after Start is pressed
# The camera focuses on the eyes during the countdown (see FOCUS_ON_EYES
# in gaze_engine.py). If it hasn't finished when the countdown ends, the
# dots wait for it, up to FOCUS_WAIT_SECONDS.
FOCUS_WAIT_SECONDS = 10
# Where the user should sit: about BEST_DISTANCE_CM from the camera. The
# READY stage shows the distance (worked out from the iris size, see
# get_face_distance_cm in gaze_engine.py) and asks the user to move
# closer or back if it is more than DISTANCE_TOLERANCE_CM off. Closer is
# more accurate: the eyes are bigger in the picture, and the same small
# error in the eye's angle moves the pointer less on the screen. On
# 9 November the face was about 55 cm away.
BEST_DISTANCE_CM = 40
DISTANCE_TOLERANCE_CM = 7
RESULT_SECONDS = 10             # how long the result stays on screen (to photograph it)
# CALIBRATING AGAIN. A result that says "please calibrate again" (Poor,
# up/down did not work, or calibration did not work) does not go on to
# the home screen: after RETRY_SECONDS, calibration starts again from the
# beginning (live preview and Start). After CALIBRATION_TRIES such results
# in a row, the app goes on anyway, so the user is never left without a
# way to talk, and the caregiver's dashboard gets a "Calibration required"
# alert (during a session). The caregiver can also leave calibration at
# any time with Stop calibration on the dashboard. CALIBRATION_TRIES = 1
# goes on after any result, as before 13 November 2026.
CALIBRATION_TRIES = 3
RETRY_SECONDS = 6
# The result's line when up/down did not calibrate (see ROW_MODE).
UPDOWN_FAILED_LINE = ("\nUp/down did not work and uses the mouse -- please calibrate again"
                      if ROW_MODE == "off" else
                      "\nUp/down did not work (Row mode will be used) -- please calibrate again")

# ACCURACY TEST (for testing, and for the results chapter). After
# calibration, the red dot visits ACCURACY_TEST_POINTS, places between
# the calibration dots, and the app measures where the gaze pointer
# lands: how far from the dot on average (accuracy) and how much it
# shakes (precision), in centimetres on the screen. Every reading, from
# calibration and from the test, is saved to a CSV file in the "logs"
# folder, so the real measurements can be analysed later. The test dots
# also make the calibration better (see TEST_DOTS_IN_CALIBRATION), so it
# can stay on for the real user sessions too; set RUN_ACCURACY_TEST to
# False only to make calibration about 20 seconds shorter.
RUN_ACCURACY_TEST = True

# USE THE TEST DOTS TOO. After the accuracy test, the calibration is fitted
# again from all 15 dots (9 calibration + 6 test dots). The result shown
# is still the 9-dot one, measured at dots the fits had not seen. Redone
# with the 22 saved calibrations that had a test (11 November), more dots
# put the pointer about 10% closer: each test dot, fitted with the 9
# calibration dots and the 5 other test dots, was 1.49 cm off on average
# instead of 1.66 cm (median 1.28 instead of 1.43), better in 17 of 22.
TEST_DOTS_IN_CALIBRATION = True

# The gaze engine's calibration, saved when the calibration screen opens
# so "Stop calibration" on the caregiver dashboard can put it back.
CALIBRATION_STATE = ("_x_points", "_y_points", "_shakes", "_x_fit", "_y_fit",
                     "_x_backup", "_y_backup", "_calib_face_sizes", "_calib_face_size",
                     "_x_heads", "_y_heads", "_calib_head", "_centre_x", "_centre_y",
                     "_pointer_shift", "updown_error_cm")
ACCURACY_TEST_POINTS = [
    (0.25, 0.65), (0.50, 0.65), (0.75, 0.65),   # upper row, left to right
    (0.75, 0.30), (0.50, 0.30), (0.25, 0.30),   # lower row, right to left
]
# Centimetres on the screen use SCREEN_PX_PER_CM_X and _Y, set in gaze_engine.py.
# The result screen rates the test in one word, against the letter keys
# (5.2 x 3.1 cm): "Good" if the pointer is off and shakes by at most
# GOOD_CM (about half a key's height, so it stays on the key), "OK" up to
# OK_CM, and "Poor" above that (calibrate again). The details of every
# test dot are printed in the terminal, and saved in the CSV file.
GOOD_CM = 1.5
OK_CM = 2.5
# EYE PICTURES (for testing): during calibration and the accuracy test,
# EYE_PICTURES_PER_DOT pictures of both eyes are saved at each dot, at the
# camera's full detail, together with the face landmarks and where the
# dot was. They go in a folder next to the CSV file (same name, ending in
# "_eyes"). With them, the eye measurements can be checked and improved
# on real eyes, away from the Pi. They are pictures of a person's eyes,
# so set SAVE_EYE_PICTURES to False for the real user sessions (Data
# Privacy Act), unless the participant agreed to it.
SAVE_EYE_PICTURES = True
EYE_PICTURES_PER_DOT = 6
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

GREEN_TEXT = rgb(120, 220, 120)  # "Face detected"
RED_TEXT = rgb(255, 120, 120)    # "Face not detected"
AMBER_TEXT = rgb(255, 200, 90)   # an "OK" calibration result

# Sizes on the calibration screens, in pixels. The screen shows 1920 x 1080
# pixels on 21.5 x 13.5 cm, so about 85 pixels make 1 cm and a 34-pixel
# text is about 4 mm tall. Made bigger on 13 November 2026, after the
# engineers' evaluation said the calibration text and result were too
# small to read from 40 cm. The live preview fills PREVIEW_WIDTH of the
# screen's width (0.5 = half) and keeps the camera picture's shape.
CALIBRATION_TEXT_SIZE = 32   # the instructions above the live preview
FACE_TEXT_SIZE = 34          # "Face detected ..." under the live preview
START_TEXT_SIZE = 38         # the Start Calibration button
COUNTDOWN_TEXT_SIZE = 72     # "Calibration starts in 3"
DOT_TEXT_SIZE = 30           # "Look at the red dot (1 of 9)" at the top edge
RESULT_TITLE_SIZE = 64       # "Calibration done: Good"
RESULT_TEXT_SIZE = 42        # "1.2 cm off (...), shaking 0.9 cm"
RESULT_DETAILS_SIZE = 30     # camera pictures a second, distance, log file
PREVIEW_WIDTH = 0.5


def _hex(color):
    """A colour (r, g, b, a from 0 to 1) as Kivy markup wants it: "78dc78"."""
    return "".join(f"{round(c * 255):02x}" for c in color[:3])


def save_gray_png(filename, image):
    """Saves a grayscale numpy picture (whole numbers 0-255) as a PNG file
    with only Python's own zlib (no extra library needed). A PNG file is
    a signature and three parts: IHDR (size; 8 bits, colour type 0 =
    grayscale), IDAT (the compressed rows, each starting with a 0 byte =
    "no filter") and IEND (the end). Each part is its length, its name,
    its data and a checksum."""
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
        self.stage = "ready"           # "ready", "countdown", "dots", "redo" or "test"
        self.summary = ""              # the calibration result, shown at the end
        self.log_rows = []             # every reading, saved to a CSV file at the end
        self.test_results = []         # per test dot: (target x, target y, pointer positions)
        self._countdown_left = 0
        self._sample_event = None
        self._preview_event = None
        self._countdown_event = None
        self._retry_event = None       # calibrating again after a poor result (see CALIBRATION_TRIES)
        self._tries = 0                # calibrations so far that said "please calibrate again"
        self._leaving = False          # True once we've decided to leave
        self.eye_pictures = []         # see SAVE_EYE_PICTURES: (stage, dot, dot x, dot y, seconds, picture)
        self._last_picture_number = None

        # Instructions at the top of the screen. Its height changes per
        # stage, so it never covers the preview or the dots.
        self.instruction_label = Label(
            font_size=CALIBRATION_TEXT_SIZE, size_hint=(1, 0.19), pos_hint={"top": 0.99, "x": 0},
            halign="center", valign="top", line_height=1.1,
        )
        self.instruction_label.bind(size=lambda w, s: setattr(w, "text_size", s))
        self.add_widget(self.instruction_label)

        # Live camera preview, with the green iris dots drawn by
        # gaze_engine, so the user can SEE their face is being tracked.
        # Only shown in the READY stage: during the dots it would pull the
        # eyes away from the dot and spoil the measurements.
        # "contain" lets the picture grow to fill its box (keeping its
        # shape); the camera sends a 768 x 432 picture for the preview.
        self.preview_image = Image(
            size_hint=(PREVIEW_WIDTH, 0.53), fit_mode="contain",
            pos_hint={"center_x": 0.5, "top": 0.8},
        )
        self.add_widget(self.preview_image)

        # "Face detected" / "Face not detected", just under the preview,
        # on two lines: what the camera sees, then what to do.
        self.face_label = Label(
            font_size=FACE_TEXT_SIZE, bold=True, size_hint=(1, 0.1),
            pos_hint={"center_x": 0.5, "y": 0.165}, halign="center", valign="middle",
        )
        self.face_label.bind(size=lambda w, s: setattr(w, "text_size", s))
        self.add_widget(self.face_label)

        # The Start button. It works by touch/click (for the caregiver) and
        # by dwell (resting the pointer on it), like every other button.
        self.start_btn = DwellButton(
            text="Start Calibration", bold=True, font_size=START_TEXT_SIZE,
            bg_color=PURPLE, text_color=WHITE, radius=16,
            size_hint=(None, None), size=(480, 115),
            pos_hint={"center_x": 0.5, "y": 0.03},
            on_selected=lambda b: self._start_countdown(),
        )
        self.start_btn.bind(on_release=lambda b: self._start_countdown())

        with self.canvas:   # the dot (see DOT_SIZE), off-screen until shown
            Color(0.9, 0.2, 0.2, 1)
            self.dot = Ellipse(pos=(-200, -200), size=(DOT_SIZE, DOT_SIZE))
            Color(1, 1, 1, 1)
            self.dot_centre = Ellipse(pos=(-200, -200), size=(DOT_CENTRE_SIZE, DOT_CENTRE_SIZE))

    # ---- stage 1: READY ------------------------------------------------
    def on_pre_enter(self, *args):
        # Kept so "Stop calibration" on the dashboard can put it back (see cancel).
        self._previous_calibration = {name: getattr(self.gaze_engine, name)
                                      for name in CALIBRATION_STATE if hasattr(self.gaze_engine, name)}
        self._tries = 0
        self._get_ready()

    def _get_ready(self):
        """The READY stage: live preview and Start. Also used to calibrate
        again after a poor result (see CALIBRATION_TRIES)."""
        self._cancel_retry()
        self.target_index = 0
        self.log_rows = []
        self.test_results = []
        self.gaze_engine.clear_calibration()
        self.stage = "ready"
        self._leaving = False
        self._hide_dot()
        self.eye_pictures = []
        self._dot_results = {}   # calibration dot -> its readings (see _keep_dot_result)
        self._redo_queue = []    # dots still to show again (see REDO_MAX_DOTS)
        self._redo_total = 0
        # Sharpen the camera picture for where the user sits now (see
        # FOCUS_ON_EYES in gaze_engine.py), so the preview is sharp too.
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
        self.preview_image.opacity = 0   # shown once the first picture arrives
        # The button is added once the first camera picture arrives.
        if self.start_btn.parent is not None:
            self.remove_widget(self.start_btn)
        self._shown_frame = None   # the camera picture on show (see _update_preview)
        self._preview_event = Clock.schedule_interval(self._update_preview, 1 / 15)

    def _update_preview(self, dt):
        """Runs 15 times a second during the READY stage: shows the newest
        camera picture and whether a face is detected."""
        if not self.gaze_engine.is_camera_ok():
            # No camera (or MediaPipe failed) -- don't leave the user stuck
            # on this screen; continue with mouse control instead.
            self._stop_preview()
            self.face_label.text = "No camera detected\nUsing mouse control instead"
            self.face_label.color = RED_TEXT
            self._leave_after(2.0)
            return

        frame = self.gaze_engine.get_debug_frame()
        if frame is None:
            return  # camera still starting up

        # Only a new picture is drawn, into the same texture each time:
        # making a new texture 15 times a second (as before) took time
        # away from the gaze engine, which shares the processor.
        if frame is not self._shown_frame:
            self._shown_frame = frame
            h, w = frame.shape[:2]
            texture = self.preview_image.texture
            if texture is None or tuple(texture.size) != (w, h):
                texture = Texture.create(size=(w, h), colorfmt="rgb")
                texture.flip_vertical()  # Kivy textures are bottom-up; camera frames are top-down
                self.preview_image.texture = texture
            texture.blit_buffer(frame.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
            self.preview_image.canvas.ask_update()
            self.preview_image.opacity = 1

        feature = self.gaze_engine.get_latest_feature()
        distance = self.gaze_engine.get_face_distance_cm()
        if self.gaze_engine.is_focusing() and distance is not None:
            # The face has been seen; the camera is focusing on the eyes.
            self.face_label.text = "Face detected\nFocusing the camera on your eyes..."
            self.face_label.color = WHITE
        elif feature is None:
            self.face_label.text = "Face not detected\nMove into the camera view"
            self.face_label.color = RED_TEXT
        else:
            # The IR glints are not needed to start: when they are not seen
            # at every calibration dot (with the camera above the screen
            # they hide under the upper eyelid at the bottom dots), the
            # iris is used instead (see _fit_axis in gaze_engine.py). So a
            # missing glint is only mentioned, not shown as a problem.
            glints = "IR glints seen" if feature[1] is not None else "no IR glints (the iris will be used)"
            advice, color = "Press Start when you are ready", GREEN_TEXT
            if distance is not None:
                glints += f", {distance:.0f} cm from the camera"
                # See BEST_DISTANCE_CM. Start still works; this is advice.
                if distance > BEST_DISTANCE_CM + DISTANCE_TOLERANCE_CM:
                    advice, color = f"Move closer, to about {BEST_DISTANCE_CM} cm", WHITE
                elif distance < BEST_DISTANCE_CM - DISTANCE_TOLERANCE_CM:
                    advice, color = f"Move back, to about {BEST_DISTANCE_CM} cm", WHITE
            self.face_label.text = f"Face detected: {glints}\n{advice}"
            self.face_label.color = color

        # The camera works, so the user may start now.
        if self.start_btn.parent is None:
            self.add_widget(self.start_btn)

    def _stop_preview(self):
        if self._preview_event is not None:
            self._preview_event.cancel()
            self._preview_event = None

    def _place_label(self, where, font_size, height, markup=False):
        """Moves the instruction text: "top" = a strip along the top edge,
        "center" = large in the middle of the screen (countdown, result).
        markup=True lets the text change size and colour line by line
        (Kivy markup such as [size=64]...[/size]); only the result uses it."""
        self.instruction_label.markup = markup
        self.instruction_label.font_size = font_size
        self.instruction_label.size_hint_y = height
        if where == "top":
            self.instruction_label.pos_hint = {"x": 0, "top": 0.99}
            self.instruction_label.valign = "top"
        else:
            self.instruction_label.pos_hint = {"x": 0, "center_y": 0.5}
            self.instruction_label.valign = "middle"

    # ---- stage 2: COUNTDOWN ----------------------------------------------
    def _start_countdown(self):
        if self.stage != "ready":
            return  # already started (e.g. tapped and dwelled at the same time)
        self.stage = "countdown"
        # The user is now in place: focus once more during the countdown.
        self.gaze_engine.request_focus()
        self._stop_preview()
        self.remove_widget(self.start_btn)
        self.preview_image.opacity = 0   # hide the preview from now on
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
        # Wait for the camera to finish focusing on the eyes (see
        # FOCUS_WAIT_SECONDS): meanwhile there are no gaze readings.
        if self.gaze_engine.is_focusing() and self._countdown_left > -FOCUS_WAIT_SECONDS:
            self.instruction_label.text = "Focusing the camera on your eyes...\nKeep your head still"
            return
        self._countdown_event.cancel()
        self._countdown_event = None
        # Smaller text in a thin strip at the top, so it never covers a dot.
        self._place_label("top", DOT_TEXT_SIZE, 0.1)
        self.stage = "dots"
        self._show_current_target()

    # ---- stage 3: DOTS (and the accuracy test) --------------------------
    def _points(self):
        return ACCURACY_TEST_POINTS if self.stage == "test" else CALIBRATION_POINTS

    def _show_current_target(self):
        # The dot's position is worked out from the screen's size NOW, not
        # when the app started: the window is still growing to full size
        # (maximizing) at start-up, and using that early, smaller size put
        # every dot in the left part of the screen.
        fx, fy = self._points()[self.target_index]
        x, y = self.width * fx, self.height * fy
        self.current_target_px = (x, y)
        self._draw_dot(0.0)
        self._ticks_after_settle = 0   # checks after the settle time (see REDO_MIN_SEEN)
        self.samples = []      # fx readings (left/right) at this dot
        self.samples_vy = []   # vy readings (up/down, glints), only when a glint is seen
        self.samples_iy = []   # iy readings (up/down, iris against the eye corners)
        self.samples_hx = []   # hx readings (left/right, glints), only when a glint is seen
        self.pointer_samples = []  # accuracy test: where the gaze pointer was
        self.smoothed_samples = [] # (fx, vy, iy, hx) as averaged for the pointer
        self._pictures_at_dot = 0  # eye pictures kept at this dot (SAVE_EYE_PICTURES)
        self._target_start_time = time.time()
        self._sample_event = Clock.schedule_interval(self._collect_sample, 1 / 30)
        Clock.schedule_once(self._finish_current_target, CALIBRATION_CAPTURE_TIME)

    def _draw_dot(self, elapsed):
        """Draws the dot at the current target, `elapsed` seconds after it
        appeared: it shrinks from DOT_START_SIZE to DOT_SIZE during the
        settle time."""
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
        # This picture's own readings, not averaged over time: averaged
        # ones still trail behind the eyes after the move from the
        # previous dot. Recomputed from 28 saved calibrations (12
        # November), using these put the test dots 1.46 cm off instead of
        # 1.74 cm on average, and 1.51 instead of 2.04 cm in the 14 that
        # ran below 6.5 pictures a second.
        raw = self.gaze_engine.get_latest_raw_feature()
        elapsed = time.time() - self._target_start_time
        if elapsed <= CALIBRATION_SETTLE_TIME + 0.1:
            self._draw_dot(elapsed)   # still shrinking
        pointer = (self.gaze_engine.get_screen_x(), self.gaze_engine.get_screen_y())
        self._log(elapsed, feature, pointer, raw)
        if elapsed >= CALIBRATION_SETTLE_TIME:
            self._ticks_after_settle += 1
            self._keep_eye_picture(elapsed)
        # Short blinks already give no readings; longer eye closures do,
        # and those readings are not where the user looks (see eyes_closed
        # in gaze_engine.py). They stay in the CSV file, but not in the dot.
        if raw is not None and self.gaze_engine.eyes_closed():
            raw = None
        if self.stage == "test" and elapsed >= CALIBRATION_SETTLE_TIME:
            self.pointer_samples.append(pointer)
        # Ignore the first moment at each dot: the eyes are still moving there.
        # (The test dots' readings are kept too, see TEST_DOTS_IN_CALIBRATION.)
        if raw is not None and elapsed >= CALIBRATION_SETTLE_TIME:
            fx, vy, iy, hx = raw
            self.samples.append(fx)
            self.samples_iy.append(iy)
            if vy is not None:
                self.samples_vy.append(vy)
            if hx is not None:
                self.samples_hx.append(hx)
        # The averaged readings, only to measure how much the pointer
        # would shake (see _add_dot_to_calibration).
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
        """Keeps a picture of the eyes at this dot (see SAVE_EYE_PICTURES),
        up to EYE_PICTURES_PER_DOT, each from a different camera picture.
        They are written to files at the end (_save_eye_pictures)."""
        if not SAVE_EYE_PICTURES or self._pictures_at_dot >= EYE_PICTURES_PER_DOT:
            return
        # Spread over the measured part of the dot's time, not all at its start.
        spacing = (CALIBRATION_CAPTURE_TIME - CALIBRATION_SETTLE_TIME) / EYE_PICTURES_PER_DOT
        if elapsed < CALIBRATION_SETTLE_TIME + self._pictures_at_dot * spacing:
            return
        picture = self.gaze_engine.get_eye_picture()
        if picture is None or picture[0] == self._last_picture_number:
            return   # no face, or the same camera picture as last time
        self._last_picture_number = picture[0]
        self._pictures_at_dot += 1
        target_x, target_y = self.current_target_px
        self.eye_pictures.append((self.stage, self.target_index + 1, round(target_x), round(target_y),
                                  round(elapsed, 3), picture))

    def _log(self, elapsed, feature, pointer, raw=None):
        """Keeps one reading for the CSV file (see RUN_ACCURACY_TEST)."""
        def rounded(value, digits):
            return None if value is None else round(value, digits)

        fx, vy, iy, hx = feature if feature is not None else (None, None, None, None)
        # The head's pose (see _head_pose in gaze_engine.py), angles in degrees.
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
            return  # calibration was stopped (see cancel): don't carry on in the background

        if self.stage == "test":
            self.test_results.append((*self.current_target_px, self.pointer_samples))
            # Stored now, but only used for the fits after the test's result
            # is worked out (see TEST_DOTS_IN_CALIBRATION).
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
        """Keeps this calibration dot's readings, how long the eyes were
        seen and how much the readings moved (see REDO_MAX_DOTS), and the
        head pose and face size now. They go to the gaze engine once all
        dots are done (see _finish_calibration), so a dot shown again
        replaces its first readings."""
        seen = len(self.samples) / self._ticks_after_settle if self._ticks_after_settle else 0.0
        spread = None
        if len(self.samples) >= 2:
            spread = float(max(np.std(self.samples), np.std(self.samples_iy)))
        self._dot_results[self.target_index] = (self._dot_reading(), self.gaze_engine.head_and_face_size(),
                                                seen, spread)
        print(f"[Calibration] dot {self.target_index + 1}: eyes seen {seen:.0%} of the time, "
              + ("no readings" if spread is None else f"spread {spread:.3f}"))

    def _dots_to_redo(self):
        """The calibration dots (indexes) at which the eyes did not stay on
        the dot, the worst first, at most REDO_MAX_DOTS (see there)."""
        missed = []
        for index, (reading, head_and_size, seen, spread) in self._dot_results.items():
            if seen < REDO_MIN_SEEN or spread is None:
                missed.append((float("inf"), index))   # eyes closed or not seen
            elif spread > REDO_MAX_SPREAD:
                missed.append((spread, index))         # the eyes wandered
        missed.sort(key=lambda m: (-m[0], m[1]))
        return [index for _, index in missed[:REDO_MAX_DOTS]]

    def _add_dot_to_calibration(self):
        """Gives the gaze engine this dot's average readings (nothing if the
        face wasn't seen here). Used for the accuracy test's dots."""
        reading = self._dot_reading()
        if reading is not None:
            self.gaze_engine.add_calibration_point(**reading)

    def _dot_reading(self):
        """This dot's average readings, as add_calibration_point's
        arguments, or None if the face wasn't seen here."""
        if not self.samples:
            return None
        avg_fx = sum(self.samples) / len(self.samples)
        target_x, target_y = self.current_target_px
        # The glint measurements are only recorded if a glint was seen
        # for at least half of the readings at this dot; otherwise
        # they're unreliable.
        avg_vy = avg_hx = None
        if len(self.samples_vy) >= len(self.samples) / 2:
            avg_vy = sum(self.samples_vy) / len(self.samples_vy)
        if len(self.samples_hx) >= len(self.samples) / 2:
            avg_hx = sum(self.samples_hx) / len(self.samples_hx)
        avg_iy = sum(self.samples_iy) / len(self.samples_iy)
        # How much each measurement wobbled at this dot (standard
        # deviation) after the averaging the pointer uses; the gaze
        # engine uses it to choose its fits.
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
            # Not enough usable samples (e.g. face wasn't visible) --
            # fall back to mouse rather than leaving gaze half-broken.
            print(f"[Calibration] {e} -- falling back to mouse control.")
            self.summary = "Calibration did not work, using the mouse -- please calibrate again"
            self._show_result()
            return
        print("[Calibration] " + fits.replace("\n", "; "))   # the fits, for testing
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
        """Works out the accuracy test's result (see RUN_ACCURACY_TEST): on
        average, and for each test dot which way the pointer was off."""
        offsets_x, offsets_y, shakes, missing, per_dot = [], [], [], 0, []
        for (target_x, target_y, pointers), (fx, fy) in zip(self.test_results, ACCURACY_TEST_POINTS):
            seen = [(x, y) for x, y in pointers if x is not None and y is not None]
            missing += len(pointers) - len(seen)
            if not seen:
                continue
            xs = np.array([x for x, y in seen]) / SCREEN_PX_PER_CM_X   # in cm
            ys = np.array([y for x, y in seen]) / SCREEN_PX_PER_CM_Y
            # Accuracy: how far the average pointer position is from the dot.
            dx = xs.mean() - target_x / SCREEN_PX_PER_CM_X
            dy = ys.mean() - target_y / SCREEN_PX_PER_CM_Y
            offsets_x.append(abs(dx))
            offsets_y.append(abs(dy))
            # Precision: how far the pointer strays from its own average.
            shake = np.sqrt(np.mean((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))
            shakes.append(shake)
            # Kivy's y grows upwards, so a positive dy means the pointer was too high.
            where = ("upper" if fy > 0.5 else "lower") + " " + \
                    ("left" if fx < 0.4 else "right" if fx > 0.6 else "middle")
            per_dot.append(f"{where}: {abs(dx):.1f} cm too far {'right' if dx > 0 else 'left'}, "
                           f"{abs(dy):.1f} cm too {'high' if dy > 0 else 'low'}, shaking {shake:.1f} cm")
        if not shakes:
            self._learn_test_dots()
            self._show_result()   # no gaze pointer: the summary already says why
            return
        distance = float(np.mean(np.hypot(offsets_x, offsets_y)))
        # Kept for Row mode (see ROW_MODE_UPDOWN_CM).
        self.gaze_engine.updown_error_cm = float(np.mean(offsets_y))
        shaking = float(np.mean(shakes))
        # The details, for testing, in the terminal (see GOOD_CM).
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
        """Fits the calibration again from all 15 dots, the 9 calibration
        dots and the 6 test dots (see TEST_DOTS_IN_CALIBRATION). The
        result on screen stays the one of the 9-dot fits, a fair test.
        If the new fits are not possible, or lose the up/down direction
        that the 9-dot fits had, the 9-dot fits are kept."""
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
            # The test dots rescued up/down: the summary's warning no longer holds.
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
        # The first line (the rating) is the biggest, in green (Good),
        # amber (OK) or red (Poor, or calibration did not work).
        title, *rest = self.summary.split("\n")
        color = (GREEN_TEXT if title.endswith(": Good") else AMBER_TEXT if title.endswith(": OK")
                 else RED_TEXT if ("Poor" in title or "did not work" in title) else WHITE)
        lines = [f"[size={RESULT_TITLE_SIZE}][b][color={_hex(color)}]{escape_markup(title)}"
                 f"[/color][/b][/size]"]
        lines += [escape_markup(line) for line in rest]
        # A Pi that slows itself down makes fewer pictures a second, and
        # the tracking suffers (see _pi_health in gaze_engine.py).
        slowdown = self.gaze_engine.get_slowdown()
        if slowdown:
            lines.append(f"[color={_hex(RED_TEXT)}]Warning: the Pi is slowed down: "
                         f"{escape_markup(slowdown)}[/color]")
        # A poor result: calibrate again, up to CALIBRATION_TRIES times.
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
        self._leave_after(RESULT_SECONDS)  # long enough to read (or photograph) the result

    def _calibrate_again(self):
        """RETRY_SECONDS after a poor result: back to the READY stage, unless
        calibration was left meanwhile (Stop calibration on the dashboard)."""
        self._retry_event = None
        if self.manager is not None and self.manager.current == self.name and self.stage == "result":
            self._get_ready()

    def _cancel_retry(self):
        if self._retry_event is not None:
            self._retry_event.cancel()
            self._retry_event = None

    def _save_log(self):
        """Saves every reading to logs/gaze_<date>_<time>.csv. Returns the
        file's name, or "" if it couldn't be saved."""
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
        """Writes the eye pictures (see SAVE_EYE_PICTURES) into `folder`:
        one PNG file per picture, and pictures.jsonl with one line per
        picture: its file, the stage, the dot and its position on the
        screen, the seconds since the dot appeared, where the picture's
        top-left corner is in the camera picture, and the 478 face
        landmarks (camera-picture pixels). The first line describes the
        camera."""
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

    # ---- leaving -----------------------------------------------------------
    def _leave_after(self, seconds):
        if self._leaving:
            return  # already on the way out
        self._leaving = True
        Clock.schedule_once(lambda dt: self._done(), seconds)

    def cancel(self):
        """"Stop calibration" on the caregiver dashboard: stops the dots and
        puts back the calibration from before (if there was one). The app
        then shows the keyboard."""
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
        """Leaves calibration, but only if it is still the screen on display.
        Calibration's timers keep running after the screen is left, and
        must never pull the user away from another screen later."""
        if self.manager is not None and self.manager.current == self.name:
            self.on_done()


# ---------------------------------------------------------------------------
# 7. APP ENTRY POINT
# ---------------------------------------------------------------------------
class EyeGlyntApp(App):
    def build(self):
        self.title = "EyeGlynt"
        keep_speaker_awake()
        start_voice()
        dashboard.start()   # the caregiver dashboard, if it is installed (dashboard_link.py)
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

        # Drives dwell selection -- real gaze (X) once calibrated, mouse
        # for everything else, with automatic fallback to full mouse
        # control if calibration didn't complete.
        self.gaze_manager = GazeManager(sm, gaze_engine=self.gaze_engine)
        self.gaze_manager.on_row_mode = self._on_row_mode
        Clock.schedule_interval(self._check_dashboard, 0.1)
        # Resting by itself (see REST_AFTER): the time counts from the last
        # selection, or from when the screen came up if that is later.
        self._screen_since = time.time()
        sm.bind(current=lambda *_: setattr(self, "_screen_since", time.time()))
        Clock.schedule_interval(self._check_rest, 1.0)
        return sm

    def _check_rest(self, dt):
        """Runs every second: rests the app after REST_AFTER seconds
        without a selection on the keyboard or Quick Fires."""
        if not REST_AFTER or self.sm.current not in ("keyboard", "quickfire"):
            return
        target = self.gaze_manager._current_target
        if target is not None and target.dwelling:
            return   # a selection is on its way
        idle = time.time() - max(self.gaze_manager._selected_at, self._screen_since)
        if idle >= REST_AFTER:
            print(f"[Rest] No selection for {REST_AFTER:.0f} s: resting")
            self.navigator.go("rest")

    def _on_row_mode(self, on):
        """Row mode turned on or off (see ROW_MODE): the strips appear or go."""
        for name in ("keyboard", "quickfire"):
            self.sm.get_screen(name).set_row_mode(on)

    def _calibration_done(self):
        # A recalibration asked for on the dashboard counts as successful if
        # the gaze engine is calibrated again (see dashboard_link.py).
        dashboard.recalibration_finished(success=self.gaze_engine.is_calibrated())
        self.sm.current = HOME_SCREEN

    def _check_dashboard(self, dt):
        """Runs 10 times a second: tells the caregiver dashboard what the
        device is doing, and carries out the dashboard's buttons."""
        calibrating = self.sm.current == "calibration"
        if not self.gaze_engine.is_camera_ok() or self.sm.current == "rest":
            state = "idle"   # no camera (mouse only), or resting (see REST_AFTER)
        else:
            state = "calibrating" if calibrating else "tracking"
        dashboard.update(state, fps=self.gaze_engine.get_fps(),
                         eyes_seen=self.gaze_engine.get_latest_feature() is not None)
        for command, details in dashboard.take_commands():
            if command == "recalibrate":
                # The caregiver pressed Recalibrate: show the calibration screen.
                dashboard.recalibration_started("manual_caregiver")
                if not calibrating:
                    self.sm.current = "calibration"
            elif command == "cancel_recalibration" and self.sm.current == "calibration":
                # The caregiver pressed Stop calibration: back to the home
                # screen with the calibration from before.
                self.sm.get_screen("calibration").cancel()
                dashboard.recalibration_cancelled()
                self.sm.current = HOME_SCREEN
            elif command in ("start_session", "set_dwell_time") and details.get("dwell_time_ms"):
                # The dwell time the caregiver picked for this session.
                set_dwell_time(details["dwell_time_ms"] / 1000)
            elif command == "end_session":
                set_dwell_time(DEVICE_DWELL_TIME)

    def on_start(self):
        # Open maximized, filling the screen but keeping the title bar, so
        # the window can still be closed or moved normally.
        Window.maximize()
        # Row mode's highlight (see ROW_MODE), over the screens.
        self.gaze_manager.row_overlay = RowOverlay()
        Window.add_widget(self.gaze_manager.row_overlay)
        if SHOW_GAZE_POINTER:
            # Added to the window after the screens, so it is drawn on top.
            self.gaze_manager.pointer = GazePointer()
            Window.add_widget(self.gaze_manager.pointer)

    def on_stop(self):
        self.gaze_engine.stop()
        dashboard.stop()


if __name__ == "__main__":
    EyeGlyntApp().run()