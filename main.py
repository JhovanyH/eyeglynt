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
import os
import time

import numpy as np

import pyttsx3
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

from gaze_engine import GazeEngine, SCREEN_PX_PER_CM
from icons import draw_icon


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
DWELL_TIME = 4.0  # seconds

# Gaze is never perfectly steady: the pointer jitters by a few millimetres
# even when the eyes are still, and a blink can throw it off for a moment.
# Without help, every jitter across a button's edge restarts the dwell, so
# a selection may never finish. Two settings make buttons "sticky":
STICKY_MARGIN = 60   # pixels (about 1 cm): the button being dwelled on still
                     # counts as looked at while the pointer is this close to it
LEAVE_DELAY = 0.3    # seconds: the pointer must stay away from the button this
                     # long before its dwell is cancelled (rides out blinks)
REPEAT_TIME = DWELL_TIME  # held keys repeat at the SAME rate as the initial dwell,
                          # so there's only one dwell-time value to document/defend.
                          # Set this to a different (smaller) number later if you
                          # decide, with testing data, that repeats should be faster.

# Calibration is least accurate at the edges of the screen, and the
# pointer often overshoots there, past the edge. A pointer outside the
# screen is pulled back to EDGE_INSET pixels inside it, so an overshoot
# still lands on the button at the edge (Clear, Space, Thank you, ...).
EDGE_INSET = 20  # pixels

# For testing only: draws a small yellow circle where the app thinks the
# person is looking, so you can see how far off the tracking is. Set it to
# False for the real user tests, because people tend to chase the circle
# instead of looking at the button.
SHOW_GAZE_POINTER = True

# ---------------------------------------------------------------------------
# 1. TEXT-TO-SPEECH
# ---------------------------------------------------------------------------
# pyttsx3 works out of the box on Windows/Mac for testing. On the Pi, you
# may end up calling espeak-ng directly instead (pyttsx3 just wraps it on
# Linux anyway) -- swap the internals of speak() later if needed, the rest
# of the app doesn't care how speak() is implemented.
def speak(text: str):
    """
    Creates a brand-new TTS engine for every single call, instead of
    reusing one. pyttsx3's Windows voice (SAPI5) has a known quirk where
    reusing one engine across multiple say()/runAndWait() calls in the
    same run often goes silent after the first utterance -- a fresh
    engine each time avoids that reliably. Slightly slower to set up
    each time, but for a phrase every second or two, it's unnoticeable.
    """
    print(f"[TTS] Speaking: {text}")
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"[TTS] Error: {e}")


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

    def __init__(self, on_selected=None, dwell_time=DWELL_TIME,
                 repeatable=False, repeat_time=REPEAT_TIME,
                 bg_color=None, border_color=None, text_color=None,
                 radius=10, icon=None, icon_color=None, icon_layout="top",
                 selectable=True, image=None, **kwargs):
        super().__init__(**kwargs)
        self.on_selected = on_selected
        self.dwell_time = dwell_time
        self.repeatable = repeatable
        self.repeat_time = repeat_time
        self._active_dwell_time = dwell_time
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
    def start_dwell(self, dwell_time=None):
        if self._event is not None:
            return  # already dwelling on this button
        self.progress = 0.0
        self._active_dwell_time = dwell_time if dwell_time is not None else self.dwell_time
        self._event = Clock.schedule_interval(self._tick, 1 / 30)

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
    """

    def __init__(self, screen_manager: ScreenManager, gaze_engine: GazeEngine = None):
        self.sm = screen_manager
        self.gaze_engine = gaze_engine
        self._current_target = None
        self._away_since = None  # when the pointer left the current button, or None
        self._blocked = None     # a new button the gaze must leave before it can be used
        self.pointer = None      # the test circle (SHOW_GAZE_POINTER), added in on_start
        Clock.schedule_interval(self._update, 1 / 30)

    def get_pointer_pos(self):
        x, y = Window.mouse_pos  # start from the mouse...
        if self.gaze_engine is not None:
            gaze_x = self.gaze_engine.get_screen_x()  # None if not available
            gaze_y = self.gaze_engine.get_screen_y()
            if gaze_x is not None:
                x = gaze_x  # ...and replace each direction gaze can provide
            if gaze_y is not None:
                y = gaze_y
        # Keep the pointer on the screen (see EDGE_INSET).
        x = min(max(x, EDGE_INSET), Window.width - EDGE_INSET)
        y = min(max(y, EDGE_INSET), Window.height - EDGE_INSET)
        return (x, y)

    def _update(self, dt):
        active_screen = self.sm.current_screen
        if active_screen is None:
            return
        pos = self.get_pointer_pos()
        if self.pointer is not None:
            # Hidden during calibration, where it would distract from the dots.
            self.pointer.move_to(pos, visible=active_screen.name != "calibration")
        buttons = [btn for btn in _find_dwell_buttons(active_screen)
                   if not btn.disabled and btn.selectable]  # greyed-out/display-only can't be selected
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
            self._blocked = self._button_at(buttons, pos)
        if self._blocked is not None:
            if self._blocked in buttons and self._is_near(self._blocked, pos, STICKY_MARGIN):
                return
            self._blocked = None

        # Sticky buttons (see STICKY_MARGIN and LEAVE_DELAY): keep dwelling
        # on the current button while the pointer is near it...
        if current in buttons and self._is_near(current, pos, STICKY_MARGIN):
            self._away_since = None
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
            if target is not None:
                target.start_dwell()

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


# ---------------------------------------------------------------------------
# 3b. NAVIGATION -- the KEYBOARD and QUICK FIRES tabs
# ---------------------------------------------------------------------------
class Navigator:
    """Switches between the Keyboard and Quick Fires screens, for the two
    tabs in the top bar of each screen."""

    def __init__(self, screen_manager):
        self.sm = screen_manager

    def go(self, name):
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
        frame = Panel(bg_color=FRAME_BG, radius=0, orientation="vertical", padding=10, spacing=10)
        top_bar = BoxLayout(orientation="horizontal", spacing=10, size_hint_y=TOP_BAR_HEIGHT)
        top_bar.add_widget(self._tab("KEYBOARD", "keyboard", "keyboard", PURPLE))
        top_bar.add_widget(self._tab("QUICK FIRES", "flame", "quickfire", ORANGE))
        controls.size_hint_x = 1 - 2 * TAB_WIDTH
        top_bar.add_widget(controls)
        frame.add_widget(top_bar)
        content.size_hint_y = 1 - TOP_BAR_HEIGHT
        frame.add_widget(content)
        self.add_widget(frame)

    def _tab(self, text, icon, screen_name, color):
        """One of the two tabs. The tab of the screen on display gets a
        white border and is not selectable; the other one switches to its
        screen."""
        own = screen_name == self.TAB
        return DwellButton(text=text, bold=True, font_size=20, icon=icon, icon_layout="top",
                           bg_color=color, text_color=WHITE, radius=12,
                           border_color=WHITE if own else None, selectable=not own,
                           size_hint_x=TAB_WIDTH,
                           on_selected=lambda b: self.nav.go(screen_name))


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
        self.text_label = Label(font_size=34, halign="left", valign="top", size_hint_y=None,
                                markup=True)  # markup: for the coloured cursor (see _refresh_text)
        # Wrap at the box width; let the height grow with the text.
        self.text_label.bind(width=lambda w, v: setattr(w, "text_size", (v, None)))
        self.text_label.bind(texture_size=self._fit_text_height)
        self.text_scroll.add_widget(self.text_label)
        box.add_widget(self.text_scroll)
        row.add_widget(box)

        action = dict(bg_color=PURPLE, text_color=WHITE, radius=12, bold=True, font_size=20,
                      icon_layout="top", size_hint_x=0.2)
        row.add_widget(DwellButton(text="Speak", icon="speak",
                                   on_selected=lambda b: self._speak_text(), **action))
        row.add_widget(DwellButton(text="Backspace", icon="erase", repeatable=True,
                                   on_selected=lambda b: self._backspace(), **action))
        # Repeatable, so that simply keeping the eyes on it gives the second selection.
        self.clear_btn = DwellButton(text="Clear", repeatable=True, on_selected=self._on_clear,
                                     **{**action, "font_size": 26})
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
            btn = DwellButton(text=word, bold=True, font_size=32, bg_color=color,
                              text_color=DARK_TEXT, radius=12,
                              on_selected=lambda b: self.say(b.text))
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
        btn.font_size = 44 if role == "char" else 30
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

    def _backspace(self):
        self.spoken = False   # Backspace after Speak edits the spoken message
        self.composed_text = self.composed_text[:-1]
        self._after_edit()

    def _on_clear(self, btn):
        """First selection: ask (see CLEAR_CONFIRM_TIME). Second: erase."""
        if self._clear_timer is None:
            btn.text = "Clear?"
            self._clear_timer = Clock.schedule_once(lambda dt: self._cancel_clear(),
                                                    CLEAR_CONFIRM_TIME)
            return
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
        # Speaking holds up the screen until the voice has finished, so it
        # starts a moment later, once the message box shows the text.
        Clock.schedule_once(lambda dt: speak(text), 0.1)

    def _speak_text(self):
        if self.composed_text.strip():
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
        self.step_label = Label(bold=True, font_size=30, color=WHITE)
        step_box = Panel(bg_color=PANEL_BG, border_color=KEY_BORDER, radius=12)
        step_box.add_widget(self.step_label)
        self.tiles_panel = Panel(bg_color=PANEL_BG, border_color=KEY_BORDER, padding=10)
        self.build_frame(step_box, self.tiles_panel)
        self._show_categories()

    def _show_categories(self):
        self.current_category = None
        self.step_label.text = "Pick a category"
        grid = GridLayout(cols=QUICKFIRE_COLUMNS, spacing=10)
        for name, (color, _phrases) in QUICKFIRE_CATEGORIES.items():
            tile = DwellButton(text=name, bold=True, font_size=26, bg_color=color,
                               text_color=DARK_TEXT, radius=12,
                               image=CATEGORY_PICTURES.get(name), icon_layout="top",
                               on_selected=self._on_category)
            tile.key_value = name
            grid.add_widget(tile)
        self._show_grid(grid)

    def _show_phrases(self, name):
        self.current_category = name
        self.step_label.text = f"{name}: pick a phrase"
        color, phrases = QUICKFIRE_CATEGORIES[name]
        grid = GridLayout(cols=QUICKFIRE_COLUMNS, spacing=10)
        for phrase, picture in phrases:
            tile = DwellButton(text=phrase, font_size=24, bg_color=color, text_color=DARK_TEXT,
                               radius=12, image=picture, icon_layout="top",
                               on_selected=self._on_phrase)
            tile.key_value = phrase
            grid.add_widget(tile)
        grid.add_widget(DwellButton(text="Back", bold=True, font_size=26, icon="arrow_left",
                                    icon_layout="left", on_selected=lambda b: self._show_categories(),
                                    **{**KEY_STYLE, "radius": 12}))
        self._show_grid(grid)

    def _show_grid(self, grid):
        self.tiles_panel.clear_widgets()
        self.tiles_panel.add_widget(grid)

    def _on_category(self, btn):
        self._show_phrases(btn.key_value)

    def _on_phrase(self, btn):
        # Shown in the top bar here and in the keyboard's message box, and spoken.
        self.step_label.text = f"Said: {btn.key_value}"
        self.nav.sm.get_screen("keyboard").say(btn.key_value)

    def on_pre_enter(self, *args):
        self._show_categories()   # always start from the categories


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
CALIBRATION_CAPTURE_TIME = 3.0  # seconds the dot stays at each point
CALIBRATION_SETTLE_TIME = 1.0   # first part of those seconds is ignored: the
                                # eyes are still moving onto the new dot. Users
                                # with motor impairments may need this time.
COUNTDOWN_SECONDS = 3           # 3, 2, 1 after Start is pressed
RESULT_SECONDS = 30             # how long the result stays on screen (to photograph it)

# ACCURACY TEST (for testing, and for the results chapter). After
# calibration, the red dot visits ACCURACY_TEST_POINTS, places between
# the calibration dots, and the app measures where the gaze pointer
# lands: how far from the dot on average (accuracy) and how much it
# shakes (precision), in centimetres on the screen. Every reading, from
# calibration and from the test, is saved to a CSV file in the "logs"
# folder, so the real measurements can be analysed later. Set
# RUN_ACCURACY_TEST to False for the real user sessions.
RUN_ACCURACY_TEST = True
ACCURACY_TEST_POINTS = [
    (0.25, 0.65), (0.50, 0.65), (0.75, 0.65),   # upper row, left to right
    (0.75, 0.30), (0.50, 0.30), (0.25, 0.30),   # lower row, right to left
]
# Centimetres on the screen use SCREEN_PX_PER_CM, set in gaze_engine.py.
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

GREEN_TEXT = rgb(120, 220, 120)  # "Face detected"
RED_TEXT = rgb(255, 120, 120)    # "Face not detected"
ORANGE_TEXT = rgb(255, 190, 90)  # "No IR glints"


class CalibrationScreen(Screen):
    def __init__(self, gaze_engine: GazeEngine, on_done, **kwargs):
        super().__init__(**kwargs)
        self.gaze_engine = gaze_engine
        self.on_done = on_done
        self.current_target_px = None  # (x, y) of the dot on display
        self.target_index = 0
        self.samples = []
        self.stage = "ready"           # "ready", "countdown", "dots" or "test"
        self.summary = ""              # the calibration result, shown at the end
        self.log_rows = []             # every reading, saved to a CSV file at the end
        self.test_results = []         # per test dot: (target x, target y, pointer positions)
        self._countdown_left = 0
        self._sample_event = None
        self._preview_event = None
        self._countdown_event = None
        self._leaving = False          # True once we've decided to leave

        # Instructions at the top of the screen. Its height changes per
        # stage, so it never covers the preview or the dots.
        self.instruction_label = Label(
            font_size=22, size_hint=(1, 0.2), pos_hint={"top": 1, "x": 0},
            halign="center", valign="top",
        )
        self.instruction_label.bind(size=lambda w, s: setattr(w, "text_size", s))
        self.add_widget(self.instruction_label)

        # Live camera preview, with the green iris dots drawn by
        # gaze_engine, so the user can SEE their face is being tracked.
        # Only shown in the READY stage: during the dots it would pull the
        # eyes away from the dot and spoil the measurements.
        self.preview_image = Image(
            size_hint=(None, None), size=(640, 360),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )
        self.add_widget(self.preview_image)

        # "Face detected" / "Face not detected", just under the preview.
        self.face_label = Label(
            font_size=24, bold=True, size_hint=(1, 0.06),
            pos_hint={"center_x": 0.5, "y": 0.2},
        )
        self.add_widget(self.face_label)

        # The Start button. It works by touch/click (for the caregiver) and
        # by dwell (resting the pointer on it), like every other button.
        self.start_btn = DwellButton(
            text="Start Calibration", bold=True, font_size=28,
            bg_color=PURPLE, text_color=WHITE, radius=16,
            size_hint=(None, None), size=(380, 90),
            pos_hint={"center_x": 0.5, "y": 0.06},
            on_selected=lambda b: self._start_countdown(),
        )
        self.start_btn.bind(on_release=lambda b: self._start_countdown())

        with self.canvas:
            Color(0.9, 0.2, 0.2, 1)
            self.dot = Ellipse(pos=(-100, -100), size=(40, 40))  # off-screen until shown

    # ---- stage 1: READY ------------------------------------------------
    def on_pre_enter(self, *args):
        self.target_index = 0
        self.log_rows = []
        self.test_results = []
        self.gaze_engine.clear_calibration()
        self.stage = "ready"
        self._leaving = False
        self.dot.pos = (-100, -100)
        self._place_label("top", 22, 0.2)
        self.instruction_label.text = (
            "Get ready for calibration\n"
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
        self._preview_event = Clock.schedule_interval(self._update_preview, 1 / 15)

    def _update_preview(self, dt):
        """Runs 15 times a second during the READY stage: shows the newest
        camera picture and whether a face is detected."""
        if not self.gaze_engine.is_camera_ok():
            # No camera (or MediaPipe failed) -- don't leave the user stuck
            # on this screen; continue with mouse control instead.
            self._stop_preview()
            self.face_label.text = "No camera detected -- using mouse control instead."
            self.face_label.color = RED_TEXT
            self._leave_after(2.0)
            return

        frame = self.gaze_engine.get_debug_frame()
        if frame is None:
            return  # camera still starting up

        h, w = frame.shape[:2]
        flipped = frame[::-1]  # Kivy textures are bottom-up; camera frames are top-down
        texture = Texture.create(size=(w, h), colorfmt="rgb")
        texture.blit_buffer(flipped.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        self.preview_image.texture = texture
        self.preview_image.opacity = 1

        feature = self.gaze_engine.get_latest_feature()
        if feature is None:
            self.face_label.text = "Face not detected -- move into the camera view"
            self.face_label.color = RED_TEXT
        elif feature[1] is None:
            # Face found, but no IR glints: up/down would fall back to the mouse.
            self.face_label.text = "Face detected, but no IR glints (white dots) -- check the IR LEDs"
            self.face_label.color = ORANGE_TEXT
        else:
            self.face_label.text = "Face and IR glints detected -- press Start when you are ready"
            self.face_label.color = GREEN_TEXT

        # The camera works, so the user may start now.
        if self.start_btn.parent is None:
            self.add_widget(self.start_btn)

    def _stop_preview(self):
        if self._preview_event is not None:
            self._preview_event.cancel()
            self._preview_event = None

    def _place_label(self, where, font_size, height):
        """Moves the instruction text: "top" = a strip along the top edge,
        "center" = large in the middle of the screen (countdown, result)."""
        self.instruction_label.font_size = font_size
        self.instruction_label.size_hint_y = height
        if where == "top":
            self.instruction_label.pos_hint = {"x": 0, "top": 1}
            self.instruction_label.valign = "top"
        else:
            self.instruction_label.pos_hint = {"x": 0, "center_y": 0.5}
            self.instruction_label.valign = "middle"

    # ---- stage 2: COUNTDOWN ----------------------------------------------
    def _start_countdown(self):
        if self.stage != "ready":
            return  # already started (e.g. tapped and dwelled at the same time)
        self.stage = "countdown"
        self._stop_preview()
        self.remove_widget(self.start_btn)
        self.preview_image.opacity = 0   # hide the preview from now on
        self.face_label.text = ""
        self._place_label("center", 48, 0.3)
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
        self._countdown_event.cancel()
        self._countdown_event = None
        # Smaller text in a thin strip at the top, so it never covers a dot.
        self._place_label("top", 20, 0.1)
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
        self.dot.pos = (x - 20, y - 20)
        self.samples = []      # fx readings (left/right) at this dot
        self.samples_vy = []   # vy readings (up/down, glints), only when a glint is seen
        self.samples_iy = []   # iy readings (up/down, iris against the eye corners)
        self.samples_hx = []   # hx readings (left/right, glints), only when a glint is seen
        self.pointer_samples = []  # accuracy test: where the gaze pointer was
        self._target_start_time = time.time()
        self._sample_event = Clock.schedule_interval(self._collect_sample, 1 / 30)
        Clock.schedule_once(self._finish_current_target, CALIBRATION_CAPTURE_TIME)

    def _collect_sample(self, dt):
        feature = self.gaze_engine.get_latest_feature()
        elapsed = time.time() - self._target_start_time
        pointer = (self.gaze_engine.get_screen_x(), self.gaze_engine.get_screen_y())
        self._log(elapsed, feature, pointer)
        if self.stage == "test":
            if elapsed >= CALIBRATION_SETTLE_TIME:
                self.pointer_samples.append(pointer)
        # Ignore the first moment at each dot: the eyes are still moving there.
        elif feature is not None and elapsed >= CALIBRATION_SETTLE_TIME:
            fx, vy, iy, hx = feature
            self.samples.append(fx)
            self.samples_iy.append(iy)
            if vy is not None:
                self.samples_vy.append(vy)
            if hx is not None:
                self.samples_hx.append(hx)

        remaining = max(0.0, CALIBRATION_CAPTURE_TIME - elapsed)
        point_num = self.target_index + 1
        total_points = len(self._points())
        face_status = "face detected" if feature is not None else "FACE NOT DETECTED -- move into camera view"
        what = "Accuracy test: look" if self.stage == "test" else "Look"
        self.instruction_label.text = (
            f"{what} at the red dot  ({point_num} of {total_points})  --  "
            f"{remaining:.1f}s  --  [{face_status}]"
        )

    def _log(self, elapsed, feature, pointer):
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
                             + [rounded(v, 2) for v in (turn, tilt, head_x, head_y)])

    def _finish_current_target(self, dt):
        if self._sample_event is not None:
            self._sample_event.cancel()
            self._sample_event = None

        if self.stage == "test":
            self.test_results.append((*self.current_target_px, self.pointer_samples))
            self.target_index += 1
            if self.target_index < len(ACCURACY_TEST_POINTS):
                self._show_current_target()
            else:
                self._finish_test()
            return

        if self.samples:
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
            # deviation); the gaze engine uses it to choose its fits.
            shakes = tuple(float(np.std(values)) if len(values) >= 2 else None
                           for values in (self.samples, self.samples_vy, self.samples_iy, self.samples_hx))
            self.gaze_engine.add_calibration_point(avg_fx, target_x, avg_vy, target_y, avg_iy, avg_hx,
                                                   shakes=shakes)

        self.target_index += 1
        if self.target_index < len(CALIBRATION_POINTS):
            self._show_current_target()
        else:
            self._finish_calibration()

    def _finish_calibration(self):
        try:
            self.summary = "Calibration complete!\n" + self.gaze_engine.fit_calibration()
        except ValueError as e:
            # Not enough usable samples (e.g. face wasn't visible) --
            # fall back to mouse rather than leaving gaze half-broken.
            print(f"[Calibration] {e} -- falling back to mouse control.")
            self.summary = f"Calibration incomplete: {e}\nUsing mouse control instead."
            self._show_result()
            return
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
            xs = np.array([x for x, y in seen]) / SCREEN_PX_PER_CM   # in cm
            ys = np.array([y for x, y in seen]) / SCREEN_PX_PER_CM
            # Accuracy: how far the average pointer position is from the dot.
            dx = xs.mean() - target_x / SCREEN_PX_PER_CM
            dy = ys.mean() - target_y / SCREEN_PX_PER_CM
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
            self.summary += "\nAccuracy test: no gaze pointer (up/down or left/right uses the mouse)"
        else:
            distance = np.mean(np.hypot(offsets_x, offsets_y))
            self.summary += (
                f"\nAccuracy test: {distance:.1f} cm off on average "
                f"(left/right {np.mean(offsets_x):.1f} cm, "
                f"up/down {np.mean(offsets_y):.1f} cm), "
                f"shaking {np.mean(shakes):.1f} cm\n" + "\n".join(per_dot)
            )
            if missing:
                self.summary += f"\n({missing} readings had no gaze pointer)"
        self._show_result()

    def _show_result(self):
        self.dot.pos = (-100, -100)
        self.stage = "result"
        self._place_label("center", 24, 0.8)
        fps = self.gaze_engine.get_fps()
        if fps is not None:
            self.summary += f"\nCamera: {fps:.0f} pictures a second"
        saved = self._save_log()
        self.instruction_label.text = self.summary + (f"\nReadings saved to {saved}" if saved else "")
        self._leave_after(RESULT_SECONDS)  # long enough to read (or photograph) the result

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
                                 "head_turn_deg", "head_tilt_deg", "head_x", "head_y"])
                writer.writerows(self.log_rows)
            return f"logs/{name}"
        except OSError as e:
            print(f"[Calibration] Could not save the readings: {e}")
            return ""

    # ---- leaving -----------------------------------------------------------
    def _leave_after(self, seconds):
        if self._leaving:
            return  # already on the way out
        self._leaving = True
        Clock.schedule_once(lambda dt: self._done(), seconds)

    def on_leave(self, *args):
        self._stop_preview()
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
        self.gaze_engine = GazeEngine()
        self.gaze_engine.start()

        sm = ScreenManager()
        self.navigator = Navigator(sm)
        sm.add_widget(KeyboardScreen(self.navigator, name="keyboard"))
        sm.add_widget(QuickFireScreen(self.navigator, name="quickfire"))
        sm.add_widget(
            CalibrationScreen(
                gaze_engine=self.gaze_engine,
                on_done=lambda: setattr(sm, "current", "keyboard"),
                name="calibration",
            )
        )
        sm.current = "calibration"

        # Drives dwell selection -- real gaze (X) once calibrated, mouse
        # for everything else, with automatic fallback to full mouse
        # control if calibration didn't complete.
        self.gaze_manager = GazeManager(sm, gaze_engine=self.gaze_engine)
        return sm

    def on_start(self):
        # Open maximized, filling the screen but keeping the title bar, so
        # the window can still be closed or moved normally.
        Window.maximize()
        if SHOW_GAZE_POINTER:
            # Added to the window after the screens, so it is drawn on top.
            self.gaze_manager.pointer = GazePointer()
            Window.add_widget(self.gaze_manager.pointer)

    def on_stop(self):
        self.gaze_engine.stop()


if __name__ == "__main__":
    EyeGlyntApp().run()