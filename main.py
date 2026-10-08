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

import os
import time

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

from gaze_engine import GazeEngine
from icons import draw_icon


def rgb(r, g, b, a=1.0):
    """Converts 0-255 colour values (as read from the design) to Kivy's 0-1 range."""
    return (r / 255, g / 255, b / 255, a)


# Colours sampled directly from the pixels of Figure 29.
FRAME_BG = rgb(58, 59, 64)       # device frame and screen background
PANEL_BG = rgb(70, 71, 80)       # keyboard and word panels
HEADER_BG = rgb(128, 128, 128)   # top title bar and bottom footer
PURPLE = rgb(106, 31, 173)       # KEYBOARD tab, Speak, Clear, Backspace
ORANGE = rgb(226, 112, 42)       # QUICK FIRES tab
KEY_FACE = rgb(99, 98, 98)       # letter keys
KEY_BORDER = rgb(154, 154, 154)  # thin outline around keys and panels
PINK = rgb(226, 169, 241)        # Quick Access words
GREEN = rgb(191, 236, 172)       # Common Words
YELLOW = rgb(255, 235, 153)      # Phrases; Basic Needs tiles
TILE_RED = rgb(250, 155, 155)    # Emergency tiles (Figures 30-31)
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

DWELL_TIME = 1.5  # seconds -- matches the 1.5s example in your flowchart section
REPEAT_TIME = DWELL_TIME  # held keys repeat at the SAME rate as the initial dwell,
                          # so there's only one dwell-time value to document/defend.
                          # Set this to a different (smaller) number later if you
                          # decide, with testing data, that repeats should be faster.

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
    """

    def __init__(self, screen_manager: ScreenManager, gaze_engine: GazeEngine = None):
        self.sm = screen_manager
        self.gaze_engine = gaze_engine
        self._current_target = None
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
        return (x, y)

    def _update(self, dt):
        active_screen = self.sm.current_screen
        if active_screen is None:
            return
        pos = self.get_pointer_pos()
        target = None
        for btn in _find_dwell_buttons(active_screen):
            if btn.disabled or not btn.selectable:
                continue  # greyed-out or display-only buttons can't be selected
            local = btn.to_widget(*pos)
            if btn.collide_point(*local):
                target = btn
                break
        if target is not self._current_target:
            if self._current_target is not None:
                self._current_target.cancel_dwell()
            self._current_target = target
            if target is not None:
                target.start_dwell()


# ---------------------------------------------------------------------------
# 3b. NAVIGATION -- Home and Back buttons
# ---------------------------------------------------------------------------
class Navigator:
    """Switches screens and remembers the route taken, so Back can return.

    Calibration is never recorded: the app leaves it once, at start-up,
    and Back should not send the user through calibration again.
    """

    def __init__(self, screen_manager, home="keyboard", max_history=20):
        self.sm = screen_manager
        self.home = home
        self.max_history = max_history
        self.history = []

    def go(self, name):
        if name == self.sm.current:
            return
        self.history.append(self.sm.current)
        self.history = self.history[-self.max_history:]  # keep the list short
        self.sm.current = name

    def back(self):
        if self.history:
            self.sm.current = self.history.pop()

    def go_home(self):
        self.go(self.home)

    def can_go_back(self):
        return bool(self.history)


# ---------------------------------------------------------------------------
# 4. KEYBOARD INTERFACE (Figure 29)
# ---------------------------------------------------------------------------
# Word sets shown in the design. Preliminary: to be confirmed by the
# caregiver and therapist interviews (Data Gathering Procedure).
QUICK_ACCESS_WORDS = ["Yes", "No", "Please", "Thank you"]
COMMON_WORDS = ["my", "want", "like", "go", "you", "it", "more", "stop"]
PHRASES = ["I need help", "I'm hungry", "I'm tired", "I love you"]

# The 26 character keys. The 123 key swaps letters for numbers and
# punctuation on the SAME buttons, so both layouts must have 10, 9 and 7
# keys per row.
LETTER_ROWS = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"]
SYMBOL_ROWS = ["1234567890", "-/:;()&@\"", ".,?!'#%"]

SENTENCE_END = (".", "?", "!")
ATTACHING_PUNCTUATION = ".,?!"  # punctuation that belongs right after a word

# Shared look of the dark keyboard keys.
KEY_STYLE = dict(bg_color=KEY_FACE, border_color=KEY_BORDER, text_color=WHITE, radius=8)


class AppScreen(Screen):
    """The frame shared by the Keyboard and Quick-Fire screens (Figures 29-31):
    a title bar with Home and Back, the KEYBOARD and QUICK FIRES tabs on
    the left, the screen's own content on the right, and a footer.

    Subclasses set TITLE and TAB, build their content, then call
    build_frame(content). Keeping this in one place means both screens
    always look and behave the same.
    """

    TITLE = ""
    TAB = ""  # "keyboard" or "quickfire": which side tab is this screen's own

    def __init__(self, navigator, **kwargs):
        super().__init__(**kwargs)
        self.nav = navigator

    def build_frame(self, content):
        frame = Panel(bg_color=FRAME_BG, radius=0, orientation="vertical")
        frame.add_widget(self._build_header())
        body = BoxLayout(orientation="horizontal", padding=[10, 10, 10, 6], spacing=10)
        body.add_widget(self._build_sidebar())
        content.size_hint_x = 0.83
        body.add_widget(content)
        frame.add_widget(body)
        frame.add_widget(self._build_footer())
        self.add_widget(frame)

    def _build_header(self):
        header = Panel(bg_color=HEADER_BG, radius=0, size_hint_y=0.08,
                       padding=[10, 6], spacing=8)
        self.home_btn = DwellButton(icon="home", icon_layout="center", bg_color=FRAME_BG,
                                    radius=22, size_hint_x=None, width=80,
                                    on_selected=lambda b: self.nav.go_home())
        self.back_btn = DwellButton(icon="arrow_left", icon_layout="center", bg_color=FRAME_BG,
                                    radius=22, size_hint_x=None, width=80,
                                    on_selected=lambda b: self.nav.back())
        header.add_widget(self.home_btn)
        header.add_widget(self.back_btn)
        header.add_widget(Label(text=self.TITLE, bold=True, font_size=30, color=WHITE))
        # An empty block as wide as the two buttons, so the title stays centred.
        header.add_widget(Widget(size_hint_x=None, width=168))
        return header

    def _build_sidebar(self):
        side = BoxLayout(orientation="vertical", size_hint_x=0.17, spacing=10)
        on_keyboard = self.TAB == "keyboard"
        # The tab of the screen on display gets a white border and is not
        # selectable; the other tab switches screens.
        side.add_widget(DwellButton(text="KEYBOARD", bold=True, font_size=20,
                                    icon="keyboard", icon_layout="top",
                                    bg_color=PURPLE, text_color=WHITE, radius=12,
                                    border_color=WHITE if on_keyboard else None,
                                    selectable=not on_keyboard, size_hint_y=0.27,
                                    on_selected=lambda b: self.nav.go("keyboard")))
        side.add_widget(DwellButton(text="QUICK FIRES", bold=True, font_size=20,
                                    icon="flame", icon_layout="top",
                                    bg_color=ORANGE, text_color=WHITE, radius=12,
                                    border_color=None if on_keyboard else WHITE,
                                    selectable=on_keyboard, size_hint_y=0.27,
                                    on_selected=lambda b: self.nav.go("quickfire")))
        side.add_widget(Widget(size_hint_y=0.46))  # empty space below the tabs
        return side

    def _build_footer(self):
        footer = Panel(bg_color=HEADER_BG, radius=0, size_hint_y=0.04)
        footer.add_widget(Label(text="EyeGlynt", font_size=14, color=WHITE))
        return footer

    def on_pre_enter(self, *args):
        # Back is greyed out (and ignored by gaze) when there is nowhere to go back to.
        self.back_btn.disabled = not self.nav.can_go_back()


class KeyboardScreen(AppScreen):
    """Figure 29: compose a message key by key or word by word, then Speak.

    Layout, top to bottom: title bar with Home and Back; a body with the
    mode tabs on the left and, on the right, the message box with Speak,
    Clear and Backspace, the letter keys, and three word panels; then a
    footer.
    """

    TITLE = "KEYBOARD"
    TAB = "keyboard"

    def __init__(self, navigator, **kwargs):
        super().__init__(navigator, **kwargs)
        self.composed_text = ""
        self.shift_on = True      # a message starts with a capital letter
        self.symbols_on = False   # False: letters shown; True: numbers and punctuation
        self.char_keys = []       # the 26 character buttons, in row order

        main = BoxLayout(orientation="vertical", spacing=8)
        main.add_widget(self._build_compose_row())
        main.add_widget(self._build_keys_panel())
        main.add_widget(self._build_words_panel())
        self.build_frame(main)

        self._refresh_keys()
        self._refresh_text()

    # -- building the layout -------------------------------------------------
    def _build_compose_row(self):
        row = BoxLayout(orientation="horizontal", size_hint_y=0.22, spacing=8)

        box = Panel(bg_color=WHITE, radius=12, padding=[14, 10], size_hint_x=0.74)
        # The message sits in a ScrollView: when it grows taller than the box,
        # the view scrolls to the end so the newest text is always visible.
        self.text_scroll = ScrollView(do_scroll_x=False, bar_width=4)
        self.text_label = Label(font_size=28, halign="left", valign="top", size_hint_y=None)
        # Wrap at the box width; let the height grow with the text.
        self.text_label.bind(width=lambda w, v: setattr(w, "text_size", (v, None)))
        self.text_label.bind(texture_size=self._fit_text_height)
        self.text_scroll.add_widget(self.text_label)
        box.add_widget(self.text_scroll)
        row.add_widget(box)

        action = dict(bg_color=PURPLE, text_color=WHITE, radius=10, bold=True, font_size=16)
        actions = GridLayout(cols=2, spacing=8, size_hint_x=0.26)
        actions.add_widget(DwellButton(text="Speak", icon="speak", icon_layout="top",
                                       on_selected=lambda b: self._speak_text(), **action))
        actions.add_widget(DwellButton(text="Clear", icon="erase", icon_layout="top",
                                       on_selected=lambda b: self._clear(), **action))
        actions.add_widget(DwellButton(text="Backspace", icon="arrow_left", icon_layout="left",
                                       repeatable=True,
                                       on_selected=lambda b: self._backspace(),
                                       **{**action, "font_size": 14}))
        actions.add_widget(Widget())  # the design leaves this cell empty
        row.add_widget(actions)
        return row

    def _make_char_key(self):
        """One character key. Its character is stored on the button
        (btn.key_value) and changes when 123 switches layouts."""
        btn = DwellButton(font_size=28, repeatable=True,
                          on_selected=self._on_char_key, **KEY_STYLE)
        btn.key_value = ""
        self.char_keys.append(btn)
        return btn

    def _build_keys_panel(self):
        panel = Panel(bg_color=PANEL_BG, border_color=KEY_BORDER, orientation="vertical",
                      size_hint_y=0.42, padding=8, spacing=6)

        row1 = BoxLayout(spacing=6)
        for _ in range(10):
            row1.add_widget(self._make_char_key())

        row2 = BoxLayout(spacing=6)
        row2.add_widget(Widget(size_hint_x=0.5))  # half-key indent, as on a real keyboard
        for _ in range(9):
            row2.add_widget(self._make_char_key())
        row2.add_widget(Widget(size_hint_x=0.5))

        row3 = BoxLayout(spacing=6)
        self.shift_btn = DwellButton(icon="shift", icon_layout="center", size_hint_x=1.5,
                                     on_selected=lambda b: self._toggle_shift(), **KEY_STYLE)
        row3.add_widget(self.shift_btn)
        for _ in range(7):
            row3.add_widget(self._make_char_key())
        row3.add_widget(DwellButton(icon="erase", icon_layout="center", size_hint_x=1.5,
                                    repeatable=True, on_selected=lambda b: self._backspace(),
                                    **KEY_STYLE))

        row4 = BoxLayout(spacing=6)
        self.mode_btn = DwellButton(text="123", font_size=24, size_hint_x=1.5,
                                    on_selected=lambda b: self._toggle_symbols(), **KEY_STYLE)
        row4.add_widget(self.mode_btn)
        row4.add_widget(DwellButton(text="Space", font_size=24, size_hint_x=7, repeatable=True,
                                    on_selected=lambda b: self._insert_text(" "), **KEY_STYLE))
        row4.add_widget(DwellButton(text="Enter", font_size=24, size_hint_x=1.5,
                                    on_selected=lambda b: self._insert_text("\n"), **KEY_STYLE))

        for row in (row1, row2, row3, row4):
            panel.add_widget(row)
        return panel

    def _word_group(self, title, words, color, cols, width_share):
        group = BoxLayout(orientation="vertical", spacing=4, size_hint_x=width_share)
        group.add_widget(Label(text=title, bold=True, font_size=18, color=WHITE, size_hint_y=0.2))
        grid = GridLayout(cols=cols, spacing=6)
        for word in words:
            btn = DwellButton(text=word, font_size=20, bg_color=color, text_color=DARK_TEXT,
                              radius=8, on_selected=self._on_word)
            btn.key_value = word
            grid.add_widget(btn)
        group.add_widget(grid)
        return group

    def _build_words_panel(self):
        panel = Panel(bg_color=FRAME_BG, border_color=KEY_BORDER, orientation="horizontal",
                      size_hint_y=0.36, padding=8, spacing=14)
        panel.add_widget(self._word_group("Quick Access", QUICK_ACCESS_WORDS, PINK, 2, 0.24))
        panel.add_widget(self._word_group("Common Words", COMMON_WORDS, GREEN, 4, 0.45))
        panel.add_widget(self._word_group("Phrases", PHRASES, YELLOW, 2, 0.31))
        return panel

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
        if self.composed_text:
            self.text_label.text = self.composed_text
            self.text_label.color = DARK_TEXT
        else:
            self.text_label.text = "Type, here..."
            self.text_label.color = PLACEHOLDER
        # Scroll once Kivy has re-measured the text, on the next frame.
        Clock.schedule_once(self._scroll_to_end, 0)

    def _refresh_keys(self):
        rows = SYMBOL_ROWS if self.symbols_on else LETTER_ROWS
        for btn, ch in zip(self.char_keys, "".join(rows)):
            btn.key_value = ch
            btn.text = ch
        self.mode_btn.text = "ABC" if self.symbols_on else "123"
        self.shift_btn.disabled = self.symbols_on  # Shift has no meaning for symbols
        highlight = self.shift_on and not self.symbols_on
        self.shift_btn.set_bg_color(PURPLE if highlight else KEY_FACE)

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
    def _on_char_key(self, btn):
        ch = btn.key_value
        if ch.isalpha():
            ch = ch.upper() if self.shift_on else ch.lower()
        if ch in ATTACHING_PUNCTUATION and self.composed_text.endswith(" "):
            # A word button leaves a trailing space. Pull the punctuation back
            # onto that word ("more ." becomes "more. ") and keep a space after
            # it, so the user doesn't spend a Backspace and a Space on it.
            self.composed_text = self.composed_text[:-1] + ch + " "
            self._after_edit()
            return
        self._insert_text(ch)

    def _on_word(self, btn):
        """Inserts a whole word or phrase, adding spaces around it as needed."""
        word = btn.key_value
        if self.shift_on:
            word = word[:1].upper() + word[1:]
        if self.composed_text and not self.composed_text.endswith((" ", "\n")):
            self.composed_text += " "
        self.composed_text += word + " "
        self._after_edit()

    def _insert_text(self, text):
        self.composed_text += text
        self._after_edit()

    def _backspace(self):
        self.composed_text = self.composed_text[:-1]
        self._after_edit()

    def _clear(self):
        self.composed_text = ""
        self._after_edit()

    def _toggle_shift(self):
        self.shift_on = not self.shift_on
        self._refresh_keys()

    def _toggle_symbols(self):
        self.symbols_on = not self.symbols_on
        self._refresh_keys()

    def _speak_text(self):
        if self.composed_text.strip():
            speak(self.composed_text)


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
QUICKFIRE_COLUMNS = 4  # 8 phrases per category -> a 4 x 2 grid of large tiles


class QuickFireScreen(AppScreen):
    """Figures 30-31, organised into the manuscript's six category tabs.

    Selecting a category tab (by dwell, like everything else) shows that
    category's phrase tiles. Selecting a tile speaks its phrase at once,
    with no confirmation step, as the manuscript specifies for urgent
    messages.
    """

    TITLE = "QUICK FIRES"
    TAB = "quickfire"

    def __init__(self, navigator, **kwargs):
        super().__init__(navigator, **kwargs)
        self.category_names = list(QUICKFIRE_CATEGORIES)
        self.category_grids = {}  # category -> its tile grid, built the first time it is opened
        self.tab_buttons = {}     # category -> its tab button
        self.current_category = None

        content = BoxLayout(orientation="vertical", spacing=8)
        content.add_widget(self._build_tab_row())
        self.tiles_panel = Panel(bg_color=PANEL_BG, border_color=KEY_BORDER,
                                 size_hint_y=0.88, padding=10)
        content.add_widget(self.tiles_panel)
        self.build_frame(content)

        self._switch_category(self.category_names[0])

    def _build_tab_row(self):
        row = BoxLayout(orientation="horizontal", size_hint_y=0.12, spacing=8)
        for name in self.category_names:
            btn = DwellButton(text=name, bold=True, font_size=17,
                              on_selected=self._on_tab, **{**KEY_STYLE, "radius": 10})
            btn.key_value = name
            self.tab_buttons[name] = btn
            row.add_widget(btn)
        return row

    def _build_tile_grid(self, name):
        color, phrases = QUICKFIRE_CATEGORIES[name]
        grid = GridLayout(cols=QUICKFIRE_COLUMNS, spacing=10)
        for phrase, picture in phrases:
            tile = DwellButton(text=phrase, font_size=20, bg_color=color, text_color=DARK_TEXT,
                               radius=12, image=picture, icon_layout="top",
                               on_selected=self._on_tile)
            tile.key_value = phrase
            grid.add_widget(tile)
        return grid

    def _on_tab(self, btn):
        self._switch_category(btn.key_value)

    def _on_tile(self, btn):
        speak(btn.key_value)

    def _switch_category(self, name):
        if name not in self.category_grids:
            self.category_grids[name] = self._build_tile_grid(name)
        self.tiles_panel.clear_widgets()
        self.tiles_panel.add_widget(self.category_grids[name])
        self.current_category = name

        # The active tab takes its category's colour with a white border and,
        # like the side tabs, is not selectable; the others stay dark.
        for tab_name, tab in self.tab_buttons.items():
            active = tab_name == name
            tab.set_bg_color(QUICKFIRE_CATEGORIES[tab_name][0] if active else KEY_FACE)
            tab.border_color = WHITE if active else KEY_BORDER
            tab.color = DARK_TEXT if active else WHITE
            tab.selectable = not active
            tab._redraw()


# ---------------------------------------------------------------------------
# 6. CALIBRATION SCREEN -- get ready, then a 9-point calibration
# ---------------------------------------------------------------------------
# The screen has three stages:
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
# Only the HORIZONTAL position is fitted for now (see gaze_engine.py), but
# 9 points still help: each of the 3 columns is measured 3 times, at
# different heights, so one bad reading has much less effect on the fit.
# The same 9 points will be reused for up/down once PCCR is added.

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
CALIBRATION_CAPTURE_TIME = 2.0  # seconds the dot stays at each point
CALIBRATION_SETTLE_TIME = 0.5   # first part of those seconds is ignored: the
                                # eyes are still moving onto the new dot
COUNTDOWN_SECONDS = 3           # 3, 2, 1 after Start is pressed

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
        self.stage = "ready"           # "ready", "countdown" or "dots"
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
        self.gaze_engine.clear_calibration()
        self.stage = "ready"
        self._leaving = False
        self.dot.pos = (-100, -100)
        self._place_label("top", 22, 0.2)
        self.instruction_label.text = (
            "Get ready for calibration\n"
            "Sit about 50 cm from the screen, with your whole face in the camera view.\n"
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

    # ---- stage 3: DOTS ---------------------------------------------------
    def _show_current_target(self):
        # The dot's position is worked out from the screen's size NOW, not
        # when the app started: the window is still growing to full size
        # (maximizing) at start-up, and using that early, smaller size put
        # every dot in the left part of the screen.
        fx, fy = CALIBRATION_POINTS[self.target_index]
        x, y = self.width * fx, self.height * fy
        self.current_target_px = (x, y)
        self.dot.pos = (x - 20, y - 20)
        self.samples = []      # fx readings (left/right) at this dot
        self.samples_vy = []   # vy readings (up/down), only when a glint is seen
        self._target_start_time = time.time()
        self._sample_event = Clock.schedule_interval(self._collect_sample, 1 / 30)
        Clock.schedule_once(self._finish_current_target, CALIBRATION_CAPTURE_TIME)

    def _collect_sample(self, dt):
        feature = self.gaze_engine.get_latest_feature()
        elapsed = time.time() - self._target_start_time
        # Ignore the first moment at each dot: the eyes are still moving there.
        if feature is not None and elapsed >= CALIBRATION_SETTLE_TIME:
            fx, vy = feature
            self.samples.append(fx)
            if vy is not None:
                self.samples_vy.append(vy)

        remaining = max(0.0, CALIBRATION_CAPTURE_TIME - elapsed)
        point_num = self.target_index + 1
        total_points = len(CALIBRATION_POINTS)
        face_status = "face detected" if feature is not None else "FACE NOT DETECTED -- move into camera view"
        self.instruction_label.text = (
            f"Look at the red dot  ({point_num} of {total_points})  --  "
            f"{remaining:.1f}s  --  [{face_status}]"
        )

    def _finish_current_target(self, dt):
        if self._sample_event is not None:
            self._sample_event.cancel()
            self._sample_event = None

        if self.samples:
            avg_fx = sum(self.samples) / len(self.samples)
            target_x, target_y = self.current_target_px
            # Up/down is only recorded if a glint was seen for at least
            # half of the readings at this dot; otherwise it's unreliable.
            avg_vy = None
            if len(self.samples_vy) >= len(self.samples) / 2:
                avg_vy = sum(self.samples_vy) / len(self.samples_vy)
            self.gaze_engine.add_calibration_point(avg_fx, target_x, avg_vy, target_y)

        self.target_index += 1
        if self.target_index < len(CALIBRATION_POINTS):
            self._show_current_target()
        else:
            self._finish_calibration()

    def _finish_calibration(self):
        self.dot.pos = (-100, -100)
        self._place_label("center", 36, 0.3)
        try:
            summary = self.gaze_engine.fit_calibration()
            self.instruction_label.text = f"Calibration complete!\n{summary}"
        except ValueError as e:
            # Not enough usable samples (e.g. face wasn't visible) --
            # fall back to mouse rather than leaving gaze half-broken.
            print(f"[Calibration] {e} -- falling back to mouse control.")
            self.instruction_label.text = "Calibration incomplete -- using mouse control instead."
        self._leave_after(3.0)  # long enough to read the result

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
        self.navigator = Navigator(sm, home="keyboard")
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

    def on_stop(self):
        self.gaze_engine.stop()


if __name__ == "__main__":
    EyeGlyntApp().run()