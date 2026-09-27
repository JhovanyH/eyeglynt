"""
EyeGlynt - Prototype On-Screen Interface
=========================================
This is a starter version of the Keyboard Interface and Quick-Fire
Interface described in Chapter III of your thesis (Figures 29-31).

It runs RIGHT NOW on your laptop (Windows/Mac/Linux) using your MOUSE
to simulate the eye-gaze pointer, so you can build and test the whole
interface before the camera + MediaPipe + PCCR gaze-tracking module is
ready. Once that module exists, you only need to change ONE method
(GazeManager.get_pointer_pos, near the bottom of section 3) to feed it
real gaze coordinates instead of the mouse position -- nothing else in
the UI needs to change.

SETUP
-----
    pip install -r requirements.txt
    python main.py

Controls: just move your mouse over a button and hold it there for
~1.5 seconds (DWELL_TIME below) -- it fills up like a progress bar
and then "selects", exactly like a gaze dwell-click will.
"""

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
from kivy.graphics import Color, Rectangle, Ellipse

from gaze_engine import GazeEngine

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
    dwell-time selection method from your Methodology (the fix for the
    Midas Touch problem) -- built once here so both interfaces reuse it.

    Set `repeatable=True` for buttons where holding the gaze in place
    should keep re-firing (typing the same letter twice, like "food") --
    after the first full `dwell_time` selection, it re-arms itself using
    the shorter `repeat_time`, and keeps repeating for as long as the
    pointer/gaze stays put, exactly like holding down a physical key.
    Leave it False (the default) for anything where an accidental repeat
    would be unsafe or annoying -- phrases, Speak, Clear, navigation.
    """

    progress = NumericProperty(0.0)  # 0.0 -> 1.0

    def __init__(self, on_selected=None, dwell_time=DWELL_TIME,
                 repeatable=False, repeat_time=REPEAT_TIME, **kwargs):
        super().__init__(**kwargs)
        self.on_selected = on_selected
        self.dwell_time = dwell_time
        self.repeatable = repeatable
        self.repeat_time = repeat_time
        self._active_dwell_time = dwell_time
        self._event = None

        with self.canvas.after:
            Color(0.2, 0.8, 0.4, 0.55)
            self._progress_rect = Rectangle(pos=self.pos, size=(0, self.height))
        self.bind(pos=self._redraw, size=self._redraw)

    def _redraw(self, *args):
        self._progress_rect.pos = self.pos
        self._progress_rect.size = (self.width * self.progress, self.height)

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
                # Re-arm immediately at the faster repeat_time. GazeManager
                # never touched this button (its target hasn't changed), so
                # this keeps firing on its own until the pointer/gaze
                # actually leaves -- at which point GazeManager calls
                # cancel_dwell() on it and this loop stops.
                self.start_dwell(dwell_time=self.repeat_time)
            return False  # stop this particular Clock event


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

    Pointer X comes from real gaze once `gaze_engine` is calibrated;
    pointer Y still comes from the mouse -- see the CURRENT LIMITATION
    note in gaze_engine.py for why (plain webcam iris tracking couldn't
    reliably measure vertical eye movement; real PCCR will replace this
    once the IR camera + LED hardware exists). Until calibration happens
    (or if no camera is available at all), this falls back to full
    mouse control, so the app is never left unusable.
    """

    def __init__(self, screen_manager: ScreenManager, gaze_engine: GazeEngine = None):
        self.sm = screen_manager
        self.gaze_engine = gaze_engine
        self._current_target = None
        Clock.schedule_interval(self._update, 1 / 30)

    def get_pointer_pos(self):
        mouse_x, mouse_y = Window.mouse_pos
        if self.gaze_engine is not None and self.gaze_engine.is_calibrated():
            gaze_x = self.gaze_engine.get_screen_x()
            if gaze_x is not None:
                return (gaze_x, mouse_y)  # real gaze for X, mouse for Y (for now)
        return (mouse_x, mouse_y)

    def _update(self, dt):
        active_screen = self.sm.current_screen
        if active_screen is None:
            return
        pos = self.get_pointer_pos()
        target = None
        for btn in _find_dwell_buttons(active_screen):
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
# 4. KEYBOARD INTERFACE (Figure 29)
# ---------------------------------------------------------------------------
SHORTHAND_WORDS = ["Yes", "No", "Please", "Thank you", "Help", "Water", "Bathroom", "Hurts"]

KEY_ROWS = [
    list("1234567890"),
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
]


class KeyboardScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.composed_text = ""

        root = BoxLayout(orientation="vertical", padding=10, spacing=10)

        # -- composed text display --
        self.text_label = Label(
            text="", font_size=32, size_hint=(1, 0.15),
            halign="left", valign="middle",
        )
        self.text_label.bind(size=lambda w, s: setattr(w, "text_size", s))
        root.add_widget(self.text_label)

        # -- shorthand / pre-loaded words row --
        shorthand_row = BoxLayout(size_hint=(1, 0.12), spacing=6)
        for word in SHORTHAND_WORDS:
            shorthand_row.add_widget(
                DwellButton(text=word, on_selected=self._make_word_handler(word))
            )
        root.add_widget(shorthand_row)

        # -- alphanumeric keyboard grid (letters/numbers are repeatable, so
        #    holding your gaze on one key types it more than once -- e.g.
        #    "food") --
        keyboard_grid = BoxLayout(orientation="vertical", size_hint=(1, 0.55), spacing=6)
        for row in KEY_ROWS:
            row_layout = BoxLayout(spacing=6)
            for ch in row:
                row_layout.add_widget(
                    DwellButton(text=ch, on_selected=self._make_char_handler(ch), repeatable=True)
                )
            keyboard_grid.add_widget(row_layout)
        root.add_widget(keyboard_grid)

        # -- controls: space / backspace / clear / speak / switch mode --
        # SPACE and BACKSPACE are repeatable (multiple spaces, or holding
        # gaze to delete several characters). CLEAR, SPEAK, and switching
        # screens are deliberately NOT repeatable -- you don't want "Speak"
        # firing the whole message twice, or accidentally bouncing between
        # screens, just because you looked a moment too long.
        controls = BoxLayout(size_hint=(1, 0.18), spacing=6)
        controls.add_widget(DwellButton(text="SPACE", on_selected=self._make_char_handler(" "), repeatable=True))
        controls.add_widget(DwellButton(text="BACKSPACE", on_selected=lambda b: self._backspace(), repeatable=True))
        controls.add_widget(DwellButton(text="CLEAR", on_selected=lambda b: self._clear()))
        controls.add_widget(DwellButton(text="SPEAK", on_selected=lambda b: self._speak_text()))
        controls.add_widget(DwellButton(text="Quick-Fire ->", on_selected=lambda b: self._go_to_quickfire()))
        root.add_widget(controls)

        self.add_widget(root)

    def _make_char_handler(self, ch):
        def handler(btn):
            self.composed_text += ch
            self.text_label.text = self.composed_text
        return handler

    def _make_word_handler(self, word):
        def handler(btn):
            self.composed_text += word + " "
            self.text_label.text = self.composed_text
        return handler

    def _backspace(self):
        self.composed_text = self.composed_text[:-1]
        self.text_label.text = self.composed_text

    def _clear(self):
        self.composed_text = ""
        self.text_label.text = self.composed_text

    def _speak_text(self):
        if self.composed_text.strip():
            speak(self.composed_text)

    def _go_to_quickfire(self):
        self.manager.current = "quickfire"


# ---------------------------------------------------------------------------
# 5. QUICK-FIRE INTERFACE (Figures 30-31)
# ---------------------------------------------------------------------------
# Preliminary phrase set -- swap these once caregiver/therapist interviews
# (Data Gathering Procedure) confirm the final list.
QUICKFIRE_CATEGORIES = {
    "Emergency": ["Call the nurse", "I need help now", "Emergency!", "Call my family"],
    "Pain": ["I am in pain", "My head hurts", "My stomach hurts", "It hurts here"],
    "Basic Needs": ["I am hungry", "I am thirsty", "I need the bathroom", "I am tired"],
    "Social/Emotional": ["I am happy", "I am sad", "I am scared", "I am bored"],
    "People/Family": ["I want to see my family", "Call my caregiver", "Who is here?"],
    "Environment": ["Too hot", "Too cold", "Turn off the light", "Turn on the TV"],
}


class QuickFireScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category_names = list(QUICKFIRE_CATEGORIES.keys())
        self.category_widgets = {}  # category name -> its ScrollView, built once
        self.tab_buttons = {}       # category name -> its DwellButton
        self.current_category = None

        root = BoxLayout(orientation="vertical", padding=10, spacing=10)

        # -- category tab row: plain DwellButtons, exactly like every other
        #    button in the app, so switching categories also requires a
        #    dwell instead of an instant click --
        self.tab_row = BoxLayout(size_hint=(1, 0.12), spacing=6)
        for category in self.category_names:
            btn = DwellButton(text=category, on_selected=self._make_tab_handler(category))
            self.tab_buttons[category] = btn
            self.tab_row.add_widget(btn)
        root.add_widget(self.tab_row)

        # -- content area: we swap what's inside this depending on which
        #    category tab was last dwell-selected --
        self.content_area = BoxLayout(size_hint=(1, 0.73))
        root.add_widget(self.content_area)

        root.add_widget(
            DwellButton(
                text="<- Keyboard", size_hint=(1, 0.15),
                on_selected=lambda b: self._go_to_keyboard(),
            )
        )
        self.add_widget(root)

        # show the first category by default
        self._switch_category(self.category_names[0])

    def _build_category_view(self, category):
        """Builds the scrollable phrase grid for one category (only once,
        the first time that category is opened)."""
        grid = GridLayout(cols=2, spacing=8, padding=8, size_hint_y=None)
        grid.bind(minimum_height=grid.setter("height"))
        for phrase in QUICKFIRE_CATEGORIES[category]:
            grid.add_widget(
                DwellButton(
                    text=phrase, size_hint_y=None, height=110,
                    on_selected=self._make_phrase_handler(phrase),
                )
            )
        scroll = ScrollView(size_hint=(1, 1))
        scroll.add_widget(grid)
        return scroll

    def _make_tab_handler(self, category):
        def handler(btn):
            self._switch_category(category)
        return handler

    def _switch_category(self, category):
        if category not in self.category_widgets:
            self.category_widgets[category] = self._build_category_view(category)

        self.content_area.clear_widgets()
        self.content_area.add_widget(self.category_widgets[category])
        self.current_category = category

        # highlight the active tab so it's obvious which category you're in
        for name, btn in self.tab_buttons.items():
            btn.background_color = (0.35, 0.55, 0.85, 1) if name == category else (1, 1, 1, 1)

    def _make_phrase_handler(self, phrase):
        def handler(btn):
            speak(phrase)
        return handler

    def _go_to_keyboard(self):
        self.manager.current = "keyboard"


# ---------------------------------------------------------------------------
# 6. CALIBRATION SCREEN -- horizontal-only 3-point calibration
# ---------------------------------------------------------------------------
# Only 3 points (left, center, right), not the full 5-point routine from
# Figure 38 -- since vertical tracking isn't usable yet (see gaze_engine.py),
# calibrating a vertical dimension that doesn't work would be misleading.
# Swap this for the real 5-point PCCR calibration once the IR hardware and
# true corneal-reflection tracking are in place.
CALIBRATION_CAPTURE_TIME = 2.0  # seconds spent looking at each dot


class CalibrationScreen(Screen):
    def __init__(self, gaze_engine: GazeEngine, on_done, **kwargs):
        super().__init__(**kwargs)
        self.gaze_engine = gaze_engine
        self.on_done = on_done
        self.targets_px = []
        self.target_index = 0
        self.samples = []
        self._sample_event = None
        self._preview_event = None

        # Instructions confined to the TOP strip of the screen only, so
        # they never overlap the dot (which sits at vertical center).
        self.instruction_label = Label(
            text="Calibration -- look at the red dot and keep still",
            font_size=22, size_hint=(1, 0.25), pos_hint={"top": 1, "x": 0},
            halign="center", valign="top",
        )
        self.instruction_label.bind(size=lambda w, s: setattr(w, "text_size", s))
        self.add_widget(self.instruction_label)

        # Live camera preview, with the same green iris dots you saw in
        # gaze_test.py -- so you can SEE it tracking your eyes, instead of
        # only trusting a text status message.
        self.preview_image = Image(
            size_hint=(None, None), size=(240, 180),
            pos_hint={"right": 0.98, "top": 0.98},
        )
        self.add_widget(self.preview_image)

        with self.canvas:
            Color(0.9, 0.2, 0.2, 1)
            self.dot = Ellipse(pos=(-100, -100), size=(40, 40))  # off-screen until shown

    def _update_preview(self, dt):
        frame = self.gaze_engine.get_debug_frame()
        if frame is None:
            return
        h, w = frame.shape[:2]
        flipped = frame[::-1]  # Kivy textures are bottom-up; camera frames are top-down
        texture = Texture.create(size=(w, h), colorfmt="rgb")
        texture.blit_buffer(flipped.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        self.preview_image.texture = texture

    def on_pre_enter(self, *args):
        w, h = Window.size
        mid_y = h * 0.5
        # left / center / right, matching gaze_engine's horizontal-only calibration
        self.targets_px = [(w * 0.08, mid_y), (w * 0.5, mid_y), (w * 0.92, mid_y)]
        self.target_index = 0
        self.gaze_engine.clear_calibration()
        self.instruction_label.text = "Starting camera..."
        self._preview_event = Clock.schedule_interval(self._update_preview, 1 / 15)

        # The camera/MediaPipe setup happens in a background thread and
        # takes a moment to either succeed or fail -- wait briefly rather
        # than checking is_camera_ok() immediately, which could catch it
        # mid-startup and wrongly assume it failed.
        Clock.schedule_once(self._check_camera_and_start, 1.0)

    def on_leave(self, *args):
        if self._preview_event is not None:
            self._preview_event.cancel()
            self._preview_event = None

    def _check_camera_and_start(self, dt):
        if not self.gaze_engine.is_camera_ok():
            # No camera available -- skip calibration entirely rather than
            # get the user stuck staring at dots that can't be measured.
            self.instruction_label.text = "No camera detected -- using mouse control instead."
            Clock.schedule_once(lambda dt: self.on_done(), 1.5)
            return
        self._show_current_target()

    def _show_current_target(self):
        x, y = self.targets_px[self.target_index]
        self.dot.pos = (x - 20, y - 20)
        self.samples = []
        self._target_start_time = time.time()
        self._sample_event = Clock.schedule_interval(self._collect_sample, 1 / 30)
        Clock.schedule_once(self._finish_current_target, CALIBRATION_CAPTURE_TIME)

    def _collect_sample(self, dt):
        feature = self.gaze_engine.get_latest_feature()
        if feature is not None:
            self.samples.append(feature[0])  # only fx matters right now

        elapsed = time.time() - self._target_start_time
        remaining = max(0.0, CALIBRATION_CAPTURE_TIME - elapsed)
        point_num = self.target_index + 1
        total_points = len(self.targets_px)
        face_status = "face detected" if feature is not None else "face NOT detected -- move into camera view"
        self.instruction_label.text = (
            f"LOOK AT THE RED DOT  ({point_num} of {total_points})\n"
            f"Hold still -- {remaining:.1f}s left\n"
            f"[{face_status}]"
        )

    def _finish_current_target(self, dt):
        if self._sample_event is not None:
            self._sample_event.cancel()
            self._sample_event = None

        if self.samples:
            avg_fx = sum(self.samples) / len(self.samples)
            target_x, _ = self.targets_px[self.target_index]
            self.gaze_engine.add_calibration_point(avg_fx, target_x)

        self.target_index += 1
        if self.target_index < len(self.targets_px):
            self._show_current_target()
        else:
            self._finish_calibration()

    def _finish_calibration(self):
        self.dot.pos = (-100, -100)
        try:
            self.gaze_engine.fit_calibration()
            self.instruction_label.text = "Calibration complete!"
        except ValueError as e:
            # Not enough usable samples (e.g. face wasn't visible) --
            # fall back to mouse rather than leaving gaze half-broken.
            print(f"[Calibration] {e} -- falling back to mouse control.")
            self.instruction_label.text = "Calibration incomplete -- using mouse control instead."
        Clock.schedule_once(lambda dt: self.on_done(), 1.0)


# ---------------------------------------------------------------------------
# 7. APP ENTRY POINT
# ---------------------------------------------------------------------------
class EyeGlyntApp(App):
    def build(self):
        self.title = "EyeGlynt - Prototype UI"
        self.gaze_engine = GazeEngine()
        self.gaze_engine.start()

        sm = ScreenManager()
        sm.add_widget(KeyboardScreen(name="keyboard"))
        sm.add_widget(QuickFireScreen(name="quickfire"))
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

    def on_stop(self):
        self.gaze_engine.stop()


if __name__ == "__main__":
    EyeGlyntApp().run()