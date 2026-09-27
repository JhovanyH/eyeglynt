# EyeGlynt - Prototype UI

A starter, working version of the **Keyboard Interface** and **Quick-Fire
Interface** (thesis Figures 29-31), built with [Kivy](https://kivy.org).
It runs today on your laptop, driven by your mouse, and is built so the
mouse can be swapped for real gaze coordinates later without touching the
rest of the UI.

## 1. Run it on your laptop (do this first)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Move your mouse over any key or phrase button and hold still for about
1.5 seconds — it fills up green and "selects" itself, printing (and
speaking, if your system's TTS voices are installed) the result. That
fill-and-trigger behavior is exactly what a gaze dwell-click will do later.

If `pip install kivy` fails on Windows, see Kivy's official install guide:
https://kivy.org/doc/stable/gettingstarted/installation.html

## 2. What's actually in `main.py`

| Section | What it does |
|---|---|
| `DwellButton` | A button that fills up while something rests on it, then fires. This is your Midas-Touch fix from the Methodology chapter, built once and reused everywhere. |
| `GazeManager` | Every 1/30th of a second, asks "where is the pointer, and which button is it over?" and starts/stops dwell timers accordingly. |
| `KeyboardScreen` | Text box + shorthand word row + full QWERTY grid + Speak/Backspace/Clear, matching Figure 29. |
| `QuickFireScreen` | Tabbed categories (Emergency, Pain, Basic Needs, Social/Emotional, People/Family, Environment) each with phrase buttons, matching Figures 30-31. |
| `speak()` | Wraps `pyttsx3`. On the Pi you'll likely rely on `espeak-ng` under the hood — swap the internals here if needed; nothing else in the app needs to change. |

## 3. The one thing you'll change later: real gaze input

Right now, `GazeManager.get_pointer_pos()` returns `Window.mouse_pos`.
When your MediaPipe + PCCR gaze-estimation module is ready, it will be
running in its own loop/thread, continuously computing a `(x, y)` screen
coordinate from the camera feed. The cleanest way to connect the two:

1. Have your gaze-tracking thread write its latest `(x, y)` into a
   thread-safe variable (a `queue.Queue` with maxsize=1, or a small
   class with a lock) every frame.
2. Change `get_pointer_pos()` to read from that variable instead of
   `Window.mouse_pos`.

Everything else — the buttons, the dwell timing, the screens, the TTS
calls — stays exactly the same. That separation (tracking produces
coordinates; UI just consumes coordinates) is also what makes it easy to
keep testing the UI on your laptop with a mouse even after the gaze
engine exists on the Pi.

## 4. Moving to the Raspberry Pi

- Copy this folder to the Pi (via `git`, USB, or `scp`), or use VS Code's
  **Remote-SSH** extension to edit it directly on the Pi from your laptop.
- Your real screen is **1280x800** (10.1" capacitive) — already set as
  `SCREEN_W, SCREEN_H` at the top of `main.py`. When you actually run on
  the Pi's screen, you'll likely also want `Config.set('graphics',
  'fullscreen', 'auto')` before the window is created, so it fills the
  display without window borders.
- To auto-launch this app when the Pi boots (so a caregiver never has to
  open a terminal), you'll eventually set up either a `systemd` service
  or a `.desktop` autostart entry — happy to walk through that when you
  get there.

## 5. Not yet included (next layers to build)

- Real gaze tracking (OpenCV + MediaPipe + regression calibration).
- Writing selection/session events to MariaDB for the Caregiver Dashboard.
- Icons for Quick-Fire buttons (thesis references Mulberry Symbols,
  CC BY-SA licensed) — swap `DwellButton(text=...)` for an image+text
  layout once you have icon assets.
- Real recalibration flow (the 5-point calibration + regression model
  from Figure 38).
