"""
EyeGlynt <-> Caregiver Dashboard link
=====================================
Tells the caregiver dashboard (the "dashboard" folder: a website on this
Pi that caregivers open on their phone) what happens on the device, and
carries out the dashboard's buttons (Recalibrate, Stop calibration).

The device and the dashboard never talk to each other directly. Both use
the Pi's database (MariaDB), through dashboard/engine_bridge/eyeglynt_db.py.
So if the dashboard stops working, the user can still communicate.

All database work is done in the background (a separate thread), so a
slow or broken database can never freeze the screen or the gaze pointer.
If the dashboard isn't installed (no dashboard/.env file) or pymysql is
missing, the app works exactly as before and prints one line saying so.

Selections are only saved while a caregiver has a session running on the
dashboard. Outside a session the device works normally and saves nothing.

One-time setup on the Pi (after dashboard/setup_dashboard.sh), in the
Terminal:
    ~/eyeglynt_env/bin/pip install pymysql

How main.py uses it:
    dashboard.start()                      once, when the app starts
    dashboard.selection(...)               every time a button is selected
    dashboard.update(...)                  many times a second: device status
    dashboard.take_commands()              the dashboard's button presses
    dashboard.recalibration_started() / recalibration_finished(ok)
    dashboard.recalibration_cancelled()    Stop calibration was pressed
    dashboard.calibration_gave_up(tries)   calibration was poor `tries` times
                                           in a row; the app went on anyway
    dashboard.stop()                       when the app closes: the
                                           dashboard then ends the session
"""

import os
import queue
import sys
import threading
import time

# The dashboard folder, next to this file.
DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")

# How often the device tells the dashboard it is on (the "heartbeat").
# The dashboard shows the device as offline after 10 s without one.
HEARTBEAT_EVERY = 2.0  # seconds

# How often the device checks for dashboard button presses (Recalibrate, ...).
COMMANDS_EVERY = 1.0  # seconds

# After a database error, wait this long before trying again.
RETRY_AFTER = 10.0  # seconds

# GAZE LOCK LOSS: when the eyes can't be found (the user closed them for a
# while, looked away or moved out of the camera view) for GAZE_LOST_AFTER
# seconds, the dashboard gets a "Gaze lock lost" alert. A shorter time
# would raise an alert at every blink. The alert's "event" time is the
# moment the eyes were first missing, so the dashboard's Detection Latency
# is about GAZE_LOST_AFTER plus the time to process one camera picture.
GAZE_LOST_AFTER = 0.3  # seconds

# NAVIGATION SELECTIONS are the ones that only move around the screens:
# opening a letter group or a Quick Fires category, Back, 123/ABC, Shift,
# the KEYBOARD / QUICK FIRES tabs. They are not saved, so the caregiver's
# Activity Log shows the user's words: only selections that type, erase or
# speak. (They used to be saved during Test Mode's free-look test; Test
# Mode was removed on 8 November.) Set this to False to save navigation
# selections too.
NAVIGATION_ONLY_IN_FREE_LOOK = True

VERSION = "eyeglynt-app"  # shown to researchers on the dashboard


def slug(text):
    """A short id made of the text's letters and digits: "I'm hungry" -> "imhungry"."""
    return "".join(c for c in text.lower() if c.isalnum())[:24]


def _read_env(path):
    """Reads the KEY=value lines of the dashboard's .env file."""
    values = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class DashboardLink:
    def __init__(self):
        self.enabled = False
        self._db = None
        self._jobs = queue.Queue()      # database work, done one by one in the background
        self._commands = queue.Queue()  # dashboard button presses, for the app to carry out
        self._status = ("idle", None)   # (tracking state, fps) sent with the next heartbeat
        self._lost_since = None         # when the eyes were first missing, or None
        self._lost_alerted = False      # True once "Gaze lock lost" has been sent
        self._recal_id = None           # the recalibration being recorded, or None
        self._recal_open = False        # True from recalibration_started() to _finished()
        self._failing = False           # True while the database can't be reached
        self._stopped = threading.Event()

    # ---- starting and stopping -----------------------------------------
    def start(self):
        """Connects to the dashboard's database, if the dashboard is installed."""
        env_file = os.path.join(DASHBOARD_DIR, ".env")
        if not os.path.exists(env_file):
            print("[Dashboard] Not installed on this computer (no dashboard/.env): "
                  "running without it")
            return
        try:
            # The bridge file is in dashboard/engine_bridge. Added at the END
            # of Python's search path, so it can't hide any other module.
            if DASHBOARD_DIR not in sys.path:
                sys.path.append(DASHBOARD_DIR)
            from engine_bridge.eyeglynt_db import EyeGlyntDB
        except ImportError as e:
            print(f"[Dashboard] Can't connect ({e}). On the Pi, run once in the Terminal: "
                  "~/eyeglynt_env/bin/pip install pymysql")
            return
        env = _read_env(env_file)
        # The gaze app logs in with the "engine" account, which may only add
        # events and read the dashboard's commands.
        self._db = EyeGlyntDB(host=env.get("DB_HOST"), port=env.get("DB_PORT"),
                              user=env.get("ENGINE_DB_USER"),
                              password=env.get("ENGINE_DB_PASSWORD", ""),
                              database=env.get("DB_NAME"))
        self.enabled = True
        # If the app was restarted during a session, use that session's dwell time.
        self._do(self._resume_session)
        threading.Thread(target=self._background, daemon=True).start()
        print("[Dashboard] Connected: the caregiver dashboard will show this device")

    def stop(self):
        """Tells the dashboard the app is closing (waits at most 2 seconds).
        The dashboard then ends the running session by itself."""
        if not self.enabled or self._failing:
            return
        self._status = ("closed", None)
        self._do(lambda db: db.heartbeat("closed", version=VERSION))
        self._do(lambda db: self._stopped.set())
        self._stopped.wait(timeout=2.0)

    # ---- called by the app -----------------------------------------------
    def selection(self, interface, kind, element_id, text, category=None,
                  dwell_s=None, pointer=None, navigation=False):
        """
        Saves one dwell selection.
          interface   "keyboard" or "quickfire"
          kind        "key", "shorthand", "phrase", "speak" or "control"
          element_id  which button, e.g. "kb:A", "kb:speak", "qf:pain:myheadhurts"
          text        what the caregiver reads in the Activity Log
          navigation  True for a selection that only moves around the
                      screens (see NAVIGATION_ONLY_IN_FREE_LOOK)
        """
        if not self.enabled:
            return
        at = time.time()  # the moment of the selection, not of the database save
        dwell_ms = None if dwell_s is None else int(round(dwell_s * 1000))
        gaze_xy = None if pointer is None else (int(round(pointer[0])), int(round(pointer[1])))
        if navigation:
            kind = "control"

        def job(db):
            if navigation and NAVIGATION_ONLY_IN_FREE_LOOK and not self._in_free_look(db):
                return
            db.log_selection(interface, kind, element_id, text, category=category,
                             dwell_ms=dwell_ms, gaze_xy=gaze_xy, at=at)
        self._do(job)

    def update(self, state, fps=None, eyes_seen=True):
        """
        Called many times a second with what the device is doing:
          state      "tracking" (keyboard or Quick Fires on screen),
                     "calibrating" or "idle" (no camera)
          eyes_seen  False while the gaze engine can't find the eyes
        Sends "Gaze lock lost" / "Gaze lock restored" alerts (see
        GAZE_LOST_AFTER) and keeps the heartbeat's status up to date.
        """
        if not self.enabled:
            return
        now = time.time()
        if state == "tracking" and not eyes_seen:
            if self._lost_since is None:
                self._lost_since = now  # the first moment without the eyes
            if not self._lost_alerted and now - self._lost_since >= GAZE_LOST_AFTER:
                self._lost_alerted = True
                event_at = self._lost_since
                self._alert("gaze_lock_loss",
                            "Eyes not detected: the user looked away, closed their eyes "
                            "or left the camera view", event_at=event_at, detected_at=now)
            if self._lost_alerted:
                state = "lost"
        else:
            if self._lost_alerted and state == "tracking":
                self._alert("gaze_lock_restored", "Eyes detected again", severity="info")
            self._lost_since = None
            self._lost_alerted = False
        self._status = (state, fps)

    def take_commands(self):
        """The dashboard's button presses since the last call, as a list of
        (command, details) pairs, e.g. ("recalibrate", {}) or
        ("cancel_recalibration", {})."""
        found = []
        while True:
            try:
                found.append(self._commands.get_nowait())
            except queue.Empty:
                return found

    def recalibration_started(self, source="manual_caregiver"):
        """Records a recalibration (Recalibration Success Rate). source:
        manual_caregiver (the dashboard's button), manual_user, auto_gaze_loss,
        auto_lighting or auto_drift."""
        if not self.enabled or self._recal_open:
            return
        self._recal_open = True

        def job(db):
            self._recal_id = db.start_recalibration(source)  # None if no session is running
        self._do(job)

    def recalibration_finished(self, success):
        """Ends the recalibration recorded by recalibration_started(), if any."""
        if not self.enabled or not self._recal_open:
            return
        self._recal_open = False

        def job(db):
            if self._recal_id is not None:
                db.finish_recalibration(self._recal_id, success)
                self._recal_id = None
        self._do(job)

    def recalibration_cancelled(self):
        """Stop calibration was pressed on the dashboard: the recalibration
        is closed without a result, so it doesn't count in the
        Recalibration Success Rate (neither a success nor a failure)."""
        if not self.enabled:
            return
        self._recal_open = False

        def job(db):
            if self._recal_id is not None:
                db.cancel_recalibration(self._recal_id)
                self._recal_id = None
            elif db.active_session():
                db.raise_alert("recalibration_cancelled", "Calibration stopped from the dashboard",
                               severity="info")
        self._do(job)

    def calibration_gave_up(self, tries):
        """Calibration was poor `tries` times in a row and the app went on
        anyway (see CALIBRATION_TRIES in main.py): tells the caregiver. A
        recalibration asked for on the dashboard ends as failed, which
        raises the "Recalibration failed" alert; otherwise a "Calibration
        required" alert is raised. Both only while a session is running."""
        if not self.enabled:
            return
        if self._recal_open:
            self.recalibration_finished(False)
            return
        self._alert("calibration_required",
                    f"Calibration was poor {tries} times in a row: check the user's position "
                    "(about 40 cm from the camera) and the lighting, then press Recalibrate",
                    severity="critical")

    # ---- background work ---------------------------------------------------
    def _do(self, job):
        """Hands a piece of database work to the background thread."""
        self._jobs.put(job)

    def _alert(self, alert_type, message, **details):
        """Sends an alert, but only while a session is running."""
        def job(db):
            if db.active_session():
                db.raise_alert(alert_type, message, **details)
        self._do(job)

    def _in_free_look(self, db):
        session = db.active_session()
        if not session or session["mode"] != "test":
            return False
        block = db.active_test_block(session["session_id"])
        return block is not None and block["block_type"] == "free_look"

    def _resume_session(self, db):
        session = db.active_session(force=True)
        if session and session.get("dwell_time_ms"):
            self._commands.put(("set_dwell_time", {"dwell_time_ms": session["dwell_time_ms"]}))

    def _background(self):
        """Runs for as long as the app is open: heartbeats, dashboard
        commands, and the waiting database work, in that order."""
        next_heartbeat = next_commands = 0.0
        while not self._stopped.is_set():   # set by stop(): the app is closing
            try:
                now = time.time()
                if now >= next_heartbeat:
                    state, fps = self._status
                    self._db.heartbeat(state, fps=None if fps is None else round(fps, 1),
                                       version=VERSION)
                    self._failing = False  # the database answered
                    next_heartbeat = now + HEARTBEAT_EVERY
                if now >= next_commands:
                    for command in self._db.poll_commands():
                        self._commands.put((command["command"], command["payload"] or {}))
                        self._db.complete_command(command["command_id"])
                    next_commands = now + COMMANDS_EVERY
                try:
                    job = self._jobs.get(timeout=0.2)
                except queue.Empty:
                    continue
                job(self._db)
                self._failing = False
            except Exception as e:
                # Database not running, wrong password, ... The app keeps
                # working; the dashboard just doesn't get this information.
                self._failing = True
                print(f"[Dashboard] Database problem ({e}); trying again in {RETRY_AFTER:.0f} s")
                time.sleep(RETRY_AFTER)


# The one link the whole app shares: "from dashboard_link import dashboard".
dashboard = DashboardLink()
