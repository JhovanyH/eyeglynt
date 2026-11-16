"""Link between the app and the caregiver dashboard (the "dashboard" folder).

Both sides only share the Pi's database (dashboard/engine_bridge/eyeglynt_db.py),
so the app keeps working if the dashboard stops. Database work runs in a
background thread. Selections are saved only during a caregiver's session.
Without the dashboard (no dashboard/.env) or pymysql, the app runs as normal.

Setup on the Pi: ~/eyeglynt_env/bin/pip install pymysql
"""

import os
import queue
import sys
import threading
import time

# The dashboard folder, next to this file.
DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")

# Heartbeat: the dashboard shows the device offline after 10 s without one.
HEARTBEAT_EVERY = 2.0  # seconds

# How often to check for dashboard button presses.
COMMANDS_EVERY = 1.0  # seconds

# Wait after a database error.
RETRY_AFTER = 10.0  # seconds

# "Gaze lock lost" alert after the eyes are missing this long (shorter would
# alert at every blink).
GAZE_LOST_AFTER = 0.3  # seconds

# Selections that only move around the screens (groups, categories, Back, tabs)
# are saved as "navigation": researchers see them, caregivers don't, and the
# metrics leave them out.

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
        self._jobs = queue.Queue()  # database jobs, run in the background
        self._commands = queue.Queue()  # dashboard button presses
        self._status = ("idle", None)  # (state, fps) for the next heartbeat
        self._lost_since = None         # when the eyes were first missing, or None
        self._lost_alerted = False  # True once "Gaze lock lost" was sent
        self._recal_id = None  # recalibration being recorded
        self._recal_open = False  # between recalibration_started() and _finished()
        self._failing = False  # True while the database can't be reached
        self._stopped = threading.Event()

    # ---- starting and stopping ----
    def start(self):
        """Connects to the dashboard's database, if the dashboard is installed."""
        env_file = os.path.join(DASHBOARD_DIR, ".env")
        if not os.path.exists(env_file):
            print("[Dashboard] Not installed on this computer (no dashboard/.env): "
                  "running without it")
            return
        try:
            # The bridge is in dashboard/engine_bridge (added at the end of the path).
            if DASHBOARD_DIR not in sys.path:
                sys.path.append(DASHBOARD_DIR)
            from engine_bridge.eyeglynt_db import EyeGlyntDB
        except ImportError as e:
            print(f"[Dashboard] Can't connect ({e}). On the Pi, run once in the Terminal: "
                  "~/eyeglynt_env/bin/pip install pymysql")
            return
        env = _read_env(env_file)
        # The "engine" account may only add events and read commands.
        self._db = EyeGlyntDB(host=env.get("DB_HOST"), port=env.get("DB_PORT"),
                              user=env.get("ENGINE_DB_USER"),
                              password=env.get("ENGINE_DB_PASSWORD", ""),
                              database=env.get("DB_NAME"))
        self.enabled = True
        # After a restart during a session, use its dwell time.
        self._do(self._resume_session)
        threading.Thread(target=self._background, daemon=True).start()
        print("[Dashboard] Connected: the caregiver dashboard will show this device")

    def stop(self):
        """Tells the dashboard the app is closing (waits at most 2 s)."""
        if not self.enabled or self._failing:
            return
        self._status = ("closed", None)
        self._do(lambda db: db.heartbeat("closed", version=VERSION))
        self._do(lambda db: self._stopped.set())
        self._stopped.wait(timeout=2.0)

    # ---- called by the app ----
    def selection(self, interface, kind, element_id, text, category=None,
                  dwell_s=None, pointer=None, navigation=False):
        """Saves one selection. kind: "key", "shorthand", "phrase", "speak" or "control";
        navigation=True for selections that only move around the screens."""
        if not self.enabled:
            return
        at = time.time()  # time of the selection, not of the save
        dwell_ms = None if dwell_s is None else int(round(dwell_s * 1000))
        gaze_xy = None if pointer is None else (int(round(pointer[0])), int(round(pointer[1])))
        if navigation:
            kind = "navigation"

        def job(db):
            db.log_selection(interface, kind, element_id, text, category=category,
                             dwell_ms=dwell_ms, gaze_xy=gaze_xy, at=at)
        self._do(job)

    def update(self, state, fps=None, eyes_seen=True):
        """Called often with the device state ("tracking", "calibrating" or "idle"); sends gaze
        lost/restored alerts and updates the heartbeat."""
        if not self.enabled:
            return
        now = time.time()
        if state == "tracking" and not eyes_seen:
            if self._lost_since is None:
                self._lost_since = now  # first moment without the eyes
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
        """Dashboard button presses since the last call, as (command, details) pairs."""
        found = []
        while True:
            try:
                found.append(self._commands.get_nowait())
            except queue.Empty:
                return found

    def recalibration_started(self, source="manual_caregiver"):
        """Records a recalibration; source is what started it (e.g. "manual_caregiver")."""
        if not self.enabled or self._recal_open:
            return
        self._recal_open = True

        def job(db):
            self._recal_id = db.start_recalibration(source)  # None if no session
        self._do(job)

    def recalibration_finished(self, success):
        """Ends the recalibration started by recalibration_started(), if any."""
        if not self.enabled or not self._recal_open:
            return
        self._recal_open = False

        def job(db):
            if self._recal_id is not None:
                db.finish_recalibration(self._recal_id, success)
                self._recal_id = None
        self._do(job)

    def recalibration_cancelled(self):
        """Stop calibration: close the recalibration without a result."""
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
        """Calibration was poor tries times in a row: fail the recalibration, or raise a
        "Calibration required" alert (during a session)."""
        if not self.enabled:
            return
        if self._recal_open:
            self.recalibration_finished(False)
            return
        self._alert("calibration_required",
                    f"Calibration was poor {tries} times in a row: check the user's position "
                    "(about 40 cm from the camera) and the lighting, then press Recalibrate",
                    severity="critical")

    # ---- background work ----
    def _do(self, job):
        """Hands database work to the background thread."""
        self._jobs.put(job)

    def _alert(self, alert_type, message, **details):
        """Sends an alert, only during a session."""
        def job(db):
            if db.active_session():
                db.raise_alert(alert_type, message, **details)
        self._do(job)

    def _resume_session(self, db):
        session = db.active_session(force=True)
        if session and session.get("dwell_time_ms"):
            self._commands.put(("set_dwell_time", {"dwell_time_ms": session["dwell_time_ms"]}))

    def _background(self):
        """Background loop: heartbeats, dashboard commands, then waiting database jobs."""
        next_heartbeat = next_commands = 0.0
        while not self._stopped.is_set():  # set by stop()
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
                # Database unreachable: the app keeps working without the dashboard.
                self._failing = True
                print(f"[Dashboard] Database problem ({e}); trying again in {RETRY_AFTER:.0f} s")
                time.sleep(RETRY_AFTER)


# Shared by the whole app: from dashboard_link import dashboard
dashboard = DashboardLink()
