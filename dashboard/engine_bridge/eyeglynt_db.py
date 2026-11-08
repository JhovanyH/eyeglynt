"""
EyeGlynt engine ↔ database bridge.

Import this from the gaze-tracking program (and the simulator). It is the ONLY
code the engine needs to talk to the Caregiver Dashboard — the two programs
never call each other directly.

    from engine_bridge.eyeglynt_db import EyeGlyntDB
    db = EyeGlyntDB()                       # reads ENGINE_DB_* from .env
    db.heartbeat("tracking", fps=28.5)      # every ~2 s
    db.log_selection("quickfire", "phrase", "qf:needs:water", "I want water", category="Basic Needs")
    for cmd in db.poll_commands():          # every ~1 s
        ...

Timestamps: pass the engine's own time.time() for event_at so Detection Latency
measures engine reaction time, not database insert time.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime

import pymysql

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional for the engine
    pass


def _ts(t: float | datetime | None) -> datetime | None:
    if t is None:
        return None
    return t if isinstance(t, datetime) else datetime.fromtimestamp(t)


class EyeGlyntDB:
    SESSION_CACHE_S = 2.0

    def __init__(self, host=None, user=None, password=None, database=None, port=None, unix_socket=None):
        self._cfg = dict(
            host=host or os.getenv("DB_HOST", "localhost"),
            port=int(port or os.getenv("DB_PORT", "3306")),
            user=user or os.getenv("ENGINE_DB_USER", "eyeglynt_engine"),
            password=password if password is not None else os.getenv("ENGINE_DB_PASSWORD", ""),
            database=database or os.getenv("DB_NAME", "eyeglynt"),
            unix_socket=unix_socket or os.getenv("DB_UNIX_SOCKET") or None,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )
        self._conn = None
        self._lock = threading.Lock()          # safe to call from camera + UI threads
        self._session_cache = (0.0, None)

    # ------------------------------------------------------------------ plumbing
    def _run(self, sql, args=None, fetch=None):
        with self._lock:
            for attempt in (1, 2):             # reconnect once if MariaDB restarted
                try:
                    if self._conn is None or not self._conn.open:
                        self._conn = pymysql.connect(**self._cfg)
                    with self._conn.cursor() as cur:
                        cur.execute(sql, args or ())
                        if fetch == "one":
                            return cur.fetchone()
                        if fetch == "all":
                            return cur.fetchall()
                        return cur.lastrowid
                except pymysql.err.OperationalError:
                    self._conn = None
                    if attempt == 2:
                        raise

    # ------------------------------------------------------------------ session
    def active_session(self, force=False) -> dict | None:
        """The running session (cached ~2 s). None means: don't log, just communicate."""
        ts, cached = self._session_cache
        if force or time.monotonic() - ts > self.SESSION_CACHE_S:
            cached = self._run(
                """SELECT session_id, patient_id, mode, dwell_time_ms, profile_id
                   FROM sessions WHERE status='active' LIMIT 1""", fetch="one")
            self._session_cache = (time.monotonic(), cached)
        return cached

    def active_test_block(self, session_id: int) -> dict | None:
        return self._run(
            """SELECT block_id, block_type, planned_trials, planned_seconds FROM test_blocks
               WHERE session_id=%s AND ended_at IS NULL ORDER BY block_id DESC LIMIT 1""",
            (session_id,), fetch="one")

    # ------------------------------------------------------------------ events
    def log_selection(self, interface: str, selection_type: str, element_id: str, output_text: str,
                      *, category: str | None = None, dwell_ms: int | None = None,
                      gaze_xy: tuple[int, int] | None = None, target_element_id: str | None = None,
                      at: float | None = None) -> int | None:
        """Record one dwell-time activation. interface: 'keyboard'|'quickfire';
        selection_type: 'key'|'shorthand'|'phrase'|'speak'|'control'."""
        s = self.active_session()
        if not s:
            return None
        block = self.active_test_block(s["session_id"]) if s["mode"] == "test" else None
        sel_id = self._run(
            """INSERT INTO selections (session_id, block_id, selected_at, interface, selection_type,
                   element_id, output_text, category, dwell_ms, gaze_x, gaze_y, target_element_id)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (s["session_id"], block["block_id"] if block else None, _ts(at or time.time()),
             interface, selection_type, element_id, output_text[:500], category, dwell_ms,
             gaze_xy[0] if gaze_xy else None, gaze_xy[1] if gaze_xy else None,
             target_element_id if (block and block["block_type"] == "accuracy") else None))
        # Close an accuracy block exactly at its planned trial count (e.g. 30) so no extra trials sneak in.
        if block and block["block_type"] == "accuracy" and block["planned_trials"]:
            self._run(
                """UPDATE test_blocks SET ended_at=NOW(3)
                   WHERE block_id=%s AND ended_at IS NULL AND
                         (SELECT COUNT(*) FROM selections WHERE block_id=%s) >= planned_trials""",
                (block["block_id"], block["block_id"]))
        return sel_id

    def raise_alert(self, alert_type: str, message: str, *, severity: str = "warning",
                    event_at: float | None = None, detected_at: float | None = None,
                    lux: float | None = None) -> int | None:
        """alert_type: lighting_shift | gaze_lock_loss | gaze_lock_restored | recalibration_started |
        recalibration_success | recalibration_failed | calibration_required | system.
        For lighting_shift / gaze_lock_loss pass event_at = time of the first affected frame."""
        s = self.active_session()
        return self._run(
            """INSERT INTO system_alerts (session_id, alert_type, severity, message, event_at,
                   detected_at, lux_reading) VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            (s["session_id"] if s else None, alert_type, severity, message[:255],
             _ts(event_at), _ts(detected_at or time.time()), lux))

    def start_recalibration(self, trigger_source: str, trigger_alert_id: int | None = None) -> int | None:
        """trigger_source: auto_gaze_loss | auto_lighting | auto_drift | manual_caregiver | manual_user"""
        s = self.active_session()
        if not s:
            return None
        rid = self._run(
            "INSERT INTO recalibrations (session_id, trigger_source, trigger_alert_id) VALUES (%s,%s,%s)",
            (s["session_id"], trigger_source, trigger_alert_id))
        self.raise_alert("recalibration_started", f"Recalibration started ({trigger_source.replace('_', ' ')})",
                         severity="info")
        return rid

    def finish_recalibration(self, recal_id: int, success: bool, *, verify_selection_id: int | None = None,
                             new_profile_id: int | None = None) -> None:
        """Thesis rule: success = tracking restored AND the next selection is correct on the first try."""
        if recal_id is None:
            return
        self._run(
            """UPDATE recalibrations SET completed_at=NOW(3), success=%s,
                   verify_selection_id=%s, new_profile_id=%s WHERE recal_id=%s""",
            (1 if success else 0, verify_selection_id, new_profile_id, recal_id))
        if success:
            self.raise_alert("recalibration_success", "Tracking restored", severity="info")
        else:
            self.raise_alert("recalibration_failed", "Tracking could not be restored — check position and lighting",
                             severity="critical")

    def cancel_recalibration(self, recal_id: int) -> None:
        """Calibration was stopped from the dashboard: closed with no result
        (success stays NULL), so it is not counted as a success or a failure."""
        if recal_id is None:
            return
        self._run("UPDATE recalibrations SET completed_at=NOW(3) WHERE recal_id=%s AND success IS NULL",
                  (recal_id,))
        self.raise_alert("recalibration_cancelled", "Calibration stopped from the dashboard", severity="info")

    def save_calibration_profile(self, patient_id: int, model_params: dict, fit_accuracy_pct: float,
                                 points_captured: int = 5) -> int:
        """Store a 5-point calibration. Becomes the patient's active profile if fit ≥ 90 %."""
        pid = self._run(
            """INSERT INTO calibration_profiles (patient_id, points_captured, model_params, fit_accuracy_pct)
               VALUES (%s,%s,%s,%s)""",
            (patient_id, points_captured, json.dumps(model_params), fit_accuracy_pct))
        if fit_accuracy_pct >= 90:
            self._run("UPDATE calibration_profiles SET is_active = (profile_id=%s) WHERE patient_id=%s",
                      (pid, patient_id))
        return pid

    # ------------------------------------------------------------------ commands + status
    def poll_commands(self) -> list[dict]:
        """Fetch pending dashboard commands and mark them acknowledged. Call complete_command() after."""
        rows = self._run(
            """SELECT command_id, session_id, command, payload FROM device_commands
               WHERE status='pending' ORDER BY command_id LIMIT 20""", fetch="all") or []
        for r in rows:
            r["payload"] = json.loads(r["payload"]) if r["payload"] else {}
            self._run("UPDATE device_commands SET status='acknowledged', handled_at=NOW(3) WHERE command_id=%s",
                      (r["command_id"],))
        if any(r["command"] in ("start_session", "end_session", "set_dwell_time") for r in rows):
            self.active_session(force=True)
        return rows

    def complete_command(self, command_id: int, ok: bool = True) -> None:
        self._run("UPDATE device_commands SET status=%s, handled_at=NOW(3) WHERE command_id=%s",
                  ("done" if ok else "failed", command_id))

    def heartbeat(self, tracking_state: str = "tracking", *, fps: float | None = None,
                  lux: float | None = None, profile_id: int | None = None, version: str = "1.0") -> None:
        """tracking_state: idle | calibrating | tracking | lost | closed (the app is closing)"""
        self._run(
            """INSERT INTO device_status (device_id, last_heartbeat, tracking_state, fps, current_lux,
                   active_profile_id, software_version) VALUES (1, NOW(3), %s, %s, %s, %s, %s)
               ON DUPLICATE KEY UPDATE last_heartbeat=NOW(3), tracking_state=VALUES(tracking_state),
                   fps=VALUES(fps), current_lux=VALUES(current_lux),
                   active_profile_id=VALUES(active_profile_id), software_version=VALUES(software_version)""",
            (tracking_state, fps, lux, profile_id, version))

