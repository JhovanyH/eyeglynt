"""Session metrics, benchmark checks (thesis Table 7) and session lifecycle helpers."""
import json

from flask import current_app

from . import db
from .constants import BENCHMARKS

METRIC_COLUMNS = (
    "total_selections, keyboard_selections, quickfire_selections, most_used_interface, "
    "gaze_accuracy_pct, accuracy_basis, accuracy_trials, false_positive_pct, "
    "unintended_selections, avg_detection_latency_ms, latency_events, "
    "recalibration_success_pct, recalibration_attempts"
)


def live_metrics(session_id):
    return db.query(
        f"SELECT {METRIC_COLUMNS} FROM v_session_metrics WHERE session_id=%s", (session_id,), one=True
    )


def saved_metrics(session_id):
    """Frozen snapshot if the session has ended, otherwise the live view."""
    row = db.query(
        f"SELECT {METRIC_COLUMNS}, computed_at FROM session_metrics WHERE session_id=%s",
        (session_id,), one=True,
    )
    return row or live_metrics(session_id)


def evaluate(metrics):
    """Return {metric_key: 'pass' | 'fail' | 'na'} against the thesis benchmarks."""
    result = {}
    for key, rule in BENCHMARKS.items():
        value = metrics.get(key) if metrics else None
        if value is None:
            result[key] = "na"
            continue
        value = float(value)
        ok = value >= rule["value"] if rule["op"] == ">=" else value < rule["value"]
        result[key] = "pass" if ok else "fail"
    return result


def to_jsonable(metrics):
    out = {}
    for k, v in (metrics or {}).items():
        out[k] = float(v) if hasattr(v, "is_finite") else (v.isoformat() if hasattr(v, "isoformat") else v)
    return out


def device_status():
    row = db.query(
        """SELECT tracking_state, fps, current_lux, software_version, last_heartbeat,
                  TIMESTAMPDIFF(SECOND, last_heartbeat, NOW(3)) AS age_s
           FROM device_status WHERE device_id=1""",
        one=True,
    ) or {}
    limit = current_app.config["DEVICE_OFFLINE_AFTER_S"]
    age = row.get("age_s")
    row["online"] = age is not None and age <= limit
    return row


def queue_command(session_id, command, issued_by, payload=None):
    return db.execute(
        "INSERT INTO device_commands (session_id, command, payload, issued_by) VALUES (%s,%s,%s,%s)",
        (session_id, command, json.dumps(payload) if payload is not None else None, issued_by),
    )


def end_session(session_id, caregiver_id, ended_at=None, app_closed=False):
    """Mark ended, freeze metrics, tell the device. Idempotent.
    app_closed=True: the EyeGlynt app was closed on the device (see
    end_if_app_closed), so the session ends at ended_at and the device
    isn't sent anything."""
    session = db.query("SELECT status, started_at FROM sessions WHERE session_id=%s FOR UPDATE",
                       (session_id,), one=True)
    if not session or session["status"] == "ended":
        db.rollback()
        return False
    db.execute("UPDATE test_blocks SET ended_at=NOW(3) WHERE session_id=%s AND ended_at IS NULL",
               (session_id,))
    if ended_at is None or ended_at < session["started_at"]:
        ended_at = None
    db.execute("UPDATE sessions SET status='ended', ended_at=COALESCE(%s, NOW(3)) WHERE session_id=%s",
               (ended_at, session_id))
    if app_closed:
        db.execute(
            """INSERT INTO system_alerts (session_id, alert_type, severity, message)
               VALUES (%s, 'session_auto_ended', 'info', %s)""",
            (session_id, "The EyeGlynt app was closed on the device, so the session ended automatically."),
        )
    db.execute(
        f"""REPLACE INTO session_metrics (session_id, {METRIC_COLUMNS})
            SELECT session_id, {METRIC_COLUMNS} FROM v_session_metrics WHERE session_id=%s""",
        (session_id,),
    )
    if not app_closed:
        queue_command(session_id, "end_session", caregiver_id)
    db.commit()
    return True


def end_if_app_closed():
    """Ends the running session if the EyeGlynt app was closed on the device
    after the session started (the app sends the state "closed" as it
    closes, see dashboard_link.py). Returns the ended session's id or None."""
    row = db.query(
        """SELECT s.session_id, d.last_heartbeat
           FROM sessions s JOIN device_status d ON d.device_id = 1
           WHERE s.status='active' AND d.tracking_state='closed'
             AND d.last_heartbeat > s.started_at
           LIMIT 1""",
        one=True,
    )
    if not row:
        return None
    end_session(row["session_id"], None, ended_at=row["last_heartbeat"], app_closed=True)
    return row["session_id"]
