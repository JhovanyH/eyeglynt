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


def close_expired_test_blocks(session_id):
    """Auto-close accuracy blocks after their planned number of selections (e.g. 30), marked or not,
    and free-look blocks past their time. Researchers can still mark trials after the block closes."""
    changed = db.execute(
        """UPDATE test_blocks b
           SET b.ended_at = NOW(3)
           WHERE b.session_id=%s AND b.ended_at IS NULL AND (
                 (b.block_type='free_look' AND b.planned_seconds IS NOT NULL
                    AND TIMESTAMPDIFF(SECOND, b.started_at, NOW(3)) >= b.planned_seconds)
              OR (b.block_type='accuracy' AND b.planned_trials IS NOT NULL
                    AND (SELECT COUNT(*) FROM selections s WHERE s.block_id=b.block_id)
                        >= b.planned_trials))""",
        (session_id,),
    )
    db.commit()
    return changed


def active_test_block(session_id):
    return db.query(
        """SELECT b.*, TIMESTAMPDIFF(SECOND, b.started_at, NOW(3)) AS elapsed_s,
                  (SELECT COUNT(*) FROM selections s WHERE s.block_id=b.block_id) AS activations,
                  (SELECT COUNT(*) FROM selections s WHERE s.block_id=b.block_id
                     AND s.target_element_id IS NOT NULL) AS trials,
                  (SELECT COALESCE(SUM(s.is_correct),0) FROM selections s WHERE s.block_id=b.block_id) AS correct
           FROM test_blocks b WHERE b.session_id=%s AND b.ended_at IS NULL
           ORDER BY b.block_id DESC LIMIT 1""",
        (session_id,), one=True,
    )


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


def end_session(session_id, caregiver_id):
    """Close test blocks, mark ended, freeze metrics, tell the device. Idempotent."""
    session = db.query("SELECT status FROM sessions WHERE session_id=%s FOR UPDATE",
                       (session_id,), one=True)
    if not session or session["status"] == "ended":
        db.rollback()
        return False
    db.execute("UPDATE test_blocks SET ended_at=NOW(3) WHERE session_id=%s AND ended_at IS NULL",
               (session_id,))
    db.execute("UPDATE sessions SET status='ended', ended_at=NOW(3) WHERE session_id=%s",
               (session_id,))
    db.execute(
        f"""REPLACE INTO session_metrics (session_id, {METRIC_COLUMNS})
            SELECT session_id, {METRIC_COLUMNS} FROM v_session_metrics WHERE session_id=%s""",
        (session_id,),
    )
    queue_command(session_id, "end_session", caregiver_id)
    db.commit()
    return True
