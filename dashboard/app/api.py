"""JSON endpoints used by the Live screen (polled every POLL_INTERVAL_MS)."""
from flask import Blueprint, abort, jsonify, request, url_for
from flask_login import current_user, login_required

from . import db, metrics
from .constants import ALERT_LABELS, CAREGIVER_ALERT_TEXT
from .sessions import get_session, session_alerts, session_selections

bp = Blueprint("api", __name__, url_prefix="/api")


def _sel_json(r):
    return {
        "id": r["selection_id"],
        "time": r["selected_at"].strftime("%H:%M:%S"),
        "interface": r["interface"],
        "type": r["selection_type"],
        "text": r["output_text"],
        "category": r["category"],
        "intent": r["intent_status"],
        "target": r["target_element_id"],
        "is_correct": r["is_correct"],
        "in_block": r["block_id"] is not None,
        "block_type": r["block_type"],
    }


def _alert_json(r, detailed=True):
    if not detailed:
        return {
            "id": r["alert_id"],
            "time": r["detected_at"].strftime("%H:%M:%S"),
            "type": r["alert_type"],
            "label": ALERT_LABELS.get(r["alert_type"], r["alert_type"]),
            "severity": r["severity"],
            "message": CAREGIVER_ALERT_TEXT.get(r["alert_type"], r["message"]),
            "latency_ms": None,
            "acknowledged": r["acknowledged_at"] is not None,
        }
    return {
        "id": r["alert_id"],
        "time": r["detected_at"].strftime("%H:%M:%S"),
        "type": r["alert_type"],
        "label": ALERT_LABELS.get(r["alert_type"], r["alert_type"]),
        "severity": r["severity"],
        "message": r["message"],
        "latency_ms": r["detection_latency_ms"],
        "acknowledged": r["acknowledged_at"] is not None,
    }


def _active_or_409(session_id):
    s = get_session(session_id)
    if s["status"] != "active":
        abort(409, description="Session has already ended.")
    return s


@bp.errorhandler(409)
@bp.errorhandler(403)
@bp.errorhandler(400)
def _json_error(err):
    return jsonify(ok=False, error=err.description), err.code


@bp.get("/sessions/<int:session_id>/updates")
@login_required
def updates(session_id):
    if metrics.end_if_app_closed() == session_id:
        return jsonify(ok=True, status="ended", app_closed=True,
                       summary_url=url_for("sessions.summary", session_id=session_id, ended=1, closed=1))
    s = get_session(session_id)
    detailed = current_user.is_researcher
    after_sel = request.args.get("after_sel", 0, type=int)
    after_alert = request.args.get("after_alert", 0, type=int)

    m = metrics.live_metrics(session_id)
    dev = metrics.device_status()
    # Still waiting for the device to open calibration: shown as "requested".
    pending_recal = db.query(
        """SELECT COUNT(*) AS n FROM device_commands
           WHERE session_id=%s AND command='recalibrate' AND status IN ('pending','acknowledged')""",
        (session_id,), one=True,
    )["n"]

    return jsonify(
        ok=True,
        status=s["status"],
        summary_url=url_for("sessions.summary", session_id=session_id, ended=1),
        elapsed_s=s["duration_s"],
        selections=[_sel_json(r) for r in session_selections(session_id, after_sel)],
        alerts=[_alert_json(r, detailed) for r in session_alerts(session_id, after_alert)],
        # Caregivers get counts only; full metrics are for researchers.
        metrics=metrics.to_jsonable(m) if detailed else {
            "total_selections": m["total_selections"],
            "keyboard_selections": int(m["keyboard_selections"] or 0),
            "quickfire_selections": int(m["quickfire_selections"] or 0),
            "most_used_interface": m["most_used_interface"],
        },
        checks=metrics.evaluate(m) if detailed else {},
        device={
            "online": dev.get("online", False),
            "state": dev.get("tracking_state"),
            "fps": float(dev["fps"]) if detailed and dev.get("fps") is not None else None,
            "lux": float(dev["current_lux"]) if detailed and dev.get("current_lux") is not None else None,
        },
        recal_pending=pending_recal > 0,
    )


@bp.post("/selections/<int:selection_id>/flag")
@login_required
def flag_selection(selection_id):
    row = db.query("SELECT session_id FROM selections WHERE selection_id=%s", (selection_id,), one=True)
    if not row:
        abort(404)
    get_session(row["session_id"])
    # "ignored": left out of every metric (not intended, not unintended).
    intent = (request.get_json(silent=True) or {}).get("intent")
    if intent not in ("intended", "unintended", "ignored"):
        abort(400, description="intent must be 'intended', 'unintended' or 'ignored'")
    db.execute(
        """UPDATE selections SET intent_status=%s, reviewed_by=%s, reviewed_at=NOW()
           WHERE selection_id=%s""",
        (intent, current_user.id, selection_id),
    )
    db.commit()
    return jsonify(ok=True, id=selection_id, intent=intent)


@bp.post("/alerts/<int:alert_id>/ack")
@login_required
def ack_alert(alert_id):
    db.execute(
        """UPDATE system_alerts SET acknowledged_by=%s, acknowledged_at=NOW()
           WHERE alert_id=%s AND acknowledged_at IS NULL""",
        (current_user.id, alert_id),
    )
    db.commit()
    return jsonify(ok=True)


@bp.post("/sessions/<int:session_id>/recalibrate")
@login_required
def recalibrate(session_id):
    _active_or_409(session_id)
    metrics.queue_command(session_id, "recalibrate", current_user.id, {"source": "manual_caregiver"})
    db.commit()
    return jsonify(ok=True)


@bp.post("/sessions/<int:session_id>/recalibrate/stop")
@login_required
def stop_recalibration(session_id):
    """Stop calibration: the device leaves the calibration screen and goes
    back to the keyboard, keeping its previous calibration. A Recalibrate
    the device hasn't picked up yet is simply withdrawn."""
    _active_or_409(session_id)
    db.execute(
        """UPDATE device_commands SET status='failed', handled_at=NOW(3)
           WHERE session_id=%s AND command='recalibrate' AND status='pending'""",
        (session_id,),
    )
    metrics.queue_command(session_id, "cancel_recalibration", current_user.id)
    db.commit()
    return jsonify(ok=True)


@bp.post("/sessions/<int:session_id>/end")
@login_required
def end(session_id):
    get_session(session_id)
    metrics.end_session(session_id, current_user.id)
    return jsonify(ok=True, redirect=url_for("sessions.summary", session_id=session_id, ended=1))
