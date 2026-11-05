"""JSON endpoints used by the Live screen (polled every POLL_INTERVAL_MS)."""
from flask import Blueprint, abort, jsonify, request, url_for
from flask_login import current_user, login_required

from . import db, metrics
from .auth import researcher_required
from .constants import ALERT_LABELS, CAREGIVER_ALERT_TEXT, DWELL_OPTIONS
from .sessions import get_session, guard_view, session_alerts, session_selections

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
    guard_view(s)
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
    s = get_session(session_id)
    guard_view(s)
    detailed = current_user.is_researcher
    after_sel = request.args.get("after_sel", 0, type=int)
    after_alert = request.args.get("after_alert", 0, type=int)

    if s["status"] == "active":
        metrics.close_expired_test_blocks(session_id)
    block = metrics.active_test_block(session_id) if s["mode"] == "test" else None
    m = metrics.live_metrics(session_id)
    dev = metrics.device_status()
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
        dwell_time_ms=s["dwell_time_ms"],
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
        block=None if not block else {
            "id": block["block_id"],
            "type": block["block_type"],
            "elapsed_s": block["elapsed_s"],
            "planned_trials": block["planned_trials"],
            "planned_seconds": block["planned_seconds"],
            "activations": block["activations"],
            "trials": int(block["trials"] or 0),
            "correct": int(block["correct"] or 0),
        },
    )


@bp.post("/selections/<int:selection_id>/flag")
@login_required
def flag_selection(selection_id):
    row = db.query("SELECT session_id FROM selections WHERE selection_id=%s", (selection_id,), one=True)
    if not row:
        abort(404)
    guard_view(get_session(row["session_id"]))
    intent = (request.get_json(silent=True) or {}).get("intent")
    if intent not in ("intended", "unintended"):
        abort(400, description="intent must be 'intended' or 'unintended'")
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


@bp.post("/sessions/<int:session_id>/dwell")
@login_required
def set_dwell(session_id):
    _active_or_409(session_id)
    dwell = (request.get_json(silent=True) or {}).get("dwell_time_ms")
    if dwell not in [d for d, _ in DWELL_OPTIONS]:
        abort(400, description="Unsupported dwell time.")
    db.execute("UPDATE sessions SET dwell_time_ms=%s WHERE session_id=%s", (dwell, session_id))
    metrics.queue_command(session_id, "set_dwell_time", current_user.id, {"dwell_time_ms": dwell})
    db.commit()
    return jsonify(ok=True, dwell_time_ms=dwell)


@bp.post("/selections/<int:selection_id>/score")
@researcher_required
def score_selection(selection_id):
    """Accuracy test: the researcher said the target word out loud, then marks the
    patient's selection Correct or Wrong. Stored as the trial's target so the
    is_correct column and Gaze Accuracy work the same as device-scored trials."""
    row = db.query(
        """SELECT x.session_id, x.element_id, b.block_type
           FROM selections x LEFT JOIN test_blocks b ON b.block_id = x.block_id
           WHERE x.selection_id=%s""",
        (selection_id,), one=True,
    )
    if not row:
        abort(404)
    _active_or_409(row["session_id"])
    if row["block_type"] != "accuracy":
        abort(400, description="Only selections made during an accuracy test can be scored.")
    result = (request.get_json(silent=True) or {}).get("result")
    if result not in ("correct", "wrong", "clear"):
        abort(400, description="result must be 'correct', 'wrong' or 'clear'")
    target = {"correct": row["element_id"], "wrong": "manual:wrong", "clear": None}[result]
    db.execute("UPDATE selections SET target_element_id=%s WHERE selection_id=%s",
               (target, selection_id))
    db.commit()
    return jsonify(ok=True, result=result)


@bp.post("/sessions/<int:session_id>/test-block")
@researcher_required
def test_block(session_id):
    s = _active_or_409(session_id)
    if s["mode"] != "test":
        abort(400, description="Test blocks are only available in Test Mode.")
    data = request.get_json(silent=True) or {}
    action = data.get("action")

    if action == "end":
        open_block = metrics.active_test_block(session_id)
        db.execute("UPDATE test_blocks SET ended_at=NOW(3) WHERE session_id=%s AND ended_at IS NULL",
                   (session_id,))
        if open_block:
            metrics.queue_command(session_id, "end_test_block", current_user.id,
                                  {"block_id": open_block["block_id"]})
        db.commit()
        return jsonify(ok=True)

    if action == "start":
        btype = data.get("type")
        if btype not in ("accuracy", "free_look"):
            abort(400, description="type must be 'accuracy' or 'free_look'")
        if metrics.active_test_block(session_id):
            abort(409, description="Finish the current test block first.")
        trials = int(data.get("trials", 30)) if btype == "accuracy" else None
        seconds = int(data.get("seconds", 120)) if btype == "free_look" else None
        block_id = db.execute(
            """INSERT INTO test_blocks (session_id, block_type, planned_trials, planned_seconds)
               VALUES (%s,%s,%s,%s)""",
            (session_id, btype, trials, seconds),
        )
        metrics.queue_command(session_id, "start_test_block", current_user.id,
                              {"block_id": block_id, "type": btype,
                               "trials": trials, "seconds": seconds})
        db.commit()
        return jsonify(ok=True, block_id=block_id)

    abort(400, description="action must be 'start' or 'end'")


@bp.post("/sessions/<int:session_id>/end")
@login_required
def end(session_id):
    guard_view(get_session(session_id))
    metrics.end_session(session_id, current_user.id)
    return jsonify(ok=True, redirect=url_for("sessions.summary", session_id=session_id, ended=1))
