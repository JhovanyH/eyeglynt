"""Session pages: Start (Fig. 34), Live (Fig. 35), Summary (Fig. 36), History."""
from datetime import datetime

from flask import (Blueprint, Response, abort, current_app, flash, redirect,
                   render_template, request, url_for)
from flask_login import current_user, login_required

from . import db, metrics
from .constants import ALERT_LABELS, BENCHMARKS, CARE_SETTINGS, DWELL_TIME_MS

bp = Blueprint("sessions", __name__)


# ---------------------------------------------------------------- helpers
def get_session(session_id):
    row = db.query(
        """SELECT s.*, p.full_name AS patient_name, p.patient_code,
                  c.first_name AS cg_first, c.last_name AS cg_last, c.role AS cg_role,
                  TIMESTAMPDIFF(SECOND, s.started_at, COALESCE(s.ended_at, NOW(3))) AS duration_s
           FROM sessions s
           JOIN patients p   ON p.patient_id = s.patient_id
           JOIN caregivers c ON c.caregiver_id = s.caregiver_id
           WHERE s.session_id=%s""",
        (session_id,), one=True,
    )
    if not row:
        abort(404)
    return row


def active_session():
    metrics.end_if_app_closed()   # the EyeGlynt app was closed: that session is over
    return db.query("SELECT session_id, mode FROM sessions WHERE status='active' LIMIT 1", one=True)


def active_session_id():
    row = active_session()
    return row["session_id"] if row else None


def session_selections(session_id, after_id=0):
    return db.query(
        """SELECT x.selection_id, x.selected_at, x.interface, x.selection_type, x.element_id,
                  x.output_text, x.category, x.target_element_id, x.is_correct, x.intent_status,
                  x.block_id, b.block_type
           FROM selections x LEFT JOIN test_blocks b ON b.block_id = x.block_id
           WHERE x.session_id=%s AND x.selection_id>%s
           ORDER BY x.selection_id""",
        (session_id, after_id),
    )


def session_alerts(session_id, after_id=0):
    return db.query(
        """SELECT alert_id, alert_type, severity, message, event_at, detected_at,
                  detection_latency_ms, acknowledged_at
           FROM system_alerts WHERE session_id=%s AND alert_id>%s ORDER BY alert_id""",
        (session_id, after_id),
    )


def next_codes():
    today = datetime.now().strftime("%Y%m%d")
    n = db.query("SELECT COUNT(*) AS n FROM sessions WHERE session_code LIKE %s",
                 (f"S-{today}-%",), one=True)["n"]
    return f"S-{today}-{n + 1:02d}"


# ---------------------------------------------------------------- routes
@bp.route("/home")
@login_required
def home():
    active = active_session()
    if not active:
        return redirect(url_for(".start"))
    return redirect(url_for(".live", session_id=active["session_id"]))


@bp.route("/session/start", methods=["GET", "POST"])
@login_required
def start():
    if active_session():
        flash("A session is already running on the device.", "info")
        return redirect(url_for(".home"))

    patients = db.query(
        """SELECT p.patient_id, p.full_name, p.patient_code, p.care_setting,
                  (SELECT COUNT(*) FROM calibration_profiles cp
                   WHERE cp.patient_id=p.patient_id AND cp.is_active=1) AS has_profile
           FROM patients p ORDER BY p.full_name"""
    )
    form, errors = {}, {}

    if request.method == "POST":
        form = {k: v.strip() for k, v in request.form.items()}
        patient_choice = form.get("patient_id", "new")
        chosen = next((p for p in patients if str(p["patient_id"]) == patient_choice), None)
        if patient_choice == "new":
            # Only researchers enrol new patients (study participants).
            if not current_user.is_researcher:
                errors["patient_id"] = "Ask the research team to add the patient."
            elif not form.get("patient_name"):
                errors["patient_name"] = "Enter the patient's full name."
        elif not chosen:
            errors["patient_id"] = "Select a patient."
        if current_user.is_researcher:
            care_setting = form.get("care_setting")
            if care_setting not in CARE_SETTINGS:
                errors["care_setting"] = "Select a care setting."
        else:
            # Caregivers can't change it: it comes from the patient's record.
            care_setting = chosen["care_setting"] if chosen else None
            if chosen and not care_setting:
                errors["care_setting"] = "This patient has no care setting yet. Ask the research team to set it."
        # One mode for everyone, and the device's fixed 4 s dwell time.
        mode, dwell = "normal", DWELL_TIME_MS

        if not errors:
            try:
                if patient_choice == "new":
                    nxt = int(db.query("SELECT COALESCE(MAX(patient_id),0)+1 AS n FROM patients", one=True)["n"])
                    patient_id = db.execute(
                        """INSERT INTO patients (patient_code, full_name, care_setting, created_by)
                           VALUES (%s,%s,%s,%s)""",
                        (f"P-{nxt:03d}", form["patient_name"], care_setting, current_user.id),
                    )
                else:
                    patient_id = int(patient_choice)
                    if current_user.is_researcher and care_setting != chosen["care_setting"]:
                        # A researcher changing the setting updates the patient's record too.
                        db.execute("UPDATE patients SET care_setting=%s WHERE patient_id=%s",
                                   (care_setting, patient_id))
                profile = db.query(
                    """SELECT profile_id FROM calibration_profiles
                       WHERE patient_id=%s AND is_active=1 ORDER BY profile_id DESC LIMIT 1""",
                    (patient_id,), one=True,
                )
                session_id = db.execute(
                    """INSERT INTO sessions (session_code, patient_id, caregiver_id, care_setting,
                                             mode, dwell_time_ms, profile_id, pre_session_notes)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (next_codes(), patient_id, current_user.id, care_setting, mode, dwell,
                     profile["profile_id"] if profile else None, form.get("notes") or None),
                )
                metrics.queue_command(session_id, "start_session", current_user.id,
                                      {"session_id": session_id, "patient_id": patient_id})
                db.commit()
            except Exception as exc:  # most likely: another session started at the same moment
                db.rollback()
                current_app.logger.exception("start session failed")
                if "uq_one_active_session" in str(exc):
                    flash("Another caregiver just started a session.", "info")
                    return redirect(url_for(".home"))
                raise
            return redirect(url_for(".live", session_id=session_id))

    return render_template(
        "start.html", form=form, errors=errors, patients=patients,
        care_settings=CARE_SETTINGS, device=metrics.device_status(),
    )


@bp.route("/session/<int:session_id>/live")
@login_required
def live(session_id):
    metrics.end_if_app_closed()
    s = get_session(session_id)
    if s["status"] == "ended":
        return redirect(url_for(".summary", session_id=session_id))
    return render_template(
        "live.html", s=s,
        poll_ms=current_app.config["POLL_INTERVAL_MS"], benchmarks=BENCHMARKS,
    )


@bp.route("/session/<int:session_id>/summary")
@login_required
def summary(session_id):
    s = get_session(session_id)
    if s["status"] == "active":
        return redirect(url_for(".live", session_id=session_id))
    m = metrics.saved_metrics(session_id)
    return render_template(
        "summary.html", s=s, m=m, detailed=current_user.is_researcher, checks=metrics.evaluate(m), benchmarks=BENCHMARKS,
        selections=session_selections(session_id), alerts=session_alerts(session_id),
        alert_labels=ALERT_LABELS,
        just_ended=request.args.get("ended") == "1",
        app_closed=request.args.get("closed") == "1",
    )


@bp.route("/session/<int:session_id>/notes", methods=["POST"])
@login_required
def save_notes(session_id):
    get_session(session_id)
    notes = request.form.get("post_session_notes", "").strip()
    db.execute("UPDATE sessions SET post_session_notes=%s WHERE session_id=%s",
               (notes or None, session_id))
    db.commit()
    flash("Notes saved.", "success")
    return redirect(url_for(".summary", session_id=session_id))


@bp.route("/session/<int:session_id>/summary.pdf")
@login_required
def summary_pdf(session_id):
    from .pdf import build_summary_pdf

    s = get_session(session_id)
    if s["status"] != "ended":
        abort(400)
    anonymise = request.args.get("anon") == "1"
    m = metrics.saved_metrics(session_id)
    pdf = build_summary_pdf(s, m, metrics.evaluate(m), session_selections(session_id),
                            session_alerts(session_id), anonymise=anonymise,
                            detailed=current_user.is_researcher)
    name = f"EyeGlynt_{s['session_code'].strip()}{'_anon' if anonymise else ''}.pdf"
    return Response(pdf, mimetype="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@bp.route("/sessions")
@login_required
def history():
    if not current_user.is_researcher:
        return redirect(url_for(".home"))     # History is for researchers only
    patient_id = request.args.get("patient", type=int)
    conds, args = [], []
    if patient_id:
        conds.append("s.patient_id=%s")
        args.append(patient_id)
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    rows = db.query(
        f"""SELECT s.session_id, s.session_code, s.status, s.mode, s.started_at,
                   TIMESTAMPDIFF(SECOND, s.started_at, COALESCE(s.ended_at, NOW(3))) AS duration_s,
                   p.full_name AS patient_name, p.patient_code,
                   CONCAT(c.first_name,' ',c.last_name) AS caregiver,
                   m.total_selections, m.gaze_accuracy_pct, m.false_positive_pct,
                   m.avg_detection_latency_ms, m.recalibration_success_pct
            FROM sessions s
            JOIN patients p ON p.patient_id=s.patient_id
            JOIN caregivers c ON c.caregiver_id=s.caregiver_id
            LEFT JOIN session_metrics m ON m.session_id=s.session_id
            {where}
            ORDER BY s.started_at DESC LIMIT 200""",
        tuple(args),
    )
    for r in rows:
        r["checks"] = metrics.evaluate(r) if r["status"] == "ended" else {}
    patients = db.query("SELECT patient_id, full_name, patient_code FROM patients ORDER BY full_name")
    return render_template("history.html", rows=rows, patients=patients, patient_id=patient_id,
                           detailed=current_user.is_researcher)
