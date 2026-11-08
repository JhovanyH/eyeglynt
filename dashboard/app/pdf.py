"""Session Summary → PDF (ReportLab, pure Python — runs fine on the Pi)."""
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .constants import ALERT_LABELS, BENCHMARKS

INK = colors.HexColor("#1F1B2E")
MUTED = colors.HexColor("#6B6578")
LINE = colors.HexColor("#E4DED2")
BRAND = colors.HexColor("#6D28D9")
PASS = colors.HexColor("#15803D")
FAIL = colors.HexColor("#B91C1C")


def _fmt(v, unit):
    if v is None:
        return "—"
    return f"{float(v):.0f} ms" if unit == "ms" else f"{float(v):.1f}%"


def _duration(sec):
    sec = int(sec or 0)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def build_summary_pdf(s, m, checks, selections, alerts, anonymise=False, detailed=True):
    """detailed=False → caregiver report: patient info, totals, activity log, notes (no metrics/alerts)."""
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"EyeGlynt Session {s['session_code'].strip()}")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=16, textColor=INK, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, textColor=BRAND,
                        spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=9, textColor=INK, leading=12)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=MUTED)

    def table(data, widths, header=True, zebra=True):
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        style = [
            ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
            ("TEXTCOLOR", (0, 0), (-1, -1), INK),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        if header:
            style += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
                      ("TEXTCOLOR", (0, 0), (-1, 0), MUTED)]
        t.setStyle(TableStyle(style))
        return t

    patient = s["patient_code"] if anonymise else f"{s['patient_name']} ({s['patient_code']})"
    if not detailed:
        return _caregiver_pdf(doc, s, m, selections, h1, h2, body, small, table)
    story = [
        Paragraph("EyeGlynt — Session Summary", h1),
        Paragraph(f"Session {s['session_code'].strip()} · "
                  f"{' · Test Mode (old)' if s['mode'] == 'test' else ''}", small),
        Spacer(1, 6),
        Paragraph("Patient &amp; session", h2),
        table([
            ["Patient", Paragraph(escape(patient), body), "Date", s["started_at"].strftime("%B %d, %Y")],
            ["Care setting", s["care_setting"], "Start – End",
             f"{s['started_at']:%I:%M %p} – {s['ended_at']:%I:%M %p}"],
            ["Duration", _duration(s["duration_s"]), "", ""],
            ["Caregiver", Paragraph(escape(f"{s['cg_first']} {s['cg_last']} ({s['cg_role']})"), body),
             "Dwell time", f"{s['dwell_time_ms'] / 1000:.1f} s"],
        ], [26 * mm, 72 * mm, 26 * mm, 50 * mm], header=False),
    ]

    # Metrics vs thesis Table 7
    rows = [["Metric", "Result", "Benchmark", "Status"]]
    for key, rule in BENCHMARKS.items():
        status = {"pass": "Meets", "fail": "Below target", "na": "No data"}[checks.get(key, "na")]
        rows.append([rule["label"], _fmt(m.get(key) if m else None, rule["unit"]), rule["target"], status])
    most = (m or {}).get("most_used_interface")
    rows.append(["Total selections", str((m or {}).get("total_selections", 0)), "—", ""])
    rows.append(["Most-used interface",
                 {"keyboard": "Keyboard", "quickfire": "Quick-Fire"}.get(most, "—"), "—", ""])
    mt = table(rows, [52 * mm, 34 * mm, 40 * mm, 48 * mm])
    for i, key in enumerate(BENCHMARKS, start=1):
        c = {"pass": PASS, "fail": FAIL}.get(checks.get(key))
        if c:
            mt.setStyle(TableStyle([("TEXTCOLOR", (3, i), (3, i), c),
                                    ("FONT", (3, i), (3, i), "Helvetica-Bold", 8.5)]))
    story += [Paragraph("Performance metrics", h2), mt]
    basis = ("Accuracy computed from Test Mode trials (correct ÷ trials)."
             if s["mode"] == "test" else
             "Accuracy counts selections as intended unless they were marked Unintended. "
             "Ignored selections are left out of every result.")
    story.append(Paragraph(basis, small))

    # Activity log
    story.append(Paragraph(f"Activity log ({len(selections)} selections)", h2))
    log = [["Time", "Interface", "Selection", "Note"]]
    for r in selections:
        note = ""
        if r["intent_status"] == "unintended":
            note = "Flagged unintended"
        elif r["intent_status"] == "ignored":
            note = "Ignored (not counted)"
        elif r["selection_type"] == "speak":
            note = "Spoken"
        elif r["target_element_id"] is not None:
            note = "Correct" if r["is_correct"] else f"Missed target {r['target_element_id']}"
        log.append([r["selected_at"].strftime("%H:%M:%S"),
                    "Quick-Fire" if r["interface"] == "quickfire" else "Keyboard",
                    Paragraph(escape(r["output_text"]), body), note])
    story.append(table(log, [20 * mm, 24 * mm, 86 * mm, 44 * mm]) if selections
                 else Paragraph("No selections were recorded.", small))

    # Alerts
    story.append(Paragraph(f"System alerts ({len(alerts)})", h2))
    al = [["Time", "Alert", "Detail", "Latency"]]
    for a in alerts:
        al.append([a["detected_at"].strftime("%H:%M:%S"), ALERT_LABELS.get(a["alert_type"], a["alert_type"]),
                   Paragraph(escape(a["message"]), body),
                   f"{a['detection_latency_ms']} ms" if a["detection_latency_ms"] is not None else ""])
    story.append(table(al, [20 * mm, 44 * mm, 86 * mm, 24 * mm]) if alerts
                 else Paragraph("No alerts were raised.", small))

    # Notes
    story.append(Paragraph("Caregiver notes", h2))
    if s.get("pre_session_notes"):
        story.append(Paragraph(f"<b>Before session:</b> {escape(s['pre_session_notes'])}", body))
    story.append(Paragraph(f"<b>After session:</b> {escape(s.get('post_session_notes') or '—')}", body))
    story += [Spacer(1, 10),
              Paragraph("Generated by the EyeGlynt Caregiver Monitoring Dashboard. "
                        "Data stored locally on the device.", small)]

    doc.build(story)
    return buf.getvalue()


def _caregiver_pdf(doc, s, m, selections, h1, h2, body, small, table):
    """Simple report for families and caregivers — mirrors the caregiver Summary screen."""
    most = {"keyboard": "Keyboard", "quickfire": "Quick-Fire"}.get((m or {}).get("most_used_interface"), "—")
    story = [
        Paragraph("EyeGlynt — Session Summary", h1),
        Paragraph(s["started_at"].strftime("%B %d, %Y"), small),
        Spacer(1, 6),
        Paragraph("Patient information", h2),
        table([
            ["Name", Paragraph(escape(s["patient_name"]), body)],
            ["Date", s["started_at"].strftime("%B %d, %Y")],
            ["Start – End", f"{s['started_at']:%I:%M %p} – {s['ended_at']:%I:%M %p}"],
            ["Duration", _duration(s["duration_s"])],
            ["Total selections", str((m or {}).get("total_selections", 0))],
            ["Interface most used", most],
            ["Unintended", str(sum(r["intent_status"] == "unintended" for r in selections))],
            ["Ignored (not counted)", str(sum(r["intent_status"] == "ignored" for r in selections))],
        ], [45 * mm, 129 * mm], header=False),
        Paragraph(f"Activity log ({len(selections)} selections)", h2),
    ]
    if selections:
        rows = [["Time", "Interface", "What was said", "Note"]]
        notes = {"unintended": "Unintended", "ignored": "Ignored"}
        for r in selections:
            note = notes.get(r["intent_status"]) or ("Spoken" if r["selection_type"] == "speak" else "")
            rows.append([r["selected_at"].strftime("%I:%M:%S %p"),
                         "Quick-Fire" if r["interface"] == "quickfire" else "Keyboard",
                         Paragraph(escape(r["output_text"]), body), note])
        story.append(table(rows, [28 * mm, 28 * mm, 92 * mm, 26 * mm]))
    else:
        story.append(Paragraph("No selections were recorded.", small))
    story.append(Paragraph("Caregiver notes", h2))
    if s.get("pre_session_notes"):
        story.append(Paragraph(f"<b>Before session:</b> {escape(s['pre_session_notes'])}", body))
    story.append(Paragraph(f"<b>After session:</b> {escape(s.get('post_session_notes') or '—')}", body))
    story += [Spacer(1, 10), Paragraph("Generated by the EyeGlynt Caregiver Dashboard.", small)]
    buf = doc.filename
    doc.build(story)
    return buf.getvalue()
