# EyeGlynt Caregiver Monitoring Dashboard — Build Plan

Flask + MariaDB 10.11, hosted on the Raspberry Pi 4, served to phones/laptops on the same LAN. Fully offline (no CDNs).

## 1. Architecture

```
 ┌──────────────────────── Raspberry Pi 4 ────────────────────────┐
 │                                                                │
 │  Gaze Engine (Python)          MariaDB "eyeglynt"     Flask    │
 │  OpenCV/MediaPipe/TFLite  ──►  selections        ◄──  app      │──► Caregiver browser
 │  dwell-time, TTS               system_alerts           :5000   │    (phone/laptop, LAN)
 │                                recalibrations                  │
 │                           ◄──  device_commands   ◄──  (Recalibrate / End)
 │  heartbeat every 2 s      ──►  device_status                   │
 └────────────────────────────────────────────────────────────────┘
```

- **The two programs never talk directly; the database is the only link between them.** If the dashboard crashes, the user keeps communicating, which matches thesis §1.3.
- **Engine → DB:** it inserts a row on every dwell activation, alert and recalibration.
- **Dashboard → engine:** the dashboard writes to `device_commands`, and the engine polls that table about every second.
- **Dashboard → browser:** the browser polls `GET /api/sessions/<id>/updates?after_sel=&after_alert=` every 2 s and receives only the new rows.

## 2. Screens → tables

| Screen (thesis fig.) | Reads | Writes |
|---|---|---|
| Register (32) | — | `caregivers` |
| Login (33) | `caregivers` | `caregivers.last_login_at` |
| Start Session (34) | `patients`, `calibration_profiles`, `device_status` | `patients` (if new), `sessions`, `device_commands(start_session)` |
| Live Monitoring (35) | `v_session_metrics`, `selections`, `system_alerts`, `device_status` | `selections.intent_status` (flag), `system_alerts.acknowledged_*`, `device_commands` |
| Session Summary (36) | `session_metrics`, `selections`, `system_alerts`, `recalibrations` | `sessions.post_session_notes` |
| *Added:* Session History | `sessions`, `session_metrics` | — |
| *Added:* Phrase Library | `phrase_categories`, `phrases` | same |

The Start Session form gains a **Condition** field, because the Live screen displays it but the thesis form never collects it. It also gains a **Mode** toggle (Normal / Test). The caregiver name and role are filled in automatically from the logged-in account.

## 3. Metric definitions (as implemented in `v_session_metrics`)

Verified against the thesis worked examples: 27/30 trials gives **90%** accuracy, latencies of 380/410/470/490/430 ms average **436 ms**, and 9/10 recalibrations gives **90%**.

| Metric | Test Mode (formal evaluation) | Normal Mode (daily use) |
|---|---|---|
| Gaze Selection Accuracy | correct ÷ trials in `accuracy` blocks | (all − flagged unintended) ÷ all selections |
| False Positive Rate | (free-look activations + flagged) ÷ all activations | flagged unintended ÷ all activations |
| Detection Latency | AVG(`detected_at − event_at`) for lighting-shift and gaze-lock-loss alerts | same |
| Recalibration Success | successful ÷ completed recalibrations (rows still running are excluded) | same |
| Most-Used Interface | larger of the keyboard and Quick-Fire counts | same |

At **End Session**, the view is copied into `session_metrics` so the Summary and PDF don't change if the log is edited later.

### ⚠ Two thesis issues to raise with your adviser

1. **False Positive procedure (p. ~83).** During the 2-minute free-look period every activation is unintended by definition. Unintended ÷ total activations within that period is therefore always 100% (or 0/0 when nothing fires). The schema computes FPR over the whole test session instead, so accuracy-block activations form part of the denominator. An alternative is to report free-look activations **per minute**. Either way, the wording in Chapter III should be updated.
2. **Normal-mode accuracy.** The device cannot know the user's intent during daily use. In Normal Mode a selection counts as intended unless the caregiver flags it, so accuracy is effectively 100 − FPR. The thesis benchmarks in Table 7 should come from **Test Mode** sessions only. The Summary screen states which basis was used.

## 4. Flask project structure

```
eyeglynt-dashboard/
├── app/
│   ├── __init__.py          # create_app(), config, blueprints
│   ├── db.py                # PyMySQL connection per request
│   ├── auth.py              # /register, /login, /logout  (werkzeug password hashing, Flask-Login)
│   ├── sessions.py          # /session/start, /session/<id>/live, /session/<id>/summary, history
│   ├── api.py               # JSON: updates polling, flag selection, ack alert, recalibrate, end
│   ├── phrases.py           # phrase library CRUD
│   ├── metrics.py           # snapshot + benchmark pass/fail (Table 7 thresholds)
│   ├── pdf.py               # Summary → PDF (ReportLab, pure Python, works on Pi)
│   ├── templates/           # Jinja2: base, auth, start, live, summary, history, phrases
│   └── static/              # css/, js/live.js (polling), fonts — all local, no CDN
├── database/
│   ├── schema.sql   seed.sql   users.sql      ✅ done
├── engine_bridge/
│   └── eyeglynt_db.py       # helper the gaze engine imports: log_selection(), raise_alert(), poll_commands(), heartbeat()
├── tools/
│   └── simulator.py         # fake gaze engine for demos/testing (uses engine_bridge)
├── .env.example             # DB creds, SECRET_KEY
├── requirements.txt         # Flask, Flask-Login, Flask-WTF, PyMySQL, reportlab, python-dotenv
└── run.py                   # dev server; production via gunicorn + systemd on the Pi
```

**Libraries:**

- **PyMySQL** (pure Python) instead of mysqlclient, so nothing needs compiling on the Pi.
- **Flask-WTF** for CSRF protection on forms.
- **ReportLab** instead of WeasyPrint for the PDF, because it's lighter on the Pi.

## 5. JSON API (dashboard ↔ browser)

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/sessions/<id>/updates?after_sel=&after_alert=` | new selections and alerts, plus live metrics and device status |
| POST | `/api/selections/<id>/flag` | `{intent: "intended"\|"unintended"}` |
| POST | `/api/alerts/<id>/ack` | acknowledge alert |
| POST | `/api/sessions/<id>/recalibrate` | queues a `device_commands` row |
| POST | `/api/sessions/<id>/test-block` | start or end an accuracy / free-look block (Test Mode) |
| POST | `/api/sessions/<id>/end` | end the session, snapshot the metrics, redirect to Summary |
| GET | `/session/<id>/summary.pdf` | PDF export |

## 6. Security and privacy (Ethical Considerations)

- **Passwords:** hashed with `werkzeug.security`. Logins are session-cookie based, and all forms are CSRF-protected.
- **Database accounts:** two least-privilege MySQL users (`users.sql`). The engine cannot read caregiver accounts.
- **Patient anonymity:** PDFs and thesis tables can use `patient_code` (P-001) in place of the patient's name, with a toggle on export.
- **Network:** LAN only. Bind to the Pi's LAN IP. Don't port-forward.

## 7. Roles

- **Caregivers** (default) use Normal Mode only. They see a simplified UI that matches the thesis mockup: no percentages, and plain-language alerts.
- **Researchers** register with an access code. They can run Test Mode and see the full metrics report.

The rules are enforced on the server, not just hidden in the page: Test Mode pages, PDFs and the test-block API return 403 for caregivers, and a crafted `mode=test` request is saved as Normal. This was verified with a two-role browser test.

## 8. Status

All build steps are done and were tested end to end against MariaDB 10.11, with the simulator and a headless browser at phone and desktop widths:

- Core: app factory, DB layer, auth
- Session screens: Start, Live, Summary, History
- Live polling: flagging, alert acknowledgement, device-online badge, dwell change
- Engine bridge and simulator
- Test Mode: accuracy blocks close automatically at exactly 30 trials, plus timed free-look blocks
- PDF export (full or anonymised) and the phrase library
- Pi deployment notes in README.md

**Next:**

1. Wire `engine_bridge` into the real gaze engine.
2. Have the device load Quick-Fire phrases from the `phrases` table.
3. Fill the phrase library from the caregiver interviews.
