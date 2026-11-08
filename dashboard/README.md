# EyeGlynt Caregiver Monitoring Dashboard

This is the Flask and MariaDB web dashboard for **EyeGlynt**. It runs on the Raspberry Pi 4, and caregivers open it from any phone or laptop on the same Wi-Fi. It works with no internet connection.

| Screen | Thesis figure | URL |
|---|---|---|
| Register / Log in | Figs. 32–33 | `/register`, `/login` |
| Start Session | Fig. 34 | `/session/start` |
| Live Monitoring | Fig. 35 | `/session/<id>/live` |
| Session Summary and PDF export | Fig. 36 | `/session/<id>/summary` |
| Session History *(added)* | — | `/sessions` |
| Research access *(added)* | — | Researcher role + access code at `/register` |

---

## 1. Install on the Raspberry Pi (Raspberry Pi OS Bookworm)

```bash
sudo apt update
sudo apt install -y mariadb-server python3-venv
sudo mysql_secure_installation          # set a root password and answer Y to the rest

git clone <your-repo> eyeglynt-dashboard   # or copy this folder to the Pi
cd eyeglynt-dashboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Database

1. Edit `database/users.sql` and replace both `CHANGE_ME_*` passwords.
2. Run the three SQL files:

```bash
sudo mariadb < database/schema.sql
sudo mariadb < database/seed.sql
sudo mariadb < database/users.sql
```

3. Copy the environment file and fill it in:

```bash
cp .env.example .env
nano .env        # set SECRET_KEY and the two DB passwords you chose
```

To generate a `SECRET_KEY`, run `python3 -c "import secrets; print(secrets.token_hex(32))"`.

### Run it

```bash
python run.py                      # development
# or
gunicorn -w 2 -b 0.0.0.0:5000 run:app
```

To open the dashboard from a phone on the same Wi-Fi:

1. Find the Pi's address by running `hostname -I` on the Pi.
2. On the phone, open `http://<pi-ip>:5000`.

---

## Account types: caregiver vs researcher

| | Caregiver (default) | Researcher |
|---|---|---|
| Who | family, nurses, therapists | the thesis team |
| Session mode | Normal only | Normal **and** Test |
| Live screen | activity log, plain-language alerts, controls | + metric tiles, test panel, technical alert details |
| Summary and PDF | patient info, most-used interface, total selections, activity log, notes | full report: all 6 metrics vs Table 7 benchmarks, alerts, anonymised PDF |
| History | Normal sessions only, without metrics | all sessions, with metrics |

**To register as a researcher:**

1. Choose the **Researcher** role on the Register page.
2. Enter the access code from `.env` (`RESEARCHER_ACCESS_CODE`, default `eyeglynt-research`). Change this code before the Pi is used with participants.

**Updating an existing database?** Run `database/migrations/002_dashboard_update.sql` once (in HeidiSQL: open it in a Query tab and press F9). It adds researcher accounts, updates the role and care-setting lists, and removes the patient condition. It is safe to run twice.

**Older instructions (migration 001 only):** Run the migration once, then promote your own account:

```
mariadb -u root -p eyeglynt < database/migrations/001_account_types.sql
mariadb -u root -p eyeglynt -e "UPDATE caregivers SET account_type='researcher' WHERE username='YOUR_USERNAME'"
```

Also add `RESEARCHER_ACCESS_CODE=eyeglynt-research` to your `.env`.

---

## 2. Try it without the hardware (simulator)

The simulator acts as the eye-tracking device. It writes selections, alerts and recalibrations into the database, and it responds to the dashboard's Recalibrate and test buttons.

```bash
python tools/simulator.py --speed 3
```

Then follow these steps in the dashboard:

1. Register and log in.
2. Start a session. Activity should appear within a few seconds.

For a repeatable demo, add `--seed 7`. Run `python tools/simulator.py --help` to see the accuracy and recalibration-rate options.

---

## 3. Connect the real gaze engine

The gaze engine and the dashboard **never call each other directly**. Both use the database through `engine_bridge/eyeglynt_db.py`, so if the dashboard goes down, the patient can still communicate.

```python
from engine_bridge.eyeglynt_db import EyeGlyntDB
db = EyeGlyntDB()

# every ~2 s
db.heartbeat("tracking", fps=fps, lux=lux)

# on every dwell-time selection (after TTS fires)
db.log_selection("quickfire", "phrase", "qf:needs:water", "I want water",
                 category="Basic Needs", dwell_ms=1520, gaze_xy=(x, y),
                 target_element_id=current_test_target)   # None outside accuracy tests

# when lighting changes / gaze lock is lost  (event_at = time of the first bad frame)
aid = db.raise_alert("lighting_shift", "Ambient light changed", event_at=t_first_bad_frame)
rid = db.start_recalibration("auto_lighting", aid)
...
db.finish_recalibration(rid, success=True)

# every ~1 s: obey the dashboard
for cmd in db.poll_commands():
    if cmd["command"] == "recalibrate":
        ...
    db.complete_command(cmd["command_id"])
```

`log_selection()` only records selections while a session is running. Outside a session, the device works normally and logs nothing.

---

## 4. How the metrics are computed

The metrics come from the `v_session_metrics` view in `database/schema.sql`. When a session ends, they are frozen into `session_metrics`.

Since 8 November 2026 there is one session mode for everyone (Test Mode was removed; old test sessions still show their results). Researchers see every metric; caregivers see the simple summary.

| Metric | How it is computed |
|---|---|
| Gaze Selection Accuracy | selections not marked Unintended ÷ all counted selections |
| False Positive Rate | selections marked Unintended ÷ all counted selections |
| Detection Latency | average of `detected_at − event_at` for lighting-shift and gaze-lock-loss alerts |
| Recalibration Success | successful ÷ finished recalibrations (a calibration stopped from the dashboard is not counted) |

Selections marked **Ignore** in the activity log are left out of every metric. The dwell time is fixed at 4 s (`DWELL_TIME` in main.py). When the EyeGlynt app is closed, the dashboard ends the running session by itself. Passwords are stored as typed (team decision), in the `password` column.

The benchmarks (from thesis Table 7) are set in `app/constants.py`. Tiles turn green when a metric meets its benchmark and red when it falls below.

---

## 5. Start automatically on boot (systemd)

Create `/etc/systemd/system/eyeglynt-dashboard.service`:

```ini
[Unit]
Description=EyeGlynt Caregiver Dashboard
After=network.target mariadb.service

[Service]
User=pi
WorkingDirectory=/home/pi/eyeglynt-dashboard
ExecStart=/home/pi/eyeglynt-dashboard/venv/bin/gunicorn -w 2 -b 0.0.0.0:5000 run:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Then enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now eyeglynt-dashboard
```

---

## Project layout

```
app/                 Flask app (auth, sessions, api, metrics, pdf, templates, static)
database/            schema.sql · seed.sql · users.sql
engine_bridge/       eyeglynt_db.py — the only file the gaze engine needs
tools/simulator.py   fake device for demos and testing
tools/check_db.py    diagnoses database login problems (.env vs MariaDB accounts)
PLAN.md              design decisions and the two thesis issues to raise
```
