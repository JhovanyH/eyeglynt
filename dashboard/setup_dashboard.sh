#!/bin/bash
# =====================================================================
# EyeGlynt Caregiver Dashboard - one-time setup on the Raspberry Pi
#
# Run it from a Terminal on the Pi:
#     bash ~/eyeglynt/dashboard/setup_dashboard.sh
#
# What it does (it is safe to run again if something goes wrong):
#   1. installs MariaDB (the database) and Python's venv tool
#   2. creates the "eyeglynt" database and its tables
#   3. makes random passwords and writes them into dashboard/.env
#   4. installs the dashboard's Python packages into dashboard/venv
#   5. makes a security certificate (HTTPS) and makes the dashboard start
#      by itself every time the Pi turns on
#   6. lets the EyeGlynt app (main.py) talk to the database (pymysql)
# =====================================================================
set -e

# The folder this script is in (the dashboard folder)
DASH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DASH"
echo "Dashboard folder: $DASH"

echo
echo "== Step 1 of 6: installing MariaDB and Python venv (needs internet, takes a few minutes) =="
sudo apt update
sudo apt install -y mariadb-server python3-venv
sudo systemctl enable --now mariadb

echo
echo "== Step 2 of 6: creating the database =="
if sudo mariadb -e "USE eyeglynt; SELECT 1 FROM caregivers LIMIT 1;" >/dev/null 2>&1; then
    echo "The eyeglynt database is already there. Keeping it (no data is deleted)."
else
    sudo mariadb < database/schema.sql
    echo "Database and tables created."
fi
# Updates for a database made by an older version (safe to run every time).
sudo mariadb < database/migrations/005_one_mode_ignore_https.sql
sudo mariadb < database/migrations/006_dwell_options.sql
echo "Database is up to date."
sudo mariadb < database/seed.sql

echo
echo "== Step 3 of 6: passwords (.env file) =="
if [ ! -f .env ]; then
    cp .env.example .env
    SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    WEBPW=$(python3 -c "import secrets; print(secrets.token_hex(12))")
    ENGPW=$(python3 -c "import secrets; print(secrets.token_hex(12))")
    sed -i "s/^SECRET_KEY=.*/SECRET_KEY=$SECRET/" .env
    sed -i "s/^DB_PASSWORD=.*/DB_PASSWORD=$WEBPW/" .env
    sed -i "s/^ENGINE_DB_PASSWORD=.*/ENGINE_DB_PASSWORD=$ENGPW/" .env
    echo "Made a new .env with random passwords."
else
    echo "Keeping the .env that is already there."
fi
WEBPW=$(grep '^DB_PASSWORD=' .env | cut -d= -f2-)
ENGPW=$(grep '^ENGINE_DB_PASSWORD=' .env | cut -d= -f2-)
# Create the two database accounts with the passwords from .env
# (users.sql itself is not changed, so git pull keeps working).
sed -e "s/CHANGE_ME_web/$WEBPW/" -e "s/CHANGE_ME_engine/$ENGPW/" database/users.sql | sudo mariadb
sudo mariadb -e "ALTER USER 'eyeglynt_web'@'localhost' IDENTIFIED BY '$WEBPW'; ALTER USER 'eyeglynt_engine'@'localhost' IDENTIFIED BY '$ENGPW'; FLUSH PRIVILEGES;"
echo "Database accounts ready."

echo
echo "== Step 4 of 6: installing the dashboard's Python packages (takes a few minutes) =="
if [ ! -x venv/bin/python ]; then
    python3 -m venv venv
fi
venv/bin/pip install -q -r requirements.txt
venv/bin/python tools/check_db.py

echo
echo "== Step 5 of 6: HTTPS certificate, and start the dashboard automatically =="
# A self-made certificate, so phones talk to the dashboard over an encrypted
# connection (https). Browsers warn about it once ("not private"): choose
# Advanced, then Proceed / Continue.
if [ ! -f certs/cert.pem ] || [ ! -f certs/key.pem ]; then
    mkdir -p certs
    SAN="DNS:localhost,DNS:$(hostname).local,IP:127.0.0.1"
    for ip in $(hostname -I); do
        case "$ip" in *:*) ;; *) SAN="$SAN,IP:$ip" ;; esac
    done
    openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
        -keyout certs/key.pem -out certs/cert.pem \
        -subj "/CN=EyeGlynt Dashboard" -addext "subjectAltName=$SAN" 2>/dev/null
    chmod 600 certs/key.pem
    echo "Made a new HTTPS certificate."
else
    echo "Keeping the HTTPS certificate that is already there."
fi
# gthread (8 threads per worker): phone browsers open extra connections
# that send nothing for a while. With the plain "sync" workers, two such
# connections froze the whole dashboard for 30 s, so pages and the Live
# screen hung. --keep-alive 5 lets the Live screen's poll every 2 s reuse
# its connection instead of setting up HTTPS again each time.
sudo tee /etc/systemd/system/eyeglynt-dashboard.service >/dev/null <<SERVICE
[Unit]
Description=EyeGlynt Caregiver Dashboard
After=network.target mariadb.service

[Service]
User=$USER
WorkingDirectory=$DASH
ExecStart=$DASH/venv/bin/gunicorn -w 2 -k gthread --threads 8 --keep-alive 5 -b 0.0.0.0:5000 --certfile $DASH/certs/cert.pem --keyfile $DASH/certs/key.pem run:app
Restart=always

[Install]
WantedBy=multi-user.target
SERVICE
sudo systemctl daemon-reload
sudo systemctl enable eyeglynt-dashboard
sudo systemctl restart eyeglynt-dashboard

echo
echo "== Step 6 of 6: connect the EyeGlynt app to the dashboard =="
# main.py runs in ~/eyeglynt_env and needs pymysql to reach the database
# (see dashboard_link.py in the eyeglynt folder).
if [ -x "$HOME/eyeglynt_env/bin/pip" ]; then
    "$HOME/eyeglynt_env/bin/pip" install -q pymysql
    echo "The EyeGlynt app can now reach the dashboard."
else
    echo "No ~/eyeglynt_env found. Later, run: <your venv>/bin/pip install pymysql"
fi
sleep 3

echo
if curl -sk -o /dev/null https://localhost:5000/; then
    echo "DONE. The dashboard is running."
else
    echo "The dashboard did not answer yet. Send a screenshot of this window to Claude."
fi
echo "Open it on a phone or laptop on the SAME Wi-Fi as the Pi:"
for ip in $(hostname -I); do
    case "$ip" in *:*) ;; *) echo "    https://$ip:5000" ;; esac
done
echo "    or https://$(hostname).local:5000"
echo "    or, when the Pi is its own Wi-Fi hotspot: https://10.42.0.1:5000"
echo "(https, not http. The first time, the browser warns that the connection"
echo " is not private: tap Advanced, then Proceed. It is your own Pi.)"
