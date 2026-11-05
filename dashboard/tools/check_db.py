#!/usr/bin/env python3
"""
EyeGlynt database connection checker.

Tests the two database logins exactly as the dashboard and the simulator use them
(reading .env), and shows WHICH MariaDB server Python is reaching.

    python tools/check_db.py
    python tools/check_db.py --root      # also try root (asks for its password)
"""
import getpass
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT, ".env")

try:
    import pymysql
    from dotenv import dotenv_values, load_dotenv
except ImportError:
    sys.exit("Missing packages. Activate the venv and run:  python -m pip install -r requirements.txt")


def show(value):
    """Reveal hidden problems (spaces, quotes) without printing the whole password."""
    if value is None:
        return "(not set)"
    flags = []
    if value != value.strip():
        flags.append("HAS LEADING/TRAILING SPACES")
    if value[:1] in "\"'" or value[-1:] in "\"'":
        flags.append("HAS QUOTES")
    masked = value[:3] + "*" * max(0, len(value) - 3)
    return f"{masked}  (length {len(value)}){'  <-- ' + ', '.join(flags) if flags else ''}"


def try_login(label, user, password, host, port):
    print(f"\n[{label}] user={user!r} host={host} port={port}")
    try:
        conn = pymysql.connect(host=host, port=port, user=user, password=password, connect_timeout=5)
        with conn.cursor() as cur:
            cur.execute("SELECT VERSION(), @@port, @@hostname, CURRENT_USER(), "
                        "(SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name='eyeglynt')")
            ver, sport, shost, cu, has_db = cur.fetchone()
        conn.close()
        print(f"  OK   server version {ver}, port {sport}, logged in as {cu}")
        print(f"       eyeglynt database {'found' if has_db else 'NOT FOUND on this server'}")
        return True
    except pymysql.err.OperationalError as e:
        code, msg = e.args[0], e.args[1] if len(e.args) > 1 else ""
        print(f"  FAIL ({code}) {msg}")
        if code == 2003:
            print("       -> No MariaDB is listening here. Is the MariaDB service running? Is DB_PORT right?")
        return False


def main():
    print("EyeGlynt DB check")
    print("=================")
    print(f".env file: {ENV_PATH}  ->  {'found' if os.path.exists(ENV_PATH) else 'NOT FOUND'}")
    if not os.path.exists(ENV_PATH):
        alt = [f for f in os.listdir(ROOT) if f.lower().startswith(".env")]
        print(f"  files starting with .env in the project folder: {alt or 'none'}")
        print("  -> Run:  copy .env.example .env   (Windows may have saved it as .env.txt)")

    raw = dotenv_values(ENV_PATH) if os.path.exists(ENV_PATH) else {}
    load_dotenv(ENV_PATH)
    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "3306"))
    web_user, web_pw = os.getenv("DB_USER", "eyeglynt_web"), os.getenv("DB_PASSWORD", "")
    eng_user, eng_pw = os.getenv("ENGINE_DB_USER", "eyeglynt_engine"), os.getenv("ENGINE_DB_PASSWORD", "")

    print(f"\nDB_HOST={host}  DB_PORT={port}")
    print(f"DB_USER={web_user}  DB_PASSWORD={show(web_pw)}")
    print(f"ENGINE_DB_USER={eng_user}  ENGINE_DB_PASSWORD={show(eng_pw)}")
    for k in ("DB_PASSWORD", "ENGINE_DB_PASSWORD"):
        if k in os.environ and raw.get(k) is not None and os.environ[k] != raw.get(k):
            print(f"  NOTE: {k} is also set as a Windows environment variable and overrides .env")

    web_ok = try_login("dashboard", web_user, web_pw, host, port)
    eng_ok = try_login("simulator/engine", eng_user, eng_pw, host, port)

    if "--root" in sys.argv:
        pw = getpass.getpass("\nroot password (typing is hidden): ")
        try_login("root", "root", pw, host, port)

    print("\nWhat this means")
    print("---------------")
    if web_ok and eng_ok:
        print("Both logins work. Restart run.py and the simulator.")
    elif web_ok and not eng_ok:
        print("Same server, but the ENGINE password/account is wrong. Re-run the CREATE/ALTER USER block")
        print("for eyeglynt_engine as root, and make sure ENGINE_DB_PASSWORD in .env matches exactly.")
    else:
        print("Neither app account works. Run:  python tools/check_db.py --root")
        print(" - If root ALSO fails with your correct root password, Python is reaching a DIFFERENT")
        print("   MariaDB/MySQL than HeidiSQL (e.g. XAMPP). In HeidiSQL run  SELECT @@port;  and put that")
        print("   number in .env as DB_PORT=...  (or stop XAMPP's MySQL).")
        print(" - If root works here, the accounts were never created on this server: run users.sql")
        print("   (or the CREATE/ALTER USER block) again while connected to THIS server.")


if __name__ == "__main__":
    main()
