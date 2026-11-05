"""EyeGlynt Caregiver Monitoring Dashboard — Flask application factory."""
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from flask import Flask, redirect, url_for
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

from . import db

csrf = CSRFProtect()
login_manager = LoginManager()


def create_app(test_config=None):
    load_dotenv()
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.getenv("SECRET_KEY", "dev-only-change-me"),
        DB_HOST=os.getenv("DB_HOST", "localhost"),
        DB_PORT=int(os.getenv("DB_PORT", "3306")),
        DB_NAME=os.getenv("DB_NAME", "eyeglynt"),
        DB_USER=os.getenv("DB_USER", "eyeglynt_web"),
        DB_PASSWORD=os.getenv("DB_PASSWORD", ""),
        DB_UNIX_SOCKET=os.getenv("DB_UNIX_SOCKET") or None,
        POLL_INTERVAL_MS=int(os.getenv("POLL_INTERVAL_MS", "2000")),
        DEVICE_OFFLINE_AFTER_S=int(os.getenv("DEVICE_OFFLINE_AFTER_S", "10")),
        RESEARCHER_ACCESS_CODE=os.getenv("RESEARCHER_ACCESS_CODE", ""),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    )
    if test_config:
        app.config.update(test_config)

    db.init_app(app)
    csrf.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message = None
    app.config["USE_SESSION_FOR_NEXT"] = True

    from .auth import bp as auth_bp, load_user  # noqa: F401  (registers user_loader)
    from .sessions import bp as sessions_bp
    from .api import bp as api_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(sessions_bp)
    app.register_blueprint(api_bp)

    register_filters(app)

    @app.errorhandler(403)
    def forbidden(_err):
        from flask import render_template
        return render_template("forbidden.html"), 403

    @app.route("/")
    def index():
        return redirect(url_for("sessions.home"))

    return app


def register_filters(app):
    @app.template_filter("hms")
    def hms(value):
        return value.strftime("%H:%M:%S") if isinstance(value, datetime) else "—"

    @app.template_filter("hm")
    def hm(value):
        return value.strftime("%I:%M %p").lstrip("0") if isinstance(value, datetime) else "—"

    @app.template_filter("date")
    def date(value):
        return value.strftime("%b %d, %Y") if isinstance(value, datetime) else "—"

    @app.template_filter("duration")
    def duration(seconds):
        if seconds is None:
            return "—"
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"

    @app.template_filter("pct")
    def pct(value):
        return "—" if value is None else f"{float(value):.0f}%"
