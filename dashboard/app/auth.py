"""Account registration and login (thesis Figs. 32–33)."""
import hmac
import re
from functools import wraps

from flask import (Blueprint, abort, current_app, flash, redirect, render_template, request,
                   session, url_for)
from flask_login import UserMixin, current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

from . import db, login_manager
from .constants import ROLES

bp = Blueprint("auth", __name__)

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.]{3,40}$")
PHONE_RE = re.compile(r"^(\+63|0)9\d{9}$")  # PH mobile: 09XXXXXXXXX or +639XXXXXXXXX


class Caregiver(UserMixin):
    def __init__(self, row):
        self.id = row["caregiver_id"]
        self.first_name = row["first_name"]
        self.last_name = row["last_name"]
        self.role = row["role"]
        self.username = row["username"]
        self.account_type = row.get("account_type", "caregiver")

    @property
    def is_researcher(self):
        return self.account_type == "researcher"

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    @property
    def initials(self):
        return (self.first_name[:1] + self.last_name[:1]).upper()


def researcher_required(view):
    """Only researcher accounts may run Test Mode or see full metrics."""
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if not current_user.is_researcher:
            abort(403)
        return view(*args, **kwargs)
    return wrapped


@login_manager.user_loader
def load_user(user_id):
    row = db.query(
        "SELECT * FROM caregivers WHERE caregiver_id=%s AND is_active=1", (user_id,), one=True
    )
    return Caregiver(row) if row else None


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("sessions.home"))
    form = {}
    if request.method == "POST":
        form = request.form
        username = form.get("username", "").strip()
        password = form.get("password", "")
        row = db.query(
            "SELECT * FROM caregivers WHERE username=%s AND is_active=1", (username,), one=True
        )
        if row and check_password_hash(row["password_hash"], password):
            login_user(Caregiver(row), remember=bool(form.get("remember")))
            db.execute("UPDATE caregivers SET last_login_at=NOW() WHERE caregiver_id=%s",
                       (row["caregiver_id"],))
            db.commit()
            nxt = request.args.get("next") or session.pop("next", None)
            return redirect(nxt if nxt and nxt.startswith("/") else url_for("sessions.home"))
        flash("Incorrect username or password.", "error")
    return render_template("auth.html", tab="login", form=form, roles=ROLES, errors={})


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("sessions.home"))
    form, errors = {}, {}
    if request.method == "POST":
        form = {k: v.strip() for k, v in request.form.items() if k not in ("password", "confirm")}
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        for field, label in [("first_name", "First name"), ("last_name", "Last name"),
                             ("role", "Role"), ("contact_number", "Contact number"),
                             ("username", "Username")]:
            if not form.get(field):
                errors[field] = f"{label} is required."
        if form.get("role") and form["role"] not in ROLES:
            errors["role"] = "Choose a role from the list."
        phone = re.sub(r"[\s-]", "", form.get("contact_number", ""))
        if phone and not PHONE_RE.match(phone):
            errors["contact_number"] = "Use a PH mobile number, e.g. 09171234567."
        if form.get("username") and not USERNAME_RE.match(form["username"]):
            errors["username"] = "3–40 letters, numbers, dots or underscores."
        if len(password) < 8:
            errors["password"] = "Use at least 8 characters."
        elif password != confirm:
            errors["confirm"] = "Passwords do not match."
        account_type = "caregiver"
        if form.get("role") == "Researcher":
            expected = current_app.config["RESEARCHER_ACCESS_CODE"]
            given = request.form.get("access_code", "").strip()
            if not expected:
                errors["access_code"] = "Researcher registration is turned off on this device."
            elif not hmac.compare_digest(given.encode(), expected.encode()):
                errors["access_code"] = "Incorrect researcher access code."
            else:
                account_type = "researcher"
        if not errors.get("username") and form.get("username") and db.query(
            "SELECT 1 FROM caregivers WHERE username=%s", (form["username"],), one=True
        ):
            errors["username"] = "That username is already taken."

        if not errors:
            db.execute(
                """INSERT INTO caregivers
                   (first_name, last_name, role, contact_number, username, password_hash, account_type)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                (form["first_name"], form["last_name"], form["role"], phone,
                 form["username"], generate_password_hash(password), account_type),
            )
            db.commit()
            flash("Account created. You can now log in.", "success")
            return redirect(url_for("auth.login"))
    return render_template("auth.html", tab="register", form=form, roles=ROLES, errors=errors)


@bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("auth.login"))
