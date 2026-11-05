"""Thin PyMySQL layer: one connection per request, dict rows, explicit commits."""
import pymysql
from flask import current_app, g


def get_db():
    if "db" not in g:
        cfg = current_app.config
        g.db = pymysql.connect(
            host=cfg["DB_HOST"],
            port=cfg["DB_PORT"],
            unix_socket=cfg["DB_UNIX_SOCKET"],
            user=cfg["DB_USER"],
            password=cfg["DB_PASSWORD"],
            database=cfg["DB_NAME"],
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=False,
        )
    return g.db


def close_db(_exc=None):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_app(app):
    app.teardown_appcontext(close_db)


def query(sql, args=None, one=False):
    with get_db().cursor() as cur:
        cur.execute(sql, args or ())
        return cur.fetchone() if one else cur.fetchall()


def execute(sql, args=None):
    """Run INSERT/UPDATE/DELETE. Returns lastrowid. Caller must commit()."""
    with get_db().cursor() as cur:
        cur.execute(sql, args or ())
        return cur.lastrowid


def commit():
    get_db().commit()


def rollback():
    get_db().rollback()
