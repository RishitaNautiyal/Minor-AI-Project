"""Local user profile API backed by SQLite.

Firebase remains the authentication provider. This service stores only the
non-sensitive profile details needed by the OffGrid dashboard.
"""

import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS


app = Flask(__name__)
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "http://127.0.0.1:5500",
                "http://localhost:5500",
                "http://127.0.0.1:5002",
                "http://localhost:5002",
                "null",
            ]
        }
    },
)

DATABASE_PATH = Path(__file__).with_name("offgrid.db")
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


@contextmanager
def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def initialize_database():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                firebase_uid TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def clean_text(value, limit):
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())[:limit]


def validate_user_payload(payload):
    if not isinstance(payload, dict):
        return None, "Request body must be a JSON object."

    firebase_uid = clean_text(payload.get("firebase_uid"), 128)
    name = clean_text(payload.get("name"), 80)
    email = clean_text(payload.get("email"), 320).lower()

    if not firebase_uid:
        return None, "Firebase user ID is required."
    if not name:
        return None, "Name is required."
    if not email or not EMAIL_PATTERN.fullmatch(email):
        return None, "A valid email is required."

    return {
        "firebase_uid": firebase_uid,
        "name": name,
        "email": email,
    }, None


@app.post("/api/users")
def save_user():
    payload, error = validate_user_payload(request.get_json(silent=True))
    if error:
        return jsonify({"error": error}), 400

    timestamp = datetime.now(timezone.utc).isoformat()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO users (firebase_uid, name, email, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(firebase_uid) DO UPDATE SET
                name = excluded.name,
                email = excluded.email,
                updated_at = excluded.updated_at
            """,
            (
                payload["firebase_uid"],
                payload["name"],
                payload["email"],
                timestamp,
                timestamp,
            ),
        )
        row = connection.execute(
            """
            SELECT firebase_uid, name, email, created_at, updated_at
            FROM users
            WHERE firebase_uid = ?
            """,
            (payload["firebase_uid"],),
        ).fetchone()

    return jsonify({"user": dict(row)}), 201


@app.get("/api/health")
def health_check():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    initialize_database()
    app.run(host="127.0.0.1", port=5002, debug=False)
