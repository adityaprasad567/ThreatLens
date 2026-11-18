"""SQLite storage for scan history (a local file - nothing leaves your machine)."""

import sqlite3
from contextlib import closing
from datetime import datetime

import pandas as pd

from .config import DB_PATH

HISTORY_COLUMNS = ["id", "url", "prediction", "risk_level", "confidence", "phishing_probability", "scanned_at"]


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """Create the table if it does not exist yet."""
    with closing(_connect()) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                prediction TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                confidence REAL NOT NULL,
                phishing_probability REAL NOT NULL,
                scanned_at TEXT NOT NULL
            )
            """
        )


def save_scan(result: dict) -> None:
    """Store one successful prediction result."""
    init_db()
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO scans (url, prediction, risk_level, confidence, phishing_probability, scanned_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                result["url"],
                result["prediction"],
                result["risk_level"],
                float(result["confidence"]),
                float(result["phishing_probability"]),
                datetime.now().isoformat(sep=" ", timespec="seconds"),
            ),
        )


def get_history(limit: int | None = None) -> pd.DataFrame:
    """All scans, newest first, as a DataFrame."""
    init_db()
    query = "SELECT id, url, prediction, risk_level, confidence, phishing_probability, scanned_at FROM scans ORDER BY id DESC"
    params: tuple = ()
    if limit:
        query += " LIMIT ?"
        params = (limit,)
    with closing(_connect()) as conn:
        return pd.read_sql_query(query, conn, params=params)


def clear_history() -> int:
    """Delete every scan. Returns how many rows were removed."""
    init_db()
    with closing(_connect()) as conn, conn:
        cursor = conn.execute("DELETE FROM scans")
        return cursor.rowcount


def count_scans() -> int:
    init_db()
    with closing(_connect()) as conn:
        return conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
