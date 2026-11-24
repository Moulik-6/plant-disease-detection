"""
SQLite Database interface for diagnosis history.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional


def get_connection(db_path: str) -> sqlite3.Connection:
    """Creates a connection with row factory enabled."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    """Initializes the SQLite schema if not already present."""
    db_file = Path(db_path).resolve()
    db_file.parent.mkdir(parents=True, exist_ok=True)

    with get_connection(str(db_file)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                model_name TEXT NOT NULL,
                image_path TEXT
            )
            """
        )
        conn.commit()


def save_prediction(
    db_path: str,
    predicted_class: str,
    confidence: float,
    status: str,
    model_name: str,
    image_path: Optional[str] = None,
) -> int:
    """Inserts a new diagnosis entry and returns its generated ID."""
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO history (created_at, predicted_class, confidence, status, model_name, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (created_at, predicted_class, float(confidence), status, model_name, image_path),
        )
        conn.commit()
        return cursor.lastrowid


def get_history(db_path: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves up to limit history entries ordered by created_at DESC."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, created_at, predicted_class, confidence, status, model_name, image_path
            FROM history
            ORDER BY id DESC
            LIMIT ?
            """,
            (max(1, min(limit, 500)),),
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def clear_history(db_path: str, upload_dir: Optional[str] = None) -> int:
    """Clears all history entries and optionally removes stored image files."""
    init_db(db_path)
    count = 0
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM history")
        count = cursor.fetchone()[0]
        cursor.execute("DELETE FROM history")
        conn.commit()

    if upload_dir and Path(upload_dir).exists():
        for f in Path(upload_dir).glob("*"):
            if f.is_file():
                try:
                    f.unlink()
                except OSError:
                    pass

    return count
