# src/db/database.py
"""SQLite database module for SocrAItes.

Provides schema initialisation and CRUD helpers for all persistent data:
  * sessions       – per‑user conversation sessions
  * messages       – individual chat turns within a session
  * weaknesses     – concepts the learner struggled with
  * schedules      – spaced‑repetition / review calendar entries
  * documents      – metadata for uploaded lecture PDFs
  * reports        – generated weekly diagnosis reports

Uses only the built‑in ``sqlite3`` module for zero‑dependency operation.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DB_PATH = os.getenv("SOCRAITES_DB_PATH", os.path.join(DB_DIR, "socraites.db"))


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with WAL mode and foreign keys enabled."""
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _now() -> str:
    """UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- 1. Sessions: groups a sequence of messages into a conversation.
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,          -- UUID
    user_id       TEXT NOT NULL DEFAULT 'default',
    title         TEXT,
    socratic_mode TEXT NOT NULL DEFAULT 'standard',  -- light | standard | deep
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

-- 2. Messages: every chat turn, linked to a session.
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT    NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content     TEXT    NOT NULL,
    metadata    TEXT,                         -- JSON blob (tool calls, depth info, etc.)
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

-- 3. Weaknesses: concepts the learner found difficult.
CREATE TABLE IF NOT EXISTS weaknesses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT    NOT NULL DEFAULT 'default',
    session_id  TEXT    REFERENCES sessions(id) ON DELETE SET NULL,
    concept     TEXT    NOT NULL,
    details     TEXT,
    severity    INTEGER NOT NULL DEFAULT 1 CHECK (severity BETWEEN 1 AND 5),
    resolved    INTEGER NOT NULL DEFAULT 0,  -- boolean 0/1
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

-- 4. Schedules: spaced‑repetition / review reminders.
CREATE TABLE IF NOT EXISTS schedules (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    weakness_id   INTEGER REFERENCES weaknesses(id) ON DELETE CASCADE,
    review_at     TEXT    NOT NULL,           -- ISO 8601 datetime
    description   TEXT,
    completed     INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL
);

-- 5. Documents: metadata for uploaded lecture PDFs.
CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    filename     TEXT    NOT NULL,
    display_name TEXT,
    num_chunks   INTEGER NOT NULL DEFAULT 0,
    file_hash    TEXT,                        -- SHA‑256 for deduplication
    uploaded_at  TEXT    NOT NULL
);

-- 6. Reports: generated diagnosis / weekly reports.
CREATE TABLE IF NOT EXISTS reports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT    NOT NULL DEFAULT 'default',
    title       TEXT    NOT NULL,
    body        TEXT    NOT NULL,             -- markdown or plain text
    created_at  TEXT    NOT NULL
);

-- 7. User Profiles: stores learning preferences and style.
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id             TEXT PRIMARY KEY,
    learning_style      TEXT,                         -- conceptual | practical | concise
    preferred_tone      TEXT,                         -- encouraging | strict | academic
    academic_background TEXT,                         -- e.g., '컴퓨터공학과 학부생'
    notes               TEXT,                         -- AI notes on user style/preferences
    strengths_summary   TEXT NOT NULL DEFAULT '',     -- dynamic paragraph summarizing user strengths
    weaknesses_summary  TEXT NOT NULL DEFAULT '',     -- dynamic paragraph summarizing user weaknesses
    updated_at          TEXT NOT NULL
);

-- 8. Strengths: concepts the learner demonstrated mastery on.
CREATE TABLE IF NOT EXISTS strengths (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT    NOT NULL DEFAULT 'default',
    session_id  TEXT    REFERENCES sessions(id) ON DELETE SET NULL,
    concept     TEXT    NOT NULL,
    details     TEXT,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);
"""


def init_db() -> None:
    """Create all tables if they do not already exist."""
    conn = get_connection()
    conn.executescript(_SCHEMA_SQL)
    
    # Migration: Add user_id column to weaknesses and strengths if missing
    for table in ["weaknesses", "strengths"]:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = [row["name"] for row in cursor.fetchall()]
        if "user_id" not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default'")
            
    # Migration: Add strengths_summary and weaknesses_summary to user_profiles if missing
    cursor = conn.execute("PRAGMA table_info(user_profiles)")
    columns = [row["name"] for row in cursor.fetchall()]
    if "strengths_summary" not in columns:
        conn.execute("ALTER TABLE user_profiles ADD COLUMN strengths_summary TEXT NOT NULL DEFAULT ''")
    if "weaknesses_summary" not in columns:
        conn.execute("ALTER TABLE user_profiles ADD COLUMN weaknesses_summary TEXT NOT NULL DEFAULT ''")

    # Seed default user profile
    now = _now()
    conn.execute(
        "INSERT OR IGNORE INTO user_profiles (user_id, learning_style, preferred_tone, academic_background, notes, strengths_summary, weaknesses_summary, updated_at) "
        "VALUES ('default', 'conceptual', 'encouraging', '대학원생', '소크라테스식 학습 진행 중', '', '', ?)",
        (now,)
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def create_session(
    user_id: str = "default",
    title: Optional[str] = None,
    socratic_mode: str = "standard",
) -> str:
    """Create a new session and return its UUID."""
    sid = str(uuid.uuid4())
    now = _now()
    conn = get_connection()
    conn.execute(
        "INSERT INTO sessions (id, user_id, title, socratic_mode, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (sid, user_id, title, socratic_mode, now, now),
    )
    conn.commit()
    conn.close()
    return sid


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Return a session dict or ``None``."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_sessions(user_id: str = "default") -> List[Dict[str, Any]]:
    """List all sessions for a user, most recent first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_session_mode(session_id: str, socratic_mode: str) -> None:
    """Change the Socratic depth mode for a session."""
    conn = get_connection()
    conn.execute(
        "UPDATE sessions SET socratic_mode = ?, updated_at = ? WHERE id = ?",
        (socratic_mode, _now(), session_id),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def log_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[str] = None,
) -> int:
    """Insert a chat message and return its row id."""
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO messages (session_id, role, content, metadata, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, metadata, _now()),
    )
    row_id = cur.lastrowid
    # Touch the session timestamp
    conn.execute(
        "UPDATE sessions SET updated_at = ? WHERE id = ?",
        (_now(), session_id),
    )
    conn.commit()
    conn.close()
    return row_id


def get_messages(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Return messages for a session in chronological order."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, role, content, metadata, created_at "
        "FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_messages(session_id: str) -> int:
    """Return the total number of messages in a session."""
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    conn.close()
    return row["cnt"] if row else 0


# ---------------------------------------------------------------------------
# Weakness helpers
# ---------------------------------------------------------------------------

def save_weakness(
    concept: str,
    details: Optional[str] = None,
    severity: int = 1,
    session_id: Optional[str] = None,
    user_id: str = "default",
) -> int:
    """Persist a weakness record and return its row id.

    If an unresolved weakness with the same concept already exists for this
    user, update it in-place (severity and details) instead of inserting a
    duplicate row.
    """
    now = _now()
    conn = get_connection()
    # Check for an existing unresolved weakness with the same concept/user
    row = conn.execute(
        "SELECT id FROM weaknesses WHERE user_id = ? AND concept = ? AND resolved = 0",
        (user_id, concept),
    ).fetchone()
    if row:
        row_id = row["id"]
        conn.execute(
            "UPDATE weaknesses SET details = ?, severity = ?, updated_at = ? WHERE id = ?",
            (details, severity, now, row_id),
        )
    else:
        cur = conn.execute(
            "INSERT INTO weaknesses (session_id, concept, details, severity, created_at, updated_at, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, concept, details, severity, now, now, user_id),
        )
        row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def resolve_weakness(weakness_id: int) -> None:
    """Mark a weakness as resolved."""
    conn = get_connection()
    conn.execute(
        "UPDATE weaknesses SET resolved = 1, updated_at = ? WHERE id = ?",
        (_now(), weakness_id),
    )
    conn.commit()
    conn.close()


def delete_weakness(weakness_id: int) -> bool:
    """Delete a weakness record. Returns True if a row was deleted."""
    conn = get_connection()
    cur = conn.execute("DELETE FROM weaknesses WHERE id = ?", (weakness_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def get_weaknesses(
    resolved: Optional[bool] = None,
    limit: int = 100,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return weakness records, optionally filtered by resolved status."""
    conn = get_connection()
    query = "SELECT * FROM weaknesses"
    params: list = []
    clauses: list[str] = []

    if user_id is not None:
        clauses.append("user_id = ?")
        params.append(user_id)

    if resolved is not None:
        clauses.append("resolved = ?")
        params.append(int(resolved))

    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_unresolved_weaknesses(
    user_id: str = "default",
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Return unresolved weaknesses for a specific user, joining with sessions."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT w.id, w.session_id, w.concept, w.details, w.severity, w.created_at "
        "FROM weaknesses w "
        "LEFT JOIN sessions s ON w.session_id = s.id "
        "WHERE w.user_id = ? AND w.resolved = 0 "
        "ORDER BY w.created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def add_schedule(
    review_at: str,
    description: Optional[str] = None,
    weakness_id: Optional[int] = None,
) -> int:
    """Create a review schedule entry and return its row id."""
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO schedules (weakness_id, review_at, description, created_at) "
        "VALUES (?, ?, ?, ?)",
        (weakness_id, review_at, description, _now()),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_pending_schedules() -> List[Dict[str, Any]]:
    """Return all uncompleted review schedules ordered by date."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM schedules WHERE completed = 0 ORDER BY review_at ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def complete_schedule(schedule_id: int) -> None:
    """Mark a schedule entry as completed."""
    conn = get_connection()
    conn.execute("UPDATE schedules SET completed = 1 WHERE id = ?", (schedule_id,))
    conn.commit()
    conn.close()


def delete_schedule(schedule_id: int) -> bool:
    """Delete a schedule record. Returns True if a row was deleted."""
    conn = get_connection()
    cur = conn.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------

def register_document(
    filename: str,
    num_chunks: int,
    display_name: Optional[str] = None,
    file_hash: Optional[str] = None,
) -> int:
    """Register an uploaded PDF document and return its row id."""
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO documents (filename, display_name, num_chunks, file_hash, uploaded_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (filename, display_name or filename, num_chunks, file_hash, _now()),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_documents() -> List[Dict[str, Any]]:
    """Return all registered documents."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def save_report(title: str, body: str, user_id: str = "default") -> int:
    """Save a generated diagnosis / weekly report and return its row id."""
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO reports (user_id, title, body, created_at) VALUES (?, ?, ?, ?)",
        (user_id, title, body, _now()),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def delete_session(session_id: str) -> None:
    """Delete a session and all its messages (CASCADE)."""
    conn = get_connection()
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()


def get_reports(user_id: str = "default", limit: int = 10) -> List[Dict[str, Any]]:
    """Return the most recent reports for a user."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM reports WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_profile(user_id: str = "default") -> Dict[str, Any]:
    """Retrieve the user profile, or return seeded defaults if not present."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    if row:
        return dict(row)
    # Default fallback
    return {
        "user_id": user_id,
        "learning_style": "conceptual",
        "preferred_tone": "encouraging",
        "academic_background": "대학원생",
        "notes": "소크라테스식 학습 진행 중",
        "strengths_summary": "",
        "weaknesses_summary": "",
    }


def save_user_profile(
    user_id: str,
    learning_style: Optional[str] = None,
    preferred_tone: Optional[str] = None,
    academic_background: Optional[str] = None,
    notes: Optional[str] = None,
    strengths_summary: Optional[str] = None,
    weaknesses_summary: Optional[str] = None,
) -> None:
    """Save or update the user's personalization profile."""
    now = _now()
    conn = get_connection()
    
    # Check if profile exists
    row = conn.execute("SELECT 1 FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
    if row:
        # Update existing
        query = "UPDATE user_profiles SET updated_at = ?"
        params = [now]
        if learning_style is not None:
            query += ", learning_style = ?"
            params.append(learning_style)
        if preferred_tone is not None:
            query += ", preferred_tone = ?"
            params.append(preferred_tone)
        if academic_background is not None:
            query += ", academic_background = ?"
            params.append(academic_background)
        if notes is not None:
            query += ", notes = ?"
            params.append(notes)
        if strengths_summary is not None:
            query += ", strengths_summary = ?"
            params.append(strengths_summary)
        if weaknesses_summary is not None:
            query += ", weaknesses_summary = ?"
            params.append(weaknesses_summary)
        query += " WHERE user_id = ?"
        params.append(user_id)
        conn.execute(query, params)
    else:
        # Insert new
        conn.execute(
            "INSERT INTO user_profiles (user_id, learning_style, preferred_tone, academic_background, notes, strengths_summary, weaknesses_summary, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                learning_style or "conceptual",
                preferred_tone or "encouraging",
                academic_background or "대학원생",
                notes or "",
                strengths_summary or "",
                weaknesses_summary or "",
                now,
            ),
        )
    conn.commit()
    conn.close()


def save_strength(
    concept: str,
    details: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: str = "default",
) -> int:
    """Persist a strength record and return its row id."""
    now = _now()
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO strengths (session_id, concept, details, created_at, updated_at, user_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, concept, details, now, now, user_id),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_user_strengths(user_id: str = "default", limit: int = 10) -> List[Dict[str, Any]]:
    """Return strengths for a specific user, joining with sessions."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT str.id, str.session_id, str.concept, str.details, str.created_at "
        "FROM strengths str "
        "LEFT JOIN sessions s ON str.session_id = s.id "
        "WHERE str.user_id = ? "
        "ORDER BY str.created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
