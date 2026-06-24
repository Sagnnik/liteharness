from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from config import settings
from memory import NESS_DIR

THREADS_DIR = NESS_DIR / "threads"
THREADS_DB = THREADS_DIR / "threads.db"
SESSION_THREAD_PREFIX = "session-"
SUBAGENT_THREAD_PREFIX = "subagent-"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived_at TEXT,
    turn_count INTEGER NOT NULL DEFAULT 0,
    model TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    total_cost_usd REAL NOT NULL DEFAULT 0.0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS events (
    thread_id TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (thread_id, seq)
);

CREATE TABLE IF NOT EXISTS subagents (
    subagent_thread_id TEXT PRIMARY KEY,
    parent_thread_id TEXT NOT NULL REFERENCES threads(thread_id),
    agent_name TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'running',
    started_at TEXT NOT NULL,
    completed_at TEXT,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    output TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_threads_updated ON threads(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_subagents_parent ON subagents(parent_thread_id, started_at DESC);
"""

_write_lock = threading.Lock()


def append_event(thread_id: str, event: dict[str, Any]) -> None:
    if not settings.auto_save_threads:
        return

    if thread_id.startswith(SUBAGENT_THREAD_PREFIX):
        if event.get("kind") == "usage":
            _rollup_subagent_usage(thread_id, event)
        return

    payload = dict(event)
    payload.setdefault("t", _now())

    with _write_lock:
        with _connect() as conn:
            _ensure_schema(conn)
            now = _now()
            conn.execute(
                """
                INSERT INTO threads (thread_id, started_at, updated_at, model)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(thread_id) DO NOTHING
                """,
                (thread_id, now, now, settings.model_name),
            )

            seq = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM events WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO events (thread_id, seq, payload) VALUES (?, ?, ?)",
                (thread_id, seq, json.dumps(payload, ensure_ascii=False)),
            )
            _apply_event_to_thread(conn, thread_id, payload, now)
            conn.commit()


def list_threads(n: int = 10) -> list[dict[str, Any]]:
    with _connect() as conn:
        _ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT
                thread_id, started_at, updated_at, archived_at,
                turn_count, model, summary,
                total_cost_usd, input_tokens, cached_input_tokens,
                cache_write_tokens, output_tokens
            FROM threads
            WHERE thread_id LIKE ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (f"{SESSION_THREAD_PREFIX}%", n),
        ).fetchall()

    return [
        {
            "thread_id": row["thread_id"],
            "started_at": row["started_at"],
            "updated_at": row["updated_at"],
            "turn_count": int(row["turn_count"]),
            "model": row["model"],
            "summary": row["summary"],
            "total_cost_usd": float(row["total_cost_usd"]),
            "input_tokens": int(row["input_tokens"]),
            "cached_input_tokens": int(row["cached_input_tokens"]),
            "cache_write_tokens": int(row["cache_write_tokens"]),
            "output_tokens": int(row["output_tokens"]),
            **({"archived_at": row["archived_at"]} if row["archived_at"] else {}),
        }
        for row in rows
    ]


def load_thread_events(thread_id: str) -> list[dict[str, Any]]:
    if not THREADS_DB.exists():
        return []

    with _connect() as conn:
        _ensure_schema(conn)
        rows = conn.execute(
            "SELECT payload FROM events WHERE thread_id = ? ORDER BY seq",
            (thread_id,),
        ).fetchall()

    return [json.loads(row[0]) for row in rows]


def archive_thread(thread_id: str) -> str:
    """Mark a thread archived in the threads table.

    Idempotent: skips when the thread was already archived with no new events since.
    Headline is derived from the first user message.
    """
    if not settings.auto_save_threads:
        return "Thread autosave disabled"

    with _write_lock:
        with _connect() as conn:
            _ensure_schema(conn)
            row = conn.execute(
                "SELECT updated_at, archived_at FROM threads WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()

            if row is None:
                return f"No thread to archive: {thread_id}"

            if row["archived_at"] and str(row["archived_at"]) >= str(row["updated_at"]):
                return f"Thread already archived: {thread_id}"

            event_rows = conn.execute(
                "SELECT payload FROM events WHERE thread_id = ? ORDER BY seq",
                (thread_id,),
            ).fetchall()
            if not event_rows:
                return f"No events to archive: {thread_id}"

            events = [json.loads(item[0]) for item in event_rows]
            headline = ""
            for event in events:
                if event.get("kind") != "user":
                    continue
                content = event.get("content")
                text = content if isinstance(content, str) else str(content)
                headline = " ".join(text.split())[:200]
                break
            raw = headline or f"Session {thread_id}"
            cleaned = " ".join(raw.split()).strip().strip('"').strip(".")
            summary = cleaned[:80].strip() or "Session"
            archived_at = _now()
            conn.execute(
                "UPDATE threads SET summary = ?, archived_at = ?, updated_at = ? WHERE thread_id = ?",
                (summary, archived_at, archived_at, thread_id),
            )
            conn.commit()

    return f"Archived thread {thread_id}"


def register_subagent(
    parent_thread_id: str,
    subagent_thread_id: str,
    *,
    agent_name: str,
    label: str = "",
) -> None:
    if not settings.auto_save_threads:
        return

    with _write_lock:
        with _connect() as conn:
            _ensure_schema(conn)
            now = _now()
            conn.execute(
                """
                INSERT INTO threads (thread_id, started_at, updated_at, model)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(thread_id) DO NOTHING
                """,
                (parent_thread_id, now, now, settings.model_name),
            )
            conn.execute(
                """
                INSERT INTO subagents (
                    subagent_thread_id, parent_thread_id, agent_name, label,
                    status, started_at
                ) VALUES (?, ?, ?, ?, 'running', ?)
                """,
                (subagent_thread_id, parent_thread_id, agent_name, label, now),
            )
            conn.commit()


def complete_subagent(
    subagent_thread_id: str,
    *,
    status: str,
    output: str = "",
    duration_ms: int = 0,
) -> None:
    if not settings.auto_save_threads:
        return

    with _write_lock:
        with _connect() as conn:
            _ensure_schema(conn)
            completed_at = _now()
            conn.execute(
                """
                UPDATE subagents
                SET status = ?, completed_at = ?, duration_ms = ?, output = ?
                WHERE subagent_thread_id = ?
                """,
                (status, completed_at, duration_ms, output, subagent_thread_id),
            )
            conn.commit()


def list_subagents(parent_thread_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        _ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT
                subagent_thread_id, parent_thread_id, agent_name, label,
                status, started_at, completed_at, duration_ms, output
            FROM subagents
            WHERE parent_thread_id = ?
            ORDER BY started_at ASC
            """,
            (parent_thread_id,),
        ).fetchall()

    return [
        {
            "subagent_thread_id": row[0],
            "parent_thread_id": row[1],
            "agent_name": row[2],
            "label": row[3],
            "status": row[4],
            "started_at": row[5],
            "completed_at": row[6],
            "duration_ms": int(row[7]),
            "output": row[8],
        }
        for row in rows
    ]


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    THREADS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(THREADS_DB, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
    finally:
        conn.close()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


def _rollup_subagent_usage(subagent_thread_id: str, event: dict[str, Any]) -> None:
    with _write_lock:
        with _connect() as conn:
            _ensure_schema(conn)
            row = conn.execute(
                "SELECT parent_thread_id FROM subagents WHERE subagent_thread_id = ?",
                (subagent_thread_id,),
            ).fetchone()
            if row is None:
                return
            _apply_event_to_thread(conn, row[0], event, _now())
            conn.commit()


def _apply_event_to_thread(
    conn: sqlite3.Connection,
    thread_id: str,
    event: dict[str, Any],
    updated_at: str,
) -> None:
    turn_delta = 1 if event.get("kind") == "user" else 0
    cost_delta = 0.0
    input_delta = 0
    cached_delta = 0
    cache_write_delta = 0
    output_delta = 0

    if event.get("kind") == "usage":
        cost_delta = float(event.get("cost_usd") or 0.0)
        input_delta = int(event.get("input_tokens") or 0)
        cached_delta = int(event.get("cached_input_tokens") or 0)
        cache_write_delta = int(event.get("cache_write_tokens") or 0)
        output_delta = int(event.get("output_tokens") or 0)

    conn.execute(
        """
        UPDATE threads SET
            updated_at = ?,
            turn_count = turn_count + ?,
            total_cost_usd = total_cost_usd + ?,
            input_tokens = input_tokens + ?,
            cached_input_tokens = cached_input_tokens + ?,
            cache_write_tokens = cache_write_tokens + ?,
            output_tokens = output_tokens + ?
        WHERE thread_id = ?
        """,
        (
            updated_at,
            turn_delta,
            cost_delta,
            input_delta,
            cached_delta,
            cache_write_delta,
            output_delta,
            thread_id,
        ),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
