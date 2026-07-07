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

CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    user_seq INTEGER NOT NULL,
    git_hash TEXT,
    modified_paths TEXT NOT NULL DEFAULT '',
    mem_snapshot TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    PRIMARY KEY (thread_id, user_seq)
);

CREATE INDEX IF NOT EXISTS idx_threads_updated ON threads(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_subagents_parent ON subagents(parent_thread_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread ON checkpoints(thread_id, user_seq ASC);
"""

_write_lock = threading.Lock()


def append_event(thread_id: str, event: dict[str, Any]) -> int | None:
    """Append an event to the durable log. Returns the assigned seq, or None when skipped.

    Skipped when autosave is off or when the target is a subagent thread (those
    only roll up usage). Returning the seq lets callers (e.g. the rollback
    checkpoint hook) key side tables off the exact user-message event.
    """
    if not settings.auto_save_threads:
        return None

    if thread_id.startswith(SUBAGENT_THREAD_PREFIX):
        if event.get("kind") == "usage":
            _rollup_subagent_usage(thread_id, event)
        return None

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
            return seq


def list_threads(n: int = 10) -> list[dict[str, Any]]:
    with _connect() as conn:
        _ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT
                thread_id, started_at, updated_at, archived_at,
                turn_count, model, summary,
                total_cost_usd, input_tokens, cached_input_tokens,
                output_tokens
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


# --- rollback checkpoints -----------------------------------------------
# Per-turn snapshot row keyed by the user message's seq. Rolls back by deleting
# events with seq >= user_seq (truncating the conversation tail) and restoring
# files / session memory from the snapshot.

_FULL_TREE_SENTINEL = "*"


def save_checkpoint(
    thread_id: str,
    user_seq: int,
    git_hash: str | None,
    mem_snapshot: str = "",
) -> None:
    """Persist a per-user-turn rollback checkpoint row.

    Called right after the user message is appended (so ``user_seq`` is known).
    Overwrites any existing row for the same (thread_id, user_seq) so a re-run
    of the same turn replaces the stale snapshot instead of stranding two.
    """
    if not settings.auto_save_threads:
        return

    with _write_lock:
        with _connect() as conn:
            _ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO checkpoints (thread_id, user_seq, git_hash, modified_paths, mem_snapshot, created_at)
                VALUES (?, ?, ?, '', ?, ?)
                ON CONFLICT(thread_id, user_seq) DO UPDATE SET
                    git_hash = excluded.git_hash,
                    mem_snapshot = excluded.mem_snapshot,
                    modified_paths = '',
                    created_at = excluded.created_at
                """,
                (thread_id, user_seq, git_hash, mem_snapshot, _now()),
            )
            conn.commit()


def add_modified_path(thread_id: str, user_seq: int, path: str) -> None:
    """Record a filesystem path the agent mutated during one user turn.

    Called from the tools node for destructive fs/shell calls. Empty path or
    the ``"*"`` sentinel marks the turn as full-tree restore (used for shell
    commands where mutated paths cannot be enumerated). The set is
    expanded idempotently; once ``*`` is set, per-path entries are dropped.
    """
    if not settings.auto_save_threads:
        return
    if not path:
        return

    with _write_lock:
        with _connect() as conn:
            _ensure_schema(conn)
            row = conn.execute(
                "SELECT modified_paths FROM checkpoints WHERE thread_id = ? AND user_seq = ?",
                (thread_id, user_seq),
            ).fetchone()
            if row is None:
                return
            raw = row[0] or ""
            paths: set[str] = set()
            if raw:
                try:
                    paths = set(json.loads(raw))
                except ValueError:
                    paths = set()
            # Once the "*" full-tree sentinel is set, no further per-path
            # recording can refine it; adding "*" again is also a no-op.
            if _FULL_TREE_SENTINEL in paths:
                return
            if path == _FULL_TREE_SENTINEL:
                paths = {_FULL_TREE_SENTINEL}
            else:
                paths.add(path)
            conn.execute(
                "UPDATE checkpoints SET modified_paths = ? WHERE thread_id = ? AND user_seq = ?",
                (json.dumps(sorted(paths)), thread_id, user_seq),
            )
            conn.commit()


def get_checkpoint(thread_id: str, user_seq: int) -> dict[str, Any] | None:
    with _connect() as conn:
        _ensure_schema(conn)
        row = conn.execute(
            """
            SELECT thread_id, user_seq, git_hash, modified_paths, mem_snapshot, created_at
            FROM checkpoints WHERE thread_id = ? AND user_seq = ?
            """,
            (thread_id, user_seq),
        ).fetchone()
    if row is None:
        return None
    return {
        "thread_id": row[0],
        "user_seq": int(row[1]),
        "git_hash": row[2],
        "modified_paths": row[3] or "",
        "mem_snapshot": row[4] or "",
        "created_at": row[5],
    }


def first_user_message(thread_id: str) -> str | None:
    """Return the headline (truncated first user message) for a thread.

    Mirrors the summary derivation in ``archive_thread`` so the /threads table
    can show a useful label for threads that have not been archived yet.
    """
    if not THREADS_DB.exists():
        return None
    with _connect() as conn:
        _ensure_schema(conn)
        row = conn.execute(
            """
            SELECT payload FROM events
            WHERE thread_id = ? AND json_extract(payload, '$.kind') = 'user'
            ORDER BY seq ASC
            LIMIT 1
            """,
            (thread_id,),
        ).fetchone()
    if row is None:
        return None
    payload = json.loads(row[0])
    content = payload.get("content")
    text = content if isinstance(content, str) else str(content)
    headline = " ".join(text.split())[:200]
    cleaned = " ".join(headline.split()).strip().strip('"').strip(".")
    return cleaned[:80].strip() or None


def list_user_turns(thread_id: str) -> list[dict[str, Any]]:
    """Return every persisted user-message event for a thread, oldest-first.

    Used to populate the /rollback picker. ``seq`` is the event seq (the
    /rollback target); ``content`` is the (possibly truncated) user text.
    """
    if not THREADS_DB.exists():
        return []
    with _connect() as conn:
        _ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT seq, payload FROM events
            WHERE thread_id = ? AND json_extract(payload, '$.kind') = 'user'
            ORDER BY seq ASC
            """,
            (thread_id,),
        ).fetchall()
    turns: list[dict[str, Any]] = []
    for row in rows:
        payload = json.loads(row[1])
        content = payload.get("content")
        text = content if isinstance(content, str) else str(content)
        turns.append({"seq": int(row[0]), "content": text})
    return turns


def truncate_after(thread_id: str, user_seq: int) -> None:
    """Delete events with seq >= user_seq and re-derive thread rollup columns.

    Also drops checkpoint rows at/after user_seq (those snapshots describe the
    abandoned tail). Hard truncate per design decision: no undo of the undo.
    """
    if not settings.auto_save_threads:
        return

    with _write_lock:
        with _connect() as conn:
            _ensure_schema(conn)
            # Delete the abandoned conversation tail and its checkpoints.
            conn.execute(
                "DELETE FROM events WHERE thread_id = ? AND seq >= ?",
                (thread_id, user_seq),
            )
            conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = ? AND user_seq >= ?",
                (thread_id, user_seq),
            )
            # Re-derive threads rollup from the remaining events so /status
            # matches the new (shorter) conversation. Cost tracker stays
            # process-global per product decision.
            remaining = conn.execute(
                "SELECT payload FROM events WHERE thread_id = ? ORDER BY seq",
                (thread_id,),
            ).fetchall()
            turn_count = 0
            total_cost = 0.0
            input_tokens = 0
            cached_tokens = 0
            output_tokens = 0
            for item in remaining:
                event = json.loads(item[0])
                if event.get("kind") == "user":
                    turn_count += 1
                elif event.get("kind") == "usage":
                    total_cost += float(event.get("cost_usd") or 0.0)
                    input_tokens += int(event.get("input_tokens") or 0)
                    cached_tokens += int(event.get("cached_input_tokens") or 0)
                    output_tokens += int(event.get("output_tokens") or 0)
            conn.execute(
                """
                UPDATE threads SET
                    updated_at = ?,
                    turn_count = ?,
                    total_cost_usd = ?,
                    input_tokens = ?,
                    cached_input_tokens = ?,
                    output_tokens = ?
                WHERE thread_id = ?
                """,
                (_now(), turn_count, total_cost, input_tokens, cached_tokens, output_tokens, thread_id),
            )
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
    output_delta = 0

    if event.get("kind") == "usage":
        cost_delta = float(event.get("cost_usd") or 0.0)
        input_delta = int(event.get("input_tokens") or 0)
        cached_delta = int(event.get("cached_input_tokens") or 0)
        output_delta = int(event.get("output_tokens") or 0)

    conn.execute(
        """
        UPDATE threads SET
            updated_at = ?,
            turn_count = turn_count + ?,
            total_cost_usd = total_cost_usd + ?,
            input_tokens = input_tokens + ?,
            cached_input_tokens = cached_input_tokens + ?,
            output_tokens = output_tokens + ?
        WHERE thread_id = ?
        """,
        (
            updated_at,
            turn_delta,
            cost_delta,
            input_delta,
            cached_delta,
            output_delta,
            thread_id,
        ),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
