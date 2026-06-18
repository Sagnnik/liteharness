from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from config import settings
from memory import NESS_DIR

THREADS_DIR = NESS_DIR / "threads"
INDEX_FILE = THREADS_DIR / "index.json"


def append_event(thread_id: str, event: dict[str, Any]) -> None:
    if not settings.auto_save_threads:
        return

    THREADS_DIR.mkdir(parents=True, exist_ok=True)
    event.setdefault("t", _now())
    path = THREADS_DIR / f"{thread_id}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    _update_index(thread_id, event)


def list_threads(n: int = 10) -> list[dict[str, Any]]:
    return _load_index()[:n]


def load_thread_events(thread_id: str) -> list[dict[str, Any]]:
    path = THREADS_DIR / f"{thread_id}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def archive_thread(thread_id: str) -> str:
    """Mark a thread archived in index.json.

    Idempotent: skips when the thread was already archived with no new events since.
    Headline is derived from the first user message.
    """
    if not settings.auto_save_threads:
        return "Thread autosave disabled"
    index = _load_index()
    entry = next((item for item in index if item.get("thread_id") == thread_id), None)
    if entry is None:
        return f"No thread to archive: {thread_id}"
    if _already_archived(entry):
        return f"Thread already archived: {thread_id}"
    events = load_thread_events(thread_id)
    if not events:
        return f"No events to archive: {thread_id}"
    first_message = _first_user_message(events)
    entry["summary"] = _truncate_headline(first_message or f"Session {thread_id}")
    entry["archived_at"] = _now()
    _save_index(index)
    return f"Archived thread {thread_id}"


def _already_archived(meta: dict[str, Any]) -> bool:
    archived_at = meta.get("archived_at")
    if not archived_at:
        return False
    updated_at = meta.get("updated_at", "")
    return bool(updated_at) and str(archived_at) >= str(updated_at)


def _first_user_message(events: list[dict[str, Any]], limit: int = 200) -> str:
    for event in events:
        if event.get("kind") != "user":
            continue
        content = event.get("content")
        text = content if isinstance(content, str) else str(content)
        return " ".join(text.split())[:limit]
    return ""


def _truncate_headline(text: str, limit: int = 80) -> str:
    cleaned = " ".join(text.split()).strip().strip('"').strip(".")
    return cleaned[:limit].strip() or "Session"


def _update_index(thread_id: str, event: dict[str, Any]) -> None:
    index = _load_index()
    entry = next((item for item in index if item.get("thread_id") == thread_id), None)
    if entry is None:
        entry = {
            "thread_id": thread_id,
            "started_at": _now(),
            "turn_count": 0,
            "model": settings.model_name,
            "summary": "",
            "total_cost_usd": 0.0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
            "output_tokens": 0,
        }
        index.insert(0, entry)
    if event.get("kind") == "user":
        entry["turn_count"] = int(entry.get("turn_count", 0)) + 1
    if event.get("kind") == "usage":
        entry["total_cost_usd"] = float(entry.get("total_cost_usd", 0.0)) + float(event.get("cost_usd") or 0.0)
        entry["input_tokens"] = int(entry.get("input_tokens", 0)) + int(event.get("input_tokens") or 0)
        entry["cached_input_tokens"] = int(entry.get("cached_input_tokens", 0)) + int(event.get("cached_input_tokens") or 0)
        entry["cache_write_tokens"] = int(entry.get("cache_write_tokens", 0)) + int(event.get("cache_write_tokens") or 0)
        entry["output_tokens"] = int(entry.get("output_tokens", 0)) + int(event.get("output_tokens") or 0)
    entry["updated_at"] = _now()
    _save_index(index)


def _load_index() -> list[dict[str, Any]]:
    if not INDEX_FILE.exists():
        return []
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))


def _save_index(index: list[dict[str, Any]]) -> None:
    THREADS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
