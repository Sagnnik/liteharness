from config import settings
from pathlib import Path
import json
from datetime import datetime, timezone

NESS = Path(settings.ness_dir)
THREADS_DIR = NESS / "threads"
INDEX_FILE = THREADS_DIR / "index.json"

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _load_index() -> list[dict]:
    if not INDEX_FILE.exists():
        return []
    return json.loads(INDEX_FILE.read_text())

def _save_index(index: list[dict]):
    THREADS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps(index, indent=2))


def append_event(thread_id:str, event:dict):
    if not settings.auto_save_threads:
        return

    THREADS_DIR.mkdir(parents=True, exist_ok=True)
    path = THREADS_DIR / f"{thread_id}.jsonl"

    event.setdefault("t", _now())

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    # update index
    index = _load_index()
    entry = next((e for e in index if e["thread_id"] == thread_id), None)
    if not entry:
        entry = {"thread_id": thread_id, "started_at": _now(), "turn_count": 0}
        index.insert(0, entry)

    entry["turn_count"] = entry.get("turn_count", 0) + 1
    entry["updated_at"] = _now()
    _save_index(index)

    

def list_threads(n: int = 10) -> list[dict]:
    return _load_index()[:n]

def load_thread_messages(thread_id:str) -> list[dict]:
    path = THREADS_DIR / f"{thread_id}.jsonl"
    if not path.exists():
        return []

    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] 
