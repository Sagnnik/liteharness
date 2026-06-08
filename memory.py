from __future__ import annotations

from pathlib import Path

from config import settings
from utils import get_project_context

"""
There are 3 main memory files:
- NESS.md: project memory (durable) (L1)
- LOG.md: episodic per-session memory (session-durable) (L2)
- USER.md: cross-repo identity / preferences (L1, human-authored only)

The location of USER.md is currently repo-local. Would be moved to a global home directory
logic under testing; Still work in progress.
"""

NESS = Path(settings.ness_dir)
MEMORY_FILE = NESS / "NESS.md"


def user_memory_path() -> Path:
    """Resolve the USER.md location.

    Repo-local now. This is the single place to switch to a global home directory (for example ~/.ness/USER.md) 
    after LiteHarness is packaged as a CLI, so the cross-repo move stays a one-line change.
    """
    return NESS / "USER.md"


USER_FILE = user_memory_path()
LOG_FILE = NESS / "LOG.md"

LOG_SECTION_MARKER = "## Session "
RECENT_LOG_SESSIONS = 8
MAX_NESS_CHARS = 12_000

# --- NESS.md: project memory (durable) (L1) ---

def load_memory() -> str:
    """Load .ness/NESS.md project memory."""
    if MEMORY_FILE.exists():
        return MEMORY_FILE.read_text(encoding="utf-8")
    return ""


def memory_char_count() -> int:
    """Return the current character length of NESS.md."""
    return len(load_memory())


def load_project_context() -> str:
    """Load repo context plus project memory (NESS.md)."""
    context = get_project_context()
    memory = load_memory()
    if not memory:
        return context
    return f"{context}\n\n--- Project Memory (.ness/NESS.md) ---\n{memory}"


def append_memory(text: str) -> str:
    """Append a note to project memory."""
    NESS.mkdir(parents=True, exist_ok=True)
    before = load_memory()
    with MEMORY_FILE.open("a", encoding="utf-8") as handle:
        handle.write(text.strip() + "\n")
    if before == load_memory():
        return f"No changes for {MEMORY_FILE}"
    return f"Appended to {MEMORY_FILE}"


def write_memory(text: str, overwrite: bool = False) -> str:
    """Write project memory, refusing to overwrite unless requested."""
    NESS.mkdir(parents=True, exist_ok=True)
    if MEMORY_FILE.exists() and not overwrite:
        return f"Error: {MEMORY_FILE} already exists"
    MEMORY_FILE.write_text(text.strip() + "\n", encoding="utf-8")
    return f"Wrote {MEMORY_FILE}"


def memory_key() -> tuple[bool, int, int]:
    """Return a cheap signature for durable memory cache invalidation."""
    if not MEMORY_FILE.exists():
        return (False, 0, 0)
    stat = MEMORY_FILE.stat()
    return (True, int(stat.st_mtime_ns), int(stat.st_size))


# --- USER.md: cross-repo identity / preferences (L1, human-authored only) ---


def load_user_memory() -> str:
    """Load .ness/USER.md cross-repo user preferences."""
    path = user_memory_path()
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def append_user_memory(text: str) -> str:
    """Append a preference note to USER.md.

    Only the /user command calls this. The reflection gate must never write
    here, since it only sees a single repo's conversation.
    """
    path = user_memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    before = load_user_memory()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.strip() + "\n")
    if before == load_user_memory():
        return f"No changes for {path}"
    return f"Appended to {path}"


def user_memory_key() -> tuple[bool, int, int]:
    """Cheap signature for L1 cache invalidation of USER.md."""
    path = user_memory_path()
    if not path.exists():
        return (False, 0, 0)
    stat = path.stat()
    return (True, int(stat.st_mtime_ns), int(stat.st_size))


# --- LOG.md: episodic per-session memory (L2) (Optional Addition) ---


def load_recent_log(n: int = RECENT_LOG_SESSIONS) -> str:
    """Return only the last n session blocks from LOG.md (episodic tail for L2)."""
    if not LOG_FILE.exists():
        return ""
    _, blocks = _partition_sections(LOG_FILE.read_text(encoding="utf-8"), LOG_SECTION_MARKER)
    if not blocks:
        return ""
    return "\n\n".join(block.strip() for block in blocks[-n:])


def append_log(entry: str, thread_id: str | None = None) -> str:
    """Append a session block to LOG.md.

    When thread_id is given, any prior block for the same thread is replaced so
    re-archiving (for example /save then /exit) upserts instead of duplicating.
    """
    NESS.mkdir(parents=True, exist_ok=True)
    block = entry.strip()
    if thread_id is not None and LOG_FILE.exists():
        _, blocks = _partition_sections(LOG_FILE.read_text(encoding="utf-8"), LOG_SECTION_MARKER)
        prefix = f"{LOG_SECTION_MARKER}{thread_id} "
        kept = [b.strip() for b in blocks if not b.strip().startswith(prefix)]
        kept.append(block)
        LOG_FILE.write_text("\n\n".join(kept) + "\n", encoding="utf-8")
        return f"Wrote {LOG_FILE}"
    has_content = LOG_FILE.exists() and LOG_FILE.stat().st_size > 0
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        if has_content:
            handle.write("\n")
        handle.write(block + "\n")
    return f"Appended to {LOG_FILE}"


def log_key() -> tuple[bool, int, int]:
    """Cheap signature for L2 cache invalidation of LOG.md."""
    if not LOG_FILE.exists():
        return (False, 0, 0)
    stat = LOG_FILE.stat()
    return (True, int(stat.st_mtime_ns), int(stat.st_size))


def _partition_sections(text: str, marker: str) -> tuple[str, list[str]]:
    """Split text into (preamble, blocks).

    preamble is the content before the first marker line. Each block in blocks
    starts with a marker line and runs until the next marker.
    """
    preamble: list[str] = []
    blocks: list[str] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith(marker):
            if current is not None:
                blocks.append("\n".join(current))
            current = [line]
        elif current is None:
            preamble.append(line)
        else:
            current.append(line)
    if current is not None:
        blocks.append("\n".join(current))
    return "\n".join(preamble), blocks
