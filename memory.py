from __future__ import annotations

from pathlib import Path

from config import settings
from utils import get_project_context

"""
There are 3 main memory files:
- NESS.md: project memory (durable, human-managed) (L1)
- sessions/mem_<thread_id>.md: episodic per-session memory (session-durable, agent-managed) (L2)
- USER.md: cross-repo identity / preferences (L1, human-authored only)
"""

NESS_DIR = Path(settings.ness_dir)
NESS_FILE = NESS_DIR / "NESS.md"
SESSIONS_DIR = NESS_DIR / "sessions"

def user_memory_path() -> Path:
    """Resolve the USER.md location (repo-local or global home fallback)."""
    return NESS_DIR / "USER.md"

USER_FILE = user_memory_path()

MAX_NESS_CHARS = 20_000


def _file_key(path: Path) -> tuple[bool, int, int]:
    # Cheap signature for cache invalidation from filesystem metadata: (T/F, last modified time (ns), size (bytes))
    if not path.exists():
        return (False, 0, 0)
    stat = path.stat()
    return (True, stat.st_mtime_ns, stat.st_size)


def _append_markdown_file(path: Path, text: str) -> str:
    # append the text to the file
    cleaned = text.strip()
    if not cleaned:
        return f"No changes for {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(cleaned + "\n")
    return f"Appended to {path}"


def _session_memory_path(thread_id: str) -> Path:
    return SESSIONS_DIR / f"mem_{thread_id}.md"


def _extract_bullets(text: str) -> list[str]:
    # extract "- " lines and return them as list[str] (bullet points)
    bullets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
    return bullets


def _format_bullets(bullets: list[str]) -> str:
    return "\n".join(f"- {bullet}" for bullet in bullets)


def load_repo_context() -> str:
    # Load repo structure and manifest snippets only (no memory files)
    return get_project_context()


# --- NESS.md: Project Memory (Durable, Human-Managed) (L1) ---

def load_ness_memory() -> str:
    """Load .ness/NESS.md project memory."""
    if NESS_FILE.exists():
        return NESS_FILE.read_text(encoding="utf-8")
    return ""


def check_ness_health() -> str | None:
    """Passive lint check to verify if NESS.md is exceeding past the threshold size. Usable by CLI for health check."""
    current_chars = len(load_ness_memory())
    if current_chars > MAX_NESS_CHARS:
        return (
            f"Warning: NESS.md is at {current_chars} characters (cache threshold: {MAX_NESS_CHARS}).\n"
            f"Consider running `/compact memory` to consolidate redundant rules."
        )
    return None


def append_ness_memory(text: str) -> str:
    """Append a note to project memory. Restricted to explicit human actions or /memory add <note>"""
    return _append_markdown_file(NESS_FILE, text)


def write_ness_memory(text: str, overwrite: bool = False) -> str:
    """Write project memory, triggered by /init"""
    NESS_DIR.mkdir(parents=True, exist_ok=True)
    if NESS_FILE.exists() and not overwrite:
        return f"Error: {NESS_FILE} already exists"
    NESS_FILE.write_text(text.strip() + "\n", encoding="utf-8")
    return f"Wrote {NESS_FILE}"


def ness_key() -> tuple[bool, int, int]:
    """Cheap signature for durable project memory cache invalidation."""
    return _file_key(NESS_FILE)


# --- USER.md: Cross-Repo Identity / Preferences (L1, Human-Authored Only) ---

def load_user_memory() -> str:
    """Load USER.md cross-repo user preferences."""
    path = NESS_DIR / "USER.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""

def append_user_memory(text: str) -> str:
    """Append a preference note to USER.md. Only called via explicit /user commands."""
    return _append_markdown_file(NESS_DIR / "USER.md", text)


def user_memory_key() -> tuple[bool, int, int]:
    """Cheap signature for L1 cache invalidation of USER.md."""
    return _file_key(NESS_DIR / "USER.md")


# --- sessions/mem_<thread_id>.md: Episodic Per-Session Memory (L2, Agent-Managed) ---

def load_session_memory(thread_id: str) -> str:
    """Return bullet lines for one thread (no header, ids, or dates)."""
    if not thread_id:
        return ""

    path = _session_memory_path(thread_id)
    if not path.exists():
        return ""

    bullets = _extract_bullets(path.read_text(encoding="utf-8"))
    return _format_bullets(bullets)


def append_session_bullets(thread_id: str, bullets: list[str]) -> bool:
    """Append up to a few new bullets to the thread's session file. Returns True if disk changed."""
    if not thread_id:
        return False

    cleaned = [item.strip() for item in bullets if item and item.strip()]
    if not cleaned:
        return False

    path = _session_memory_path(thread_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _extract_bullets(path.read_text(encoding="utf-8")) if path.exists() else []
    merged = list(existing)
    for item in cleaned:
        if item not in merged:
            merged.append(item)

    new_content = _format_bullets(merged)
    if new_content:
        new_content += "\n"

    existing_text = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing_text == new_content:
        return False

    if new_content:
        path.write_text(new_content, encoding="utf-8")
    elif path.exists():
        path.unlink()
    return True


def memory_key(thread_id: str) -> tuple[bool, int, int]:
    """Cheap signature for L2 cache invalidation of one thread's session memory file."""
    if not thread_id:
        return (False, 0, 0)
    return _file_key(_session_memory_path(thread_id))
