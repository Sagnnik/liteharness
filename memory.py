from pathlib import Path
from config import settings
from utils import get_project_context

NESS = Path(settings.ness_dir)
MEMORY_FILE = NESS / "NESS.md"

def load_memory() -> str:
    """Load NESS.md Memory file"""
    if MEMORY_FILE.exists():
        return MEMORY_FILE.read_text(encoding="utf-8")
    return ""

def load_project_context() -> str:
    ctx = get_project_context()
    mem = load_memory()
    if mem:
        return f"{ctx}\n\n--- Project Memory (NESS.md) ---\n{mem}"

    return ctx

def append_memory(text: str) -> str:
    NESS.mkdir(parents=True, exist_ok=True)
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(text.strip() + "\n")

    return f"Appended to {MEMORY_FILE}"
