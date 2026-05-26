import subprocess
import re
import os
from langchain_core.tools import tool
from pathlib import Path
from tools.common import validate_path
import shutil


@tool
def grep(pattern: str, path:str = ".", glob: str | None = None) -> str:
    """Search files with ripgrep (fallback: Python regrex walk)"""
    path = validate_path(path)
    if shutil.which("rg"):
        cmd = ["rg", "-n", "--no-heading", pattern, path]
        if glob:
            cmd.extend(["-g", glob])

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out = r.stdout.strip()
        return out[:8000] if out else "No matches found"

    # fallback
    matches = []
    rx = re.compile(pattern)
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in (".git", ".venv", ".pytest_cache", "__pycache__")]
        for fname in files:
            fp = os.path.join(root, fname)
            try:
                for i, line in enumerate(open(fp, "r", encoding="utf-8"), 1):
                    if rx.search(line):
                        matches.append(f"{fp}:{i}: {line.rstrip()}")

            except Exception:
                pass
            if len(matches) >= 50:
                break

    return "\n".join(matches[:50]) if matches else "No matches found"