from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from dotenv import set_key
from prompt_toolkit.application import get_app

from cli.constants import ENV_PATH


def write_env(key: str, value: str) -> None:
    ENV_PATH.touch(exist_ok=True)
    set_key(str(ENV_PATH), key, value, quote_mode="never")


def term_size() -> tuple[int, int]:
    try:
        app = get_app()
        size = app.output.get_size()
        if size.columns > 0 and size.rows > 0:
            return max(40, size.columns), max(24, size.rows)
    except Exception:
        pass
    fallback = shutil.get_terminal_size(fallback=(100, 24))
    return max(40, fallback.columns), max(24, fallback.lines)


def term_width() -> int:
    return term_size()[0]


def term_height() -> int:
    return term_size()[1]


def align(left: str, right: str) -> str:
    width = term_width()
    gap = max(1, width - len(left) - len(right))
    return left + (" " * gap) + right


def context_bar(used: int, total: int, width: int = 10) -> str:
    if total <= 0:
        return "░" * width
    filled = round(min(used / total, 1.0) * width)
    return "█" * filled + "░" * (width - filled)


def model_footer_name(slug: str) -> str:
    return slug.rsplit("/", 1)[-1]


def display_cwd() -> str:
    path = Path.cwd()
    home = Path.home()
    try:
        display = f"~/{path.relative_to(home)}"
    except ValueError:
        display = str(path)
    branch = os.environ.get("LITEHARNESS_WORKTREE") or git_branch()
    return f"{display} ({branch})" if branch else display


def git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=0.25,
            check=False,
        )
    except Exception:
        return ""
    return result.stdout.strip()
