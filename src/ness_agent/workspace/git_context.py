"""Git worktree context helpers for harness overlays.

Used by the agent L3 overlay for branch/dirty-state snapshots. For checkpoint
restore see rollback.py; for parallel sessions see worktree.py.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path.cwd()

_GIT_SUMMARY_MAX_PATHS = 5


def _git(args: list[str], timeout: int = 30, cwd: Path = PROJECT_ROOT) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        return f"Error: git {' '.join(args)} failed with exit {result.returncode}\n{output}"
    return output or "(ok)"


def git_worktree_summary(cwd: Path = PROJECT_ROOT) -> str:
    """Compact branch + dirty-state snapshot for per-turn system-reminder overlays."""
    branch_out = _git(["branch", "--show-current"], timeout=5, cwd=cwd)
    if branch_out.startswith("Error:"):
        return ""
    branch_name = branch_out if branch_out != "(ok)" else "(detached)"

    status_out = _git(["status", "--porcelain"], timeout=5, cwd=cwd)
    if status_out.startswith("Error:"):
        return f"branch: {branch_name}"
    if status_out == "(ok)":
        return f"branch: {branch_name}; working tree clean"

    lines = [line for line in status_out.splitlines() if line.strip()]
    if not lines:
        return f"branch: {branch_name}; working tree clean"

    paths = [line[3:].strip() for line in lines[:_GIT_SUMMARY_MAX_PATHS]]
    remainder = len(lines) - len(paths)
    suffix = f" (+{remainder} more)" if remainder else ""
    return (
        f"branch: {branch_name}; {len(lines)} changed file(s): "
        f"{', '.join(paths)}{suffix}"
    )


def auto_git_snapshot(message: str = "agent: auto-snapshot") -> bool:
    """Stage all changes and commit with --no-verify. Returns True if committed or clean."""
    add = _git(["add", "-A"])
    if add.startswith("Error:"):
        return False
    result = _git(["commit", "-m", message, "--no-verify"], timeout=60)
    if result.startswith("Error:"):
        return "nothing to commit" in result.lower()
    return True
