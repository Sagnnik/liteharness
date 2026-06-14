from __future__ import annotations

import subprocess
from pathlib import Path

from langchain_core.tools import tool

from permissions import PROJECT_ROOT, relative_to_root


def _git(args: list[str], timeout: int = 30, cwd: Path = PROJECT_ROOT) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        return f"Error: git {' '.join(args)} failed with exit {result.returncode}\n{output}"
    return output or "(ok)"


_GIT_SUMMARY_MAX_PATHS = 5
_GIT_LOG_DEFAULT_COUNT = 20
_GIT_LOG_MAX_COUNT = 100


def git_worktree_summary(cwd: Path = PROJECT_ROOT) -> str:
    """Compact branch + dirty-state snapshot for per-turn working-state overlays."""
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


@tool
def git_status() -> str:
    """Show concise git working tree status."""
    return _git(["status", "--short"])


@tool
def git_diff(path: str = "", cached: bool = False, stat: bool = False) -> str:
    """Show git diff for the working tree, optionally scoped to one path or staged changes."""
    args = ["diff"]
    if cached:
        args.append("--cached")
    if stat:
        args.append("--stat")
    if path:
        args.extend(["--", relative_to_root(path)])
    return _git(args)[:16000]


def _clamp_log_count(n: int | str) -> int:
    try:
        count = int(n)
    except (TypeError, ValueError):
        count = _GIT_LOG_DEFAULT_COUNT
    return min(_GIT_LOG_MAX_COUNT, max(1, count))


def _invalid_option_like(value: str, label: str) -> str | None:
    if not value:
        return f"Error: {label} must not be empty"
    if value.startswith("-"):
        return f"Error: {label} must not start with '-'"
    return None


def _invalid_branch_name(value: str | None, label: str = "branch name") -> str | None:
    if value is None or value == "":
        return f"Error: {label} must not be empty"
    if value.startswith("-"):
        return f"Error: {label} must not start with '-'"
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
        return f"Error: {label} must not contain whitespace or control characters"
    return None


@tool
def git_log(n: int | str = 20, path: str = "", grep: str = "") -> str:
    """Show recent git commit history."""
    args = ["log", f"-{_clamp_log_count(n)}", "--oneline"]
    if grep:
        args.extend(["--grep", grep])
    if path:
        args.extend(["--", relative_to_root(path)])
    return _git(args)


@tool
def git_show(rev: str = "HEAD") -> str:
    """Show metadata for a git revision."""
    if error := _invalid_option_like(rev, "revision"):
        return error
    return _git(["show", "--no-ext-diff", "--no-patch", rev])[:16000]


@tool
def git_commit(message: str, paths: str = "") -> str:
    """Create a git commit, optionally staging selected whitespace-separated paths first."""
    if paths:
        rels = [relative_to_root(path) for path in paths.split()]
        add = _git(["add", *rels])
        if add.startswith("Error:"):
            return add
    return _git(["commit", "-m", message], timeout=60)


@tool
def git_checkout(branch: str, create: bool = False) -> str:
    """Check out a git branch, optionally creating it."""
    if error := _invalid_branch_name(branch):
        return error
    args = ["checkout"]
    if create:
        args.append("-b")
    args.append(branch)
    return _git(args)


@tool
def git_branch(name: str | None = None) -> str:
    """List branches or create a branch by name."""
    if name is None:
        return _git(["branch"])
    if error := _invalid_branch_name(name):
        return error
    return _git(["branch", name])


@tool
def git_stash(action: str = "list", message: str = "") -> str:
    """Run a limited git stash action: list, push, pop, or apply."""
    if action not in {"list", "push", "pop", "apply"}:
        return "Error: action must be one of list, push, pop, apply"
    if action == "push" and not message:
        return "Error: push requires a message"
    args = ["stash", action]
    if action == "push":
        args.extend(["-m", message])
    return _git(args)
