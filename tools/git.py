from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

from permissions import PROJECT_ROOT, relative_to_root

GIT_READ_ACTIONS = frozenset({"status", "diff", "log", "show"})


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


def _git_status() -> str:
    return _git(["status", "--short"])


def _git_diff(path: str = "", cached: bool = False, stat: bool = False) -> str:
    args = ["diff"]
    if cached:
        args.append("--cached")
    if stat:
        args.append("--stat")
    if path:
        args.extend(["--", relative_to_root(path)])
    return _git(args)[:16000]


def _git_log(n: int | str = 20, path: str = "", grep: str = "") -> str:
    args = ["log", f"-{_clamp_log_count(n)}", "--oneline"]
    if grep:
        args.extend(["--grep", grep])
    if path:
        args.extend(["--", relative_to_root(path)])
    return _git(args)


def _git_show(rev: str = "HEAD") -> str:
    if error := _invalid_option_like(rev, "revision"):
        return error
    return _git(["show", "--no-ext-diff", "--no-patch", rev])[:16000]


def _git_commit(message: str, paths: str = "") -> str:
    if not message:
        return "Error: commit requires a message"
    if paths:
        rels = [relative_to_root(path) for path in paths.split()]
        add = _git(["add", *rels])
        if add.startswith("Error:"):
            return add
    return _git(["commit", "-m", message], timeout=60)


def _git_checkout(branch: str, create: bool = False) -> str:
    if error := _invalid_branch_name(branch):
        return error
    args = ["checkout"]
    if create:
        args.append("-b")
    args.append(branch)
    return _git(args)


def _git_branch(name: str | None = None) -> str:
    if not name:
        return _git(["branch"])
    if error := _invalid_branch_name(name):
        return error
    return _git(["branch", name])


def _git_stash(stash_action: str = "list", message: str = "") -> str:
    if stash_action not in {"list", "push", "pop", "apply"}:
        return "Error: stash action must be one of list, push, pop, apply"
    if stash_action == "push" and not message:
        return "Error: push requires a message"
    args = ["stash", stash_action]
    if stash_action == "push":
        args.extend(["-m", message])
    return _git(args)


@tool
def git(
    action: Literal[
        "status", "diff", "log", "show", "commit", "checkout", "branch", "stash"
    ],
    path: str = "",
    cached: bool = False,
    stat: bool = False,
    n: int | str = 20,
    grep: str = "",
    rev: str = "HEAD",
    message: str = "",
    paths: str = "",
    branch: str = "",
    create: bool = False,
    name: str = "",
    stash_action: str = "list",
) -> str:
    """Run a git operation. Read actions (status, diff, log, show) need no approval.

    Actions and their relevant parameters:
      - 'status': working tree status.
      - 'diff': working tree diff. Optional: path, cached, stat.
      - 'log': recent history. Optional: n, path, grep.
      - 'show': revision metadata. Optional: rev.
      - 'commit': create a commit. Required: message. Optional: paths (whitespace-separated to stage first).
      - 'checkout': switch branch. Required: branch. Optional: create.
      - 'branch': list branches, or create one by name. Optional: name.
      - 'stash': stash management. Optional: stash_action (list, push, pop, apply), message (required for push).
    """
    if action == "status":
        return _git_status()
    if action == "diff":
        return _git_diff(path=path, cached=cached, stat=stat)
    if action == "log":
        return _git_log(n=n, path=path, grep=grep)
    if action == "show":
        return _git_show(rev=rev)
    if action == "commit":
        return _git_commit(message=message, paths=paths)
    if action == "checkout":
        return _git_checkout(branch=branch, create=create)
    if action == "branch":
        return _git_branch(name=name or None)
    if action == "stash":
        return _git_stash(stash_action=stash_action, message=message)
    return f"Error: unknown git action {action}"
