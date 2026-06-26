from __future__ import annotations

import difflib
import fnmatch
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

import permissions
from config import settings
from permissions import relative_to_root, validate_path
from utils import is_ignored_dir

READ_FILE_DEFAULT_LIMIT = 400
READ_FILE_MAX_LIMIT = 2000
PROTECTED_WRITE_DIRS = frozenset({".git", ".ness"})

@tool
def read_file(path: str, offset: int = 1, limit: int | None = None) -> str:
    """Read a UTF-8 text file with 1-based line numbers and optional offset/limit."""
    try:
        abs_path = validate_path(path)
        lines = Path(abs_path).read_text(encoding="utf-8").splitlines()
        start = max(0, int(offset) - 1)
        requested_limit = READ_FILE_DEFAULT_LIMIT if limit is None else max(0, int(limit))
        effective_limit = min(requested_limit, READ_FILE_MAX_LIMIT)
        end = start + effective_limit
        chunk = lines[start:end]
        if not chunk:
            return "(empty file)"
        output = "\n".join(f"{i + start + 1:4d}| {line}" for i, line in enumerate(chunk))
        if end < len(lines):
            output += (
                f"\n... (truncated; showing {len(chunk)} of {len(lines)} lines, "
                f"use offset={end + 1} to continue)"
            )
        return output
    except Exception as exc:
        return f"Error: {exc}"


def is_protected_write_path(path: str) -> bool:
    """Return True for project-relative paths under reserved writable state dirs."""
    rel = path.replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    return any(rel == name or rel.startswith(f"{name}/") for name in PROTECTED_WRITE_DIRS)


def _reject_protected_write(rel_path: str, action: str) -> str | None:
    if is_protected_write_path(rel_path):
        return f"Error: refusing to {action} protected path {rel_path}"
    return None


@tool
def delete_file(path: str) -> str:
    """Delete a single file inside the project. Prefer this over shell rm commands."""
    try:
        abs_path = validate_path(path)
        rel = relative_to_root(abs_path)
        if error := _reject_protected_write(rel, "delete"):
            return error
        p = Path(abs_path)
        if not p.exists():
            return f"Error: {rel} does not exist"
        if p.is_dir():
            return f"Error: {rel} is a directory; delete_file only removes files"
        p.unlink()
        return f"Deleted {rel}"
    except Exception as exc:
        return f"Error: {exc}"


@tool
def write_file(path: str, content: str) -> str:
    """Write a complete UTF-8 file atomically, creating parent directories."""
    try:
        abs_path = validate_path(path)
        rel = relative_to_root(abs_path)
        if error := _reject_protected_write(rel, "write"):
            return error
        p = Path(abs_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(p, content)
        auto_format(abs_path)
        return f"Wrote {len(content)} chars to {rel}"
    except Exception as exc:
        return f"Error: {exc}"


class EditItem(BaseModel):
    old_string: str = Field(description="Exact text to find in the file.")
    new_string: str = Field(description="Replacement text.")
    replace_all: bool = Field(default=False, description="Replace every occurrence in the file.")


@tool
def edit(path: str, edits: list[EditItem]) -> str:
    """Apply one or more exact-text replacements to a single file atomically.

    Each edit uses exact match first with a conservative fuzzy fallback. All edits
    are applied in order; if any edit finds no match, the file is left unchanged.
    A single replacement is just a one-item edits list.
    """
    try:
        abs_path = validate_path(path)
        rel = relative_to_root(abs_path)
        if error := _reject_protected_write(rel, "edit"):
            return error
        if not edits:
            return "Error: edits must contain at least one edit"
        p = Path(abs_path)
        content = p.read_text(encoding="utf-8")
        total = 0
        fuzzy_any = False
        for idx, item in enumerate(edits, 1):
            content, count, fuzzy = _replace_content(
                content, item.old_string, item.new_string, item.replace_all
            )
            if count == 0:
                return f"No match for edit {idx} in {rel}; file was not changed"
            total += count
            fuzzy_any = fuzzy_any or fuzzy
        _atomic_write(p, content)
        auto_format(abs_path)
        suffix = " using fuzzy match" if fuzzy_any else ""
        return (
            f"Applied {len(edits)} edit{'s' if len(edits) != 1 else ''} "
            f"({total} replacement{'s' if total != 1 else ''}) to {rel}{suffix}"
        )
    except Exception as exc:
        return f"Error: {exc}"


@tool
def glob_files(pattern: str) -> str:
    """Find files matching a glob pattern under the project root."""
    try:
        matches = _git_glob(pattern) if is_git_repo(str(permissions.PROJECT_ROOT)) else []
        if not matches:
            matches = _filesystem_glob(pattern)
        return "\n".join(sorted(matches)[:300]) or "No matches"
    except Exception as exc:
        return f"Error: {exc}"


@tool
def list_files(path: str = ".") -> str:
    """List a directory inside the project root."""
    try:
        abs_path = validate_path(path)
        entries = []
        with os.scandir(abs_path) as it:
            for entry in it:
                if is_ignored_dir(entry.name):
                    continue
                entries.append(entry.name + ("/" if entry.is_dir() else ""))
        return "\n".join(sorted(entries)[:300]) or "(empty directory)"
    except Exception as exc:
        return f"Error: {exc}"


def _replace_content(content: str, old: str, new: str, replace_all: bool) -> tuple[str, int, bool]:
    if not old:
        return content, 0, False
    if old in content:
        count = content.count(old) if replace_all else 1
        return content.replace(old, new, -1 if replace_all else 1), count, False

    # fuzzy match:
    # 1. split file and old into line
    # 2. Slide a window of len(old_lines) through the file
    # 3. At each position, join those lines into a block and compare to old using SequenceMatcher
    # 4. Keep the block with the highest similarity ratio
    # 5. Replace the block with the new lines if score >= 0.90
    # 6. Otherwise fail (count = 0)
    lines = content.splitlines()
    old_lines = old.splitlines()
    if replace_all or not old_lines or len(old_lines) > len(lines):
        return content, 0, False

    best_ratio = 0.0
    best_idx = -1
    for idx in range(len(lines) - len(old_lines) + 1):
        block = "\n".join(lines[idx : idx + len(old_lines)])
        ratio = difflib.SequenceMatcher(None, old, block).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = idx

    if best_ratio < 0.90:
        return content, 0, False

    replacement_lines = new.splitlines()
    new_lines = lines[:best_idx] + replacement_lines + lines[best_idx + len(old_lines) :]
    return "\n".join(new_lines) + ("\n" if content.endswith("\n") else ""), 1, True


def _atomic_write(path: Path, content: str) -> None:
    # does atomic replace of the temporary file with the content
    # create the temp file in parent directory of the path
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        # write the content to the temporary file
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        # replace the original file with the temporary file
        os.replace(tmp, path)
    # clean up the temporary file
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _run_formatter(cmd: list[str]) -> None:
    if not cmd or not shutil.which(cmd[0]):
        return
    try:
        subprocess.run(
            cmd,
            cwd=permissions.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        pass


def auto_format(path: str) -> None:
    """Run a formatter for known file types when the tool is available on PATH."""
    if not settings.format_on_write:
        return

    p = Path(path)
    suffix = p.suffix.lower()
    path_str = str(p)

    formatters: dict[str, list[str]] = {
        ".py": ["python", "-m", "black", "--quiet", "--", path_str],
        ".ts": ["npx", "--", "prettier", "--write", path_str],
        ".tsx": ["npx", "--", "prettier", "--write", path_str],
        ".js": ["npx", "--", "prettier", "--write", path_str],
        ".jsx": ["npx", "--", "prettier", "--write", path_str],
        ".json": ["npx", "--", "prettier", "--write", path_str],
        ".md": ["npx", "--", "prettier", "--write", path_str],
        ".rs": ["rustfmt", path_str],
        ".go": ["gofmt", "-w", path_str],
        ".c": ["clang-format", "-i", path_str],
        ".h": ["clang-format", "-i", path_str],
        ".cpp": ["clang-format", "-i", path_str],
        ".cc": ["clang-format", "-i", path_str],
        ".cxx": ["clang-format", "-i", path_str],
        ".hpp": ["clang-format", "-i", path_str],
        ".hh": ["clang-format", "-i", path_str],
    }
    cmd = formatters.get(suffix)
    if cmd:
        _run_formatter(cmd)


def is_git_repo(path: str = ".") -> bool:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


def _filesystem_glob(pattern: str) -> list[str]:
    root = permissions.PROJECT_ROOT
    return [str(p.relative_to(root)) for p in root.glob(pattern) if p.is_file()]


def _git_glob(pattern: str) -> list[str]:
    # git ls-files is a lot faster than the filesystem glob
    # if the repo is not a git repo then fallback to the filesystem glob
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=permissions.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if fnmatch.fnmatch(line, pattern)]
