from __future__ import annotations

import difflib
import os
import subprocess
from pathlib import Path
from typing import Iterable

from permissions import PROJECT_ROOT
from tools.common import MANIFEST_FILES, discover_manifest_files, is_ignored_dir

_READ_ERRORS = (OSError, UnicodeDecodeError)
_SNIPPET_LIMIT = 1200


def get_project_context(max_files: int = 80) -> str:
    """Return a compact project tree and key manifest snippets."""
    files = _git_files(max_files)
    if not files:
        files = _walk_files(max_files)

    tree = _render_tree(files)
    summaries = []
    for label, path in _manifest_paths():
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")[:_SNIPPET_LIMIT]
            summaries.append(f"--- {label} ---\n{text}\n")
        except _READ_ERRORS:
            continue

    return f"Project structure (top {max_files} files):\n{tree}\n\n{''.join(summaries)}"


def preview_diff(tool: str, args: dict) -> str:
    """Preview a proposed file edit without writing it."""
    path = args.get("path", "")
    if tool == "apply_patch":
        return str(args.get("patch", ""))
    if not path:
        return f"{tool}({args})"

    p = Path(path)
    try:
        old = p.read_text(encoding="utf-8") if p.exists() else ""
    except _READ_ERRORS as exc:
        return f"Cannot read {path}: {exc}"

    if tool == "write_file":
        new = str(args.get("content", ""))
    elif tool == "edit_file":
        new = old.replace(str(args.get("old_string", "")), str(args.get("new_string", "")), 1)
    elif tool == "multi_edit":
        new = old
        for edit in args.get("edits", []):
            old_s = str(edit.get("old_string", ""))
            new_s = str(edit.get("new_string", ""))
            count = -1 if edit.get("replace_all") else 1
            new = new.replace(old_s, new_s, count)
    else:
        return f"{tool}({args})"

    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def _manifest_paths() -> list[tuple[str, Path]]:
    paths = [(name, PROJECT_ROOT / name) for name in MANIFEST_FILES]
    paths.extend((str(p.relative_to(PROJECT_ROOT)), p) for p in discover_manifest_files())
    return paths


def _git_files(max_files: int) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line and not _is_runtime_path(line)][:max_files]


def _walk_files(max_files: int) -> list[str]:
    files = []
    for dirpath, dirs, names in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if not is_ignored_dir(d) and not d.startswith(".")]
        for name in names:
            files.append(str((Path(dirpath) / name).relative_to(PROJECT_ROOT)))
            if len(files) >= max_files:
                return files
    return files


def _render_tree(files: Iterable[str]) -> str:
    tree: dict = {}
    for file in files:
        parts = Path(file).parts
        current = tree
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current.setdefault("__files__", []).append(parts[-1])

    def render(node: dict, prefix: str = "") -> list[str]:
        lines = []
        for key, value in sorted(node.items(), key=lambda item: (item[0] == "__files__", item[0])):
            if key == "__files__":
                lines.extend(f"{prefix}- {name}" for name in sorted(value))
            else:
                lines.append(f"{prefix}- {key}/")
                lines.extend(render(value, prefix + "  "))
        return lines

    return "\n".join(render(tree))


def _is_runtime_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized.startswith((".ness/threads/", ".ness/shells/", ".ness/worktrees/"))
