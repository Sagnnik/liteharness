"""@-mention menu support for the CLI input buffer.

Typing ``@`` in the input opens a file-completion menu (driven by the
MenuMixin chrome). Selecting a file inserts a visible ``@<relative/path>``
token in the buffer. This module only ships the menu primitives
(``index_files`` / ``filter_files``); the actual ``@file`` expansion into
``<document>`` blocks lives on the adapter side in
:mod:`liteharness_cli.mentions` and runs inside ``CodingSession.run_turn``
(and again on resume/rollback replay), so file content always reflects
current disk.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cli.models import MenuItem

# Project root for path resolution (the CLI process cwd after any worktree
# bootstrap chdir). Mirrors the old root-level permissions.PROJECT_ROOT.
_PROJECT_ROOT = Path.cwd().resolve()

# Skip these directory names for the non-git walk fallback. The git path
# underneath already respects .gitignore via ``git ls-files``.
_WALK_SKIP_DIRS = frozenset(
    {
        ".git",
        ".ness",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".ruff_cache",
        ".pytest_cache",
        ".mypy_cache",
        "dist",
        "build",
        ".tox",
    }
)

# Cap on the number of paths surfaced from index_files so the menu and the
# in-memory cached list stay manageable in huge repos.
_INDEX_LIMIT = 2000

# Index cache keyed only on (PROJECT_ROOT identity, mtime of .git, in-repo flag).
# Refresh is cheap and only happens on first call or when a file tree changes.
_index_cache: dict[str, object] = {}


def index_files(limit: int = _INDEX_LIMIT) -> list[Path]:
    """Return up to ``limit`` candidate file paths under the project root.

    In a git repo: ``git ls-files`` (fast, respects .gitignore, no extra deps).
    Otherwise: a filtered os.walk that skips the convention dirs in
    ``_WALK_SKIP_DIRS``. The result is cached on the module to keep the menu
    responsive; refresh happens when ``.git`` mtime changes or the call lands
    more than 30s after the last build.
    """
    import time

    git_dir = _PROJECT_ROOT / ".git"
    cache_key = (
        str(_PROJECT_ROOT),
        git_dir.stat().st_mtime_ns if git_dir.exists() else 0,
        bool(git_dir.exists()),
    )
    now = time.monotonic()
    cached = _index_cache.get("key")
    if cached == cache_key and _index_cache.get("expires", 0) > now:
        return list(_index_cache.get("files", []))  # type: ignore[arg-type]

    files = _git_ls_files(limit) if git_dir.exists() else _walk_files(limit)

    _index_cache.clear()
    _index_cache["key"] = cache_key
    _index_cache["expires"] = now + 30.0
    _index_cache["files"] = list(files)
    return list(_index_cache["files"])  # type: ignore[arg-type]


def _git_ls_files(limit: int) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=False,
            timeout=10,
        )
    except Exception:
        return _walk_files(limit)

    if result.returncode != 0:
        return _walk_files(limit)

    out: list[Path] = []
    for entry in result.stdout.split(b"\x00"):
        if not entry:
            continue
        rel = entry.decode("utf-8", errors="replace").strip()
        if not rel:
            continue
        out.append(_PROJECT_ROOT / rel)
        if len(out) >= limit:
            break
    return out


def _walk_files(limit: int) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirs, names in os.walk(_PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in _WALK_SKIP_DIRS and not d.startswith(".")]
        for name in names:
            out.append(Path(dirpath) / name)
            if len(out) >= limit:
                return out
    return out


def filter_files(query: str, files: list[Path], limit: int) -> "list[MenuItem]":
    """Score ``files`` against ``query`` and return the top ``limit`` MenuItems.

    Empty query: most-recently-modified first (mtime desc), so the menu
    surfaces what the user just touched.
    """
    from cli.models import MenuItem

    query = (query or "").lower()

    if not query:
        try:
            scored = sorted(files, key=lambda p: _mtime_or_zero(p), reverse=True)
        except Exception:
            scored = list(files)
        items: list[MenuItem] = []
        for p in scored[:limit]:
            rel = _relative_posix(p)
            if not rel:
                continue
            items.append(MenuItem(key=rel, label=rel, description=_parent_dir(p)))
        return items

    matches: list[tuple[int, int, float, Path]] = []
    for p in files:
        name = p.name.lower()
        rel = _relative_posix(p).lower()
        if query not in rel:
            continue
        depth = rel.count("/")
        if name == query:
            score = (0, depth, -_mtime_or_zero(p))
        elif name.startswith(query):
            score = (1, depth, -_mtime_or_zero(p))
        elif "/" not in query and query in name:
            score = (2, depth, -_mtime_or_zero(p))
        else:
            score = (3, depth, -_mtime_or_zero(p))
        matches.append((*score, p))

    matches.sort()
    items = []
    for _, _, _, p in matches[:limit]:
        rel = _relative_posix(p)
        if not rel:
            continue
        items.append(MenuItem(key=rel, label=rel, description=_parent_dir(p)))
    return items


def _mtime_or_zero(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _relative_posix(p: Path) -> str:
    try:
        rel = p.relative_to(_PROJECT_ROOT)
    except ValueError:
        return ""
    return rel.as_posix()


def _parent_dir(p: Path) -> str:
    try:
        rel = p.relative_to(_PROJECT_ROOT)
    except ValueError:
        return ""
    parent = rel.parent
    if str(parent) in ("", "."):
        return ""
    return parent.as_posix()


def refresh_index_cache() -> None:
    """Force the next ``index_files`` call to rebuild the cache."""
    _index_cache.clear()
