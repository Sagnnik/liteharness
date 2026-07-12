from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from liteharness.session_context import get_session_context
from liteharness.workspace.project_context import is_ignored_dir

GREP_MAX_MATCHES = 200
GREP_MAX_OUTPUT_CHARS = 12000


@tool
def grep(pattern: str, include: str | None = None, path: str = ".") -> str:
    """Search file contents using regular expressions."""
    try:
        if not pattern:
            return "Error: pattern is empty. Provide a regex to search for."
        try:
            re.compile(pattern)
        except re.error as exc:
            return f"Error: invalid regex {pattern!r}: {exc}. Fix the pattern and retry."
        rt = get_session_context()
        abs_path = rt.permissions.validate_path(path)
        if shutil.which("rg"):
            cmd = ["rg", "-n", "--no-heading"]
            if include:
                cmd.extend(["-g", include])
            cmd.extend(["--", pattern, abs_path])
            result = subprocess.run(
                cmd,
                cwd=rt.project_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode not in (0, 1):
                return f"Error: {result.stderr or result.stdout}"
            return _cap_output(result.stdout.strip() or "No matches found")

        return _python_grep(pattern, abs_path, include)
    except Exception as exc:
        return f"Error: {exc}"


def _python_grep(pattern: str, path: str, include: str | None) -> str:
    rt = get_session_context()
    rx = re.compile(pattern)
    matches = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not is_ignored_dir(d)]
        for filename in files:
            fp = os.path.join(root, filename)
            rel = os.path.relpath(fp, rt.project_root)
            if include and not _matches_glob(rel, include):
                continue
            try:
                with open(fp, "r", encoding="utf-8") as handle:
                    for line_no, line in enumerate(handle, 1):
                        if rx.search(line):
                            matches.append(f"{rel}:{line_no}: {line.rstrip()}")
                            if len(matches) >= GREP_MAX_MATCHES:
                                return _format_matches(matches, truncated=True)
            except Exception:
                continue
    return _format_matches(matches, truncated=False)


def _matches_glob(rel_path: str, glob: str) -> bool:
    posix_path = rel_path.replace(os.sep, "/")
    return fnmatch.fnmatch(posix_path, glob) or Path(posix_path).match(glob)


def _format_matches(matches: list[str], truncated: bool) -> str:
    if not matches:
        return "No matches found"
    output = "\n".join(matches)
    if truncated:
        output += f"\n... (truncated after {GREP_MAX_MATCHES} matches)"
    return _cap_output(output)


def _cap_output(output: str) -> str:
    if len(output) <= GREP_MAX_OUTPUT_CHARS:
        return output
    return output[:GREP_MAX_OUTPUT_CHARS] + "\n... (truncated)"
