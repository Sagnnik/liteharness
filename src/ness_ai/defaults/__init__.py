"""Packaged default files seeded into a project ``.ness/`` on first setup."""

from __future__ import annotations

from pathlib import Path

_AGENTS_DIR = Path(__file__).resolve().parent / "agents"


def default_agent_profiles() -> dict[str, str]:
    """Return ``{filename: content}`` for built-in subagent profiles.

    Filenames are relative to ``.ness/agents/`` (e.g. ``explore.md``).
    """
    profiles: dict[str, str] = {}
    if not _AGENTS_DIR.is_dir():
        return profiles
    for path in sorted(_AGENTS_DIR.glob("*.md")):
        if path.is_file():
            profiles[path.name] = path.read_text(encoding="utf-8")
    return profiles
