from __future__ import annotations

import json
from pathlib import Path

from liteharness.permissions import DEFAULT_RULES

NESS_SUBDIRS = (
    "sessions",
    "agents",
    "commands",
    "skills",
    "plans",
    "threads",
    "shells",
)


def setup_ness_structure(ness_dir: Path) -> list[str]:
    """Create the standard ``.ness/`` tree and default config files.

    Creates directories (sessions/agents/commands/skills/plans/threads/shells)
    and default ``permissions.json``, ``hooks.json``, and ``mcp.json`` when
    missing. Returns a list of paths that were created.
    """
    created: list[str] = []
    root = Path(ness_dir)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        created.append(str(root))

    for name in NESS_SUBDIRS:
        p = root / name
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            created.append(str(p))

    for path, content in {
        root / "permissions.json": json.dumps(DEFAULT_RULES, indent=2) + "\n",
        root / "hooks.json": "{}\n",
        root / "mcp.json": json.dumps({"servers": {}}, indent=2) + "\n",
    }.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            created.append(str(path))

    return created
