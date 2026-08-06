"""CLI trust policy for project MCP configuration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

from ness_cli.config_store import load_configs, write_config
from ness_cli.mcp_manager import MCPTrustPreview, ProjectMCPManager
from ness_cli.terminal import terminal_safe_text

_TRUST_KEY = "mcp_trust"


def is_mcp_trusted(
    manager: ProjectMCPManager,
    *,
    config_dir: Path,
) -> bool:
    """Return whether the current normalized runnable config was approved."""
    preview = manager.load()
    if not preview.has_runnable_servers:
        return True
    entries = load_configs(config_dir).get(_TRUST_KEY, {})
    if not isinstance(entries, dict):
        return False
    entry = entries.get(str(manager.project_root.resolve()))
    return (
        isinstance(entry, dict)
        and entry.get("config_path") == str(preview.config_path)
        and entry.get("fingerprint") == preview.fingerprint
    )


def authorize_mcp_interactively(
    manager: ProjectMCPManager,
    *,
    config_dir: Path,
) -> bool:
    """Prompt once for a changed config and persist an affirmative decision."""
    preview = manager.load()
    if not preview.has_runnable_servers or is_mcp_trusted(manager, config_dir=config_dir):
        return True

    typer.echo(
        terminal_safe_text(
            f"MCP configuration requests permission: {preview.config_path}"
        )
    )
    for summary in preview.servers:
        typer.echo(terminal_safe_text(f"  - {summary}"))
    approved = typer.confirm(
        "Allow these MCP servers for this exact configuration?",
        default=False,
        abort=False,
    )
    if not approved:
        manager.mark_untrusted()
        return False
    _persist_trust(manager, preview, config_dir=config_dir)
    return True


def _persist_trust(
    manager: ProjectMCPManager,
    preview: MCPTrustPreview,
    *,
    config_dir: Path,
) -> None:
    configs = load_configs(config_dir)
    current = configs.get(_TRUST_KEY, {})
    entries: dict[str, Any] = dict(current) if isinstance(current, dict) else {}
    entries[str(manager.project_root.resolve())] = {
        "config_path": str(preview.config_path),
        "fingerprint": preview.fingerprint,
        "trusted_at": datetime.now(timezone.utc).isoformat(),
    }
    write_config(_TRUST_KEY, entries, config_dir)
