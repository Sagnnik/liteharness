from __future__ import annotations

import json
from pathlib import Path

from ness_cli.config_store import load_configs
from ness_cli.mcp_import import (
    execute_mcp_import,
    plan_mcp_import,
    provenance_for_server,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def test_import_servers_source_to_canonical_mcpservers(tmp_path: Path):
    source = tmp_path / ".cursor" / "mcp.json"
    destination = tmp_path / ".ness" / "mcp.json"
    _write(source, {"servers": {"remote": {"url": "https://example.com/mcp"}}})
    plan = plan_mcp_import(source, destination, project_root=tmp_path)
    assert plan.valid
    assert plan.entries[0].action == "add"
    execute_mcp_import(plan, config_dir=tmp_path / "config")
    result = json.loads(destination.read_text(encoding="utf-8"))
    assert result == {"mcpServers": {"remote": {"url": "https://example.com/mcp"}}}


def test_dry_plan_has_no_side_effects_and_redacts_secrets(tmp_path: Path):
    source = tmp_path / "source.json"
    destination = tmp_path / ".ness" / "mcp.json"
    _write(
        source,
        {
            "mcpServers": {
                "remote": {
                    "url": "https://example.com/mcp?secret=query",
                    "headers": {"Authorization": "Bearer literal-secret"},
                }
            }
        },
    )
    before = source.read_bytes()
    plan = plan_mcp_import(source, destination, project_root=tmp_path)
    rendered = plan.render()
    assert plan.valid
    assert "literal-secret" not in rendered
    assert "secret=query" not in rendered
    assert "literal credential" in rendered
    assert not destination.exists()
    assert source.read_bytes() == before


def test_conflict_fails_atomically_and_explicit_replace_succeeds(tmp_path: Path):
    source = tmp_path / "source.json"
    destination = tmp_path / ".ness" / "mcp.json"
    _write(source, {"mcpServers": {"one": {"command": "new"}, "two": {"command": "two"}}})
    _write(destination, {"mcpServers": {"one": {"command": "old"}}})
    original = destination.read_bytes()
    conflict = plan_mcp_import(source, destination, project_root=tmp_path)
    assert not conflict.valid
    assert destination.read_bytes() == original

    replacement = plan_mcp_import(
        source,
        destination,
        project_root=tmp_path,
        replace={"one"},
    )
    assert replacement.valid
    execute_mcp_import(replacement, config_dir=tmp_path / "config")
    servers = json.loads(destination.read_text(encoding="utf-8"))["mcpServers"]
    assert servers["one"]["command"] == "new"
    assert servers["two"]["command"] == "two"


def test_identical_is_unchanged_and_replace_is_rejected(tmp_path: Path):
    source = tmp_path / "source.json"
    destination = tmp_path / ".ness" / "mcp.json"
    value = {"mcpServers": {"one": {"command": "python"}}}
    _write(source, value)
    _write(destination, value)
    plan = plan_mcp_import(source, destination, project_root=tmp_path)
    assert plan.valid and not plan.changes
    invalid = plan_mcp_import(source, destination, project_root=tmp_path, replace={"one"})
    assert not invalid.valid


def test_selection_invalid_source_and_legacy_destination(tmp_path: Path):
    source = tmp_path / "source.json"
    destination = tmp_path / ".ness" / "mcp.json"
    _write(
        source,
        {"mcpServers": {"ok": {"command": "python"}, "sse": {"type": "sse", "url": "https://x/sse"}}},
    )
    selected = plan_mcp_import(
        source,
        destination,
        project_root=tmp_path,
        selected={"ok"},
    )
    assert selected.valid
    all_entries = plan_mcp_import(source, destination, project_root=tmp_path)
    assert not all_entries.valid
    _write(destination, {"servers": {}})
    legacy = plan_mcp_import(source, destination, project_root=tmp_path, selected={"ok"})
    assert not legacy.valid
    assert "unsupported 'servers'" in legacy.errors[0]


def test_provenance_saved_and_manual_edit_detected(tmp_path: Path):
    source = tmp_path / "source.json"
    destination = tmp_path / ".ness" / "mcp.json"
    config_dir = tmp_path / "config"
    entry = {"command": "python"}
    _write(source, {"mcpServers": {"one": entry}})
    plan = plan_mcp_import(source, destination, project_root=tmp_path)
    execute_mcp_import(plan, config_dir=config_dir)
    metadata = load_configs(config_dir)["mcp_imports"][str(tmp_path.resolve())]["one"]
    assert metadata["source_path"] == str(source.resolve())
    assert "command" not in json.dumps(metadata)
    provenance = provenance_for_server(
        config_dir=config_dir,
        project_root=tmp_path,
        name="one",
        entry=entry,
    )
    assert provenance and not provenance["modified"]
    changed = provenance_for_server(
        config_dir=config_dir,
        project_root=tmp_path,
        name="one",
        entry={"command": "uv"},
    )
    assert changed and changed["modified"]


def test_unresolved_placeholders_import_with_warning(tmp_path: Path):
    source = tmp_path / "source.json"
    destination = tmp_path / ".ness" / "mcp.json"
    _write(source, {"mcpServers": {"remote": {"url": "${MCP_URL}/mcp"}}})
    plan = plan_mcp_import(source, destination, project_root=tmp_path)
    assert plan.valid
    assert "unresolved placeholders" in plan.entries[0].warnings[0]
