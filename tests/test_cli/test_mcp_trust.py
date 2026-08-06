from __future__ import annotations

import json
from pathlib import Path

from ness_cli.config_store import load_configs
from ness_cli import mcp_trust
from ness_cli.mcp_manager import ProjectMCPManager


def _manager(tmp_path: Path) -> ProjectMCPManager:
    path = tmp_path / ".ness" / "mcp.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "remote": {
                        "url": "https://example.com/mcp?secret=query",
                        "headers": {"Authorization": "Bearer literal-secret"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return ProjectMCPManager(path, project_root=tmp_path)


def test_interactive_approval_persists_and_skips_next_prompt(tmp_path: Path, monkeypatch, capsys):
    manager = _manager(tmp_path)
    config_dir = tmp_path / "config"
    calls = 0

    def approve(*args, **kwargs):
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(mcp_trust.typer, "confirm", approve)
    assert mcp_trust.authorize_mcp_interactively(manager, config_dir=config_dir)
    assert mcp_trust.authorize_mcp_interactively(manager, config_dir=config_dir)
    assert calls == 1
    entry = load_configs(config_dir)["mcp_trust"][str(tmp_path.resolve())]
    assert entry["fingerprint"] == manager.trust_preview.fingerprint
    assert "literal-secret" not in json.dumps(load_configs(config_dir))
    output = capsys.readouterr().out
    assert "literal-secret" not in output
    assert "secret=query" not in output


def test_denial_is_not_persisted_and_marks_pending(tmp_path: Path, monkeypatch):
    manager = _manager(tmp_path)
    config_dir = tmp_path / "config"
    monkeypatch.setattr(mcp_trust.typer, "confirm", lambda *args, **kwargs: False)
    assert not mcp_trust.authorize_mcp_interactively(manager, config_dir=config_dir)
    assert manager.servers["remote"]["status"] == "pending_trust"
    assert "mcp_trust" not in load_configs(config_dir)


def test_config_change_requires_new_approval(tmp_path: Path, monkeypatch):
    manager = _manager(tmp_path)
    config_dir = tmp_path / "config"
    monkeypatch.setattr(mcp_trust.typer, "confirm", lambda *args, **kwargs: True)
    assert mcp_trust.authorize_mcp_interactively(manager, config_dir=config_dir)

    path = manager.mcp_file
    assert path is not None
    data = json.loads(path.read_text(encoding="utf-8"))
    data["mcpServers"]["second"] = {"command": "python"}
    path.write_text(json.dumps(data), encoding="utf-8")
    changed = ProjectMCPManager(path, project_root=tmp_path)
    assert not mcp_trust.is_mcp_trusted(changed, config_dir=config_dir)


def test_empty_config_is_implicitly_trusted(tmp_path: Path, monkeypatch):
    path = tmp_path / "mcp.json"
    path.write_text('{"mcpServers": {}}', encoding="utf-8")
    manager = ProjectMCPManager(path, project_root=tmp_path)
    monkeypatch.setattr(
        mcp_trust.typer,
        "confirm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prompted")),
    )
    assert mcp_trust.authorize_mcp_interactively(manager, config_dir=tmp_path / "config")
