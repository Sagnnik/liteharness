from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from ness_cli import mcp_cli
from ness_cli.tui import main as tui_main


def _paths(tmp_path: Path):
    return SimpleNamespace(
        project_root=tmp_path,
        ness_dir=tmp_path / ".ness",
        config_dir=tmp_path / "config",
    )


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_root_help_advertises_mcp_management_commands():
    result = CliRunner().invoke(tui_main.app, ["--help"])

    assert result.exit_code == 0
    assert "ness mcp status [SERVER]" in result.output
    assert "ness mcp login SERVER" in result.output
    assert "ness mcp logout SERVER" in result.output
    assert "ness mcp import PATH" in result.output
    assert "ness mcp --help" in result.output


def test_import_cli_dry_run_and_apply(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mcp_cli, "prepare_paths", lambda: _paths(tmp_path))
    source = tmp_path / "cursor.json"
    _write(source, {"mcpServers": {"remote": {"url": "https://example.com/mcp"}}})
    destination = tmp_path / ".ness" / "mcp.json"
    runner = CliRunner()

    dry = runner.invoke(mcp_cli.app, ["import", str(source), "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "Dry run" in dry.output
    assert not destination.exists()

    applied = runner.invoke(mcp_cli.app, ["import", str(source), "--yes"])
    assert applied.exit_code == 0, applied.output
    assert "Execution trust was not granted" in applied.output
    assert json.loads(destination.read_text(encoding="utf-8"))["mcpServers"]["remote"]


def test_import_cli_conflict_exit_2(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mcp_cli, "prepare_paths", lambda: _paths(tmp_path))
    source = tmp_path / "source.json"
    destination = tmp_path / ".ness" / "mcp.json"
    _write(source, {"mcpServers": {"one": {"command": "new"}}})
    _write(destination, {"mcpServers": {"one": {"command": "old"}}})
    result = CliRunner().invoke(mcp_cli.app, ["import", str(source), "--yes"])
    assert result.exit_code == 2
    assert "--replace one" in result.output
    assert json.loads(destination.read_text(encoding="utf-8"))["mcpServers"]["one"]["command"] == "old"


def test_status_and_logout_cli(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mcp_cli, "prepare_paths", lambda: _paths(tmp_path))
    _write(
        tmp_path / ".ness" / "mcp.json",
        {
            "mcpServers": {
                "remote": {
                    "url": "https://example.com/mcp",
                    "oauth": {"clientId": "client"},
                }
            }
        },
    )
    runner = CliRunner()
    status = runner.invoke(mcp_cli.app, ["status", "remote"])
    assert status.exit_code == 0, status.output
    assert "oauth=static" in status.output
    assert "logged_in=no" in status.output
    logout = runner.invoke(mcp_cli.app, ["logout", "remote"])
    assert logout.exit_code == 0, logout.output
    assert "were not revoked" in logout.output


def test_login_rejects_stdio_without_browser(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mcp_cli, "prepare_paths", lambda: _paths(tmp_path))
    _write(
        tmp_path / ".ness" / "mcp.json",
        {"mcpServers": {"local": {"command": "python"}}},
    )
    monkeypatch.setattr(
        mcp_cli.webbrowser,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("browser opened")),
    )
    result = CliRunner().invoke(mcp_cli.app, ["login", "local"])
    assert result.exit_code == 2
    assert "only for HTTP" in result.output
