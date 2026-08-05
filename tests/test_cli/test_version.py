"""``ness --version`` wiring."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as dist_version

from typer.testing import CliRunner

from ness_cli.tui import main as tui_main


def _installed_version() -> str | None:
    try:
        return dist_version("ness-agent")
    except PackageNotFoundError:
        return None


def test_cli_version_flag_prints_version_and_exits():
    result = CliRunner().invoke(tui_main.app, ["--version"])
    assert result.exit_code == 0, result.output
    installed = _installed_version()
    if installed is not None:
        assert result.output.strip() == f"ness {installed}"
    else:
        assert "version unknown" in result.output


def test_cli_version_flag_is_eager(monkeypatch):
    """--version short-circuits before any session/headless work runs."""
    monkeypatch.setattr(
        tui_main,
        "run_headless",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")),
    )
    result = CliRunner().invoke(tui_main.app, ["--version", "-p", "hi"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("ness ")
