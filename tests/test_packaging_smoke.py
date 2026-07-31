"""Clean-install smoke: build the wheel, install into a temp venv, assert basics.

Opt-in (slow / network for dependency install):

    PACKAGING_SMOKE=1 uv run pytest -m packaging
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.packaging
@pytest.mark.skipif(
    os.environ.get("PACKAGING_SMOKE") != "1",
    reason="set PACKAGING_SMOKE=1 to build and install a clean wheel",
)
def test_wheel_clean_install_smoke(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    venv = tmp_path / "venv"
    check_cwd = tmp_path / "cwd"
    check_cwd.mkdir()
    config_home = tmp_path / "config"
    config_home.mkdir()

    subprocess.run(
        ["uv", "build", "--wheel", "-o", str(dist)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(dist.glob("ness_ai-*.whl"))
    assert len(wheels) == 1, f"expected one wheel, got {wheels}"
    wheel = wheels[0]

    subprocess.run(
        [sys.executable, "-m", "venv", str(venv)],
        check=True,
        capture_output=True,
        text=True,
    )
    venv_python = venv / ("Scripts" if os.name == "nt" else "bin") / "python"
    venv_ness = venv / ("Scripts" if os.name == "nt" else "bin") / "ness"

    subprocess.run(
        ["uv", "pip", "install", "--python", str(venv_python), str(wheel)],
        check=True,
        capture_output=True,
        text=True,
    )

    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "MODEL_NAME",
            "REFLECTION_MODEL_NAME",
            "PYTHONPATH",
            "VIRTUAL_ENV",
        }
    }
    env["NESS_AI_CONFIG_DIR"] = str(config_home)
    env["HOME"] = str(tmp_path / "home")
    Path(env["HOME"]).mkdir(exist_ok=True)

    probe = r"""
from ness_ai.defaults import default_agent_profiles
from ness_cli.config import Settings
from ness_cli.instructions import default_instruction_files

import ness_ai
import ness_cli

assert ness_ai.__name__ == "ness_ai"
assert ness_cli.__name__ == "ness_cli"

settings = Settings()
assert settings.model_name == "deepseek/deepseek-v4-flash", settings.model_name

profiles = default_agent_profiles()
assert "explore.md" in profiles, sorted(profiles)
assert profiles["explore.md"].strip(), "explore.md body is empty"

instructions = default_instruction_files()
assert "l0_harness.md" in instructions, sorted(instructions)
assert instructions["l0_harness.md"].strip(), "l0_harness.md body is empty"
print("ok")
"""
    result = subprocess.run(
        [str(venv_python), "-c", probe],
        cwd=check_cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "ok" in result.stdout

    help_result = subprocess.run(
        [str(venv_ness), "--help"],
        cwd=check_cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stdout + "\n" + help_result.stderr
