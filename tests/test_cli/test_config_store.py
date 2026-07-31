"""Tests for the global JSON config/secrets store and Settings sourcing."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from ness_cli.config_store import (
    ensure_secrets_file,
    load_configs,
    load_secrets,
    migrate_env_once,
    write_config,
    write_secret,
)


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch) -> Path:
    cfg = tmp_path / "cfg"
    monkeypatch.setenv("NESS_AI_CONFIG_DIR", str(cfg))
    return cfg


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_write_config_roundtrip_and_lazy_creation(config_dir: Path):
    assert not (config_dir / "configs.json").exists()
    write_config("model_name", "openai/gpt-4o")
    write_config("enable_approval", False)
    assert load_configs() == {"enable_approval": False, "model_name": "openai/gpt-4o"}


def test_write_secret_permissions(config_dir: Path):
    write_secret("openai_api_key", "sk-test")
    path = config_dir / "secrets.json"
    assert load_secrets() == {"openai_api_key": "sk-test"}
    assert _mode(path) == 0o600


def test_write_none_deletes_key(config_dir: Path):
    write_config("openai_base_url", "https://example.com/v1")
    write_config("openai_base_url", None)
    assert load_configs() == {}
    # Deleting a missing key is a no-op and does not create the file.
    (config_dir / "configs.json").unlink()
    write_config("openai_base_url", None)
    assert not (config_dir / "configs.json").exists()


def test_corrupt_json_tolerated(config_dir: Path):
    config_dir.mkdir(parents=True)
    (config_dir / "configs.json").write_text("{not json", encoding="utf-8")
    (config_dir / "secrets.json").write_text("[1, 2]", encoding="utf-8")
    assert load_configs() == {}
    assert load_secrets() == {}
    # A corrupt file is replaced wholesale on next write.
    write_config("model_name", "x")
    assert load_configs() == {"model_name": "x"}


def test_ensure_secrets_file_creates_once(config_dir: Path):
    created = ensure_secrets_file()
    assert created is not None and created.is_file()
    assert json.loads(created.read_text()) == {}
    assert _mode(created) == 0o600
    assert ensure_secrets_file() is None


def test_migrate_env_once_imports_and_marks(config_dir: Path, tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text(
        "OPENAI_API_KEY=sk-from-env\n"
        "EXA_API_KEY=exa-from-env\n"
        "MODEL_NAME=openai/gpt-4o\n"
        "ENABLE_APPROVAL=false\n"
        "API_MAX_RETRIES=5\n"
        "UNRELATED_VAR=ignored\n"
        "EMPTY_VALUE=\n",
        encoding="utf-8",
    )
    from ness_cli.config import settings_field_env_map

    imported = migrate_env_once(env, field_for_alias=settings_field_env_map())
    assert set(imported) == {
        "openai_api_key",
        "exa_api_key",
        "model_name",
        "enable_approval",
        "api_max_retries",
    }
    assert load_secrets() == {"openai_api_key": "sk-from-env", "exa_api_key": "exa-from-env"}
    assert load_configs() == {
        "model_name": "openai/gpt-4o",
        "enable_approval": False,
        "api_max_retries": 5,
    }
    assert _mode(config_dir / "secrets.json") == 0o600
    # .env left untouched; marker prevents a second run.
    assert "sk-from-env" in env.read_text(encoding="utf-8")
    assert migrate_env_once(env, field_for_alias=settings_field_env_map()) == []


def test_migrate_env_existing_json_wins(config_dir: Path, tmp_path: Path):
    write_config("model_name", "deepseek/deepseek-chat", config_dir)
    env = tmp_path / ".env"
    env.write_text("MODEL_NAME=openai/gpt-4o\n", encoding="utf-8")
    from ness_cli.config import settings_field_env_map

    imported = migrate_env_once(env, field_for_alias=settings_field_env_map())
    assert imported == []
    assert load_configs() == {"model_name": "deepseek/deepseek-chat"}


def test_settings_reads_global_json(config_dir: Path, monkeypatch):
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    write_config("model_name", "openai/gpt-4o")
    write_secret("openai_api_key", "sk-json")
    from ness_cli.config import Settings

    fresh = Settings()
    assert fresh.model_name == "openai/gpt-4o"
    assert fresh.openai_api_key == "sk-json"


def test_settings_env_overrides_json(config_dir: Path, monkeypatch):
    write_config("model_name", "openai/gpt-4o")
    monkeypatch.setenv("MODEL_NAME", "deepseek/deepseek-chat")
    from ness_cli.config import Settings

    assert Settings().model_name == "deepseek/deepseek-chat"


def test_settings_defaults_without_json(config_dir: Path, monkeypatch):
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from ness_cli.config import Settings

    fresh = Settings()
    assert fresh.model_name == "deepseek/deepseek-v4-flash"
    assert fresh.openai_api_key is None
