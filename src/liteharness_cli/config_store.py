"""Global JSON config/secrets storage for the CLI adapter.

Values live in the global config dir alongside ``USER.md`` and ``plans/``:

- ``configs.json``: non-secret adapter settings (0644)
- ``secrets.json``: API keys and other secrets (0600)

Files are written lazily: only values the user explicitly sets are
persisted. Defaults stay on the ``Settings`` class (see
:mod:`liteharness_cli.config`), which layers the JSON files between
process env vars and its own defaults.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from liteharness_cli.paths import config_dir_from_env

# Settings field names routed to ``secrets.json`` instead of ``configs.json``.
SECRET_KEYS: frozenset[str] = frozenset({"openai_api_key", "exa_api_key"})

_CONFIGS_NAME = "configs.json"
_SECRETS_NAME = "secrets.json"
_MIGRATION_MARKER = ".env.migrated"


def configs_path(config_dir: Path | None = None) -> Path:
    return (config_dir or config_dir_from_env()) / _CONFIGS_NAME


def secrets_path(config_dir: Path | None = None) -> Path:
    return (config_dir or config_dir_from_env()) / _SECRETS_NAME


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_configs(config_dir: Path | None = None) -> dict[str, Any]:
    """Read ``configs.json`` (missing/corrupt file -> ``{}``)."""
    return _read_json(configs_path(config_dir))


def load_secrets(config_dir: Path | None = None) -> dict[str, Any]:
    """Read ``secrets.json`` (missing/corrupt file -> ``{}``)."""
    return _read_json(secrets_path(config_dir))


def _atomic_write(path: Path, data: dict[str, Any], *, secret: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if secret:
            os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _write_value(path: Path, key: str, value: Any, *, secret: bool) -> None:
    data = _read_json(path)
    if value is None:
        if key not in data:
            return
        del data[key]
    else:
        data[key] = value
    _atomic_write(path, data, secret=secret)


def write_config(key: str, value: Any, config_dir: Path | None = None) -> None:
    """Persist a non-secret value; ``None`` deletes the key."""
    _write_value(configs_path(config_dir), key, value, secret=False)


def write_secret(key: str, value: Any, config_dir: Path | None = None) -> None:
    """Persist a secret value (0600); ``None`` deletes the key."""
    _write_value(secrets_path(config_dir), key, value, secret=True)


def ensure_secrets_file(config_dir: Path | None = None) -> Path | None:
    """Create an empty ``secrets.json`` (0600) if missing. Returns path if created."""
    path = secrets_path(config_dir)
    if path.exists():
        return None
    _atomic_write(path, {}, secret=True)
    return path


def _coerce_scalar(raw: str) -> Any:
    """Best-effort typed value for a migrated ``.env`` string."""
    text = raw.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return raw


def migrate_env_once(
    env_path: Path,
    *,
    config_dir: Path | None = None,
    field_for_alias: dict[str, str],
    secret_keys: frozenset[str] = SECRET_KEYS,
) -> list[str]:
    """One-time import of known keys from a project ``.env`` into global JSON.

    Runs at most once per global config dir (tracked via a marker file).
    Existing JSON values win; the ``.env`` file is left untouched. Returns
    the list of imported Settings field names.
    """
    cfg_dir = config_dir or config_dir_from_env()
    marker = cfg_dir / _MIGRATION_MARKER
    if marker.exists():
        return []
    cfg_dir.mkdir(parents=True, exist_ok=True)
    imported: list[str] = []
    configs = load_configs(cfg_dir)
    secrets = load_secrets(cfg_dir)
    configs_changed = False
    secrets_changed = False
    if env_path.is_file():
        for alias, raw in dotenv_values(env_path).items():
            field = field_for_alias.get(alias)
            if field is None or raw is None or raw.strip() == "":
                continue
            value = _coerce_scalar(raw)
            if field in secret_keys:
                if field not in secrets:
                    secrets[field] = value
                    secrets_changed = True
                    imported.append(field)
            elif field not in configs:
                configs[field] = value
                configs_changed = True
                imported.append(field)
    if configs_changed:
        _atomic_write(configs_path(cfg_dir), configs, secret=False)
    if secrets_changed:
        _atomic_write(secrets_path(cfg_dir), secrets, secret=True)
    marker.write_text(f"migrated from {env_path}\n", encoding="utf-8")
    return imported
