from __future__ import annotations

from typing import Any

from ness_cli.config_store import atomic_write_json, configs_path, locked_path, read_json_document


def provider_profiles() -> dict[str, dict[str, Any]]:
    raw = read_json_document(configs_path()).get("provider_profiles", {})
    if not isinstance(raw, dict):
        return {}
    return {str(key): dict(value) for key, value in raw.items() if isinstance(value, dict)}


def provider_profile(provider_id: str) -> dict[str, Any]:
    return dict(provider_profiles().get(provider_id, {}))


def update_provider_profile(provider_id: str, values: dict[str, Any]) -> None:
    path = configs_path()
    with locked_path(path):
        document = read_json_document(path)
        profiles = document.get("provider_profiles")
        if not isinstance(profiles, dict):
            profiles = {}
        current = profiles.get(provider_id)
        if not isinstance(current, dict):
            current = {}
        for key, value in values.items():
            if value is None:
                current.pop(key, None)
            else:
                current[key] = value
        profiles[provider_id] = current
        document["provider_profiles"] = profiles
        atomic_write_json(path, document)
