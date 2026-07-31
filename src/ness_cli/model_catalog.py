from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

from ness_cli.paths import cache_dir_from_env

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
MODELS_DEV_URL = "https://models.dev/api.json"
CATALOG_TTL_SECONDS = 24 * 60 * 60
CATALOG_VERSION = 1
_CACHE_NAME = "openrouter-models-v1.json"


@dataclass(frozen=True, slots=True)
class ModelRecord:
    id: str
    name: str
    input_modalities: tuple[str, ...] = ("text",)
    output_modalities: tuple[str, ...] = ("text",)
    context_length: int | None = None
    input_price: float | None = None
    output_price: float | None = None
    cache_read_price: float | None = None
    supported_parameters: tuple[str, ...] = ()
    reasoning_efforts: tuple[str, ...] = ()
    supports_anthropic_messages: bool = False

    @property
    def supports_vision(self) -> bool:
        return "image" in self.input_modalities

    @property
    def cache_read_ratio(self) -> float:
        if not self.input_price or self.cache_read_price is None:
            return 0.1
        return self.cache_read_price / self.input_price


@dataclass(frozen=True, slots=True)
class RefreshResult:
    refreshed: bool
    models: int
    error: str | None = None


_records: dict[str, ModelRecord] | None = None
_fetched_at: float = 0.0
_refresh_task: asyncio.Task[RefreshResult] | None = None
_provider_aliases = {
    "z-ai": "zhipuai",
    "moonshotai": "moonshotai",
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google",
    "deepseek": "deepseek",
    "mistralai": "mistral",
}


def catalog_cache_path() -> Path:
    return cache_dir_from_env() / _CACHE_NAME


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _per_million(value: Any) -> float | None:
    parsed = _number(value)
    return parsed * 1_000_000 if parsed is not None else None


def _reasoning_efforts(
    model_id: str,
    models_dev: dict[str, Any],
) -> tuple[str, ...]:
    author, _, slug = model_id.partition("/")
    candidates: list[dict[str, Any] | None] = []
    provider = models_dev.get(_provider_aliases.get(author, author), {})
    candidates.append((provider.get("models") or {}).get(slug))
    candidates.append((provider.get("models") or {}).get(model_id))
    # OpenRouter is a useful exact-ID fallback, but primary-provider metadata
    # wins so values such as GLM's "max" remain literal.
    candidates.append(
        ((models_dev.get("openrouter") or {}).get("models") or {}).get(model_id)
    )
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        for option in entry.get("reasoning_options") or []:
            if option.get("type") == "effort":
                return tuple(
                    str(value)
                    for value in option.get("values") or []
                    if value is not None
                )
    return ()


def parse_catalog(
    openrouter_payload: dict[str, Any],
    models_dev_payload: dict[str, Any] | None = None,
) -> list[ModelRecord]:
    models_dev = models_dev_payload or {}
    records: list[ModelRecord] = []
    for raw in openrouter_payload.get("data") or []:
        if not isinstance(raw, dict):
            continue
        architecture = raw.get("architecture") or {}
        inputs = tuple(str(item) for item in architecture.get("input_modalities") or [])
        outputs = tuple(str(item) for item in architecture.get("output_modalities") or [])
        parameters = tuple(str(item) for item in raw.get("supported_parameters") or [])
        if "text" not in inputs or "text" not in outputs or "tools" not in parameters:
            continue
        model_id = str(raw.get("id") or "").strip()
        if not model_id:
            continue
        pricing = raw.get("pricing") or {}
        records.append(
            ModelRecord(
                id=model_id,
                name=str(raw.get("name") or model_id),
                input_modalities=inputs,
                output_modalities=outputs,
                context_length=int(raw["context_length"]) if raw.get("context_length") else None,
                input_price=_per_million(pricing.get("prompt")),
                output_price=_per_million(pricing.get("completion")),
                cache_read_price=_per_million(
                    pricing.get("input_cache_read") or pricing.get("cache_read")
                ),
                supported_parameters=parameters,
                reasoning_efforts=_reasoning_efforts(model_id, models_dev),
                supports_anthropic_messages=model_id.startswith("anthropic/"),
            )
        )
    return sorted(records, key=lambda item: item.id.lower())


def _load_cache() -> None:
    global _records, _fetched_at
    if _records is not None:
        return
    _records = {}
    path = catalog_cache_path()
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != CATALOG_VERSION:
            return
        _fetched_at = float(payload.get("fetched_at") or 0)
        for raw in payload.get("models") or []:
            raw = dict(raw)
            for key in (
                "input_modalities",
                "output_modalities",
                "supported_parameters",
                "reasoning_efforts",
            ):
                raw[key] = tuple(raw.get(key) or ())
            record = ModelRecord(**raw)
            _records[record.id] = record
    except (OSError, ValueError, TypeError):
        _records = {}
        _fetched_at = 0.0


def cached_models() -> tuple[ModelRecord, ...]:
    _load_cache()
    return tuple((_records or {}).values())


def model_record(model_id: str) -> ModelRecord | None:
    _load_cache()
    return (_records or {}).get(model_id)


def catalog_is_stale(now: float | None = None) -> bool:
    _load_cache()
    return not _records or (now or time.time()) - _fetched_at >= CATALOG_TTL_SECONDS


def _write_cache(records: Iterable[ModelRecord]) -> None:
    path = catalog_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CATALOG_VERSION,
        "fetched_at": time.time(),
        "models": [asdict(record) for record in records],
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def fetch_catalog(timeout: float = 20.0) -> list[ModelRecord]:
    headers = {"User-Agent": "ness_ai/0.1"}
    openrouter = requests.get(
        OPENROUTER_MODELS_URL,
        headers=headers,
        timeout=timeout,
    )
    openrouter.raise_for_status()
    models_dev_payload: dict[str, Any] = {}
    try:
        models_dev = requests.get(MODELS_DEV_URL, headers=headers, timeout=timeout)
        models_dev.raise_for_status()
        models_dev_payload = models_dev.json()
    except requests.RequestException:
        pass
    records = parse_catalog(openrouter.json(), models_dev_payload)
    if not records:
        raise ValueError("OpenRouter returned no text tool-capable models")
    _write_cache(records)
    return records


async def refresh_catalog(*, force: bool = False) -> RefreshResult:
    global _refresh_task
    if not force and not catalog_is_stale():
        return RefreshResult(False, len(cached_models()))
    if _refresh_task is not None and not _refresh_task.done():
        return await _refresh_task

    async def run() -> RefreshResult:
        global _records, _fetched_at
        try:
            records = await asyncio.to_thread(fetch_catalog)
            _records = {record.id: record for record in records}
            _fetched_at = time.time()
            return RefreshResult(True, len(records))
        except Exception as exc:
            return RefreshResult(False, len(cached_models()), str(exc))

    _refresh_task = asyncio.create_task(run())
    try:
        return await _refresh_task
    finally:
        _refresh_task = None


def reset_catalog_cache_for_tests() -> None:
    global _records, _fetched_at, _refresh_task
    _records = None
    _fetched_at = 0.0
    _refresh_task = None
