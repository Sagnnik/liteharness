from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

from langchain_openrouter import ChatOpenRouter

from config import settings


@dataclass(frozen=True)
class ModelOverrides:
    """Optional CLI/runtime overrides for model construction."""

    model_name: str | None = None
    reflection_model_name: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openrouter_session_id: str | None = None


_overrides: ModelOverrides | None = None


def configure_model(overrides: ModelOverrides | None = None) -> None:
    """Apply runtime overrides that take precedence over settings."""
    global _overrides
    _overrides = overrides


def set_active_model(model_name: str) -> None:
    """Switch the active chat model at runtime.

    Updates both the override (used when constructing the chat model) and
    ``settings.model_name`` so cost and context-window lookups follow the switch.
    Callers must rebuild the graph afterwards to bind the new model.
    """
    global _overrides
    base = _overrides or ModelOverrides()
    _overrides = replace(base, model_name=model_name)
    settings.model_name = model_name


def _resolved(field: str) -> str | int | None:
    if _overrides is not None:
        value = getattr(_overrides, field, None)
        if value is not None:
            return value
    return getattr(settings, field)


def active_model_name() -> str:
    return _resolved("model_name")


def effective_openrouter_session_id(thread_id: str, *, suffix: str = "") -> str:
    base = _resolved("openrouter_session_id") or thread_id
    if suffix:
        return f"{base}:{suffix}"
    return base


def build_chat_model(
    thread_id: str,
    *,
    model_name: str | None = None,
    session_suffix: str = "",
) -> ChatOpenRouter:
    resolved_model = model_name or _resolved("model_name")
    model_kwargs: dict[str, str] = {
        "model": resolved_model,
        "api_key": _resolved("openai_api_key"),
        "session_id": effective_openrouter_session_id(thread_id, suffix=session_suffix),
    }
    base_url = _resolved("openai_base_url")
    if base_url:
        model_kwargs["base_url"] = base_url

    model_kwargs["max_retries"] = _resolved("api_max_retries")
    return ChatOpenRouter(**model_kwargs)


def create_model(thread_id: str) -> ChatOpenRouter:
    return build_chat_model(thread_id)


def create_compaction_model(thread_id: str) -> ChatOpenRouter:
    return build_chat_model(thread_id, session_suffix="compaction")


def create_reflection_model(thread_id: str) -> ChatOpenRouter:
    return build_chat_model(
        thread_id,
        model_name=_resolved("reflection_model_name"),
        session_suffix="reflection",
    )


def model_overrides_from_args(args: argparse.Namespace) -> ModelOverrides | None:
    fields = {
        "model_name": args.model,
        "reflection_model_name": args.reflection_model,
        "openai_api_key": args.api_key,
        "openai_base_url": args.base_url,
        "openrouter_session_id": args.openrouter_session_id,
    }
    active = {key: value for key, value in fields.items() if value is not None}
    if not active:
        return None
    return ModelOverrides(**active)


def add_model_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        help="Chat model name (overrides MODEL_NAME)",
    )
    parser.add_argument(
        "--reflection-model",
        help="Reflection model name (overrides REFLECTION_MODEL_NAME)",
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI-compatible API key (overrides OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible base URL (overrides OPENAI_BASE_URL)",
    )
    parser.add_argument(
        "--openrouter-session-id",
        help="Stable OpenRouter prompt-cache session id (overrides OPENROUTER_SESSION_ID)",
    )
