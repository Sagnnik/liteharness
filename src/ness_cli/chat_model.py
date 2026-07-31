from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_openrouter import ChatOpenRouter

from ness_cli.config import (
    ReasoningEffort,
    coerce_reasoning_effort,
    model_supports_reasoning,
    reasoning_efforts_for_model,
    settings,
)
from ness_cli.anthropic_messages import OpenRouterAnthropicMessages


@dataclass(frozen=True)
class ModelOverrides:
    """Optional CLI/runtime overrides for model construction."""

    model_name: str | None = None
    reflection_model_name: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openrouter_session_id: str | None = None
    reasoning_effort: ReasoningEffort | None = None


_overrides: ModelOverrides | None = None

# Placeholder used when no provider key is configured: lets the TUI boot so
# the user can set the key via /config; the first model call fails with a
# provider auth error until then.
_MISSING_API_KEY = "sk-missing-api-key"


def provider_key_missing() -> bool:
    """True when no provider API key is configured (env, JSON, or CLI arg)."""
    return not _resolved_setting("openai_api_key")


def _resolved_api_key() -> str:
    return cast(str | None, _resolved_setting("openai_api_key")) or _MISSING_API_KEY


def configure_model(overrides: ModelOverrides | None = None) -> None:
    """Apply runtime overrides that take precedence over settings."""
    global _overrides
    _overrides = overrides


def set_active_model(model_name: str) -> str | None:
    """Switch the active chat model at runtime.

    Updates both the override (used when constructing the chat model) and
    ``settings.model_name`` so cost and context-window lookups follow the switch.
    Callers must rebuild the graph afterwards to bind the new model.

    Returns the coerced reasoning effort when it changed for the new model.
    """
    global _overrides
    base = _overrides or ModelOverrides()
    current_effort = cast(str | None, _resolved_setting("reasoning_effort"))
    coerced = coerce_reasoning_effort(model_name, current_effort)
    if coerced != current_effort:
        if coerced is not None:
            settings.reasoning_effort = cast(ReasoningEffort, coerced)
        _overrides = replace(
            base,
            model_name=model_name,
            reasoning_effort=cast(ReasoningEffort, coerced) if coerced is not None else base.reasoning_effort,
        )
    else:
        _overrides = replace(base, model_name=model_name)
    settings.model_name = model_name
    return coerced if coerced != current_effort else None


def set_active_reasoning_effort(reasoning_effort: ReasoningEffort) -> None:
    """Switch the active OpenRouter reasoning effort at runtime."""
    allowed = reasoning_efforts_for_model(active_model_name())
    if reasoning_effort not in allowed:
        allowed_text = ", ".join(allowed) if allowed else "(none)"
        raise ValueError(f"invalid reasoning effort for model: {reasoning_effort} (allowed: {allowed_text})")
    global _overrides
    base = _overrides or ModelOverrides()
    _overrides = replace(base, reasoning_effort=reasoning_effort)
    settings.reasoning_effort = reasoning_effort


def _resolved_setting(field: str) -> str | int | bool | None:
    if _overrides is not None:
        value = getattr(_overrides, field, None)
        if value is not None:
            return value
    return getattr(settings, field)


def active_model_name() -> str:
    return cast(str, _resolved_setting("model_name"))


def active_reasoning_effort() -> ReasoningEffort:
    return cast(ReasoningEffort, _resolved_setting("reasoning_effort"))


def openrouter_session(thread_id: str, *, suffix: str = "") -> str:
    base = _resolved_setting("openrouter_session_id") or thread_id
    if suffix:
        return f"{base}:{suffix}"
    return cast(str, base)


def _reasoning_kwargs(model_name: str, reasoning_effort: str | int | None) -> dict[str, Any]:
    if not model_supports_reasoning(model_name):
        return {}
    effort = str(reasoning_effort) if reasoning_effort else None
    if not effort or effort == "none":
        return {}
    allowed = reasoning_efforts_for_model(model_name)
    if effort not in allowed:
        effort = coerce_reasoning_effort(model_name, effort)
    if not effort or effort == "none":
        return {}
    return {"reasoning": {"effort": effort}}


def build_chat_model(
    thread_id: str,
    *,
    model_name: str | None = None,
    session_suffix: str = "",
) -> BaseChatModel:
    resolved_model = cast(str, model_name or _resolved_setting("model_name"))
    session_id = openrouter_session(thread_id, suffix=session_suffix)
    base_url = cast(str | None, _resolved_setting("openai_base_url"))
    is_openrouter = not base_url or "openrouter.ai" in base_url
    reasoning = _reasoning_kwargs(
        resolved_model,
        _resolved_setting("reasoning_effort"),
    ).get("reasoning")
    if (
        resolved_model.startswith("anthropic/")
        and is_openrouter
        and settings.openrouter_anthropic_messages
    ):
        return OpenRouterAnthropicMessages(
            model=resolved_model,
            api_key=_resolved_api_key(),
            base_url=(base_url or "https://openrouter.ai/api/v1").rstrip("/"),
            session_id=session_id,
            cache_ttl=settings.openrouter_cache_ttl,
            max_retries=int(_resolved_setting("api_max_retries") or 3),
            reasoning=reasoning,
        )
    model_kwargs: dict[str, Any] = {
        "model": resolved_model,
        "api_key": _resolved_api_key(),
        "session_id": session_id,
    }
    model_kwargs.update(_reasoning_kwargs(resolved_model, _resolved_setting("reasoning_effort")))
    if base_url:
        model_kwargs["base_url"] = base_url
    if resolved_model.startswith("anthropic/") and is_openrouter:
        model_kwargs["model_kwargs"] = {
            "cache_control": {
                "type": "ephemeral",
                "ttl": settings.openrouter_cache_ttl,
            }
        }

    model_kwargs["max_retries"] = _resolved_setting("api_max_retries")
    return ChatOpenRouter(**model_kwargs)


def create_model(thread_id: str) -> BaseChatModel:
    return build_chat_model(thread_id)


def create_compaction_model(thread_id: str) -> BaseChatModel:
    return build_chat_model(thread_id, session_suffix="compaction")


def create_reflection_model(thread_id: str) -> BaseChatModel:
    return build_chat_model(
        thread_id,
        model_name=cast(str, _resolved_setting("reflection_model_name")),
        session_suffix="reflection",
    )


def validate_effort(model_name: str, reasoning_effort: str) -> None:
    allowed = reasoning_efforts_for_model(model_name)
    if not allowed:
        return
    if reasoning_effort not in allowed:
        raise ValueError(
            f"reasoning effort must be one of: {', '.join(allowed)} for model {model_name!r}"
        )


def model_overrides_from_args(args: argparse.Namespace) -> ModelOverrides | None:
    fields = {
        "model_name": args.model,
        "reflection_model_name": args.reflection_model,
        "openai_api_key": args.api_key,
        "openai_base_url": args.base_url,
        "openrouter_session_id": args.openrouter_session_id,
        "reasoning_effort": getattr(args, "reasoning_effort", None),
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
    parser.add_argument(
        "--reasoning-effort",
        help="Provider-literal reasoning effort (overrides REASONING_EFFORT)",
    )
