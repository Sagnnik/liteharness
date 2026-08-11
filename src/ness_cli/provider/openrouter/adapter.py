from __future__ import annotations

import sys
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openrouter import ChatOpenRouter

from ness_cli.config import coerce_reasoning_effort, model_supports_reasoning, reasoning_efforts_for_model, settings
from ness_cli.provider.openrouter.catalog import cached_models, refresh_catalog
from ness_cli.provider.base import (
    AuthState,
    LoginMethod,
    LoginResult,
    ModelInfo,
    ProviderAdapter,
    ProviderStatus,
)
from ness_cli.provider.openrouter.chat_model import OpenRouterAnthropicMessages

_MISSING_API_KEY = "sk-missing-api-key"


def _runtime_setting(field: str) -> Any:
    """Read process-local CLI overrides without owning them in this provider."""
    facade = sys.modules.get("ness_cli.chat_model")
    resolver = getattr(facade, "_resolved_setting", None)
    return resolver(field) if callable(resolver) else getattr(settings, field)


def _chat_openrouter_class():
    # The facade attribute remains a compatibility patch point for embedders
    # and the pre-provider test suite; construction still lives here.
    facade = sys.modules.get("ness_cli.chat_model")
    return getattr(facade, "ChatOpenRouter", ChatOpenRouter)


class OpenRouterProviderAdapter(ProviderAdapter):
    id = "openrouter"
    display_name = "OpenRouter"
    login_description = "API key"
    selection_priority = 20
    billing_label = "API billing"

    def is_authenticated(self) -> bool:
        return bool(_runtime_setting("openai_api_key"))

    def build_chat_model(
        self,
        thread_id: str,
        *,
        model_name: str,
        reasoning_effort: str | None,
        session_suffix: str = "",
    ) -> BaseChatModel:
        base = _runtime_setting("openrouter_session_id") or thread_id
        session_id = f"{base}:{session_suffix}" if session_suffix else base
        base_url = _runtime_setting("openai_base_url")
        is_openrouter = not base_url or "openrouter.ai" in base_url
        reasoning: dict[str, str] | None = None
        if model_supports_reasoning(model_name) and reasoning_effort and reasoning_effort != "none":
            effort = reasoning_effort
            if effort not in reasoning_efforts_for_model(model_name):
                effort = coerce_reasoning_effort(model_name, effort)
            if effort and effort != "none":
                reasoning = {"effort": effort}
        api_key = _runtime_setting("openai_api_key") or _MISSING_API_KEY
        if model_name.startswith("anthropic/") and is_openrouter and settings.openrouter_anthropic_messages:
            return OpenRouterAnthropicMessages(
                model=model_name,
                api_key=api_key,
                base_url=(base_url or "https://openrouter.ai/api/v1").rstrip("/"),
                session_id=session_id,
                cache_ttl=settings.openrouter_cache_ttl,
                max_retries=int(_runtime_setting("api_max_retries") or 3),
                reasoning=reasoning,
            )
        kwargs: dict[str, Any] = {
            "model": model_name,
            "api_key": api_key,
            "session_id": session_id,
            "max_retries": int(_runtime_setting("api_max_retries") or 3),
        }
        if reasoning:
            kwargs["reasoning"] = reasoning
        if base_url:
            kwargs["base_url"] = base_url
        if model_name.startswith("anthropic/") and is_openrouter:
            kwargs["model_kwargs"] = {"cache_control": {"type": "ephemeral", "ttl": settings.openrouter_cache_ttl}}
        return _chat_openrouter_class()(**kwargs)

    async def models(self, *, refresh: bool = False) -> tuple[ModelInfo, ...]:
        if refresh:
            await refresh_catalog()
        return tuple(
            ModelInfo(
                id=item.id,
                name=item.name,
                reasoning_efforts=item.reasoning_efforts,
                supports_vision=item.supports_vision,
            )
            for item in cached_models()
        )

    async def status(self, *, refresh: bool = False) -> ProviderStatus:
        return ProviderStatus(
            provider=self.display_name,
            auth=AuthState(self.is_authenticated(), "API key", "configured" if self.is_authenticated() else "missing"),
        )

    def login_methods(self) -> tuple[LoginMethod, ...]:
        return (
            LoginMethod(
                "api_key",
                "API key",
                description="Stored securely in Ness configuration",
                default=True,
                input_kind="secret",
                input_label="OpenRouter API key",
                input_example="sk-or-v1-...",
            ),
        )

    async def login(
        self, *, method: str = "api_key", secret: str | None = None
    ) -> LoginResult:
        if method != "api_key":
            return LoginResult("error", f"Unsupported OpenRouter login method: {method}")
        key = (secret or "").strip()
        if not key:
            return LoginResult("cancelled", "OpenRouter sign-in was cancelled.")
        from ness_cli.config_store import write_secret

        write_secret("openai_api_key", key)
        settings.openai_api_key = key
        return LoginResult("complete", "Saved the OpenRouter API key.")

    async def logout(self) -> str:
        from ness_cli.config_store import write_secret

        write_secret("openai_api_key", None)
        settings.openai_api_key = None
        return "Removed the saved OpenRouter API key."
