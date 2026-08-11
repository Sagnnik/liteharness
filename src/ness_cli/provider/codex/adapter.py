from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from ness_cli.config import settings
from ness_cli.provider.base import (
    AccountDetails,
    AuthState,
    LoginMethod,
    LoginResult,
    ModelInfo,
    ProviderAdapter,
    ProviderStatus,
    RateLimitBucket,
)
from ness_cli.provider.codex.app_server import CodexAppServer
from ness_cli.provider.codex.auth import CodexAuth, codex_home
from ness_cli.provider.codex.browser import open_auth_url
from ness_cli.provider.codex.catalog import load_models
from ness_cli.provider.codex.chat_model import CodexSubscriptionChatModel


class CodexProviderAdapter(ProviderAdapter):
    id = "codex"
    display_name = "Codex subscription"
    login_description = "ChatGPT subscription"
    selection_priority = 10
    billing_label = "subscription"

    _device_login_guidance = (
        "Device-code login must first be enabled in ChatGPT Settings > Security: "
        'turn on "Device code authorization for Codex." If you cannot enable it, '
        'choose "Open browser" instead.'
    )

    def __init__(self) -> None:
        self.server = CodexAppServer(codex_home())
        self.auth = CodexAuth(self.server)
        self._models: tuple[ModelInfo, ...] = ()

    def is_authenticated(self) -> bool:
        return self.auth.is_authenticated()

    def build_chat_model(
        self,
        thread_id: str,
        *,
        model_name: str,
        reasoning_effort: str | None,
        session_suffix: str = "",
    ) -> BaseChatModel:
        del thread_id, session_suffix
        return CodexSubscriptionChatModel(
            model=model_name,
            reasoning_effort=reasoning_effort,
            max_retries=settings.api_max_retries,
            auth=self.auth,
        )

    async def models(self, *, refresh: bool = False) -> tuple[ModelInfo, ...]:
        if refresh or not self._models:
            await self.server.start()
            self._models = await load_models(self.server)
        return self._models

    def login_methods(self) -> tuple[LoginMethod, ...]:
        return (
            LoginMethod(
                "browser",
                "Browser sign-in",
                description="Complete sign-in with ChatGPT",
                default=True,
            ),
            LoginMethod(
                "device",
                "Device code",
                description="Requires ChatGPT Settings > Security",
                guidance=self._device_login_guidance,
            ),
        )

    @classmethod
    def _login_error(cls, error: object) -> str:
        message = str(error or "Codex sign-in failed.")
        normalized = message.casefold()
        if "enable device code authorization" in normalized:
            return (
                "Device-code authorization is disabled for this ChatGPT account. "
                + cls._device_login_guidance
                + " Then run /login again."
            )
        return message

    async def login(
        self, *, method: str = "browser", secret: str | None = None
    ) -> LoginResult:
        del secret
        await self.server.start()
        params = (
            {"type": "chatgptDeviceCode"}
            if method == "device"
            else {
                "type": "chatgpt",
                "appBrand": "codex",
                "codexStreamlinedLogin": False,
                "useHostedLoginSuccessPage": True,
            }
        )
        try:
            response = await self.server.request("account/login/start", params)
        except RuntimeError as exc:
            return LoginResult("error", self._login_error(exc))
        return LoginResult(
            "pending",
            "Complete sign-in in your browser.",
            auth_url=response.get("authUrl"),
            login_id=response.get("loginId"),
            user_code=response.get("userCode"),
            verification_url=response.get("verificationUrl"),
        )

    async def wait_for_login(self, login_id: str) -> LoginResult:
        params = await self.server.wait_notification(
            "account/login/completed",
            predicate=lambda item: not item.get("loginId") or item.get("loginId") == login_id,
        )
        if params.get("success"):
            # The completion notification may arrive just before account/read
            # and auth.json converge. Do not force-refresh a brand-new token.
            await self.auth.wait_until_ready()
            return LoginResult("complete", "Signed in with your Codex subscription.")
        return LoginResult("error", self._login_error(params.get("error")))

    async def cancel_login(self, login_id: str) -> None:
        await self.server.request("account/login/cancel", {"loginId": login_id})

    async def open_login_url(self, url: str) -> bool:
        return open_auth_url(url)

    async def logout(self) -> str:
        await self.server.start()
        await self.server.request("account/logout", {})
        self._models = ()
        return "Signed out of Codex subscription."

    async def status(self, *, refresh: bool = False) -> ProviderStatus:
        if not self.is_authenticated():
            return ProviderStatus(self.display_name, AuthState(False, "ChatGPT", "signed out"))
        await self.server.start()
        account_response = await self.server.request("account/read", {"refreshToken": refresh})
        account = account_response.get("account") or {}
        warning: str | None = None
        try:
            limits_response = await self.server.request("account/rateLimits/read", {})
        except Exception as exc:
            limits_response = {}
            warning = f"Usage limits unavailable: {exc}"
        snapshots: list[tuple[str, dict[str, Any]]] = []
        by_id = limits_response.get("rateLimitsByLimitId")
        if isinstance(by_id, dict):
            snapshots.extend(
                (str(key), value) for key, value in by_id.items() if isinstance(value, dict)
            )
        legacy = limits_response.get("rateLimits")
        if isinstance(legacy, dict):
            snapshots.append((str(legacy.get("limitId") or "codex"), legacy))

        buckets: list[RateLimitBucket] = []
        seen: set[tuple[str, int | None, int | None]] = set()
        for limit_id, snapshot in snapshots:
            limit_name = str(snapshot.get("limitName") or limit_id)
            reached = bool(snapshot.get("rateLimitReachedType"))
            for slot, label in (("primary", "primary"), ("secondary", "secondary")):
                window = snapshot.get(slot)
                if not isinstance(window, dict):
                    continue
                duration = window.get("windowDurationMins")
                resets_at = window.get("resetsAt")
                key = (limit_id, int(duration) if duration is not None else None, int(resets_at) if resets_at is not None else None)
                if key in seen:
                    continue
                seen.add(key)
                used = float(window.get("usedPercent") or 0)
                buckets.append(
                    RateLimitBucket(
                        name=f"{limit_name} {label}",
                        window_minutes=int(duration) if duration is not None else None,
                        used_percent=used,
                        remaining_percent=max(0.0, 100.0 - used),
                        resets_at=int(resets_at) if resets_at is not None else None,
                        reached=reached or used >= 100,
                    )
                )
        credits = limits_response.get("rateLimitResetCredits")
        credits_text = None
        if isinstance(credits, dict):
            credits_text = f"{int(credits.get('availableCount') or 0)} available"
        return ProviderStatus(
            provider=self.display_name,
            auth=AuthState(True, "ChatGPT", "managed by Codex CLI"),
            account=AccountDetails(
                email=account.get("email") if isinstance(account, dict) else None,
                tier=account.get("planType") if isinstance(account, dict) else None,
            ),
            limits=tuple(buckets),
            credits=credits_text,
            warning=warning,
        )

    async def close(self) -> None:
        await self.server.close()
