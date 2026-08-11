from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.language_models import BaseChatModel


@dataclass(frozen=True)
class AuthState:
    authenticated: bool
    method: str = ""
    detail: str = ""


@dataclass(frozen=True)
class AccountDetails:
    email: str | None = None
    tier: str | None = None


@dataclass(frozen=True)
class RateLimitBucket:
    name: str
    window_minutes: int | None = None
    used_percent: float | None = None
    remaining_percent: float | None = None
    resets_at: int | None = None
    reached: bool = False


@dataclass(frozen=True)
class ProviderStatus:
    provider: str
    auth: AuthState
    account: AccountDetails = field(default_factory=AccountDetails)
    limits: tuple[RateLimitBucket, ...] = ()
    credits: str | None = None
    warning: str | None = None


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    default_reasoning_effort: str | None = None
    reasoning_efforts: tuple[str, ...] = ()
    supports_vision: bool = False
    is_default: bool = False


@dataclass(frozen=True)
class LoginResult:
    status: Literal["complete", "pending", "cancelled", "error"]
    message: str
    auth_url: str | None = None
    login_id: str | None = None
    user_code: str | None = None
    verification_url: str | None = None


@dataclass(frozen=True)
class LoginMethod:
    id: str
    label: str
    description: str = ""
    default: bool = False
    guidance: str | None = None
    input_kind: Literal["none", "secret"] = "none"
    input_label: str | None = None
    input_example: str = ""


class ProviderAdapter(ABC):
    """Boundary between provider-specific behavior and the rest of the CLI."""

    id: str
    display_name: str
    login_description: str = ""
    selection_priority: int = 100
    billing_label: str = "unknown"

    @abstractmethod
    def is_authenticated(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def build_chat_model(
        self,
        thread_id: str,
        *,
        model_name: str,
        reasoning_effort: str | None,
        session_suffix: str = "",
    ) -> BaseChatModel:
        raise NotImplementedError

    @abstractmethod
    async def models(self, *, refresh: bool = False) -> tuple[ModelInfo, ...]:
        raise NotImplementedError

    @abstractmethod
    async def status(self, *, refresh: bool = False) -> ProviderStatus:
        raise NotImplementedError

    def login_methods(self) -> tuple[LoginMethod, ...]:
        """Describe interactive login choices without leaking provider details into the TUI."""
        return ()

    async def login(
        self, *, method: str = "browser", secret: str | None = None
    ) -> LoginResult:
        del method, secret
        return LoginResult("error", f"{self.display_name} does not support interactive login.")

    async def wait_for_login(self, login_id: str) -> LoginResult:
        return LoginResult("error", f"{self.display_name} does not support interactive login.")

    async def cancel_login(self, login_id: str) -> None:
        return None

    async def open_login_url(self, url: str) -> bool:
        """Open a provider login URL, returning whether launch was attempted."""
        return False

    async def logout(self) -> str:
        return f"{self.display_name} does not support logout."

    async def close(self) -> None:
        return None
