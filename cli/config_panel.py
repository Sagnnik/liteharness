"""Shared /config result helpers for the TUI config flow."""

from __future__ import annotations

from dataclasses import dataclass, field

from config import settings
from model import active_model_name, active_reasoning_effort
from cli import render


@dataclass
class ConfigResult:
    rebuild: bool = False
    session_update: bool = False
    messages: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        self.messages.append(message)

    def mark_session_update(self) -> None:
        self.session_update = True


def note_model_reasoning_changes(
    result: ConfigResult,
    *,
    model_changed: bool,
    model_name: str,
    reasoning_changed: bool,
    reasoning: str,
) -> None:
    parts: list[str] = []
    if model_changed:
        parts.append(f"Model switched to {model_name}")
    if reasoning_changed:
        parts.append(f"Reasoning effort set to {reasoning}")
    if parts:
        result.note("; ".join(parts) + ".")


def current_config_lines() -> list[str]:
    return [
        f"model         {active_model_name()}",
        f"reasoning     {active_reasoning_effort()}",
        f"reflection    {settings.reflection_model_name}",
        f"base url      {settings.openai_base_url or '(default)'}",
        f"approval      {'on' if settings.enable_approval else 'off'}",
        f"autosave      {'on' if settings.auto_save_threads else 'off'}",
        f"session end reflection  {'on' if settings.session_end_reflection else 'off'}",
        f"provider key  {'set' if settings.openai_api_key else 'missing'}",
        f"Exa key       {'set' if settings.exa_api_key else 'missing'}",
    ]


async def run_config() -> ConfigResult:
    """Run the active TUI config picker."""
    sink = render.get_sink()
    if sink is None:
        result = ConfigResult()
        result.note("/config is available only in the interactive TUI.")
        return result
    return await sink.run_config()
