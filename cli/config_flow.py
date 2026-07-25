from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from cli import render
from cli.constants import FORM_LABELS
from cli.utils import write_env
from liteharness_cli.chat_model import (
    active_model_name,
    active_reasoning_effort,
    set_active_model,
    set_active_reasoning_effort,
)
from liteharness_cli.config import reasoning_efforts_for_model, reload_settings, settings


# --- shared /config data + delegator ---------------------------------------
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
    """Run the active TUI config picker; fall back to a notice when headless."""
    sink = render.get_sink()
    if sink is None:
        result = ConfigResult()
        result.note("/config is available only in the interactive TUI.")
        return result
    return await sink.run_config()


# --- interactive /config picker + credential forms ------------------------
class ConfigFlowMixin:
    """Interactive /config picker and credential forms."""

    async def run_config(self) -> ConfigResult:
        if self._config_future is not None and not self._config_future.done():
            return await self._config_future
        self._config_future = asyncio.get_running_loop().create_future()
        self._config_result = ConfigResult()
        self._model_pick_changed = False
        self._model_pick_name = ""
        self._open_picker("config_action", "/config", index=0)
        result = await self._config_future
        self._config_future = None
        self._config_result = None
        return result

    def _finish_config(self) -> None:
        if self._config_future is not None and not self._config_future.done():
            self._config_future.set_result(self._config_result or ConfigResult())
        self._close_menu()
        self._close_form(reset_buffer=True)
        self._reset_buffer()

    def _config_note(self, message: str) -> None:
        if self._config_result is not None:
            self._config_result.note(message)

    def _config_rebuild(self) -> None:
        if self._config_result is not None:
            self._config_result.rebuild = True

    def _config_session_update(self) -> None:
        if self._config_result is not None:
            self._config_result.mark_session_update()

    def _apply_config_action(self, key: str) -> None:
        if key == "model":
            current = self._current_model_slug()
            index = next((i for i, item in enumerate(self._config_model_items()) if item.key == current), 0)
            self._open_picker("config_models", "/config", index=index)
            return
        if key == "reasoning":
            self._open_config_reasoning_picker()
            return
        if key == "approval":
            settings.enable_approval = not settings.enable_approval
            write_env("ENABLE_APPROVAL", "true" if settings.enable_approval else "false")
            self._config_session_update()
            self._config_note(f"Approval {'on' if settings.enable_approval else 'off'}.")
            self._finish_config()
            return
        if key == "autosave":
            settings.auto_save_threads = not settings.auto_save_threads
            write_env("AUTO_SAVE_THREADS", "true" if settings.auto_save_threads else "false")
            self._config_session_update()
            self._config_note(f"Autosave {'on' if settings.auto_save_threads else 'off'}.")
            self._finish_config()
            return
        if key == "session_end_reflection":
            settings.session_end_reflection = not settings.session_end_reflection
            write_env("SESSION_END_REFLECTION", "true" if settings.session_end_reflection else "false")
            self._config_session_update()
            self._config_note(f"Session end reflection {'on' if settings.session_end_reflection else 'off'}.")
            self._finish_config()
            return
        if key == "view":
            self._config_note("\n".join(current_config_lines()))
            self._finish_config()
            return
        if key in FORM_LABELS:
            self._open_form(key, FORM_LABELS[key])

    def _open_config_reasoning_picker(self) -> None:
        model_name = active_model_name()
        efforts = reasoning_efforts_for_model(model_name)
        if not efforts:
            self._config_note("Current model does not support reasoning effort.")
            self._finish_config()
            return
        current_effort = active_reasoning_effort()
        index = next((i for i, item in enumerate(self._config_reasoning_items()) if item.key == current_effort), 0)
        self._open_picker("config_reasoning", "/config", index=index)

    def _apply_config_model(self, model_name: str) -> None:
        current = self._current_model_slug()
        self._model_pick_changed = model_name != current
        self._model_pick_name = model_name
        coerced: str | None = None
        if self._model_pick_changed:
            coerced = set_active_model(model_name)
            write_env("MODEL_NAME", model_name)
            if coerced:
                write_env("REASONING_EFFORT", coerced)
            self._config_rebuild()
            self._config_session_update()
        efforts = reasoning_efforts_for_model(model_name)
        if not efforts:
            if self._config_result is not None:
                note_model_reasoning_changes(
                    self._config_result,
                    model_changed=self._model_pick_changed,
                    model_name=self._model_pick_name,
                    reasoning_changed=bool(coerced),
                    reasoning=coerced or active_reasoning_effort(),
                )
            self._finish_config()
            return
        self._open_config_reasoning_picker()

    def _apply_config_reasoning(self, effort: str) -> None:
        current = active_reasoning_effort()
        reasoning_changed = effort != current
        if reasoning_changed:
            set_active_reasoning_effort(effort)  # type: ignore[arg-type]
            write_env("REASONING_EFFORT", effort)
            self._config_rebuild()
        if self._config_result is not None:
            note_model_reasoning_changes(
                self._config_result,
                model_changed=self._model_pick_changed,
                model_name=self._model_pick_name,
                reasoning_changed=reasoning_changed,
                reasoning=effort,
            )
        self._finish_config()

    def _open_form(self, kind: str, label: str) -> None:
        self._close_menu()
        self._form_kind = kind
        self._form_label = label
        self._form_buffer.text = ""
        self._form_buffer.cursor_position = 0
        self._set_buffer_text("/config")
        self._focus_form_field()
        self.invalidate()

    def _close_form(self, *, reset_buffer: bool = True) -> None:
        had_form = self._form_kind is not None
        self._form_kind = None
        self._form_label = ""
        self._form_buffer.text = ""
        if had_form and reset_buffer:
            self._reset_buffer()
        self._focus_command_input()

    def _submit_form(self) -> None:
        kind = self._form_kind
        value = self._form_buffer.text.strip()
        if kind is None:
            return
        if kind == "openai_key":
            if not value:
                self.append_error("Provider API key cannot be empty.")
                return
            write_env("OPENAI_API_KEY", value)
            reload_settings()
            self._config_rebuild()
            self._config_note("Provider API key saved to .env.")
        elif kind == "exa_key":
            if not value:
                self.append_error("Exa API key cannot be empty.")
                return
            write_env("EXA_API_KEY", value)
            reload_settings()
            self._config_note("Exa API key saved to .env.")
        elif kind == "base_url":
            write_env("OPENAI_BASE_URL", value)
            reload_settings()
            self._config_rebuild()
            self._config_note("Base URL saved to .env." if value else "Base URL cleared.")
        self._finish_config()
