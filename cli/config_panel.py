"""Interactive /config flows: API keys, model switch, and toggles.

Changes are persisted to the project .env via python-dotenv so they survive
across runs, then re-applied to the live settings. Model/key/base-url changes
require the caller to rebuild the graph (signalled by ConfigResult.rebuild).

The pickers are custom prompt_toolkit overlays that share the look and key
handling of the /menu overlay (black background, gray modal, white text,
arrow keys to move, Enter to select, Esc to cancel) rather than the built-in
radiolist_dialog/input_dialog, which ship the default blue dialog theme and a
focus model that is awkward to drive with the keyboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import set_key
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import BeforeInput, PasswordProcessor
from prompt_toolkit.layout.screen import Point
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame

from config import AVAILABLE_MODELS, reload_settings, settings
from model import active_model_name, set_active_model
from permissions import PROJECT_ROOT
from cli.theme import PTK_STYLE_RULES

ENV_PATH = PROJECT_ROOT / ".env"
_STYLE = Style.from_dict(PTK_STYLE_RULES)


@dataclass
class ConfigResult:
    rebuild: bool = False
    messages: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        self.messages.append(message)


def _write_env(key: str, value: str) -> None:
    ENV_PATH.touch(exist_ok=True)
    set_key(str(ENV_PATH), key, value, quote_mode="never")


# --- shared overlays --------------------------------------------------------


async def _select(title: str, text: str, values: list[tuple[str, str]]) -> str | None:
    """Arrow-key picker styled like /menu. Returns the chosen key or None."""
    if not values:
        return None
    state = {"index": 0}

    def list_fragments():
        frags = []
        for i, (_, label) in enumerate(values):
            current = i == state["index"]
            style = "class:menu.item.current" if current else "class:menu.item"
            prefix = "› " if current else "  "
            frags.append((style, f"{prefix}{label}\n"))
        return frags

    def cursor_position() -> Point:
        return Point(0, state["index"])

    body = Window(
        content=FormattedTextControl(list_fragments, focusable=True, get_cursor_position=cursor_position),
        always_hide_cursor=True,
        wrap_lines=False,
        height=Dimension(min=1, max=18),
        style="class:menu.body",
    )

    layout = Layout(
        HSplit(
            [
                Window(FormattedTextControl([("class:menu.hint", f"  {text}")]), height=1, style="class:menu.bar"),
                Frame(body, title=title, style="class:menu.frame"),
                Window(
                    FormattedTextControl([("class:menu.hint", "  ↑/↓ move   ⏎ select   esc cancel")]),
                    height=1,
                    style="class:menu.bar",
                ),
            ],
            style="class:menu.screen",
        )
    )

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("c-p")
    def _up(event) -> None:
        state["index"] = (state["index"] - 1) % len(values)

    @kb.add("down")
    @kb.add("c-n")
    def _down(event) -> None:
        state["index"] = (state["index"] + 1) % len(values)

    @kb.add("enter")
    def _ok(event) -> None:
        event.app.exit(result=values[state["index"]][0])

    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event) -> None:
        event.app.exit(result=None)

    app: Application = Application(
        layout=layout,
        key_bindings=kb,
        style=_STYLE,
        full_screen=True,
        mouse_support=False,
    )
    return await app.run_async()


async def _prompt_text(title: str, text: str, *, password: bool = False) -> str | None:
    """Single-line text entry styled like /menu. Returns text or None if cancelled."""
    buffer = Buffer(multiline=False)
    processors = [BeforeInput("  ")]
    if password:
        processors.append(PasswordProcessor())
    input_window = Window(
        content=BufferControl(buffer=buffer, input_processors=processors, focusable=True),
        height=1,
        style="class:menu.body",
    )

    layout = Layout(
        HSplit(
            [
                Window(FormattedTextControl([("class:menu.hint", f"  {text}")]), height=1, style="class:menu.bar"),
                Frame(input_window, title=title, style="class:menu.frame"),
                Window(
                    FormattedTextControl([("class:menu.hint", "  ⏎ submit   esc cancel")]),
                    height=1,
                    style="class:menu.bar",
                ),
            ],
            style="class:menu.screen",
        )
    )

    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event) -> None:
        event.app.exit(result=buffer.text)

    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event) -> None:
        event.app.exit(result=None)

    app: Application = Application(
        layout=layout,
        key_bindings=kb,
        style=_STYLE,
        full_screen=True,
        mouse_support=False,
        focused_element=input_window,
    )
    return await app.run_async()


def current_config_lines() -> list[str]:
    return [
        f"model         {active_model_name()}",
        f"reflection    {settings.reflection_model_name}",
        f"base url      {settings.openai_base_url or '(default)'}",
        f"approval      {'on' if settings.enable_approval else 'off'}",
        f"autosave      {'on' if settings.auto_save_threads else 'off'}",
        f"OpenAI key    {'set' if settings.openai_api_key else 'missing'}",
        f"Exa key       {'set' if settings.exa_api_key else 'missing'}",
    ]


async def run_config() -> ConfigResult:
    """Top-level /config action picker. Returns the accumulated effects."""
    result = ConfigResult()
    action = await _select(
        title="Config",
        text="What would you like to change?",
        values=[
            ("model", "Switch chat model"),
            ("openai_key", "Set OpenAI / OpenRouter API key"),
            ("exa_key", "Set Exa API key (web search)"),
            ("base_url", "Set OpenAI-compatible base URL"),
            ("approval", f"Toggle approval (now: {'on' if settings.enable_approval else 'off'})"),
            ("autosave", f"Toggle thread autosave (now: {'on' if settings.auto_save_threads else 'off'})"),
            ("view", "View current config"),
        ],
    )

    if action is None:
        return result
    if action == "model":
        await _switch_model(result)
    elif action == "openai_key":
        await _set_secret(result, "OPENAI_API_KEY", "OpenAI / OpenRouter API key", rebuild=True)
    elif action == "exa_key":
        await _set_secret(result, "EXA_API_KEY", "Exa API key", rebuild=False)
    elif action == "base_url":
        await _set_base_url(result)
    elif action == "approval":
        _toggle(result, "enable_approval", "ENABLE_APPROVAL", "Approval")
    elif action == "autosave":
        _toggle(result, "auto_save_threads", "AUTO_SAVE_THREADS", "Autosave")
    elif action == "view":
        result.messages.extend(current_config_lines())
    return result


async def _switch_model(result: ConfigResult) -> None:
    current = active_model_name()
    values = [(name, f"{name}{'  (current)' if name == current else ''}") for name in AVAILABLE_MODELS]
    if current not in AVAILABLE_MODELS:
        values.insert(0, (current, f"{current}  (current)"))
    chosen = await _select(
        title="Switch model",
        text="Select the chat model:",
        values=values,
    )
    if not chosen or chosen == current:
        result.note("Model unchanged.")
        return
    set_active_model(chosen)
    _write_env("MODEL_NAME", chosen)
    result.rebuild = True
    result.note(f"Model switched to {chosen}.")


async def _set_secret(result: ConfigResult, env_key: str, label: str, *, rebuild: bool) -> None:
    value = await _prompt_text(
        title=label,
        text=f"Enter {label} (stored in .env):",
        password=True,
    )
    if not value:
        result.note(f"{label} unchanged.")
        return
    _write_env(env_key, value.strip())
    reload_settings()
    result.rebuild = rebuild
    result.note(f"{label} saved to .env.")


async def _set_base_url(result: ConfigResult) -> None:
    value = await _prompt_text(
        title="Base URL",
        text="Enter OpenAI-compatible base URL (blank to clear):",
    )
    if value is None:
        result.note("Base URL unchanged.")
        return
    _write_env("OPENAI_BASE_URL", value.strip())
    reload_settings()
    result.rebuild = True
    result.note("Base URL saved to .env." if value.strip() else "Base URL cleared.")


def _toggle(result: ConfigResult, attr: str, env_key: str, label: str) -> None:
    new_value = not bool(getattr(settings, attr))
    setattr(settings, attr, new_value)
    _write_env(env_key, "true" if new_value else "false")
    result.note(f"{label} {'on' if new_value else 'off'}.")
