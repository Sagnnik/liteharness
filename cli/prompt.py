"""prompt_toolkit input layer: history, slash completion, bottom toolbar and the
Shift+Tab plan/normal mode toggle.

Kept separate from rendering so the read side (input) and write side (output) do
not entangle. The PromptController pulls live state via callbacks supplied by the
SessionApp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from cli.menu import COMMAND_NAMES, get_command
from cli.theme import PTK_STYLE_RULES


class SlashCompleter(Completer):
    """Complete kept slash-command names once the line starts with '/'."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        word = text[1:]
        for name in COMMAND_NAMES:
            if name.startswith(word):
                spec = get_command(name)
                yield Completion(
                    name,
                    start_position=-len(word),
                    display=f"/{name}",
                    display_meta=spec.summary if spec else "",
                )


class PromptController:
    def __init__(
        self,
        *,
        get_mode: Callable[[], str],
        get_model: Callable[[], str],
        get_usage: Callable[[], dict[str, Any] | None],
        toggle_mode: Callable[[], None],
        history_path: Path,
    ) -> None:
        self._get_mode = get_mode
        self._get_model = get_model
        self._get_usage = get_usage
        self._toggle_mode = toggle_mode

        history_path.parent.mkdir(parents=True, exist_ok=True)
        self.session: PromptSession = PromptSession(
            history=FileHistory(str(history_path)),
            completer=SlashCompleter(),
            complete_while_typing=True,
            bottom_toolbar=self._bottom_toolbar,
            style=Style.from_dict(PTK_STYLE_RULES),
            key_bindings=self._key_bindings(),
            message=self._prompt_fragments,
        )

    def _key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("s-tab")
        def _toggle(event) -> None:
            self._toggle_mode()
            event.app.invalidate()

        return kb

    def _prompt_fragments(self):
        mode = self._get_mode()
        style = "class:prompt.mode.plan" if mode == "plan" else "class:prompt.mode"
        return [(style, f"{mode} "), ("class:prompt", "› ")]

    def _bottom_toolbar(self):
        mode = self._get_mode()
        model = self._get_model()
        usage = self._get_usage() or {}
        frags: list[tuple[str, str]] = [("class:bottom-toolbar", " ")]
        frags.append(("class:bottom-toolbar.key", "mode "))
        frags.append(("class:bottom-toolbar", mode))
        frags.append(("class:bottom-toolbar.sep", "  •  "))
        frags.append(("class:bottom-toolbar.key", "model "))
        frags.append(("class:bottom-toolbar", model))
        if usage:
            inp = int(usage.get("input_tokens", 0) or 0)
            out = int(usage.get("output_tokens", 0) or 0)
            frags.append(("class:bottom-toolbar.sep", "  •  "))
            frags.append(("class:bottom-toolbar.key", "last "))
            frags.append(("class:bottom-toolbar", f"↑{inp:,} ↓{out:,}"))
        frags.append(("class:bottom-toolbar.sep", "  •  Shift+Tab mode  /menu"))
        return frags

    async def ask(self) -> str:
        return await self.session.prompt_async()
