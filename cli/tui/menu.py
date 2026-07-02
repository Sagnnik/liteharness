from __future__ import annotations

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document

from cli.config_panel import ConfigResult
from cli.menu import COMMAND_CATALOG
from cli.tui.constants import MENU_DESC_COL, MENU_MAX_ROWS, PICKER_MODES
from cli.tui.models import MenuItem
from cli.tui.utils import term_width
from config import AVAILABLE_MODELS, reasoning_efforts_for_model, settings
from model import active_model_name, active_reasoning_effort


class MenuMixin:
    """Slash completion, config pickers, and menu rendering."""

    def _slash_filter(self) -> str:
        text = self._buffer.text
        return text[1:] if text.startswith("/") else ""

    def _on_buffer_changed(self, buffer: Buffer) -> None:
        if self._ignore_buffer_menu or self._menu_kind in PICKER_MODES or self._form_kind or self._prompt_kind:
            return
        text = buffer.text
        if text.startswith("/") and " " not in text and self._slash_menu_items():
            was_slash = self._menu_kind == "slash"
            self._menu_kind = "slash"
            if not was_slash:
                self._menu_index = 0
                self._menu_scroll = 0
            self._sync_slash_index()
        elif self._menu_kind == "slash":
            self._close_menu()

    def _sync_slash_index(self) -> None:
        query = self._slash_filter().lower()
        if not query:
            return
        for i, item in enumerate(self._slash_menu_items()):
            if item.key.lower() == query:
                self._menu_index = i
                self._clamp_menu_index()
                return

    def _current_model_slug(self) -> str:
        current = active_model_name()
        for slug in AVAILABLE_MODELS:
            if slug == current or slug.endswith(f"/{current}"):
                return slug
        return current

    def _config_action_items(self) -> list[MenuItem]:
        items = [
            MenuItem("model", "Switch chat model"),
            MenuItem("reasoning", "Switch reasoning effort"),
            MenuItem("openai_key", "Set provider API key"),
            MenuItem("exa_key", "Set Exa API key (web search)"),
            MenuItem("base_url", "Set OpenAI-compatible base URL"),
            MenuItem("approval", f"Toggle approval (now: {'on' if settings.enable_approval else 'off'})"),
            MenuItem("autosave", f"Toggle thread autosave (now: {'on' if settings.auto_save_threads else 'off'})"),
            MenuItem(
                "session_end_reflection",
                f"Toggle session end reflection (now: {'on' if settings.session_end_reflection else 'off'})",
            ),
            MenuItem("view", "View current config"),
        ]
        return items

    def _config_model_items(self) -> list[MenuItem]:
        current = self._current_model_slug()
        items = [MenuItem(slug, slug, suffix="(current)" if slug == current else "") for slug in AVAILABLE_MODELS]
        if current not in AVAILABLE_MODELS:
            items.insert(0, MenuItem(current, current, suffix="(current)"))
        return items

    def _config_reasoning_items(self) -> list[MenuItem]:
        current = active_reasoning_effort()
        levels = reasoning_efforts_for_model(active_model_name())
        return [MenuItem(level, level, suffix="(current)" if level == current else "") for level in levels]

    def _slash_menu_items(self) -> list[MenuItem]:
        query = self._slash_filter().lower()
        if query == self._slash_menu_cache_query:
            return self._slash_menu_cache_items
        self._slash_menu_cache_query = query
        self._slash_menu_cache_items = [
            MenuItem(spec.name, spec.name, description=spec.summary)
            for spec in COMMAND_CATALOG
            if spec.name.startswith(query)
        ]
        return self._slash_menu_cache_items

    def _invalidate_slash_menu_cache(self) -> None:
        self._slash_menu_cache_query = None
        self._slash_menu_cache_items = []

    def _visible_menu_items(self) -> list[MenuItem]:
        builders = {
            "config_models": self._config_model_items,
            "config_reasoning": self._config_reasoning_items,
            "config_action": self._config_action_items,
            "slash": self._slash_menu_items,
            "approval": lambda: self._prompt_items,
            "question": lambda: self._prompt_items,
        }
        builder = builders.get(self._menu_kind or "")
        return builder() if builder else []

    def _menu_body_height(self) -> int:
        if not self._menu_kind:
            return 0
        rows = min(MENU_MAX_ROWS, max(1, len(self._visible_menu_items())))
        if self._prompt_detail_lines:
            rows += min(5, len(self._prompt_detail_lines))
        return rows

    def _clamp_menu_scroll(self) -> None:
        items = self._visible_menu_items()
        if not items:
            self._menu_scroll = 0
            return
        if self._menu_index < self._menu_scroll:
            self._menu_scroll = self._menu_index
        elif self._menu_index >= self._menu_scroll + MENU_MAX_ROWS:
            self._menu_scroll = self._menu_index - MENU_MAX_ROWS + 1

    def _clamp_menu_index(self) -> None:
        items = self._visible_menu_items()
        if not items:
            self._menu_index = 0
            self._menu_scroll = 0
            return
        self._menu_index = max(0, min(self._menu_index, len(items) - 1))
        self._clamp_menu_scroll()

    def _wheel_menu_index(self, delta: int) -> None:
        items = self._visible_menu_items()
        if not items:
            return
        self._menu_index = (self._menu_index + delta) % len(items)
        self._clamp_menu_scroll()

    def _menu_header_fragments(self):
        headers = {
            "config_models": "/config > models - Select the chat model:",
            "config_reasoning": "/config > reasoning - Select reasoning effort:",
            "config_action": "/config - What would you like to change:",
            "approval": self._prompt_title,
            "question": self._prompt_title,
        }
        title = headers.get(self._menu_kind or "")
        if not title:
            return []
        right = self._prompt_hint or "↑/↓ scroll"
        gap = max(1, term_width() - len(title) - len(right))
        return [("class:chrome.menu.header", title), ("class:chrome.menu.hint", (" " * gap) + right)]

    def _menu_row_fragments(self, item: MenuItem, *, selected: bool) -> list[tuple[str, str]]:
        width = term_width()
        prefix = "-> " if selected else "   "
        left = f"{prefix}{item.label}"
        if item.suffix:
            left = f"{left}  {item.suffix}"
        if not selected:
            row = left.ljust(MENU_DESC_COL) + item.description if item.description else left
            return [("class:chrome.menu.row", row[:width].ljust(width))]
        frags: list[tuple[str, str]] = [
            ("class:chrome.menu.row.current", prefix[:1]),
            ("class:chrome.menu.arrow", prefix[1:]),
            ("class:chrome.menu.label.current", item.label),
        ]
        if item.suffix:
            frags.extend([("class:chrome.menu.row.current", "  "), ("class:chrome.menu.suffix", item.suffix)])
        if item.description:
            used = len(prefix) + len(item.label) + (len(item.suffix) + 2 if item.suffix else 0)
            frags.append(("class:chrome.menu.row.current", " " * max(1, MENU_DESC_COL - used)))
            desc = item.description[: max(0, width - MENU_DESC_COL)]
            frags.extend(
                [
                    ("class:chrome.menu.desc.current", desc),
                    ("class:chrome.menu.row.current", " " * max(0, width - MENU_DESC_COL - len(desc))),
                ]
            )
        else:
            frags.append(("class:chrome.menu.row.current", " " * max(0, width - len(left))))
        return frags

    def _menu_body_fragments(self):
        items = self._visible_menu_items()
        if not self._menu_kind:
            return []
        if not items:
            return [("class:chrome.menu.row", "   no options")]
        self._clamp_menu_index()
        visible = items[self._menu_scroll : self._menu_scroll + MENU_MAX_ROWS]
        fragments: list[tuple[str, str]] = []
        for offset, item in enumerate(visible):
            index = self._menu_scroll + offset
            fragments.extend(self._menu_row_fragments(item, selected=index == self._menu_index and not self._prompt_note_active))
            fragments.append(("class:chrome.menu.row", "\n"))
        for line in self._prompt_detail_lines[:5]:
            fragments.append(("class:chrome.menu.hint", f"   {line[:term_width() - 3]}\n"))
        return fragments

    def _close_menu(self) -> None:
        self._menu_kind = None
        self._menu_index = 0
        self._menu_scroll = 0
        self._invalidate_slash_menu_cache()

    def _cancel_menu(self) -> None:
        if self._config_future is not None and not self._config_future.done():
            self._config_future.set_result(self._config_result or ConfigResult())
        if self._prompt_future is not None and not self._prompt_future.done():
            self._prompt_future.set_result(None)
        self._clear_prompt()
        self._close_menu()
        self._reset_buffer()

    def _write_buffer_text(self, text: str) -> None:
        self._buffer.set_document(Document(text, cursor_position=len(text)), bypass_readonly=True)

    def _complete_slash_selection(self) -> None:
        items = self._visible_menu_items()
        if not items:
            return
        item = items[self._menu_index]
        self._write_buffer_text(f"/{item.key}")

    def _set_buffer_text(self, text: str) -> None:
        self._ignore_buffer_menu = True
        saved_kind = self._menu_kind
        try:
            self._menu_kind = None
            self._write_buffer_text(text)
        finally:
            self._menu_kind = saved_kind
            self._ignore_buffer_menu = False

    def _reset_buffer(self) -> None:
        self._set_buffer_text("")

    def _open_picker(self, kind: str, buffer_text: str, *, index: int = 0) -> None:
        self._ignore_buffer_menu = True
        try:
            self._menu_kind = None
            self._write_buffer_text(buffer_text)
            self._menu_kind = kind
            self._menu_index = index
            self._menu_scroll = 0
        finally:
            self._ignore_buffer_menu = False
        self._clamp_menu_index()
        self._focus_command_input()
        self.invalidate()

    def _selected_slash_command(self) -> str | None:
        items = self._visible_menu_items()
        if not items:
            return None
        item = items[self._menu_index]
        return f"/{item.key}"

    def _apply_picker_selection(self) -> None:
        items = self._visible_menu_items()
        if not items:
            return
        item = items[self._menu_index]
        if self._menu_kind == "config_action":
            self._apply_config_action(item.key)
        elif self._menu_kind == "config_models":
            self._apply_config_model(item.key)
        elif self._menu_kind == "config_reasoning":
            self._apply_config_reasoning(item.key)
        elif self._menu_kind == "approval":
            self._apply_approval_selection(item.key)
        elif self._menu_kind == "question":
            self._submit_question()
