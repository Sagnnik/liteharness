from __future__ import annotations

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document

from ness_cli.tui import mentions as mention_mod
from ness_cli.tui.config_flow import ConfigResult
from ness_cli.tui.config_registry import (
    SECTION_DESCRIPTIONS,
    SECTION_TITLES,
    SPEC_BY_KEY,
    format_current,
    specs_for_section,
)
from ness_cli.tui.command_catalog import COMMAND_CATALOG
from ness_cli.tui.constants import (
    MENU_DESC_COL,
    MENU_MAX_ROWS,
    MENTION_MAX_ROWS,
    MENTION_MENU,
    PICKER_MODES,
)
from ness_cli.tui.models import MenuItem
from ness_cli.tui.utils import term_width
from ness_cli.chat_model import active_model_name, active_reasoning_effort
from ness_cli.config import (
    available_model_ids,
    reasoning_efforts_for_model,
    settings,
)
from ness_cli.model_catalog import model_record

# Characters allowed inside an @mention token after the `@`.
_PATH_TOKEN_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./_-")


class MenuMixin:
    """Slash completion, config pickers, and menu rendering."""

    def _slash_filter(self) -> str:
        text = self._buffer.text
        return text[1:] if text.startswith("/") else ""

    def _on_buffer_changed(self, buffer: Buffer) -> None:
        if self._collapsing_paste:
            return
        if self._ignore_buffer_menu or self._form_kind or self._prompt_kind:
            return
        if self._menu_kind == "config_models":
            self._menu_index = 0
            self._menu_scroll = 0
            self._clamp_menu_index()
            self.invalidate()
            return
        if self._menu_kind in PICKER_MODES:
            return
        if self._maybe_collapse_paste(buffer):
            self.invalidate()
            return
        text = buffer.text
        if text.startswith("/") and " " not in text and self._slash_menu_items():
            was_slash = self._menu_kind == "slash"
            self._menu_kind = "slash"
            if not was_slash:
                self._menu_index = 0
                self._menu_scroll = 0
            self._sync_slash_index()
        else:
            if self._menu_kind == "slash":
                self._close_menu()
            # @-mention trigger: only when the cursor sits inside an active
            # `@token` we can extend. The buffer stays editable so the user
            # can keep typing prose around the token — the menu reads the
            # token span (from `@` up to the cursor) live.
            query = self._active_mention_query(buffer)
            if query is None:
                if self._menu_kind == MENTION_MENU:
                    self._close_menu()
            else:
                was_mention = self._menu_kind == MENTION_MENU
                self._menu_kind = MENTION_MENU
                if not was_mention:
                    self._menu_index = 0
                    self._menu_scroll = 0
                self._invalidate_mention_cache()
                self._clamp_menu_index()

    def _maybe_collapse_paste(self, buffer: Buffer) -> bool:
        text = buffer.text
        if self._pending_paste is None:
            if "\n" not in text:
                return False
            self._pending_paste = text
            self._write_paste_placeholder()
            return True
        n = self._pending_paste.count("\n") + 1
        expected = f"[pasted {n} lines]"
        if text == expected:
            return True
        if expected not in text:
            # The user deleted the marker token — they intend to discard the
            # paste, so drop the stashed content and let normal buffer handling
            # proceed. Never leave a stale ``[pasted N lines]`` literal behind.
            self._pending_paste = None
            return False
        if "\n" in text:
            # Additional multiline content was pasted on top of the marker;
            # fold it into the stashed paste and re-collapse.
            new_part = text.replace(expected, "", 1).strip("\n")
            if new_part:
                self._pending_paste = self._pending_paste + "\n" + new_part
            self._write_paste_placeholder()
        # Marker is still present (possibly with single-line prose typed around
        # it). Keep the paste stashed and leave the user's prose intact so the
        # marker can be expanded back into the real content at submit time.
        return True

    def _expand_paste(self, text: str) -> str:
        """Replace the ``[pasted N lines]`` marker with the stashed content.

        If no paste is pending or the marker is absent, ``text`` is returned
        unchanged. This is the single source of truth for turning the visible
        placeholder back into the real pasted text on submit, so the literal
        ``[pasted N lines]`` string is never sent to the model.
        """
        if self._pending_paste is None:
            return text
        n = self._pending_paste.count("\n") + 1
        marker = f"[pasted {n} lines]"
        if marker in text:
            return text.replace(marker, self._pending_paste, 1)
        return text

    def _write_paste_placeholder(self) -> None:
        if self._pending_paste is None:
            return
        n = self._pending_paste.count("\n") + 1
        placeholder = f"[pasted {n} lines]"
        self._collapsing_paste = True
        try:
            self._write_buffer_text(placeholder)
        finally:
            self._collapsing_paste = False

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
        for slug in available_model_ids():
            if slug == current or slug.endswith(f"/{current}"):
                return slug
        return current

    def _active_config_model_slug(self) -> str:
        """Current model slug for the active model-picker target field."""
        target = getattr(self, "_config_model_target", "model_name") or "model_name"
        if target == "model_name":
            return self._current_model_slug()
        current = getattr(settings, target, None) or ""
        if not current:
            return ""
        for slug in available_model_ids():
            if slug == current or slug.endswith(f"/{current}"):
                return slug
        return current

    def _config_action_items(self) -> list[MenuItem]:
        items = [
            MenuItem(
                section_id,
                title,
                description=SECTION_DESCRIPTIONS.get(section_id, ""),
            )
            for section_id, title in SECTION_TITLES.items()
        ]
        items.append(MenuItem("view", "View current config"))
        return items

    def _config_section_items(self) -> list[MenuItem]:
        return [
            MenuItem(spec.key, spec.label, suffix=format_current(spec))
            for spec in specs_for_section(self._config_section)
        ]

    def _config_select_items(self) -> list[MenuItem]:
        spec = SPEC_BY_KEY.get(self._config_select_key)
        if spec is None:
            return []
        current = str(getattr(settings, spec.key))
        return [
            MenuItem(choice, choice, suffix="(current)" if choice == current else "")
            for choice in spec.choices
        ]

    def _config_model_items(self) -> list[MenuItem]:
        current = self._active_config_model_slug()
        target = getattr(self, "_config_model_target", "model_name") or "model_name"
        spec = SPEC_BY_KEY.get(target)
        query = self._buffer.text.strip().lower() if self._menu_kind == "config_models" else ""
        ranked: list[tuple[int, str]] = []
        for slug in available_model_ids():
            record = model_record(slug)
            name = record.name if record is not None else slug
            provider = slug.partition("/")[0]
            haystacks = (slug.lower(), name.lower(), provider.lower())
            if query:
                if any(value == query for value in haystacks):
                    rank = 0
                elif any(value.startswith(query) for value in haystacks):
                    rank = 1
                elif any(query in value for value in haystacks):
                    rank = 2
                else:
                    continue
            else:
                rank = -1 if slug == current else 3
            ranked.append((rank, slug))
        ranked.sort(key=lambda item: (item[0], item[1].lower()))
        items: list[MenuItem] = []
        for _, slug in ranked:
            record = model_record(slug)
            description = ""
            if record is not None:
                modality = "VLM" if record.supports_vision else "LLM"
                context = (
                    f"{record.context_length // 1000}k"
                    if record.context_length
                    else "? context"
                )
                description = f"{modality} · {context}"
            items.append(
                MenuItem(
                    slug,
                    slug,
                    suffix="(current)" if slug == current else "",
                    description=description,
                )
            )
        if current not in {item.key for item in items} and not query:
            if current:
                items.insert(0, MenuItem(current, current, suffix="(current)"))
        if spec is not None and spec.optional and not query:
            items.insert(0, MenuItem("", "(unset) - use default"))
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

    # --- @mention autocomplete -------------------------------------------
    def _active_mention_query(self, buffer: Buffer) -> str | None:
        """Return the in-progress `@token` query at the cursor, or None.

        The token must:
        - start with `@` (matched by scanning backward from the cursor),
        - be preceded by a word boundary (start, whitespace, newline),
        - contain no whitespace (a trailing space closes the mention).
        """
        text = buffer.text
        cursor = buffer.cursor_position
        if cursor <= 0 or cursor > len(text):
            return None
        i = cursor - 1
        while i >= 0 and text[i] in _PATH_TOKEN_CHARS:
            i -= 1
        if i < 0 or text[i] != "@":
            return None
        at_index = i
        if at_index > 0 and text[at_index - 1] not in (" ", "\n", "\t"):
            return None
        token = text[at_index + 1 : cursor]
        return token

    def _mention_filter(self) -> str:
        if self._menu_kind != MENTION_MENU:
            return ""
        return self._active_mention_query(self._buffer) or ""

    def _mention_items(self) -> list[MenuItem]:
        query = self._mention_filter()
        if query == self._mention_cache_query:
            return list(self._mention_cache_items)
        self._mention_cache_query = query
        try:
            files = mention_mod.index_files()
        except Exception:
            files = []
        self._mention_cache_items = mention_mod.filter_files(query, files, limit=MENTION_MAX_ROWS)
        return list(self._mention_cache_items)

    def _invalidate_mention_cache(self) -> None:
        self._mention_cache_query = None
        self._mention_cache_items = []

    def _complete_mention_selection(self) -> None:
        """Replace the active `@token` with `@<selected path>` and a trailing space."""
        items = self._visible_menu_items()
        if not items:
            return
        item = items[self._menu_index]
        buffer = self._buffer
        text = buffer.text
        cursor = buffer.cursor_position
        if cursor <= 0 or cursor > len(text):
            return
        i = cursor - 1
        while i >= 0 and text[i] in _PATH_TOKEN_CHARS:
            i -= 1
        if i < 0 or text[i] != "@":
            return
        at_index = i
        before = text[:at_index]
        after = text[cursor:]
        replacement = "@" + item.key + " "
        self._ignore_buffer_menu = True
        try:
            buffer.set_document(
                Document(before + replacement + after, cursor_position=len(before) + len(replacement)),
                bypass_readonly=True,
            )
        finally:
            self._ignore_buffer_menu = False
        self._close_menu()
        self._focus_command_input()

    def _visible_menu_items(self) -> list[MenuItem]:
        builders = {
            "config_models": self._config_model_items,
            "config_reasoning": self._config_reasoning_items,
            "config_action": self._config_action_items,
            "config_section": self._config_section_items,
            "config_select": self._config_select_items,
            "slash": self._slash_menu_items,
            MENTION_MENU: self._mention_items,
            "approval": lambda: self._prompt_items,
            "question": lambda: self._prompt_items,
            "rollback": lambda: self._prompt_items,
            "threads": lambda: self._prompt_items,
            "fork": lambda: self._prompt_items,
        }
        builder = builders.get(self._menu_kind or "")
        return builder() if builder else []

    def _menu_body_height(self) -> int:
        if not self._menu_kind:
            return 0
        max_rows = MENTION_MAX_ROWS if self._menu_kind == MENTION_MENU else MENU_MAX_ROWS
        rows = min(max_rows, max(1, len(self._visible_menu_items())))
        if self._prompt_summary_lines:
            rows += min(3, len(self._prompt_summary_lines)) + 1
        if self._prompt_detail_lines:
            rows += min(5, len(self._prompt_detail_lines))
        return rows

    def _clamp_menu_scroll(self) -> None:
        items = self._visible_menu_items()
        if not items:
            self._menu_scroll = 0
            return
        max_rows = MENTION_MAX_ROWS if self._menu_kind == MENTION_MENU else MENU_MAX_ROWS
        if self._menu_index < self._menu_scroll:
            self._menu_scroll = self._menu_index
        elif self._menu_index >= self._menu_scroll + max_rows:
            self._menu_scroll = self._menu_index - max_rows + 1

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
        section_title = SECTION_TITLES.get(self._config_section, "")
        select_spec = SPEC_BY_KEY.get(self._config_select_key)
        headers = {
            "config_models": "/config > models - Type to search, Enter to select:",
            "config_reasoning": "/config > reasoning - Select reasoning effort:",
            "config_action": "/config - What would you like to change:",
            "config_section": f"/config > {section_title} - Enter to edit, left/right toggles:",
            "config_select": f"/config > {select_spec.label if select_spec else ''} - Select a value:",
            MENTION_MENU: "files - @mention autocomplete",
            "approval": self._prompt_title,
            "question": self._prompt_title,
            "rollback": self._prompt_title,
            "threads": self._prompt_title,
            "fork": self._prompt_title,
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
        for line in self._prompt_summary_lines[:3]:
            fragments.append(
                ("class:chrome.approval.command", f"   {line[:term_width() - 3]}\n")
            )
        if self._prompt_summary_lines:
            fragments.append(("class:chrome.menu.row", "\n"))
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
        self._pending_paste = None
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
        elif self._menu_kind == "config_section":
            self._apply_config_section_item(item.key)
        elif self._menu_kind == "config_models":
            self._apply_config_model(item.key)
        elif self._menu_kind == "config_reasoning":
            self._apply_config_reasoning(item.key)
        elif self._menu_kind == "config_select":
            self._apply_config_select(item.key)
        elif self._menu_kind == "approval":
            self._apply_approval_selection(item.key)
        elif self._menu_kind == "question":
            self._submit_question()
        elif self._menu_kind in {"rollback", "threads", "fork"}:
            if self._prompt_future is not None and not self._prompt_future.done():
                self._prompt_future.set_result(item.key)
