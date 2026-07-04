from __future__ import annotations

import string

from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition, has_selection
from prompt_toolkit.key_binding import KeyBindings

from cli.tui.constants import PICKER_MODES

def build_key_bindings(ui) -> KeyBindings:
    kb = KeyBindings()

    def _editable_input_at_start() -> bool:
        try:
            buff = get_app().current_buffer
        except Exception:
            return False
        if buff is not ui._buffer and buff is not ui._form_buffer:
            return False
        return buff.cursor_position == 0

    _silence_backspace = Condition(_editable_input_at_start) & ~has_selection

    @kb.add("backspace", filter=_silence_backspace, eager=True)
    @kb.add("c-h", filter=_silence_backspace, eager=True)
    def _noop_backspace_at_start(event) -> None:
        pass

    @kb.add("s-tab")
    def _toggle_mode(event) -> None:
        if ui._prompt_kind or ui._form_kind:
            return
        ui.session.toggle_mode()
        event.app.invalidate()

    @kb.add("down", filter=ui._menu_navigation_open)
    def _menu_down(event) -> None:
        ui._wheel_menu_index(1)
        event.app.invalidate()

    @kb.add("up", filter=ui._menu_navigation_open)
    def _menu_up(event) -> None:
        ui._wheel_menu_index(-1)
        event.app.invalidate()

    @kb.add("tab", filter=ui._question_prompt_open)
    def _toggle_question_note(event) -> None:
        ui._prompt_note_active = not ui._prompt_note_active
        if ui._prompt_note_active:
            ui._focus_form_field()
        else:
            ui._focus_command_input()
        event.app.invalidate()

    @kb.add("escape", filter=ui._line_prompt_open, eager=True)
    def _line_cancel(event) -> None:
        if ui._prompt_future is not None and not ui._prompt_future.done():
            ui._prompt_future.set_result("")
        ui._reset_buffer()
        event.app.invalidate()

    @kb.add("escape", filter=ui._form_open, eager=True)
    def _form_cancel(event) -> None:
        if ui._prompt_kind == "question":
            ui._prompt_note_active = False
            ui._focus_command_input()
        elif ui._form_kind:
            ui._finish_config()
        event.app.invalidate()

    @kb.add("escape", filter=ui._menu_open, eager=True)
    def _menu_cancel(event) -> None:
        ui._cancel_menu()
        event.app.invalidate()

    @kb.add("escape", filter=ui._transcript_focused, eager=True)
    def _transcript_unfocus(event) -> None:
        ui._refocus_input()
        event.app.invalidate()

    @kb.add("enter", filter=ui._transcript_focused)
    def _transcript_enter_refocus(event) -> None:
        ui._refocus_input()
        event.app.invalidate()

    @kb.add("backspace", filter=ui._transcript_focused, eager=True)
    def _transcript_backspace_refocus(event) -> None:
        ui._refocus_input()
        target = ui._form_buffer if ui._form_visible() and (ui._form_kind or ui._prompt_note_active) else ui._buffer
        if target.text:
            target.delete_before_cursor(count=1)
        event.app.invalidate()

    for char in string.ascii_letters + string.digits + string.punctuation + " ":
        @kb.add(char, filter=ui._transcript_focused, eager=True)
        def _transcript_type_refocus(event, ch=char) -> None:
            ui._insert_from_transcript_focus(ch)
            event.app.invalidate()

    @kb.add("pageup", filter=ui._transcript_scroll_open, eager=True)
    def _transcript_page_up(event) -> None:
        ui._scroll_transcript_by(-max(1, ui._transcript_viewport_lines() // 2))
        event.app.invalidate()

    @kb.add("pagedown", filter=ui._transcript_scroll_open, eager=True)
    def _transcript_page_down(event) -> None:
        ui._scroll_transcript_by(max(1, ui._transcript_viewport_lines() // 2))
        event.app.invalidate()

    @kb.add("c-home", filter=ui._transcript_scroll_open, eager=True)
    def _transcript_top(event) -> None:
        ui._scroll_transcript_to_top()
        event.app.invalidate()

    @kb.add("c-end", filter=ui._transcript_scroll_open, eager=True)
    @kb.add("end", filter=ui._transcript_scroll_open, eager=True)
    def _transcript_end(event) -> None:
        ui._resume_transcript_follow()
        event.app.invalidate()

    @kb.add("tab", filter=ui._slash_menu_open)
    def _tab_complete_slash(event) -> None:
        ui._complete_slash_selection()
        event.app.invalidate()

    @kb.add("enter")
    def _submit_line(event) -> None:
        buff = event.app.current_buffer
        if ui._prompt_kind == "line" and buff is ui._buffer:
            if ui._prompt_future is not None and not ui._prompt_future.done():
                ui._prompt_future.set_result(buff.text.strip())
            ui._reset_buffer()
            event.app.invalidate()
            return

        if ui._prompt_kind == "question":
            ui._submit_question()
            event.app.invalidate()
            return

        if buff is ui._form_buffer and ui._form_kind:
            ui._submit_form()
            event.app.invalidate()
            return

        if buff is not ui._buffer:
            return

        if ui._menu_kind in PICKER_MODES and ui._visible_menu_items():
            ui._apply_picker_selection()
            event.app.invalidate()
            return

        if ui._menu_kind == "approval" and ui._visible_menu_items():
            ui._apply_picker_selection()
            event.app.invalidate()
            return

        if ui._menu_kind == "rollback" and ui._visible_menu_items():
            ui._apply_picker_selection()
            event.app.invalidate()
            return

        if ui._menu_kind == "slash" and ui._visible_menu_items():
            command = ui._selected_slash_command()
            ui._close_menu()
            ui._reset_buffer()
            if command:
                ui._schedule_submit(command)
            event.app.invalidate()
            return

        text = buff.text.strip()
        if text:
            buff.append_to_history()
        ui._reset_buffer()
        ui._close_menu()
        ui._schedule_submit(text)
        event.app.invalidate()

    @kb.add("c-c", filter=ui._transcript_selection_active, eager=True)
    def _copy_transcript_selection(event) -> None:
        data = ui._copy_transcript_selection()
        if data is not None:
            event.app.clipboard.set_data(data)
        ui._refocus_input()
        event.app.invalidate()

    @kb.add("c-c", filter=has_selection)
    def _copy_selection(event) -> None:
        data = event.current_buffer.copy_selection()
        event.app.clipboard.set_data(data)
        ui._refocus_input()
        event.app.invalidate()

    @kb.add("c-c", filter=~has_selection & ~ui._transcript_selection_active)
    def _clear_or_cancel(event) -> None:
        # Priority: (1) dismiss any open approval/question/line prompt so its
        # asyncio.Future is resolved instead of stranded, (2) cancel the
        # active turn cooperatively (with a 2s hard-escalation backstop),
        # (3) clear the prompt queue, (4) clear the input buffer.
        if ui._cancel_open_prompt():
            event.app.invalidate()
            return
        if ui._cancel_active_task():
            event.app.invalidate()
            return
        cleared = ui.session.clear_prompt_queue()
        if cleared:
            ui.append_notice("queue", f"cleared {cleared} queued prompt(s)")
            event.app.invalidate()
            return
        buff = event.current_buffer
        if buff is ui._buffer and buff.text:
            ui._reset_buffer()
            event.app.invalidate()

    @kb.add("c-q")
    def _quit(event) -> None:
        event.app.exit()

    return kb
