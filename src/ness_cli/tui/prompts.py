from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from ness_cli.tui.models import MenuItem
from ness_cli.tui.tool_display import format_tool_args
from ness_cli.tui.utils import term_width
from ness_agent.utils import preview_diff


def default_question_index(options: list[dict]) -> int:
    for index, option in enumerate(options):
        if option.get("recommended"):
            return index
    return 0


def format_thread_updated_at(value: object) -> str:
    """Format a stored UTC ISO timestamp in the machine's local timezone."""
    try:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            return ""
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return ""


class PromptMixin:
    """Approval, question, and line prompts."""

    async def ask_picker(
        self,
        title: str,
        items: list[MenuItem],
        *,
        initial_key: str | None = None,
        hint: str = "↑/↓ select · Enter confirm · Esc back",
    ) -> str | None:
        """Show a compact native picker without question/note affordances."""
        self._prompt_future = asyncio.get_running_loop().create_future()
        self._prompt_kind = "picker"
        self._prompt_title = title
        self._prompt_hint = hint
        self._prompt_items = list(items)
        self._prompt_summary_lines = []
        self._prompt_detail_lines = []
        index = next(
            (i for i, item in enumerate(items) if item.key == initial_key),
            0,
        )
        self._open_picker("picker", "/login", index=index)
        result = await self._prompt_future
        self._clear_prompt()
        return str(result) if result else None

    async def ask_approval(self, name: str, args: dict) -> str:
        self._prompt_future = asyncio.get_running_loop().create_future()
        self._prompt_kind = "approval"
        self._prompt_title = f"approval needed: {name}"
        self._prompt_hint = "↑/↓ select · Enter submit · Esc deny"
        self._prompt_items = [
            MenuItem("yes", "approve once", "Run this tool call."),
            MenuItem("session", "approve session", "Allow matching calls this session."),
            MenuItem("always", "always allow", "Persist an allow rule."),
            MenuItem("no", "deny once", "Skip this call."),
            MenuItem("never", "never allow", "Persist a deny rule."),
            MenuItem("diff", "show diff", "Preview file changes."),
            MenuItem("show", "show args", "Inspect full arguments."),
        ]
        preview = format_tool_args(name, args) or json.dumps(args, ensure_ascii=False)
        self._prompt_summary_lines = [
            line[: max(20, term_width() - 6)] for line in preview.splitlines()[:3]
        ]
        self._prompt_detail_lines = []
        self._prompt_question = {"name": name, "args": args}
        self._open_picker("approval", "", index=0)
        result = await self._prompt_future
        self._clear_prompt()
        return str(result or "no")

    def _apply_approval_selection(self, key: str) -> None:
        if key == "diff":
            name = self._prompt_question.get("name", "") if self._prompt_question else ""
            args = self._prompt_question.get("args", {}) if self._prompt_question else {}
            diff = preview_diff(str(name), dict(args)) or "(no diff)"
            self._prompt_detail_lines = diff.splitlines()[:5]
            self.invalidate()
            return
        if key == "show":
            args = self._prompt_question.get("args", {}) if self._prompt_question else {}
            self._prompt_detail_lines = json.dumps(args, ensure_ascii=False, indent=2).splitlines()[:5]
            self.invalidate()
            return
        if self._prompt_future is not None and not self._prompt_future.done():
            self._prompt_future.set_result(key)

    @staticmethod
    def _cancelled_question_answer(index: int, question: dict) -> dict:
        return {
            "id": question.get("id", str(index)),
            "selected": None,
            "note": "cancelled by user",
        }

    @staticmethod
    def _is_cancelled_question_answer(answer: dict) -> bool:
        return answer.get("selected") is None and answer.get("note") == "cancelled by user"

    async def ask_questions(self, questions: list[dict]) -> list[dict]:
        answers: list[dict] = []
        for index, question in enumerate(questions, 1):
            answer = await self._ask_question(index, question)
            answers.append(answer)
            if self._is_cancelled_question_answer(answer):
                for rest_index, rest in enumerate(questions[index:], index + 1):
                    answers.append(self._cancelled_question_answer(rest_index, rest))
                break
        return answers

    async def _ask_question(self, index: int, question: dict) -> dict:
        options = list(question.get("options", []))
        self._prompt_future = asyncio.get_running_loop().create_future()
        self._prompt_kind = "question"
        self._prompt_title = f"question {index}: {question.get('prompt', '')}"
        self._prompt_hint = (
            "↑/↓ option · Tab note · Enter submit · Esc cancel"
            if question.get("allow_note", True)
            else "↑/↓ option · Enter submit · Esc cancel"
        )
        self._prompt_items = [
            MenuItem(str(i), str(option.get("label", "")), "(recommended)" if option.get("recommended") else "")
            for i, option in enumerate(options)
        ]
        self._prompt_question = question
        self._prompt_note_active = False
        self._form_buffer.text = ""
        self._open_picker("question", "", index=default_question_index(options))
        result = await self._prompt_future
        self._clear_prompt()
        if result is None or (isinstance(result, dict) and result.get("cancelled")):
            return self._cancelled_question_answer(index, question)
        selected_index = int(result["index"])
        selected = options[selected_index]
        return {
            "id": question.get("id", str(index)),
            "selected": {"id": selected.get("id"), "label": selected.get("label")},
            "note": result.get("note", ""),
        }

    def _submit_question(self) -> None:
        if self._prompt_future is None or self._prompt_future.done():
            return
        if not self._prompt_items:
            self._prompt_future.set_result({"index": 0, "note": self._form_buffer.text.strip()})
            return
        self._prompt_future.set_result({"index": self._menu_index, "note": self._form_buffer.text.strip()})

    async def ask_line(self, message: str) -> str:
        self._prompt_future = asyncio.get_running_loop().create_future()
        self._prompt_kind = "line"
        self._prompt_title = message
        self._prompt_hint = "Enter submit - Esc cancel"
        self._prompt_items = []
        self._prompt_summary_lines = []
        self._prompt_detail_lines = []
        self._reset_buffer()
        self._focus_command_input()
        self.invalidate()
        result = await self._prompt_future
        self._clear_prompt()
        return str(result or "")

    async def ask_secret(self, label: str, *, example: str = "") -> str:
        """Prompt in the password-masked form field and return the value."""
        self._prompt_future = asyncio.get_running_loop().create_future()
        self._prompt_kind = "secret"
        self._form_kind = "openai_api_key"  # registered masked form kind
        self._form_label = label
        self._form_example = example
        self._form_buffer.text = ""
        self._set_buffer_text("/login")
        self._focus_form_field()
        self.invalidate()
        result = await self._prompt_future
        self._form_kind = None
        self._clear_prompt()
        return str(result or "")

    async def request_rollback_picker(self, turns: list[dict]) -> str:
        """Open the /rollback picker over user turns; return the chosen seq or "" on cancel.

        ``turns`` is the list from ``session.list_user_turns``: each item has
        ``seq`` (int) and ``content`` (str). Selected item's key resolves the
        prompt future with the seq as a string; Esc/Ctrl+C resolves it with "".
        """
        self._prompt_future = asyncio.get_running_loop().create_future()
        self._prompt_kind = "rollback"
        self._prompt_title = "rollback to user message"
        self._prompt_hint = "↑/↓ select · Enter rollback · Esc cancel"
        self._prompt_items = [
            MenuItem(
                str(turn["seq"]),
                "[user] " + str(turn.get("content", "")).strip().splitlines()[0][:80]
                if str(turn.get("content", "")).strip()
                else "[user] (empty)",
            )
            for turn in turns
        ]
        self._prompt_detail_lines = []
        # Open near the bottom (most recent turn) — that's what users typically
        # want to roll back to.
        self._open_picker("rollback", "", index=max(0, len(turns) - 1))
        result = await self._prompt_future
        self._clear_prompt()
        return str(result or "")

    async def request_threads_picker(
        self,
        threads: list[dict],
        *,
        current_thread_id: str,
    ) -> str:
        self._prompt_future = asyncio.get_running_loop().create_future()
        self._prompt_kind = "threads"
        self._prompt_title = "saved threads"
        self._prompt_hint = "↑/↓ select · Enter switch · Esc cancel"
        self._prompt_items = []
        current_index = 0
        for index, thread in enumerate(threads):
            thread_id = str(thread.get("thread_id") or "")
            label = str(
                thread.get("label")
                or thread.get("name")
                or thread.get("summary")
                or "(no messages)"
            )
            updated_at = format_thread_updated_at(thread.get("updated_at"))
            if updated_at:
                label = f"{updated_at}  {label}"
            suffixes: list[str] = []
            if thread_id == current_thread_id:
                suffixes.append("(current)")
                current_index = index
            # Root shows how many forks exist; each child shows its ordinal so
            # the original conversation is distinct from the fork sequence.
            if thread.get("fork_parent_id"):
                suffixes.append(f"fork #{int(thread.get('fork_index') or 1)}")
            else:
                forks = int(thread.get("fork_count") or 0)
                if forks:
                    suffixes.append(f"×{forks}")
            self._prompt_items.append(
                MenuItem(thread_id, label[:100], suffix=" ".join(suffixes))
            )
        self._prompt_summary_lines = []
        self._prompt_detail_lines = []
        self._open_picker("threads", "", index=current_index)
        result = await self._prompt_future
        self._clear_prompt()
        return str(result or "")

    async def request_fork_picker(self, turns: list[dict]) -> str:
        self._prompt_future = asyncio.get_running_loop().create_future()
        self._prompt_kind = "fork"
        self._prompt_title = "fork before user message"
        self._prompt_hint = "↑/↓ select · Enter fork · Esc cancel"
        self._prompt_items = [
            MenuItem(
                str(turn["seq"]),
                "[user] " + (
                    str(turn.get("content") or "").strip().splitlines()[0][:90]
                    or "(empty)"
                ),
            )
            for turn in turns
        ]
        self._prompt_summary_lines = []
        self._prompt_detail_lines = []
        self._open_picker("fork", "", index=max(0, len(turns) - 1))
        result = await self._prompt_future
        self._clear_prompt()
        return str(result or "")

    def _clear_prompt(self) -> None:
        self._prompt_kind = None
        self._prompt_title = ""
        self._prompt_hint = ""
        self._prompt_items = []
        self._prompt_summary_lines = []
        self._prompt_detail_lines = []
        self._prompt_question = None
        self._prompt_note_active = False
        self._form_buffer.text = ""
        self._close_menu()
        self._focus_command_input()
        self.invalidate()
