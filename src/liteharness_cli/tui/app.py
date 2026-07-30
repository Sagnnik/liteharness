from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.clipboard.pyperclip import PyperclipClipboard
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import FileHistory
from prompt_toolkit.layout import VSplit, Window
from prompt_toolkit.styles import Style

from liteharness.types import SessionEvent
from liteharness_cli.chat_model import active_model_name
from liteharness_cli.config import settings

from liteharness_cli.tui import render
from liteharness_cli.tui.commands import dispatch
from liteharness_cli.tui.config_flow import ConfigResult
from liteharness_cli.tui.theme import PTK_STYLE_RULES
from liteharness_cli.tui.chrome import ChromeMixin
from liteharness_cli.tui.config_flow import ConfigFlowMixin
from liteharness_cli.tui.constants import ESCAPE_KEY_FLUSH_TIMEOUT, KEY_BINDING_TIMEOUT, PICKER_MODES
from liteharness_cli.tui.keys import build_key_bindings
from liteharness_cli.tui.pickers import MenuMixin
from liteharness_cli.tui.models import MenuItem, TranscriptLine
from liteharness_cli.tui.prompts import PromptMixin
from liteharness_cli.tui.transcript import TranscriptMixin
from liteharness_cli.tui.turn_renderer import TurnRenderer
from liteharness_cli.tui.utils import display_cwd
from liteharness_cli.tui.widgets import TranscriptStore, TranscriptViewportControl

if TYPE_CHECKING:
    from liteharness.mcp import MCPManager
    from liteharness_cli import CodingSession

CommandDispatcher = Callable[["TuiApp", str], Awaitable[None]]


def _new_thread_id() -> str:
    return f"session-{uuid.uuid4().hex[:8]}"


class TuiApp(TranscriptMixin, ChromeMixin, MenuMixin, ConfigFlowMixin, PromptMixin):
    """Full-screen prompt_toolkit UI with transcript above and input chrome pinned at the bottom.

    Wired directly to a :class:`~liteharness_cli.CodingSession`: the SDK
    owns the turn loop, cancel finalisation, and thread state; this class
    owns the TUI-side session state (prompt queue, exit flag, staged skills,
    assistant history for /copy) and renders the SessionEvent stream.
    """

    def __init__(
        self,
        coding: CodingSession,
        *,
        history_path: Path,
        mcp: MCPManager | None = None,
        command_dispatcher: CommandDispatcher = dispatch,
    ) -> None:
        self.coding = coding
        self.mcp = mcp
        self._dispatch = command_dispatcher
        # TUI-owned session state (formerly SessionApp): input queue, exit
        # flag, assistant text history for /copy. Skills stage via
        # coding.stage_skills → Session._pending_skills.
        self.should_exit = False
        self.prompt_queue: list[str] = []
        self.assistant_history: list[str] = []
        self._cwd_line = display_cwd()
        history_path.parent.mkdir(parents=True, exist_ok=True)
        # The startup header (gradient logo + dashboard panel + hints) is
        # appended via ``self.render_header()`` once the transcript pane's
        # real width is known (see ``_start_initial_header_task``), so the
        # pre-wrapped TranscriptLines aren't built against a stale fallback
        # width and re-wrap into a half-screen artifact on first render.
        self._lines: list[TranscriptLine] = []

        self._busy = False
        self._submit_task: asyncio.Task | None = None
        # Hard-escalation backstop handle: when the cooperative cancel token
        # fails to break the stream loop within 2s (e.g. a long blocked LLM
        # call), a call_later fires _hard_cancel to fall back to the legacy
        # asyncio.Task.cancel() behaviour. Stored so the cooperative path can
        # cancel it on clean exit and avoid firing on a finished task.
        self._cancel_backstop_handle: asyncio.TimerHandle | None = None
        self._config_future: asyncio.Future[ConfigResult] | None = None
        self._config_result: ConfigResult | None = None
        self._prompt_future: asyncio.Future[Any] | None = None
        self._prompt_kind: str | None = None
        self._prompt_title = ""
        self._prompt_hint = ""
        self._prompt_items: list[MenuItem] = []
        self._prompt_summary_lines: list[str] = []
        self._prompt_detail_lines: list[str] = []
        self._prompt_question: dict[str, Any] | None = None
        self._prompt_note_active = False

        self._menu_kind: str | None = None
        self._menu_index = 0
        self._menu_scroll = 0
        self._form_kind: str | None = None
        self._form_label = ""
        self._ignore_buffer_menu = False
        self._pending_paste: str | None = None
        self._collapsing_paste: bool = False
        self._pending_images: list[str] = []
        self._image_counter: int = 0
        self._follow_transcript = True
        self._transcript_revision = 0
        self._slash_menu_cache_query: str | None = None
        self._slash_menu_cache_items: list[MenuItem] = []
        self._mention_cache_query: str | None = None
        self._mention_cache_items: list[MenuItem] = []
        self._catalog_refresh_task: asyncio.Task | None = None
        self._catalog_refresh_warned = False
        self._working_active = False
        self._worked_label: str | None = None
        self._worked_elapsed = 0.0
        self._working_started_at: float | None = None
        self._working_frame = 0
        self._working_task: asyncio.Task | None = None
        self._turn_working = False
        self._transcript_render_width = 0
        self._transcript_viewport_height = 0
        self._layout_term_width = 0
        # Set on the first render once the transcript pane's real width is
        # known. The --resume startup path awaits this before replaying the
        # saved conversation so markdown/user lines are wrapped to the actual
        # pane width instead of the raw terminal width.
        self._transcript_ready = asyncio.Event()
        self._stats_line_cache_key: tuple[Any, ...] | None = None
        self._stats_line_cache: list[tuple[str, str]] = []
        self._path_line_cache_key: tuple[Any, ...] | None = None
        self._path_line_cache: list[tuple[str, str]] = []

        self._todos_block_start: int | None = None
        self._todos_block_count = 0
        # Startup header block: tracked so /config refreshes it in place
        # (instead of re-appending a duplicate banner mid-conversation) and
        # so a terminal resize re-flows it at the new width. Shape:
        # {"start": int, "count": int, "width": int, "source": dict}.
        self._header_block: dict | None = None
        # Reasoning (CoT) blocks: one per LLM call that emitted reasoning_content.
        # Each span: {"start": int, "count": int, "text": str, "elapsed": float}.
        # Collapsed by default; Ctrl+T flips ``_show_reasoning`` and re-emits
        # every span bottom-to-top so later spans' index shifts don't disturb
        # the not-yet-processed earlier ones.
        self._show_reasoning = False
        self._reasoning_spans: list[dict] = []
        self._transcript_store = TranscriptStore(self._lines)

        self._buffer = Buffer(history=FileHistory(str(history_path)), multiline=True)
        self._buffer.read_only = Condition(self._main_buffer_read_only)
        self._form_buffer = Buffer()
        self._form_buffer.password = Condition(
            lambda: self._form_kind in ("openai_key", "exa_key")
        )
        self._buffer.on_text_changed.add_handler(self._on_buffer_changed)

        self._transcript_control: TranscriptViewportControl | None = None
        self._transcript_inner: Window | None = None
        self._transcript_pane: Window | None = None
        self._input_window: Window | None = None
        self._form_pad_window: Window | None = None
        self._form_label_window: Window | None = None
        self._form_field_window: Window | None = None
        self._form_row: VSplit | None = None
        self._form_hint_window: Window | None = None
        self._menu_header_window: Window | None = None
        self._menu_body_window: Window | None = None

        self._layout = self._build_layout()
        self._menu_open = Condition(lambda: self._menu_kind is not None)
        self._menu_navigation_open = Condition(
            lambda: self._menu_kind is not None and not self._prompt_note_active
        )
        self._slash_menu_open = Condition(lambda: self._menu_kind == "slash")
        self._mention_menu_open = Condition(lambda: self._menu_kind == "mention")
        self._transcript_scroll_open = Condition(
            lambda: self._menu_kind is None and not self._form_visible()
        )
        self._transcript_focused = Condition(
            lambda: self._layout.current_control is self._transcript_control
        )
        self._transcript_selection_active = Condition(self._transcript_has_selection)
        self._form_open = Condition(self._form_visible)
        self._line_prompt_open = Condition(lambda: self._prompt_kind == "line")
        self._question_prompt_open = Condition(lambda: self._prompt_kind == "question")

        self._app = Application(
            layout=self._layout,
            key_bindings=build_key_bindings(self),
            style=Style.from_dict(PTK_STYLE_RULES),
            full_screen=True,
            mouse_support=True,
            enable_page_navigation_bindings=False,
            clipboard=PyperclipClipboard(),
            after_render=lambda _: self._after_render(),
            terminal_size_polling_interval=0.1,
        )
        self._configure_escape_timeouts(self._app)

    @staticmethod
    def _configure_escape_timeouts(app: Application) -> None:
        app.ttimeoutlen = ESCAPE_KEY_FLUSH_TIMEOUT
        app.timeoutlen = KEY_BINDING_TIMEOUT

    def _main_buffer_read_only(self) -> bool:
        if self._menu_kind in PICKER_MODES and self._menu_kind != "config_models":
            return True
        if self._form_kind is not None:
            return True
        if self._prompt_kind is not None and self._prompt_kind != "line":
            return True
        return False

    def _form_visible(self) -> bool:
        return self._form_kind is not None or self._prompt_kind == "question"

    def invalidate(self) -> None:
        with suppress(Exception):
            self._app.invalidate()

    def _focus_form_field(self) -> None:
        if self._form_field_window is not None:
            self._layout.focus(self._form_field_window)

    def _focus_command_input(self) -> None:
        if self._input_window is not None:
            self._layout.focus(self._input_window)

    def _focus_transcript(self) -> None:
        if self._transcript_inner is not None:
            self._layout.focus(self._transcript_inner)

    def _refocus_input(self) -> None:
        self._clear_transcript_selection()
        if self._form_visible() and (self._form_kind or self._prompt_note_active):
            self._focus_form_field()
        else:
            self._focus_command_input()

    def _insert_from_transcript_focus(self, text: str) -> None:
        self._refocus_input()
        if self._form_visible() and (self._form_kind or self._prompt_note_active):
            self._form_buffer.insert_text(text)
        else:
            self._buffer.insert_text(text)

    def _transcript_has_selection(self) -> bool:
        return bool(
            self._transcript_control and self._transcript_control.has_selection()
        )

    def _copy_transcript_selection(self):
        if self._transcript_control is None:
            return None
        selected = self._transcript_control.selected_text()
        if not selected:
            return None
        from prompt_toolkit.clipboard.base import ClipboardData

        return ClipboardData(selected)

    def _clear_transcript_selection(self) -> None:
        if self._transcript_control is not None:
            self._transcript_control.clear_selection()

    def _schedule_submit(self, text: str) -> None:
        if not text:
            return
        if self._busy:
            if text.startswith(("/", "!")):
                self._app.create_background_task(self._busy_dispatch(text))
            else:
                self.enqueue_prompt(text)
                count = len(self.prompt_queue)
                preview = text.strip().splitlines()[0][:48]
                if len(text.strip().splitlines()[0]) > 48:
                    preview += "..."
                self.append_notice("queue", f"added prompt #{count}: {preview}")
            self._reset_buffer()
            self.invalidate()
            return
        self._submit_task = self._app.create_background_task(self._submit_async(text))

    async def _busy_dispatch(self, text: str) -> None:
        try:
            if text.startswith("!"):
                await self._run_shell(text[1:])
            else:
                await self._dispatch(self, text, busy=True)
        except Exception as exc:
            self.append_error(f"{type(exc).__name__}: {exc}")
        finally:
            self._reset_buffer()
            self.invalidate()

    # --- shell escape (!command) ------------------------------------------
    _SHELL_OUTPUT_CAP = 20_000
    _SHELL_TIMEOUT = 60.0

    async def _run_shell(self, command: str) -> None:
        """Run a user-typed `!command` through the shell and show its output.

        This is an explicit user shell escape (like vim's `:!`): it runs with
        the user's normal environment and bypasses the agent approval system by
        design, since the user invoked it directly rather than the model.
        """
        command = command.strip()
        if not command:
            return
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:
            self.append_error(f"shell: {type(exc).__name__}: {exc}")
            return
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=self._SHELL_TIMEOUT
            )
        except asyncio.TimeoutError:
            with suppress(ProcessLookupError):
                proc.kill()
            self.append_warning(f"shell: timed out after {self._SHELL_TIMEOUT:.0f}s")
            return
        stdout = stdout_b.decode(errors="replace") if stdout_b else ""
        stderr = stderr_b.decode(errors="replace") if stderr_b else ""
        code = proc.returncode
        parts: list[str] = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(stderr.rstrip())
        body = "\n".join(parts)
        if len(body) > self._SHELL_OUTPUT_CAP:
            body = (
                body[: self._SHELL_OUTPUT_CAP]
                + f"\n... (truncated, {len(body)} chars total)"
            )
        if not body:
            body = "(no output)"
        render.render_panel_text(body, title="shell")
        if code and code != 0:
            self.append_warning(f"shell: exit {code}")

    async def _submit_async(self, text: str) -> None:
        self._busy = True
        self.invalidate()
        try:
            is_slash = text.startswith("/")
            is_shell = text.startswith("!")
            if not is_slash and not is_shell:
                self.append_user(text)
            if is_slash:
                await self._dispatch(self, text, busy=False)
            elif is_shell:
                await self._run_shell(text[1:])
            else:
                await self._run_turn(text, list(self._pending_images))
            while not self.should_exit:
                queued = self.dequeue_prompt()
                if queued is None:
                    break
                self.append_user(queued)
                await self._run_turn(queued, [])
            if self.should_exit:
                self._app.exit()
        except asyncio.CancelledError:
            # Hard-escalation path: the cooperative cancel failed to break
            # the stream loop within the backstop window, so the TUI fell
            # back to asyncio.Task.cancel(). ``_run_turn`` already rendered
            # the interrupt banner and the SDK finalised the checkpoint, so
            # there's nothing to surface here — just let it propagate.
            raise
        except Exception as exc:
            self.append_error(f"{type(exc).__name__}: {exc}")
        finally:
            self._cancel_hard_cancel_backstop()
            self._busy = False
            self._close_menu()
            self._reset_buffer()
            self._pending_images.clear()
            self._image_counter = 0
            self._focus_command_input()
            self._refresh_cwd_line_if_changed()
            self.invalidate()

    def _cancel_open_prompt(self) -> bool:
        """Resolve any open approval/question/line prompt future via the cancel path.

        Runs before turn-cancel/queue-clear so Ctrl+C during a permission or
        clarification prompt dismisses the prompt instead of stranding its
        ``asyncio.Future``. Mirrors the existing Esc/_cancel_menu behaviour
        but resolves question prompts to option 0 (the recommended default)
        instead of ``None`` so ``_ask_question`` doesn't crash on indexing.
        """
        if self._prompt_future is not None and not self._prompt_future.done():
            if self._prompt_kind == "question":
                self._prompt_future.set_result({"index": 0, "note": ""})
            else:
                self._prompt_future.set_result("")
            self._clear_prompt()
            return True
        return False

    def _cancel_hard_cancel_backstop(self) -> None:
        if self._cancel_backstop_handle is not None:
            self._cancel_backstop_handle.cancel()
            self._cancel_backstop_handle = None

    def _schedule_hard_cancel_backstop(self) -> None:
        """Arm a safety net that falls back to ``asyncio.Task.cancel``.

        The cooperative cancel_token path is preferred because it lets
        ``run_turn`` flush partial state cleanly. If the in-flight LLM call
        keeps the stream loop from reaching its ``is_set()`` check, the
        backstop escalates to the legacy hard cancel.

        The window is intentionally generous (10s): a cold first LLM call
        with no prefix cache or a slow OpenRouter RTT can easily exceed 2s,
        which would trip a hard cancel on a turn the cooperative path could
        have finalised cleanly given a little more time. ``run_turn``'s
        ``except CancelledError`` handler now finalises on hard cancel too,
        so the cost of a too-long window is only a slower UX response, not
        a dirty checkpoint.
        """
        self._cancel_hard_cancel_backstop()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._cancel_backstop_handle = loop.call_later(10.0, self._hard_cancel)

    def _hard_cancel(self) -> None:
        if self._submit_task is not None and not self._submit_task.done():
            self._submit_task.cancel()
        self._cancel_backstop_handle = None

    def _cancel_active_task(self) -> bool:
        if self._submit_task is not None and not self._submit_task.done():
            cleared = self.clear_prompt_queue()
            # Cooperative cancel preferred: lets the SDK break out at the
            # next event boundary and flush partial state. A call_later
            # backstop is scheduled so a stuck call still gets hard-cancelled.
            self.coding.cancel()
            self._schedule_hard_cancel_backstop()
            if cleared:
                self.append_notice("queue", f"cleared {cleared} queued prompt(s)")
            return True
        return False

    async def run_async(self, *, resume_thread_id: str | None = None) -> None:
        await self.coding.refresh_context_snapshot()
        render.set_sink(self)
        self._configure_escape_timeouts(self._app)
        self._start_initial_header_task()
        if resume_thread_id:
            self._start_resume_task(resume_thread_id)
        try:
            await self._app.run_async()
        finally:
            render.set_sink(None)
            self.stop_working()
            if self._submit_task is not None and not self._submit_task.done():
                self._submit_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._submit_task

    def _start_resume_task(self, thread_id: str) -> None:
        """Replay a saved thread into the transcript after the first render.

        The replay must run only once the transcript pane's real render width
        is known (set in ``_on_transcript_render_size``); otherwise the
        pre-wrapped markdown/user lines would be built against the raw
        terminal width and render misaligned in the narrower pane.
        """

        async def _resume() -> None:
            await self._transcript_ready.wait()
            await self.resume_thread(thread_id)

        self._app.create_background_task(_resume())

    def _start_initial_header_task(self) -> None:
        """Render the startup header once the transcript pane's real width is known.

        ``session.render_header()`` is normally called from ``liteharness_cli.tui.main._main``
        before the TUI starts running. At that point ``_transcript_render_width``
        is still 0, so ``append_header`` falls back to ``shutil``s terminal width,
        which can differ from the actual transcript pane width reported on the
        first render. Building the header lines against the wrong width then
        re-wrapping them once the real width arrives leaves each header row
        split across two visual rows, one only partially filled ("half screen"
        artifact). Waiting on ``_transcript_ready`` (set from
        ``_on_transcript_render_size`` -> ``set_width``) guarantees the width
        used to build the lines matches the width used to slice them.
        """

        async def _header() -> None:
            await self._transcript_ready.wait()
            self.render_header()

        self._app.create_background_task(_header())

    # ------------------------------------------------------------------
    # Session facade: queue, mode/header, thread management, the turn
    # ------------------------------------------------------------------
    # These used to live on SessionApp; with the TUI wired directly to
    # CodingSession they live here so the mixins (chrome/keys) and slash
    # commands keep one obvious ``app`` surface. Pure pass-throughs delegate
    # to ``self.coding``; anything with TUI side effects (queue, transcript,
    # history) is owned here.

    # --- prompt queue ----------------------------------------------------
    def enqueue_prompt(self, text: str) -> None:
        if text:
            self.prompt_queue.append(text)

    def dequeue_prompt(self) -> str | None:
        if self.prompt_queue:
            return self.prompt_queue.pop(0)
        return None

    def clear_prompt_queue(self) -> int:
        count = len(self.prompt_queue)
        self.prompt_queue.clear()
        return count

    @property
    def queued_prompt(self) -> str:
        return self.prompt_queue[-1] if self.prompt_queue else ""

    @queued_prompt.setter
    def queued_prompt(self, value: str) -> None:
        if value:
            self.prompt_queue = [value]
        else:
            self.prompt_queue.clear()

    # --- CodingSession pass-throughs --------------------------------------
    @property
    def thread_id(self) -> str:
        return self.coding.thread_id

    @property
    def mode(self) -> str:
        return self.coding.mode

    @property
    def turn_count(self) -> int:
        return self.coding.turn_count

    @property
    def context_used(self) -> int:
        return self.coding.context_used

    @property
    def context_total(self) -> int:
        return self.coding.context_total

    @property
    def model(self):
        """The raw chat model (used by /memory create's one-shot NESS.md draft)."""
        return self.coding.agent.config.model

    def toggle_mode(self) -> None:
        self.coding.toggle_mode()

    def render_header(self) -> None:
        options = getattr(getattr(self.coding, "cfg", None), "options", None)
        render.render_header(
            mode=self.coding.mode,
            model=active_model_name(),
            approval=getattr(options, "enable_approval", settings.enable_approval),
            yolo=getattr(options, "yolo_mode", False),
            autosave=settings.auto_save_threads,
            session_end_reflection=settings.session_end_reflection,
        )

    async def refresh_context_snapshot(self) -> None:
        await self.coding.refresh_context_snapshot()
        self._stats_line_cache_key = None
        self.invalidate()

    def rebuild_graph(self) -> None:
        """/config model or reasoning switch: rebind models + recompile."""
        self.coding.reload_model()

    def save_thread(self) -> str:
        return self.coding.save_thread()

    def request_compact(self) -> None:
        self.coding.request_compact()

    # --- thread management -------------------------------------------------
    async def reset_thread(self) -> None:
        """Archive the current thread and start a fresh one (``/new``)."""
        await self.coding.reset(_new_thread_id())
        self.coding.active_skills([])
        await self._reload_session_view([])

    async def resume_thread(self, thread_id: str) -> None:
        """Resume a saved thread: replay its transcript, then rebuild state.

        The visible transcript is cleared and re-rendered from the durable
        events first (no stale messages from the abandoned thread); the
        adapter then rebuilds the live graph from the same events.
        """
        events = self.coding.thread_store.load_thread_events(thread_id)
        if not events:
            render.render_error(f"No saved thread: {thread_id}")
            return
        if not await self.coding.resume(thread_id):
            render.render_error(f"No saved thread: {thread_id}")
            return
        await self._reload_session_view(events)

    async def rollback_to(self, user_seq: int) -> None:
        """Roll the thread back to checkpoint ``user_seq`` (``/rollback``)."""
        status = await self.coding.rollback_to(user_seq)
        if status.startswith(("Invalid", "No checkpoint")):
            render.render_error(status)
        else:
            events = self.coding.thread_store.load_thread_events(self.thread_id)
            await self._reload_session_view(events)
            render.render_notice(status, title="rollback")

    async def fork_thread(self, user_seq: int) -> None:
        """Fork before a user event, switch sessions, and prefill its prompt."""
        target, prompt, events = await self.coding.fork_before(user_seq)
        await self._reload_session_view(events)
        self._set_buffer_text(prompt)
        render.render_notice(f"Forked into {target}. Edit and submit the prompt.", title="fork")

    async def run_goal(self, goal: str) -> None:
        """Run a bounded worker–judge goal loop in the current transcript."""
        from liteharness_cli.goal import GoalCoordinator

        coordinator = GoalCoordinator(self.coding)

        async def worker_turn(instruction: str) -> None:
            render.render_user_echo(instruction)
            await self._run_turn(instruction, [])

        def on_status(role: str, message: str) -> None:
            render.render_notice(message, title=f"goal {role}")
            # Worker turns manage the spinner via ``_run_turn`` begin/finish.
            # The judge phase runs outside that path, so keep the spinner alive
            # while verifying — otherwise the UI looks frozen.
            if role == "judge":
                self.start_working()

        try:
            result = await coordinator.run(
                goal,
                worker_turn=worker_turn,
                on_status=on_status,
            )
        finally:
            self.finish_turn()

        verdict = result.verdict
        lines = [
            f"result: {'passed' if result.passed else 'stopped without passing'}",
            f"attempts: {result.attempts}",
        ]
        if verdict.evidence:
            lines.append("evidence:\n- " + "\n- ".join(verdict.evidence))
        if verdict.unmet:
            lines.append("unmet:\n- " + "\n- ".join(verdict.unmet))
        render.render_panel_text("\n".join(lines), title="goal verdict")

    async def _reload_session_view(self, events: list[dict]) -> None:
        """Atomically rebuild all transcript-derived state for one session."""
        self.clear_transcript()
        self.assistant_history = [
            str(event.get("content"))
            for event in events
            if event.get("kind") == "assistant" and event.get("content")
        ]
        self.render_header()
        self._replay_events_to_transcript(events)
        await self.refresh_context_snapshot()

    def _replay_events_to_transcript(self, events: list[dict]) -> None:
        """Render a saved event stream into the live transcript on resume.

        Mirrors the live-turn render path: user echoes, assistant panels,
        and tool call/result rows in order. Usage events are skipped (costs
        are replayed into the tracker by the adapter's resume).
        """
        for event in events:
            kind = event.get("kind")
            if kind == "user":
                content = event.get("content", "")
                text = content if isinstance(content, str) else str(content)
                if text.strip():
                    render.render_user_echo(text)
            elif kind == "assistant":
                tool_calls_raw = event.get("tool_calls") or []
                if tool_calls_raw:
                    calls = [
                        {
                            "name": tc.get("name"),
                            "args": tc.get("args", {}),
                            "id": tc.get("id"),
                            "type": tc.get("type", "tool_call"),
                        }
                        for tc in tool_calls_raw
                    ]
                    render.render_tool_calls(calls)
                content = event.get("content")
                text = "" if content is None else str(content)
                if text.strip():
                    render.render_assistant_panel(text)
            elif kind == "tool":
                tool_name = str(event.get("tool") or "")
                result = str(event.get("result") or "")
                if tool_name:
                    render.render_tool_result(tool_name, result)

    # --- the turn -----------------------------------------------------------
    async def _run_turn(self, text: str, images: list[str]) -> None:
        """Drive one CodingSession turn, rendering its SessionEvent stream."""
        renderer = TurnRenderer()
        # Omit active_skills= so Session._pending_skills (staged via /skill →
        # coding.stage_skills) is consumed by the SDK payload builder.
        render.begin_turn()
        try:
            async for ev in self.coding.run_turn(
                text,
                images=images or None,
            ):
                renderer.feed(ev)
        except asyncio.CancelledError:
            # Hard-escalation path: the cooperative cancel didn't break the
            # stream in time and the submit task was cancelled. The SDK has
            # already finalised checkpoint state; render the interrupt UX
            # here since no ``interrupted`` event will be consumed now.
            if not renderer.interrupted:
                renderer.feed(SessionEvent("interrupted", {"partial_text": ""}))
            raise
        finally:
            render.finish_turn()
        self.assistant_history.extend(renderer.assistant_texts)
        if not renderer.interrupted:
            render.render_usage_footer(renderer.usage)
            render.render_todos(await self.coding.get_todos())
