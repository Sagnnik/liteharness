"""SessionApp: the CLI controller that owns app state and drives the graph.

Responsibilities: build/rebuild the LangGraph app, run a turn (stream + render +
per-turn usage), themed permission prompts, mode toggling with the plan->act
compaction checkpoint, and full-transcript resume / reset / save.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from agent import _effective_conversation, build_graph
from compaction import (
    PLAN_COMPACTION_CHECKPOINT_RATIO,
    apply_force_floor,
    calculate_context_pressure,
    compaction_label,
    resolve_usable_context_budget,
)
from config import cost_tracker, settings
from memory import NESS_DIR
from model import (
    active_model_name,
    active_reasoning_effort,
    create_model,
    create_reflection_model,
    model_supports_reasoning,
)
from reflection import finalize_session_reflection
from rollback import (
    create_file_checkpoint,
    read_mem_file,
    restore_mem_file,
    restore_paths,
)
from session import (
    append_event,
    archive_thread,
    get_checkpoint,
    list_subagents,
    list_user_turns,
    load_thread_events,
    save_checkpoint,
    truncate_after,
)
from tools.subagents import subagent_runs_active
from tools.todo import get_thread_todos

from cli import render
from cli.mentions import expand_documents


def new_thread_id() -> str:
    return f"session-{uuid.uuid4().hex[:8]}"


class CancelToken:
    """Cooperative cancellation flag for the active turn's stream loop.

    A thin wrapper over ``asyncio.Event`` so the TUI keybinding layer can
    request a clean break-out of the ``astream_events`` loop instead of
    raising ``CancelledError`` mid-flight. The loop polls ``is_set()`` after
    each event and performs partial-state cleanup before returning normally;
    a ``call_later`` hard-escalation backstop (``asyncio.Task.cancel()``) is
    scheduled by the TUI in case the cooperative path is stuck on a long
    blocked LLM call.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def trigger(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def reset(self) -> None:
        self._event.clear()

    async def wait(self) -> None:
        await self._event.wait()


def graph_rebuild_needed(current_thread: str, built_thread: str) -> bool:
    """A graph rebuild is needed only when the thread id changed.

    Mode is enforced at runtime via state["agent_mode"] and the full tool set is
    always bound, so a plan<->act switch never requires a rebuild.
    """
    return current_thread != built_thread


class SessionApp:
    def __init__(self, *, git_available: bool) -> None:
        self.git_available = git_available
        self.thread_id = new_thread_id()
        self.agent_mode = "act"
        self.should_exit = False
        self.prompt_queue: list[str] = []
        self.pending_skills: list[str] = []
        self.assistant_history: list[str] = []
        self.last_usage: dict[str, Any] | None = None
        self.turn_count = 0
        self.context_used = 0
        self.context_total = resolve_usable_context_budget(model_name=settings.model_name)

        self._force_compact = False
        self._pending_act_checkpoint = False
        self._bootstrap: dict[str, list[BaseMessage]] = {}
        self._seen: dict[str, int] = {}
        self._plan_turn_texts: list[str] = []

        # Cooperative cancellation flag for the active turn. Reset at the
        # start of each run_turn so a stale trigger from a prior turn cannot
        # bleed into the next one.
        self.cancel_token = CancelToken()

        self.checkpointer = MemorySaver()

        self.model = create_model(self.thread_id)
        self._built_thread_id = self.thread_id
        self.app = self._build()

    # --- graph lifecycle ---------------------------------------------------
    def _build(self):
        return build_graph(
            self.model,
            thread_id=self.thread_id,
            agent_mode=self.agent_mode,
            git_available=self.git_available,
            checkpointer=self.checkpointer,
            approval_handler=self._approval,
            question_handler=self._ask_questions,
        )

    def rebuild_graph(self) -> None:
        self.model = create_model(self.thread_id)
        self.app = self._build()
        self._built_thread_id = self.thread_id

    def _ensure_graph(self) -> None:
        if graph_rebuild_needed(self.thread_id, self._built_thread_id):
            self.rebuild_graph()

    # --- mode toggle -------------------------------------------------------
    def toggle_mode(self) -> None:
        if self.agent_mode == "act":
            self.agent_mode = "plan"
            # An act checkpoint only makes sense entering act mode; dropping
            # back to plan cancels any pending one so it can't strand a stale
            # compaction prompt on a later plan->act switch.
            self._pending_act_checkpoint = False
        else:
            self.agent_mode = "act"
            self._pending_act_checkpoint = True

    # --- header ------------------------------------------------------------
    def render_header(self) -> None:
        render.render_header(
            mode=self.agent_mode,
            model=active_model_name(),
            approval=settings.enable_approval,
            autosave=settings.auto_save_threads,
            session_end_reflection=settings.session_end_reflection,
        )

    # --- per-turn cost helpers --------------------------------------------
    @staticmethod
    def _cost_snapshot() -> tuple[int, int, int, float]:
        return (
            cost_tracker.input_tokens,
            cost_tracker.output_tokens,
            cost_tracker.cached_input_tokens,
            cost_tracker.cost_usd,
        )

    @staticmethod
    def _usage_delta(before, after) -> dict[str, Any]:
        return {
            "input_tokens": after[0] - before[0],
            "output_tokens": after[1] - before[1],
            "cached_input_tokens": after[2] - before[2],
            "cost_usd": after[3] - before[3],
        }

    # --- the turn ----------------------------------------------------------
    async def run_turn(self, user_text: str) -> None:
        # Fresh token per turn; a stale trigger from the prior turn must not
        # immediately abort this one.
        self.cancel_token.reset()
        mode_switch = ""
        # Consume the plan->act toggle BEFORE the (awaitable) checkpoint
        # prompt so a Ctrl+C during that prompt doesn't strand the toggle
        # for the next turn. The checkpoint ask itself runs inside the try
        # below so a cancel there is finalised cleanly and ``mode_switch``
        # is still emitted on this turn's payload.
        pending_switch = self._pending_act_checkpoint and self.agent_mode == "act"
        if pending_switch:
            self._pending_act_checkpoint = False
            mode_switch = "plan->act"
        self._ensure_graph()

        # Hoisted before the try so the ``except CancelledError`` handler can
        # finalise even if the cancel lands before the payload is built (e.g.
        # during the plan->act checkpoint prompt).
        # recursion_limit caps the number of supersteps per turn. LangGraph's
        # default of 25 is too tight for long-running agentic turns. 75 gives real work headroom
        config = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 75}
        before = self._cost_snapshot()
        stream: render.AssistantStream | None = None
        streamed_any = False
        # ``reasoning_enabled`` is a fast-path guard only — it short-circuits
        # the ``additional_kwargs`` dict thunk on the streaming hot path for
        # non-reasoning models and when the user has explicitly set
        # effort="none". The actual gate for whether to emit a block is the
        # buffer-emptiness check inside ``AssistantStream.finalize_reasoning``;
        # some providers (Anthropic via OpenRouter) emit reasoning_content
        # even at minimal effort, so we trust the observed data and only
        # fast-path-skip the obvious no-op case.
        reasoning_enabled = model_supports_reasoning(active_model_name()) and active_reasoning_effort() != "none"
        self._plan_turn_texts = []

        render.begin_turn()

        try:
            if pending_switch:
                await self._maybe_checkpoint_before_act()

            user_message, persist_text = self._build_user_message(user_text)

            # Per-turn rollback checkpoint: snapshot files (git stash create) +
            # the per-thread session memory file BEFORE the agent acts, then
            # key the row by the soon-to-be-appended user event seq. The stash
            # commits leave the user's index/branch/working tree untouched,
            # so this is safe to run unconditionally on every turn even when
            # git is unavailable (create_file_checkpoint returns None then).
            git_hash = await asyncio.to_thread(create_file_checkpoint)
            mem_snapshot = await asyncio.to_thread(read_mem_file, NESS_DIR, self.thread_id)
            user_seq = append_event(self.thread_id, {"kind": "user", "content": _event_content(persist_text)})
            if user_seq is not None:
                save_checkpoint(self.thread_id, user_seq, git_hash, mem_snapshot)
            self.turn_count += 1

            initial = self._bootstrap.pop(self.thread_id, [])
            activate_skills = self.pending_skills
            self.pending_skills = []
            payload = {
                "messages": [*initial, user_message],
                "approval_declined": False,
                "agent_mode": self.agent_mode,
                "force_compact": self._consume_force_compact(),
                "activate_skills": activate_skills,
                "mode_switch": mode_switch,
                "current_user_seq": user_seq or 0,
            }

            async for event in self.app.astream_events(payload, config=config, version="v2"):
                etype = event.get("event")
                name = event.get("name", "")

                if subagent_runs_active() > 0:
                    if etype == "on_chat_model_end" and stream is not None:
                        stream.stop()
                        stream = None
                        streamed_any = False
                    continue

                if etype == "on_chat_model_start":
                    stream = render.AssistantStream()
                    streamed_any = False
                elif etype == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    # Reasoning capture (OpenRouter): the model's chain-of-thought
                    # arrives on ``chunk.additional_kwargs["reasoning_content"]``
                    # as a stream of string fragments (``content`` may be empty
                    # for pure-reasoning chunks). The facade reserves a slot
                    # above the assistant stream's reserved markdown slot
                    # (Anthropic/OpenCode convention) on the first fragment;
                    # finalize at ``on_chat_model_end`` re-emits the collapsed
                    # ``+ Thinking: n sec`` line.
                    if reasoning_enabled:
                        ak = getattr(chunk, "additional_kwargs", None) or {}
                        rtext = ak.get("reasoning_content") if isinstance(ak, dict) else None
                        if isinstance(rtext, str) and rtext and stream is not None:
                            stream.feed_reasoning(rtext)
                    text = getattr(chunk, "content", "")
                    if isinstance(text, str) and text and stream is not None:
                        stream.feed(text)
                        streamed_any = True
                elif etype == "on_chat_model_end":
                    if stream is not None:
                        stream.stop()
                        if streamed_any and stream.text.strip():
                            self._record_assistant(stream.text)
                        stream.finalize_reasoning()
                        stream = None
                # on_chain_end -> each langgraph node end, name -> node name
                elif etype == "on_chain_end" and name == "agent":
                    self._render_agent_output(event, streamed_any)
                elif etype == "on_chain_end" and name == "tools":
                    self._render_tool_results(event)

                # Cooperative cancellation: break out between events so the
                # post-loop cleanup can flush partial state cleanly. The hard
                # backstop (asyncio.Task.cancel) is the TUI's fallback if a
                # long blocked call keeps the loop from reaching this check.
                if self.cancel_token.is_set():
                    break

            if self.cancel_token.is_set():
                await self._finalize_cancelled_turn(stream, streamed_any, config)
            else:
                after = self._cost_snapshot()
                self.last_usage = self._usage_delta(before, after)
                render.render_usage_footer(self.last_usage)
                render.render_todos(get_thread_todos(self.thread_id))
                self._autosave_plan_turn()
                await self.refresh_context_snapshot()
        except asyncio.CancelledError:
            # Hard-escalation path: the cooperative cancel token failed to
            # break the stream loop within the backstop window, or a cancel
            # landed during the plan->act checkpoint prompt. Best-effort
            # finalisation before re-raising so the task is still properly
            # cancelled and the checkpoint isn't left dirty. ``shield`` keeps
            # the finalisation running even if a second Ctrl+C arrives.
            try:
                await asyncio.shield(
                    self._finalize_cancelled_turn(stream, streamed_any, config)
                )
            except asyncio.CancelledError:
                pass  # double-cancel; state may be dirty, nothing we can do
            except Exception:
                pass
            raise
        finally:
            render.finish_turn()

    async def _finalize_cancelled_turn(
        self,
        stream: render.AssistantStream | None,
        streamed_any: bool,
        config: dict,
    ) -> None:
        """Flush partial state after a cooperative or hard cancel.

        Mirrors how OpenCode/Codex settle an interrupted turn:

        - persist any partial assistant text annotated as interrupted,
        - synthesise a *failed* ``ToolMessage`` for every pending tool call so
          the checkpoint stays consistent, and
        - when neither partial text nor pending tool calls exist (cancel landed
          mid-LLM-call after all tools already returned, or before any
          assistant token streamed this turn), inject an ``AIMessage``
          interruption marker so the model does not silently resume the
          abandoned request on the next turn.

        The marker is an ``AIMessage`` (not ``HumanMessage``) to preserve
        strict user/assistant alternation: the last checkpoint message may be
        a ``HumanMessage`` (empty-stream turn) or a ``ToolMessage``
        (just-completed tools), and an ``AIMessage`` cap is valid in both
        cases, while a second ``HumanMessage`` risks back-to-back humans that
        some providers reject.
        """
        recorded_text = False
        partial_reasoning: tuple[str | None, float] = (None, 0.0)
        if stream is not None:
            # Capture partial reasoning before ``stop()`` discards any
            # in-flight reasoning state held by the facade.
            partial_reasoning = stream.reasoning_state()
            stream.stop()
            if streamed_any and stream.text.strip():
                self._record_assistant(stream.text.strip() + " … [interrupted]")
                recorded_text = True

        # Flush any partial reasoning captured before the cancel so the
        # interruption marker follows after the CoT block, not above an empty
        # region. The ``… [interrupted]`` suffix mirrors the assistant-text
        # convention; ``partial_reasoning[0] is None`` means the cancel landed
        # before reasoning started (or the model was non-reasoning) so nothing
        # is emitted.
        if partial_reasoning[0]:
            render.render_reasoning(partial_reasoning[0] + " … [interrupted]", elapsed=partial_reasoning[1])

        synthetic: list[BaseMessage] = []
        try:
            snapshot = await self.app.aget_state(config)
        except Exception:
            snapshot = None
        has_pending = False
        if snapshot is not None:
            messages = list((snapshot.values or {}).get("messages", []))
            answered_ids = {
                getattr(m, "tool_call_id", None)
                for m in messages
                if isinstance(m, ToolMessage)
            }
            # Find the most recent AIMessage and synthesise failure responses
            # for any tool_call on it that has no matching ToolMessage yet.
            for msg in reversed(messages):
                if not isinstance(msg, AIMessage):
                    continue
                for tc in (msg.tool_calls or []):
                    call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    if not call_id or str(call_id) in answered_ids:
                        continue
                    has_pending = True
                    tool_name = (
                        tc.get("name")
                        if isinstance(tc, dict)
                        else getattr(tc, "name", "tool")
                    )
                    synthetic.append(
                        ToolMessage(
                            tool_call_id=str(call_id),
                            name=str(tool_name or "tool"),
                            content="Tool execution interrupted",
                        )
                    )
                break

        # No partial assistant text AND no pending tool calls -> the cancel
        # landed during a pure LLM call with no observable output this turn.
        # Without an explicit marker the model has no signal that the prior
        # request was abandoned and will silently resume it. An ``AIMessage``
        # cap keeps alternation valid regardless of what precedes it.
        if not recorded_text and not has_pending:
            synthetic.append(
                AIMessage(
                    content="… [turn interrupted by user; "
                    "the previous request was abandoned — do not continue it.]"
                )
            )

        if synthetic:
            try:
                await self.app.aupdate_state(config, {"messages": synthetic})
            except Exception as exc:
                render.render_warning(f"Cancel state flush failed: {exc}")

        # Persist any partial plan text so an interrupted plan turn isn't
        # silently lost (the success path autosaves via _autosave_plan_turn).
        self._autosave_plan_turn()

        render.render_notice("Turn interrupted by user.", title="cancel")

    def _record_assistant(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        self.assistant_history.append(cleaned)
        if self.agent_mode == "plan":
            self._plan_turn_texts.append(cleaned)

    def _autosave_plan_turn(self) -> None:
        plan_text = plan_autosave_text(self._plan_turn_texts)
        if plan_text is not None:
            self._save_plan(plan_text)

    def _render_agent_output(self, event: dict, streamed_any: bool) -> None:
        for msg in _messages_from_event(event):
            if getattr(msg, "type", None) not in {"ai", "assistant"}:
                continue
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                render.render_tool_calls(tool_calls)
            elif msg.content and not streamed_any:
                text = str(msg.content)
                render.render_assistant_panel(text)
                self._record_assistant(text)

    def _render_tool_results(self, event: dict) -> None:
        todo_updated = False
        for msg in _messages_from_event(event):
            if getattr(msg, "type", None) != "tool":
                continue
            if (getattr(msg, "additional_kwargs", None) or {}).get("hidden"):
                continue
            name = getattr(msg, "name", "tool")
            if str(name or "") == "todo":
                todo_updated = True
            render.render_tool_result(name, str(msg.content))
        if todo_updated:
            render.render_todos(get_thread_todos(self.thread_id))

    # --- approval handler --------------------------------------------------
    async def _approval(self, name: str, args: dict) -> str:
        return await render.ask_approval(name, args)

    # --- interactive MCQ clarification handler -----------------------------
    async def _ask_questions(self, questions: list[dict]) -> list[dict]:
        return await render.ask_questions(questions)

    async def _ask_line(self, message: str) -> str:
        return await render.ask_line(message)

    async def refresh_context_snapshot(self) -> None:
        self.context_total = resolve_usable_context_budget(model_name=settings.model_name)
        try:
            snapshot = await self.app.aget_state({"configurable": {"thread_id": self.thread_id}})
        except Exception:
            return
        state = dict(snapshot.values or {})
        messages = list(state.get("messages", []))
        if not messages:
            self.context_used = 0
            return
        conversation = _effective_conversation(messages, state)
        pressure = calculate_context_pressure(
            conversation,
            known_input_tokens=state.get("last_input_tokens") or None,
            model_name=settings.model_name,
        )
        self.context_used = pressure.token_count
        self.context_total = pressure.usable_budget

    # --- compaction checkpoint (plan -> act) ----------------------------
    async def _maybe_checkpoint_before_act(self) -> None:
        try:
            snapshot = await self.app.aget_state({"configurable": {"thread_id": self.thread_id}})
        except Exception:
            return

        state = dict(snapshot.values or {})
        messages = list(state.get("messages", []))
        if not messages:
            return
        conversation = _effective_conversation(messages, state)
        pressure = calculate_context_pressure(
            conversation,
            known_input_tokens=state.get("last_input_tokens") or None,
            model_name=settings.model_name,
        )
        if pressure.ratio < PLAN_COMPACTION_CHECKPOINT_RATIO:
            return

        rest_count = sum(1 for message in conversation if message.type != "system")
        action, keep_recent = apply_force_floor(pressure.action, pressure.keep_recent, rest_count)
        info = (
            f"Context ~{pressure.token_count:,} tokens of {pressure.usable_budget:,} budget "
            f"({pressure.ratio:.0%}). Compaction if run: {compaction_label(action, keep_recent)}."
        )
        if pressure.hard_threshold_reached:
            self._force_compact = True
            append_event(self.thread_id, {"kind": "compact", "content": "pre-execution hard-threshold compaction requested"})
            render.render_notice(info + " Hard threshold reached; compacting before execution.", title="compaction")
            return
        render.render_notice(info, title="pre-execution checkpoint")
        answer = (await self._ask_line("compact before execution? [y/N] ")).strip().lower()
        if answer in {"y", "yes"}:
            self._force_compact = True
            append_event(self.thread_id, {"kind": "compact", "content": "pre-execution compaction requested"})

    # --- prompt queue ------------------------------------------------------
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

    # --- compaction / reflection helpers -----------------------------------
    def request_compact(self) -> None:
        self._force_compact = True
        append_event(self.thread_id, {"kind": "compact", "content": "manual compaction requested"})

    def _consume_force_compact(self) -> bool:
        value = self._force_compact
        self._force_compact = False
        return value

    async def finalize_reflection(self) -> None:
        if not settings.session_end_reflection:
            return
        try:
            await finalize_session_reflection(self.app, self.thread_id, create_reflection_model(self.thread_id))
        except Exception as exc:
            render.render_warning(f"Reflection finalize skipped: {exc}")

    # --- thread management -------------------------------------------------
    def save_thread(self) -> str:
        return archive_thread(self.thread_id)

    async def reset_thread(self) -> None:
        await self.finalize_reflection()
        archive_thread(self.thread_id)
        self.thread_id = new_thread_id()
        self._seen.pop(self.thread_id, None)
        self.assistant_history.clear()
        self.turn_count = 0
        self.rebuild_graph()
        await self.refresh_context_snapshot()

    async def resume_thread(self, thread_id: str) -> None:
        events = load_thread_events(thread_id)
        if not events:
            render.render_error(f"No saved thread: {thread_id}")
            return
        if thread_id != self.thread_id:
            await self.finalize_reflection()
            archive_thread(self.thread_id)

        await self._bootstrap_from_events(thread_id, replay_cost=True)
        render.render_notice(f"Resumed thread {thread_id}.", title="resume")

    async def _bootstrap_from_events(self, thread_id: str, *, replay_cost: bool) -> None:
        """Rebuild the live graph from persisted events for ``thread_id``.

        Shared by /resume (cost replay enabled) and /rollback (cost replay
        skipped — the in-process cost_tracker intentionally retains abandoned
        turns' usage). Reads events -> messages, sets per-thread aux state, and
        rebuilds the bound model/graph so the next turn replays from the
        truncated history.
        """
        events = load_thread_events(thread_id)
        subagents = list_subagents(thread_id)
        messages = _events_to_messages_full(events, subagents)
        if replay_cost:
            _restore_cost_from_events(events)

        self.thread_id = thread_id
        self._bootstrap[thread_id] = messages
        self._seen[thread_id] = len(messages)
        self.assistant_history = [
            str(m.content) for m in messages if getattr(m, "type", None) in {"ai", "assistant"} and m.content
        ]
        self.turn_count = sum(1 for m in messages if getattr(m, "type", None) == "human")
        self.rebuild_graph()
        await self.refresh_context_snapshot()

    async def rollback_to(self, user_seq: int) -> None:
        """Restore the thread to the state captured at checkpoint ``user_seq``.

        Three things happen, in order:

        1. Files: surgically restore the paths the agent's tools mutated at or
           after this turn from the shadow git stash. Falls back to conversation-
           only restore when git is unavailable or no paths were recorded.
        2. Session memory: overwrite ``sessions/mem_<thread_id>.md`` from the
           checkpoint snapshot so reflection bullets written by abandoned turns
           are discarded.
        3. Conversation: hard-truncate the events tail at ``user_seq`` (deleting
           the abandoned user message and everything after it) and rebuild the
           live graph from the remaining events. The in-process cost_tracker is
           intentionally left untouched — its tally is process-global billing.
        """
        if user_seq < 0:
            render.render_error("Invalid rollback target.")
            return

        checkpoint = get_checkpoint(self.thread_id, user_seq)
        if checkpoint is None:
            render.render_error(f"No checkpoint for seq {user_seq} in this thread.")
            return

        # 1. Files (surgical restore; full tree when no paths recorded).
        git_hash = checkpoint.get("git_hash")
        if git_hash:
            import json as _json

            raw = checkpoint.get("modified_paths") or ""
            paths: list[str] = []
            if raw:
                try:
                    paths = list(_json.loads(raw))
                except ValueError:
                    paths = []
            try:
                output = await asyncio.to_thread(restore_paths, git_hash, paths)
                if output and not output.startswith("("):
                    render.render_warning(f"File restore note: {output}")
            except Exception as exc:
                render.render_warning(f"File restore failed: {exc}")
        else:
            render.render_notice("No git snapshot for this checkpoint; files not reverted.", title="rollback")

        # 2. Session memory (mem_<thread_id>.md) from the snapshot.
        try:
            await asyncio.to_thread(restore_mem_file, NESS_DIR, self.thread_id, checkpoint.get("mem_snapshot") or "")
        except Exception as exc:
            render.render_warning(f"Session memory restore failed: {exc}")

        # 3. Truncate the durable conversation tail and rebuild the live graph
        #    from the surviving events. cost_tracker is intentionally preserved.
        truncate_after(self.thread_id, user_seq)
        await self._bootstrap_from_events(self.thread_id, replay_cost=False)
        render.render_notice(
            f"Rolled back to turn @ seq {user_seq}. Files, memory, and conversation restored.",
            title="rollback",
        )

    # --- message building --------------------------------------------------
    def _build_user_message(self, text: str) -> tuple[HumanMessage, str]:
        """Return the ``HumanMessage`` sent to the model and the ``persist_text``
        that goes to the events table. The persist form keeps the visible
        ``@mention`` tokens (image-syntax stripped) so resume/rollback can
        re-expand fresh from disk; the message form has ``<document>`` blocks
        prepended via ``expand_documents`` and the prose still shows the
        ``@token`` so the model knows where the user pointed.
        """
        cleaned, inline_images = _extract_inline_images(text)
        persist_text = cleaned
        # Expand @file mentions into <document> blocks prepended to the user
        # text. ``expand_documents`` is a no-op when no mention is present.
        expanded = expand_documents(cleaned)
        if not inline_images:
            return HumanMessage(content=expanded), persist_text
        if not settings.supports_vision:
            render.render_warning("Current model is not marked vision-capable; sending text only.")
            return HumanMessage(content=expanded), persist_text
        blocks: list[dict[str, Any]] = [{"type": "text", "text": expanded or "Please inspect this image."}]
        for image_path in inline_images:
            blocks.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}})
        return HumanMessage(content=blocks), persist_text

    def _save_plan(self, text: str) -> Path:
        plans_dir = Path(settings.ness_dir) / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        stamp = re.sub(r"[-:.TZ]", "", datetime.utcnow().isoformat(timespec="seconds"))
        path = plans_dir / f"{stamp}-{self.thread_id}.md"
        path.write_text(text.strip() + "\n", encoding="utf-8")
        return path


# --- module helpers ---------------------------------------------------------
def plan_autosave_text(assistant_texts: list[str]) -> str | None:
    """Return the last non-empty assistant text from a plan turn, if any."""
    cleaned = [text.strip() for text in assistant_texts if text.strip()]
    return cleaned[-1] if cleaned else None


def _messages_from_event(event: dict) -> list[BaseMessage]:
    output = event.get("data", {}).get("output")
    if isinstance(output, dict):
        messages = output.get("messages")
        if isinstance(messages, list):
            return messages
    return []


def _events_to_messages_full(
    events: list[dict],
    subagents: list[dict[str, Any]] | None = None,
) -> list[BaseMessage]:
    """Rebuild the LangGraph transcript from saved events."""
    subagents = subagents or []
    messages: list[BaseMessage] = []
    pending_calls: list[dict[str, Any]] = []

    for event in events:
        kind = event.get("kind")
        if kind == "user":
            content = event.get("content", "")
            text = content if isinstance(content, str) else str(content)
            # Re-expand @file mentions against current disk on replay
            # (resume/rollback) so attached file content always reflects the
            # latest state rather than the snapshot at send time.
            text = expand_documents(text)
            messages.append(HumanMessage(content=text))
        elif kind == "assistant":
            tool_calls_raw = event.get("tool_calls") or []
            tool_calls = [
                {
                    "name": tc.get("name"),
                    "args": tc.get("args", {}),
                    "id": tc.get("id"),
                    "type": tc.get("type", "tool_call"),
                }
                for tc in tool_calls_raw
            ]
            content = event.get("content")
            text = "" if content is None else str(content)
            if text or tool_calls:
                messages.append(AIMessage(content=text, tool_calls=tool_calls))
                pending_calls = list(tool_calls)
        elif kind == "tool":
            call_id = str(event.get("call_id") or "")
            if not call_id and pending_calls:
                call_id = str(pending_calls[0].get("id") or "")
                pending_calls = pending_calls[1:]
            tool_name = str(event.get("tool") or "")
            result = _maybe_enrich_spawn_subagent_result(
                tool_name,
                str(event.get("result") or ""),
                subagents,
            )
            messages.append(
                ToolMessage(tool_call_id=call_id, name=tool_name, content=result)
            )

    return messages


def _subagent_batch_text(subagents: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, item in enumerate(subagents, start=1):
        heading = (
            f"[{index}] name={item.get('agent_name', '')} status={item.get('status', '')} "
            f"duration_ms={item.get('duration_ms', 0)} "
            f"thread_id={item.get('subagent_thread_id', '')}"
        )
        label = item.get("label")
        if label:
            heading += f" label={label}"
        lines.extend([heading, str(item.get("output") or "").strip(), ""])
    return "\n".join(lines).strip()


def _maybe_enrich_spawn_subagent_result(
    tool_name: str,
    result: str,
    subagents: list[dict[str, Any]],
) -> str:
    if tool_name != "spawn_subagent" or not subagents:
        return result
    enriched = _subagent_batch_text(subagents)
    if len(enriched) > len(result):
        return enriched
    return result


def _restore_cost_from_events(events: list[dict]) -> None:
    """Replay usage events into the cost tracker so totals continue after resume."""
    for event in events:
        if event.get("kind") != "usage":
            continue
        cost_tracker.input_tokens += int(event.get("input_tokens", 0) or 0)
        cost_tracker.uncached_input_tokens += int(event.get("uncached_input_tokens", 0) or 0)
        cost_tracker.cached_input_tokens += int(event.get("cached_input_tokens", 0) or 0)
        cost_tracker.output_tokens += int(event.get("output_tokens", 0) or 0)
        cost_tracker.cost_usd += float(event.get("cost_usd", 0.0) or 0.0)
        cost_tracker.calls += 1
        model = event.get("model")
        if model:
            cost_tracker.model_name = model


def _extract_inline_images(text: str) -> tuple[str, list[str]]:
    pattern = r"@image:([^\s]+)"
    images = re.findall(pattern, text)
    cleaned = re.sub(pattern, "", text).strip()
    return cleaned, images


def _image_to_data_url(path: str) -> str:
    p = Path(path).expanduser()
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    return f"data:{mime};base64,{data}"


def _event_content(content: Any) -> Any:
    if isinstance(content, (str, int, float, bool)) or content is None:
        return content
    return str(content)
