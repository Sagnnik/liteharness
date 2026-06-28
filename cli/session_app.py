"""SessionApp: the CLI controller that owns app state and drives the graph.

Responsibilities: build/rebuild the LangGraph app, run a turn (stream + render +
per-turn usage), themed permission prompts, mode toggling with the plan->act
compaction checkpoint, and full-transcript resume / reset / save.
"""

from __future__ import annotations

import base64
import mimetypes
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style

from agent import _effective_conversation, build_graph
from compaction import (
    PLAN_COMPACTION_CHECKPOINT_RATIO,
    apply_force_floor,
    calculate_context_pressure,
    compaction_label,
)
from config import cost_tracker, settings
from model import active_model_name, create_model, create_reflection_model
from reflection import finalize_session_reflection
from session import append_event, archive_thread, list_subagents, load_thread_events
from tools import EDIT_TOOLS
from tools.todo import get_thread_todos
from utils import preview_diff

from cli import render
from cli.theme import PTK_STYLE_RULES


def new_thread_id() -> str:
    return f"session-{uuid.uuid4().hex[:8]}"


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
        self.pending_image = ""
        self.queued_prompt = ""
        self.assistant_history: list[str] = []
        self.last_usage: dict[str, Any] | None = None

        self._force_compact = False
        self._pending_act_checkpoint = False
        self._bootstrap: dict[str, list[BaseMessage]] = {}
        self._seen: dict[str, int] = {}

        self.checkpointer = MemorySaver()
        self._line_session: PromptSession | None = None

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
        if self._pending_act_checkpoint and self.agent_mode == "act":
            await self._maybe_checkpoint_before_act()
            self._pending_act_checkpoint = False
        self._ensure_graph()

        user_message = self._build_user_message(user_text)
        append_event(self.thread_id, {"kind": "user", "content": _event_content(user_message.content)})

        config = {"configurable": {"thread_id": self.thread_id}}
        initial = self._bootstrap.pop(self.thread_id, [])
        payload = {
            "messages": [*initial, user_message],
            "approval_declined": False,
            "agent_mode": self.agent_mode,
            "force_compact": self._consume_force_compact(),
        }

        before = self._cost_snapshot()
        stream: render.AssistantStream | None = None
        streamed_any = False

        async for event in self.app.astream_events(payload, config=config, version="v2"):
            etype = event.get("event")
            name = event.get("name", "")

            if etype == "on_chat_model_start":
                stream = render.AssistantStream()
                streamed_any = False
            elif etype == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                text = getattr(chunk, "content", "")
                if isinstance(text, str) and text and stream is not None:
                    stream.feed(text)
                    streamed_any = True
            elif etype == "on_chat_model_end":
                if stream is not None:
                    stream.stop()
                    if streamed_any and stream.text.strip():
                        self._record_assistant(stream.text)
                    stream = None
            elif etype == "on_chain_end" and name == "agent":
                self._render_agent_output(event, streamed_any)
            elif etype == "on_chain_end" and name == "tools":
                self._render_tool_results(event)

        after = self._cost_snapshot()
        self.last_usage = self._usage_delta(before, after)
        render.render_usage_footer(self.last_usage)
        render.render_todos(get_thread_todos(self.thread_id))

    def _record_assistant(self, text: str) -> None:
        self.assistant_history.append(text)
        if self.agent_mode == "plan":
            self._save_plan(text)

    def _render_agent_output(self, event: dict, streamed_any: bool) -> None:
        for msg in _messages_from_event(event):
            if getattr(msg, "type", None) not in {"ai", "assistant"}:
                continue
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                for call in tool_calls:
                    render.render_tool_call(call.get("name", "?"), call.get("args", {}))
            elif msg.content and not streamed_any:
                text = str(msg.content)
                render.render_assistant_panel(text)
                self._record_assistant(text)

    def _render_tool_results(self, event: dict) -> None:
        for msg in _messages_from_event(event):
            if getattr(msg, "type", None) != "tool":
                continue
            if (getattr(msg, "additional_kwargs", None) or {}).get("hidden"):
                continue
            render.render_tool_result(getattr(msg, "name", "tool"), str(msg.content))

    # --- approval handler --------------------------------------------------
    async def _approval(self, name: str, args: dict) -> str:
        render.console.print()
        render.render_warning(f"approval needed: {name}")
        render.render_tool_call(name, args)
        if name in EDIT_TOOLS:
            diff = preview_diff(name, args)
            if diff and diff.strip():
                render.render_diff(diff)
        while True:
            choice = (await self._ask_line("approve? [y=yes S=session a=always n=no N=never d=diff s=show] ")).strip()
            low = choice.lower()
            if low in {"d", "diff"}:
                render.render_diff(preview_diff(name, args) or "(no diff)")
                continue
            if low in {"s", "show"}:
                render.render_panel_text(_format_args_full(args), title=f"{name} args", style="usage.value")
                continue
            if low in {"a", "always"}:
                return "always"
            if choice == "S" or low == "session":
                return "session"
            if choice == "N" or low == "never":
                return "never"
            if low in {"y", "yes"}:
                return "yes"
            if low in {"n", "no"}:
                return "no"
            render.render_warning("Choose y, S, a, n, N, d, or s.")

    # --- interactive MCQ clarification handler -----------------------------
    async def _ask_questions(self, questions: list[dict]) -> list[dict]:
        answers: list[dict] = []
        for q_index, question in enumerate(questions, 1):
            options = question.get("options", [])
            render.console.print()
            render.render_notice(question.get("prompt", ""), title=f"question {q_index}")
            for opt_index, option in enumerate(options, 1):
                line = render.Text()
                line.append(f"  {opt_index}. ", style="usage")
                line.append(str(option.get("label", "")), style="usage.value")
                if option.get("recommended"):
                    line.append("  (recommended)", style="mode.plan")
                render.console.print(line)

            choice = await self._ask_option_choice(len(options))
            selected = options[choice]
            note = ""
            if question.get("allow_note", True):
                note = (await self._ask_line("note (optional, Enter to skip): ")).strip()

            answers.append(
                {
                    "id": question.get("id", str(q_index)),
                    "selected": {"id": selected.get("id"), "label": selected.get("label")},
                    "note": note,
                }
            )
        return answers

    async def _ask_option_choice(self, count: int) -> int:
        while True:
            raw = (await self._ask_line(f"choose [1-{count}]: ")).strip()
            if raw.isdigit():
                idx = int(raw)
                if 1 <= idx <= count:
                    return idx - 1
            render.render_warning(f"Enter a number between 1 and {count}.")

    async def _ask_line(self, message: str) -> str:
        if self._line_session is None:
            self._line_session = PromptSession(style=Style.from_dict(PTK_STYLE_RULES))
        return await self._line_session.prompt_async([("class:prompt", message)])

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

    # --- compaction / reflection helpers -----------------------------------
    def request_compact(self) -> None:
        self._force_compact = True
        append_event(self.thread_id, {"kind": "compact", "content": "manual compaction requested"})

    def _consume_force_compact(self) -> bool:
        value = self._force_compact
        self._force_compact = False
        return value

    async def finalize_reflection(self) -> None:
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
        self.rebuild_graph()

    async def resume_thread(self, thread_id: str) -> None:
        events = load_thread_events(thread_id)
        if not events:
            render.render_error(f"No saved thread: {thread_id}")
            return
        if thread_id != self.thread_id:
            await self.finalize_reflection()
            archive_thread(self.thread_id)

        subagents = list_subagents(thread_id)
        messages = _events_to_messages_full(events, subagents)
        _restore_cost_from_events(events)

        self.thread_id = thread_id
        self._bootstrap[thread_id] = messages
        self._seen[thread_id] = len(messages)
        self.assistant_history = [
            str(m.content) for m in messages if getattr(m, "type", None) in {"ai", "assistant"} and m.content
        ]
        self.rebuild_graph()
        render.render_notice(f"Resumed thread {thread_id} ({len(messages)} messages restored).", title="resume")

    # --- message building --------------------------------------------------
    def _build_user_message(self, text: str) -> HumanMessage:
        pending = self.pending_image
        self.pending_image = ""
        cleaned, inline_images = _extract_inline_images(text)
        images = ([pending] if pending else []) + inline_images
        if not images:
            return HumanMessage(content=cleaned)
        if not settings.supports_vision:
            render.render_warning("Current model is not marked vision-capable; sending text only.")
            return HumanMessage(content=cleaned)
        blocks: list[dict[str, Any]] = [{"type": "text", "text": cleaned or "Please inspect this image."}]
        for image_path in images:
            blocks.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}})
        return HumanMessage(content=blocks)

    def _save_plan(self, text: str) -> Path:
        plans_dir = Path(settings.ness_dir) / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        stamp = re.sub(r"[-:.TZ]", "", datetime.utcnow().isoformat(timespec="seconds"))
        path = plans_dir / f"{stamp}-{self.thread_id}.md"
        path.write_text(text.strip() + "\n", encoding="utf-8")
        return path


# --- module helpers ---------------------------------------------------------
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
            messages.append(HumanMessage(content=content if isinstance(content, str) else str(content)))
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
        cost_tracker.cache_write_tokens += int(event.get("cache_write_tokens", 0) or 0)
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


def _format_args_full(args: Any) -> str:
    import json

    try:
        return json.dumps(args, indent=2, ensure_ascii=False)
    except TypeError:
        return str(args)


def _event_content(content: Any) -> Any:
    if isinstance(content, (str, int, float, bool)) or content is None:
        return content
    return str(content)
