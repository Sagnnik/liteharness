from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from liteharness.compaction import (
    apply_force_floor,
    calculate_context_pressure,
    compaction_label,
)
from liteharness.graph.builder import build_graph
from liteharness.graph.helpers import _effective_conversation
from liteharness.session_context import SessionContext, set_session_context
from liteharness.types import ApprovalHandler, RunResult, SessionEvent, UsageEvent

_active_session: ContextVar["Session | None"] = ContextVar(
    "liteharness_active_session", default=None
)

PLAN_COMPACTION_CHECKPOINT_RATIO = 0.75


def _messages_from_event(event: dict) -> list[Any]:
    """Extracts the messages from the event data."""
    data = event.get("data") or {}
    output = data.get("output")
    if isinstance(output, dict) and "messages" in output:
        return list(output.get("messages") or [])
    if hasattr(output, "get") and callable(output.get):
        msgs = output.get("messages")
        if msgs is not None:
            return list(msgs)
    if isinstance(output, list):
        return list(output)
    return []


def _ensure_config_event_bridges(cfg: Any) -> None:
    """
    It installs one-time wrapper callbacks on the config object that bridge async events 
    (approval requests, questions, usage) into the active Session's event queue.
    """
    # skip if already installed
    if getattr(cfg, "_event_bridges_installed", False):
        return

    # get the original handlers
    original_approval = cfg.approval_handler
    original_question = cfg.question_handler
    original_usage = cfg.on_usage

    # Wraps approval_handler -> when called, emits an approval_required event 
    # into the active session's queue, then delegates to the original.
    if original_approval is not None:
        class _ApprovalHandler(ApprovalHandler):
            async def __call__(self, name: str, args: dict) -> str:
                sess = _active_session.get()
                if sess is not None:
                    sess._add_queue("approval_required", {"tool": name, "args": args})
                return await original_approval(name, args)
        cfg.approval_handler = _ApprovalHandler()

    # Same pattern for question_handler -> emits question_required event.
    if original_question is not None:
        async def _question(questions: list[dict]) -> list[dict]:
            sess = _active_session.get()
            if sess is not None:
                sess._add_queue("question_required", {"questions": questions})
            return await original_question(questions)
        # store the wrapped approval handler so langgraph can use this
        cfg.question_handler = _question

    # Wraps on_usage -> stores _last_usage on the session, 
    # emits a usage event with token/cost details, then calls the original handler.
    def _usage(event: UsageEvent) -> None:
        sess = _active_session.get()
        if sess is not None:
            sess._last_usage = event
            sess._add_queue(
                "usage",
                {
                    "model": event.model,
                    "input_tokens": event.input_tokens,
                    "uncached_input_tokens": event.uncached_input_tokens,
                    "cached_input_tokens": event.cached_input_tokens,
                    "output_tokens": event.output_tokens,
                    "cost_usd": event.cost_usd,
                },
            )
        if original_usage is not None:
            original_usage(event)

    # store the wrapped usage handler so langgraph can use this
    cfg.on_usage = _usage
    # mark as installed to avoid re-wrappings
    cfg._event_bridges_installed = True


class Session:
    def __init__(self, agent, *, thread_id, agent_mode="act", metadata=None, git_available=None):
        """Create a new interaction session bound to a single thread.

        Args:
            agent: A :class:`NessAgent` instance whose config drives tools,
                   models, prompts, and permissions.
            thread_id: Unique identifier for this conversation thread.
            agent_mode: Initial mode (``"act"`` or ``"plan"``).
            metadata: Arbitrary key-value pairs surfaced in the system prompt.
            git_available: Whether the project has a git repo.
        """
        self.agent = agent
        self.thread_id = thread_id
        self.agent_mode = agent_mode
        self.metadata = dict(metadata or {})
        self.git_available = git_available
        self._cfg = agent.config
        self._force_compact = False
        self._pending_act_checkpoint = False
        self._pending_skills: list[str] = []
        self.turn_count = 0
        self.context_used = 0
        self.context_total = 0
        self._event_queue: asyncio.Queue[SessionEvent] = asyncio.Queue()  # async queue to store SessionEvent objects
        self._last_usage: UsageEvent | None = None
        
        self.checkpointer = (
            self._cfg.checkpoint_factory() if self._cfg.checkpoint_factory else MemorySaver()
        )
        self._skill_loader = self._cfg.skill_loader
        
        _ensure_config_event_bridges(self._cfg)
        
        self._app = self._build_graph()

    def _add_queue(self, kind: str, data: dict[str, Any] | None = None) -> None:
        # add the event to the queue; non-blocking; of type SessionEvent
        self._event_queue.put_nowait(SessionEvent(kind, dict(data or {})))  # type: ignore[arg-type]

    def _drain_queue(self) -> list[SessionEvent]:
        # drain the queue and return a list of session events
        out: list[SessionEvent] = []
        while True:
            try:
                # grab items and adds to the list; non-blocking
                out.append(self._event_queue.get_nowait()) 
            except asyncio.QueueEmpty:
                break
        return out

    def _install_session_runtime(self) -> None:
        # sets up a runtime context for this session
        # it has everything specifications loaded into it (tools, models, options, permissions, threads etc.)
        cfg = self._cfg
        project_root = (cfg.options.project_root or Path.cwd()).resolve()
        ness_dir = (cfg.options.ness_dir or (project_root / ".ness")).resolve()
        set_session_context(
            SessionContext(
                permissions=cfg.permission_store,
                options=cfg.options,
                thread_store=cfg.thread_store,
                ness_dir=ness_dir,
                project_root=project_root,
                agent_config=cfg,
                all_skills=self._skill_loader.load(),
            )
        )

    def _build_graph(self):
        return build_graph(
            self._cfg,
            thread_id=self.thread_id,
            agent_mode=self.agent_mode,
            git_available=self.git_available,
            checkpointer=self.checkpointer,
            metadata=self.metadata,
        )

    @property
    def app(self):
        """The compiled langgraph application for this session."""
        return self._app

    def rebuild_graph(self) -> None:
        """Recompile the langgraph application (e.g. after config changes)."""
        self._app = self._build_graph()

    def set_mode(self, mode: str) -> None:
        """Switch the session to *mode* (``"act"`` or ``"plan"``).

        When switching from ``"plan"`` to ``"act"``, a pre-flight compaction
        checkpoint is scheduled for the next :meth:`run` or :meth:`stream`.
        """
        if mode == self.agent_mode:
            return
        if mode == "act" and self.agent_mode == "plan":
            self._pending_act_checkpoint = True
        else:
            self._pending_act_checkpoint = False
        self.agent_mode = mode

    def toggle_mode(self) -> str:
        """Flip ``"act"`` ↔ ``"plan"`` (CLI Shift+Tab semantics). Returns the new mode."""
        if self.agent_mode == "act":
            self.set_mode("plan")
        else:
            self.set_mode("act")
        return self.agent_mode

    @property
    def mode(self) -> str:
        """The current session mode (``"act"`` or ``"plan"``)."""
        return self.agent_mode

    def active_skills(self, names: Sequence[str]) -> None:
        """Sets the active skills for the session."""
        self._pending_skills = list(names)

    def request_compact(self) -> None:
        """Requests a compaction of the session."""
        self._force_compact = True

    def _consume_force_compact(self) -> bool:
        """Consumes the force compact flag and returns its current value."""
        v, self._force_compact = self._force_compact, False
        return v

    async def finalize_reflection(self) -> None:
        """Finalizes the session reflection."""
        if not self._cfg.options.session_end_reflection:
            return
        from liteharness.reflection import finalize_session_reflection

        await finalize_session_reflection(
            self._app,
            self.thread_id,
            self._cfg.reflection_model or self._cfg.model,
            memory=self._cfg.memory_store,
            persistence=self._cfg.thread_store,
            task_prompts=self._cfg.task_prompts,
            tracer=self._cfg.tracer,
        )

    async def aget_todos(self) -> list[dict[str, Any]]:
        """Gets the todos from the current graph state."""
        cfg = {"configurable": {"thread_id": self.thread_id}}
        try:
            snap = await self.app.aget_state(cfg)
        except Exception:
            return []
        return list((snap.values or {}).get("todos", []))

    async def refresh_context_snapshot(self) -> dict[str, Any]:
        """Refreshes the Session's token usage metrics by inspecting the current graph state."""
        # fetch the current graph state
        cfg = {"configurable": {"thread_id": self.thread_id}}
        try:
            snap = await self.app.aget_state(cfg)
        except Exception:
            return {}
        
        # convert the snapshot to a dictionary
        state = dict(snap.values or {})
        msgs = list(state.get("messages", []))
        if not msgs:
            self.context_used = 0
            return state
        
        # convert the messages to a conversation and resolve the model name
        conv = _effective_conversation(msgs, state)
        model_name = getattr(self._cfg.model, "model", "") or getattr(
            self._cfg.model, "model_name", ""
        )
        # calculate the context pressure
        pressure = calculate_context_pressure(
            conv,
            known_input_tokens=state.get("last_input_tokens") or None,
            model_name=model_name,
            options=self._cfg.options,
        )
        # update the context used and total
        self.context_used = pressure.token_count
        self.context_total = pressure.usable_budget
        return state

    def _user_message(
        self, message: str, images: Sequence[str] | None
    ) -> HumanMessage:
        """Builds a HumanMessage object with the message and images."""
        if not images:
            return HumanMessage(content=message)
        return HumanMessage(
            content=[{"type": "text", "text": message}]
            + [{"type": "image_url", "image_url": {"url": u}} for u in images]
        )

    async def _maybe_checkpoint_before_act(self) -> None:
        """
        Pre-execution compaction checkpoint when switching plan→act.
        It measures context pressure and either auto-compacts (hard threshold), 
        asks the user via callback, or emits an event so the frontend can decide.
        """
        # fetch the current state of the graph
        try:
            snapshot = await self.app.aget_state(
                {"configurable": {"thread_id": self.thread_id}}
            )
        except Exception:
            return

        # convert the snapshot to a dictionary
        state = dict(snapshot.values or {})
        messages = list(state.get("messages", []))
        if not messages:
            return
        # convert the messages to a conversation
        conversation = _effective_conversation(messages, state)
        model_name = getattr(self._cfg.model, "model", "") or getattr(
            self._cfg.model, "model_name", ""
        )
        # calculate the context pressure
        pressure = calculate_context_pressure(
            conversation,
            known_input_tokens=state.get("last_input_tokens") or None,
            model_name=model_name,
            options=self._cfg.options,
        )
        if pressure.ratio < PLAN_COMPACTION_CHECKPOINT_RATIO:
            return

        # count the number of non-system messages
        rest_count = sum(1 for message in conversation if message.type != "system")
        # apply the force floor
        action, keep_recent = apply_force_floor(
            pressure.action, pressure.keep_recent, rest_count
        )
        # create a string with the context pressure information
        info = (
            f"Context ~{pressure.token_count:,} tokens of {pressure.usable_budget:,} budget "
            f"({pressure.ratio:.0%}). Compaction if run: {compaction_label(action, keep_recent)}."
        )

        # if the hard threshold is reached then force a compaction
        if pressure.hard_threshold_reached:
            self._force_compact = True
            self._cfg.thread_store.append_event(
                self.thread_id,
                {"kind": "compact", "content": "pre-execution hard-threshold compaction requested"},
            )
            self._add_queue(
                "compaction",
                {
                    "reason": "pre_act_hard_threshold",
                    "info": info,
                    "forced": True,
                },
            )
            return
        # if the user has a custom compaction handler then use it
        should_compact = False
        if self._cfg.on_pre_act_compact is not None:
            should_compact = bool(await self._cfg.on_pre_act_compact(pressure))
        # if the user wants to compact then force a compaction
        if should_compact:
            self._force_compact = True
            self._cfg.thread_store.append_event(
                self.thread_id,
                {"kind": "compact", "content": "pre-execution compaction requested"},
            )
            # add a compaction event to the queue
            self._add_queue(
                "compaction",
                {"reason": "pre_act_user", "info": info, "forced": True},
            )
        else:
            # if the user does not want to compact then add a compaction event to the queue
            self._add_queue(
                "compaction",
                {"reason": "pre_act_checkpoint", "info": info, "forced": False, "ask": True},
            )

    async def _build_run_payload(
        self,
        message: str,
        *,
        images: Sequence[str] | None,
        active_skills: Sequence[str] | None,
        mode_switch: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:

        skills = list(active_skills or self._pending_skills)

        payload = {
            "messages": [self._user_message(message, images)],
            "approval_declined": False,
            "agent_mode": self.agent_mode,
            "force_compact": self._consume_force_compact(),
            "activate_skills": skills,
            "mode_switch": mode_switch,
            "current_user_seq": 0,
        }
        cfg = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 75}
        return payload, cfg

    def _dispatch_stream_event(
        self, ev: dict, assistant_text: str
    ) -> list[tuple[SessionEvent, str]]:
        """Map one astream_events chunk to SessionEvent pairs."""
        out: list[tuple[SessionEvent, str]] = []
        ek = ev.get("event")
        name = ev.get("name", "")

        # handle streaming token chunks
        if ek == "on_chat_model_stream":
            chunk = ev.get("data", {}).get("chunk")
            
            if chunk is None:
                return out
            
            data: dict[str, Any] = {}
            ak = getattr(chunk, "additional_kwargs", None) or {}
            
            # get the reasoning content
            rtext = ak.get("reasoning_content") if isinstance(ak, dict) else None
            if isinstance(rtext, str) and rtext:
                data["reasoning"] = rtext
            
            # get the main content of the chunk
            text = getattr(chunk, "content", "")
            if isinstance(text, str) and text:
                assistant_text = assistant_text + text
                data["text"] = text
            
            if data:
                # add the event to the output
                out.append((SessionEvent("assistant_delta", data), assistant_text))
            return out
        
        # handle the end of the agent chain
        if ek == "on_chat_model_end" and name == "agent":
            # add the final event to the output
            out.append(
                (SessionEvent("assistant_final", {"content": assistant_text}), assistant_text)
            )
            return out

        # handle the on_chain_end event -> end of any langgraph node
        # this handles tool_start events or the end of agent node
        if ek == "on_chain_end" and name == "agent":
            for msg in _messages_from_event(ev):
                if getattr(msg, "type", None) not in {"ai", "assistant"}:
                    continue
                for tc in getattr(msg, "tool_calls", None) or []:
                    # add the tool start event to the output
                    out.append(
                        (
                            SessionEvent(
                                "tool_start",
                                {
                                    "name": tc.get("name", "unknown"),
                                    "args": tc.get("args", {}),
                                    "id": tc.get("id"),
                                },
                            ),
                            assistant_text,
                        )
                    )
            return out

        # handle the on_end_chain event that emits tool_end or the end of tools node
        if ek == "on_chain_end" and name == "tools":
            for msg in _messages_from_event(ev):
                if getattr(msg, "type", None) != "tool" and not isinstance(msg, ToolMessage):
                    continue
                if (getattr(msg, "additional_kwargs", None) or {}).get("hidden"):
                    continue
                out.append(
                    (
                        SessionEvent(
                            "tool_end",
                            {
                                "name": getattr(msg, "name", "tool"),
                                "content": str(getattr(msg, "content", "")),
                                "id": getattr(msg, "tool_call_id", None),
                            },
                        ),
                        assistant_text,
                    )
                )
            return out

        return out

    async def _iter_events(
        self,
        message: str,
        *,
        images: Sequence[str] | None = None,
        active_skills: Sequence[str] | None = None,
        mode: str | None = None,
    ) -> AsyncIterator[tuple[SessionEvent, str]]:
        """Yield (event, assistant_text_so_far) pairs from the graph stream."""
        
        # sets up a runtime context for this session
        self._install_session_runtime()
        # reset the last usage
        self._last_usage = None
        # drain the queue and return a list of session events
        self._drain_queue()
        # set the active session context var
        token = _active_session.set(self)

        try:
            # handle mode switch and pre-flight operations before building the payload
            if mode and mode != self.agent_mode:
                self.set_mode(mode)
            mode_switch = ""
            if self._pending_act_checkpoint and self.agent_mode == "act":
                self._pending_act_checkpoint = False
                mode_switch = "plan->act"
                await self._maybe_checkpoint_before_act()

            try:
                # get the payload for the run and the langgraph config
                payload, cfg = await self._build_run_payload(
                    message, images=images, active_skills=active_skills, mode_switch=mode_switch
                )
            except Exception as exc:
                yield SessionEvent("error", {"message": str(exc)}), ""
                return

            for queued in self._drain_queue():
                yield queued, ""

            assistant_text = ""
            try:
                async for ev in self.app.astream_events(payload, config=cfg, version="v2"):
                    # first yield queued events like usage, compact, etc.
                    for queued in self._drain_queue():
                        yield queued, assistant_text
                    # then dispatch the stream events
                    for event, assistant_text in self._dispatch_stream_event(ev, assistant_text):
                        yield event, assistant_text
                # finally yield any remaining queued events
                for queued in self._drain_queue():
                    yield queued, assistant_text
            except Exception as exc:
                yield SessionEvent("error", {"message": str(exc)}), assistant_text
        finally:
            # remove the active session object from memory or contextvar
            _active_session.reset(token)

    async def run(
        self,
        message: str,
        *,
        images: Sequence[str] | None = None,
        active_skills: Sequence[str] | None = None,
        mode: str | None = None,
    ) -> RunResult:
        """Send a message and collect the full response as a :class:`RunResult`.

        This is the batched (non-streaming) entry point.  It drains the
        entire event stream, yields a single ``RunResult`` with the
        assistant text, usage stats, todos, and all intermediate events.

        Args:
            message: The user message text.
            images: Optional list of image URLs to attach.
            active_skills: Skill names to activate this turn.
            mode: Override the session mode for this turn only.
        """
        events: list[SessionEvent] = []
        assistant_text = ""
        
        async for event, assistant_text in self._iter_events(
            message, images=images, active_skills=active_skills, mode=mode
        ):
            events.append(event)
        
        self.turn_count += 1
        await self.refresh_context_snapshot()
        todos = await self.aget_todos()
        return RunResult(
            assistant_message=assistant_text,
            usage=self._last_usage,
            todos=todos,
            events=events,
        )

    async def stream(
        self,
        message: str,
        *,
        images: Sequence[str] | None = None,
        active_skills: Sequence[str] | None = None,
        mode: str | None = None,
    ) -> AsyncIterator[SessionEvent]:
        """Send a message and yield :class:`SessionEvent` objects as they arrive.

        This is the streaming entry point.  Each ``SessionEvent``
        represents a discrete milestone (token delta, tool start/end,
        usage, error, etc.).  The caller is responsible for consuming the
        iterator.

        Args:
            message: The user message text.
            images: Optional list of image URLs to attach.
            active_skills: Skill names to activate this turn.
            mode: Override the session mode for this turn only.
        """
        async for event, _ in self._iter_events(
            message, images=images, active_skills=active_skills, mode=mode
        ):
            yield event
        self.turn_count += 1
        await self.refresh_context_snapshot()
