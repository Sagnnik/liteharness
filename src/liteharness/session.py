from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from liteharness.compaction import (
    apply_force_floor,
    calculate_context_pressure,
    compaction_label,
)
from liteharness.graph.builder import build_graph
from liteharness.graph.helpers import _effective_conversation
from liteharness.session_context import SessionContext, set_session_context
from liteharness.tracing.semconv import (
    AGENT_MODE,
    COST_USD,
    INPUT_TOKENS,
    OUTPUT_TOKENS,
    THREAD_ID,
    TURN,
    TURN_COUNT,
    GEN_AI_SYSTEM_VALUE,
    GEN_AI_SYSTEM,
    GEN_AI_OPERATION_NAME,
)
from liteharness.types import (
    ApprovalHandler,
    InterruptHandler,
    PlanTurnHandler,
    RunResult,
    SessionEvent,
    UsageEvent,
)

_active_session: ContextVar["Session | None"] = ContextVar(
    "liteharness_active_session", default=None
)

PLAN_COMPACTION_CHECKPOINT_RATIO = 0.75

# [Image #N] placeholders that the TUI inserts when the user pastes an
# image into the input buffer. They are stripped from both the model-facing
# text and the persisted transcript text before the turn payload is built.
_IMAGE_PLACEHOLDER_RE = re.compile(r"\[Image #\d+\]")


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


def _extract_text_from_blocks(content: Any) -> str:
    """Join all ``text`` blocks from a list-content message into a string.

    Used by ``_strip_prior_image_blocks`` to rewrite list-content (text +
    image_url) HumanMessages back to text-only once the attached images are
    no longer needed for the running turn.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p)
    return str(content)

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

    # Internal usage channel: the agent node reports per-call token/cost
    # usage here; the bridge stores ``_last_usage`` on the active session
    # (feeding ``RunResult.usage`` / tracing spans) and queues a ``usage``
    # SessionEvent for the caller. Not a user-facing hook — the same data
    # already reaches consumers via the SessionEvent stream, the durable
    # ``usage`` log, and the cost tracker.
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

    # store the usage bridge so the agent node can use this
    cfg._usage_bridge = _usage
    # mark as installed to avoid re-wrappings
    cfg._event_bridges_installed = True


class Session:
    def __init__(
        self,
        agent,
        *,
        thread_id,
        agent_mode="act",
        metadata=None,
        git_available=None,
        vision: bool | None = None,
        on_plan_turn: PlanTurnHandler | None = None,
        on_interrupt: InterruptHandler | None = None,
    ):
        """Create a new interaction session bound to a single thread.

        Args:
            agent: A :class:`NessAgent` instance whose config drives tools,
                   models, prompts, and permissions.
            thread_id: Unique identifier for this conversation thread.
            agent_mode: Initial mode (``"act"`` or ``"plan"``).
            metadata: Arbitrary key-value pairs surfaced in the system prompt.
            git_available: Whether the project has a git repo.
            vision: Whether image attachments should be forwarded to the model.

                * ``None`` — caller-built :class:`HumanMessage` content is
                  forwarded verbatim (today's default; the SDK is shape-blind).
                * ``True`` — image blocks are sent to the model.
                * ``False`` — image blocks are dropped to text-only and a
                  ``warning`` SessionEvent is emitted so the caller can surface
                  it.

                The model-name heuristic that decides this belongs to the
                adapter; the SDK just honours the flag.
            on_plan_turn: Per-Session hook called at the end of a successful
                plan-mode turn with the assistant text. Fire-and-forget; used by
                the adapter to autosave the plan file. When unset, a
                ``plan_turn`` SessionEvent is emitted instead so the caller
                can still observe the text. Success path only — interrupted
                plan turns flow through ``on_interrupt`` / the ``interrupted``
                SessionEvent instead, so there is exactly one interrupt path.
            on_interrupt: Per-Session hook called with the captured partial
                assistant text on interruption; returns the text to surface on
                the ``interrupted`` SessionEvent (returning ``None``/falsy
                keeps the original partial text). When unset, the SDK still
                synthesises the interruption marker itself.
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
        self._event_queue: asyncio.Queue[SessionEvent] = asyncio.Queue()
        self._last_usage: UsageEvent | None = None

        self.checkpointer = (
            self._cfg.checkpoint_factory() if self._cfg.checkpoint_factory else MemorySaver()
        )
        self._skill_loader = self._cfg.skill_loader

        _ensure_config_event_bridges(self._cfg)

        self._app = self._build_graph()

        # Per-Session runtime hooks.
        # Stored on the Session so concurrent threads on the same NessAgent do
        # not clobber each other via the shared NessAgentConfig.
        self.on_plan_turn = on_plan_turn
        self.on_interrupt = on_interrupt

        # Vision gate
        # explicit bool decides whether image blocks reach the model.
        self._vision = vision

        # Bootstrap messages seeded by bootstrap() and consumed once on the next turn's payload.
        self._pending_bootstrap: list[Any] = []

        # Cancellation flag for the active turn. cancel() sets it 
        # _iter_events polls is_cancelled() between yields and then
        # finalises partial state via _finalize_cancelled_turn. 
        # Reset at the top of each run so a stale trigger cannot bleed into the next turn.
        self._cancel_token: asyncio.Event = asyncio.Event()

    def _add_queue(self, kind: str, data: dict[str, Any] | None = None) -> None:
        # add the event to the queue; non-blocking; of type SessionEvent
        self._event_queue.put_nowait(SessionEvent(kind, dict(data or {})))

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

    def reset_checkpointer(self) -> None:
        """Drop all checkpointed graph state and recompile.

        Required before a :meth:`bootstrap` replay (resume / rollback): the
        default ``MemorySaver`` keeps prior turns for this thread in memory,
        so replaying events into a reused saver would resurrect truncated
        messages and duplicate the replayed prefix (``add_messages`` appends
        by fresh id). Swapping in a fresh saver makes the durable event log
        the single source of truth for the rebuilt state.

        Caveat: with a custom ``checkpoint_factory`` backed by a persistent
        store, a new saver instance still points at the same backing store;
        the factory should scope savers per session (or clear the thread
        server-side) for replay-style flows.
        """
        self.checkpointer = (
            self._cfg.checkpoint_factory() if self._cfg.checkpoint_factory else MemorySaver()
        )
        self._app = self._build_graph()

    def bootstrap(self, messages: Sequence[Any]) -> None:
        """Seed the next turn's payload with prior messages.

        Consumed exactly once — the bootstrap list is prepended to the next
        turn's ``messages`` payload alongside the new user message, then
        cleared. This is the safe resume/rollback primitive: it mirrors the
        proven payload-seed path (CLI ``SessionApp._bootstrap``) without
        bypassing the graph entry via direct ``aupdate_state`` writes on a
        fresh checkpointer that has no prior checkpoint.
        """
        self._pending_bootstrap = list(messages)

    def cancel(self) -> None:
        """Request a cooperative break-out of the active turn's stream loop.

        ``_iter_events`` polls ``is_cancelled()`` between yields and, on a set
        token, performs partial-state cleanup via
        :meth:`_finalize_cancelled_turn` before returning normally. The TUI's
        hard-escalation backstop (``asyncio.Task.cancel``) lands as
        ``CancelledError`` and is handled by the same finaliser, shielded.
        """
        self._cancel_token.set()

    def is_cancelled(self) -> bool:
        """Whether :meth:`cancel` was requested for the active turn."""
        return self._cancel_token.is_set()

    def is_subagent_active(self) -> bool:
        """Whether a child subagent run is currently in flight.

        Polled lazily from :mod:`liteharness.tools.subagents` so the SDK stays
        decoupled from the tools package at import time. Returns ``False`` if
        the signal is unavailable, preserving today's no-suppression behavior
        on pure-SDK usage without subagents.
        """
        try:
            from liteharness.tools.subagents import subagent_runs_active
        except ImportError:
            return False
        try:
            return subagent_runs_active() > 0
        except Exception:
            return False

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
            tracing=self._cfg.tracing,
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
    ) -> tuple[HumanMessage, str]:
        """Build a HumanMessage and return ``(message, cleaned_text)``.

        ``[Image #N]`` placeholders (TUI-inserted image markers) are stripped
        from both the model text and the returned text so callers can persist
        the clean transcript. When ``self._vision is False`` and images were
        supplied, the blocks are dropped to text-only and a ``warning``
        SessionEvent is queued for the caller. When ``None`` (default), the
        caller-built content shape is forwarded verbatim — the SDK is
        shape-blind and trusts the adapter's gating decision.
        """
        cleaned = _IMAGE_PLACEHOLDER_RE.sub("", message or "").strip()
        if not images:
            return HumanMessage(content=cleaned), cleaned
        if self._vision is False:
            self._add_queue(
                "warning",
                {"message": "Session vision is disabled; sending text only."},
            )
            return (
                HumanMessage(content=cleaned or "[image omitted — model is text-only]"),
                cleaned,
            )
        return (
            HumanMessage(
                content=[{"type": "text", "text": cleaned or "Please inspect this image."}]
                + [{"type": "image_url", "image_url": {"url": u}} for u in images]
            ),
            cleaned,
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
            self._add_queue(
                "compaction",
                {
                    "reason": "pre_act_hard_threshold",
                    "info": info,
                    "forced": True,
                },
            )
            return
        # Soft checkpoint: no interactive ask — emit a passive notice so the
        # caller can surface "context is filling up" and the user can run
        # /compact before execution if they want.
        self._add_queue(
            "compaction",
            {"reason": "pre_act_checkpoint", "info": info, "forced": False, "ask": True},
        )

    async def _build_run_payload(
        self,
        user_message: HumanMessage,
        *,
        active_skills: Sequence[str] | None,
        mode_switch: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build the turn payload and run config.

        Takes a *pre-built* ``user_message`` (constructed by
        :meth:`_user_message` and stripped of image placeholders by the caller
        in :meth:`_iter_events`). Any pending bootstrap messages are prepended
        to the payload's ``messages`` and consumed once here.
        """
        skills = list(active_skills if active_skills is not None else self._pending_skills)
        if active_skills is None:
            self._pending_skills = []
        initial = list(self._pending_bootstrap)
        if initial:
            self._pending_bootstrap = []
        payload = {
            "messages": [*initial, user_message],
            "approval_declined": False,
            "agent_mode": self.agent_mode,
            "force_compact": self._consume_force_compact(),
            "activate_skills": skills,
            "mode_switch": mode_switch,
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

        # handle the on_chain_end event for the agent node -> emits the
        # authoritative assistant output (final text + tool calls). The agent
        # node runs the model via a non-streaming ainvoke and returns the
        # AIMessage; astream_events surfaces it here as the node's output
        # messages. 
        # on_chain_end (name "agent") carries only the agent's response message
        if ek == "on_chain_end" and name == "agent":
            for msg in _messages_from_event(ev):
                if getattr(msg, "type", None) not in {"ai", "assistant"}:
                    continue
                text = str(getattr(msg, "content", "") or "")
                if text.strip():
                    assistant_text = text
                    out.append(
                        (SessionEvent("assistant_final", {"content": text}), assistant_text)
                    )
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

    async def _strip_prior_image_blocks(self, cfg: dict) -> None:
        """Rewrite prior list-content HumanMessages to text-only (by id).

        Walks the checkpointer state; for each ``HumanMessage`` whose
        ``.content`` is a list (i.e. carries image_url blocks) AND that is
        followed by an ``AIMessage`` (i.e. the turn was answered), replaces it
        with a text-only :class:`HumanMessage` carrying the same id so the
        ``add_messages`` reducer swaps it in-place. The trailing image message
        from a resumed crashed turn (no following AIMessage) is left intact so
        the model can still see the image.

        Called once per turn at the top of :meth:`_iter_events`, before the
        payload is built — so large base64 payloads are not re-sent on every
        turn after the image was first answered.
        """
        try:
            snapshot = await self.app.aget_state(cfg)
        except Exception:
            return
        messages = (snapshot.values or {}).get("messages") or []
        if not messages:
            return
        answered_image_ids: list[tuple[str, str]] = []
        for i, msg in enumerate(messages):
            if (
                getattr(msg, "type", None) == "human"
                and isinstance(getattr(msg, "content", None), list)
            ):
                followed_by_ai = any(
                    getattr(messages[j], "type", None) in ("ai", "assistant")
                    for j in range(i + 1, len(messages))
                )
                if followed_by_ai and msg.id:
                    text_block = _extract_text_from_blocks(msg.content)
                    answered_image_ids.append((msg.id, text_block))
        if not answered_image_ids:
            return
        replacements = [
            HumanMessage(content=text, id=mid) for mid, text in answered_image_ids
        ]
        try:
            await self.app.aupdate_state(cfg, {"messages": replacements})
        except Exception:
            # The image blocks only cost extra tokens if
            # the swap silently fails — the turn still proceeds.
            pass

    async def _finalize_cancelled_turn(self, assistant_text: str, cfg: dict) -> None:
        """Flush partial state after a cooperative or hard cancel.

        Pure graph-mutation (no ``render``): synthesises a *failed*
        ``ToolMessage`` for every pending tool call so the checkpoint stays
        consistent, and when neither partial text nor pending tool calls exist,
        injects an ``AIMessage`` interruption marker so the model does not
        silently resume the abandoned request next turn. Emits an
        ``interrupted`` SessionEvent so the caller can surface it.

        The marker is an ``AIMessage`` (not ``HumanMessage``) to preserve
        strict user/assistant alternation — the last checkpoint message may be
        a ``HumanMessage`` (empty-stream turn) or a ``ToolMessage`` (just
        completed tools), and an ``AIMessage`` cap is valid in both cases,
        while a second ``HumanMessage`` risks back-to-back humans that some
        providers reject.
        """
        recorded_text = bool(assistant_text and assistant_text.strip())

        synthetic: list[Any] = []
        has_pending = False
        try:
            snapshot = await self.app.aget_state(cfg)
        except Exception:
            snapshot = None
        if snapshot is not None:
            messages = list((snapshot.values or {}).get("messages", []))
            # find and get the answered tool call ids
            answered_ids = {
                getattr(m, "tool_call_id", None)
                for m in messages
                if isinstance(m, ToolMessage)
            }
            # Find the last AIMessage and its tool calls
            for msg in reversed(messages):
                if not isinstance(msg, AIMessage):
                    continue
                # iterate over the tool calls
                for tc in (msg.tool_calls or []):
                    call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    if not call_id or str(call_id) in answered_ids:
                        continue
                    has_pending = True
                    tool_name = (
                        tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "tool")
                    )
                    # add synthetic tool message
                    synthetic.append(
                        ToolMessage(
                            tool_call_id=str(call_id),
                            name=str(tool_name or "tool"),
                            content="Tool execution interrupted",
                        )
                    )
                break

        if not recorded_text and not has_pending:
            # add interruption marker if there is no assistant text and no pending tool calls
            # Cancel during a pure LLM call before any tokens streamed.
            # Cancel after tools finished but before the model answered again 
            # (last message might be ToolMessage, not partial AI text in assistant_text).
            synthetic.append(
                AIMessage(content=self._cfg.options.interruption_marker)
            )

        if synthetic:
            # update the state with the synthetic messages
            try:
                await self.app.aupdate_state(cfg, {"messages": synthetic})
            except Exception:
                # the checkpoint may be left dirty but the next turn will still proceed.
                pass

        interrupted_surface = assistant_text
        if self.on_interrupt is not None:
            try:
                interrupted_surface = self.on_interrupt(assistant_text) or assistant_text
            except Exception:
                pass

        self._add_queue("interrupted", {"partial_text": interrupted_surface})
        # NOTE: interrupted plan turns are NOT routed through ``on_plan_turn``
        # here — that hook is the success-path contract (see ``_iter_events``).
        # The partial text reaches the caller exactly once, via the
        # ``on_interrupt`` hook above and the ``interrupted`` SessionEvent, so
        # an adapter that archives plan text has a single place to do it.

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
        # reset the cooperative cancel token — a stale trigger from a prior
        # turn must not abort this one.
        self._cancel_token.clear()
        # drain the queue and return a list of session events
        self._drain_queue()
        # set the active session context var
        token = _active_session.set(self)

        tracer = self._cfg.tracer
        with tracer.start_span(
            TURN,
            attributes={
                THREAD_ID: self.thread_id,
                AGENT_MODE: self.agent_mode,
                GEN_AI_SYSTEM: GEN_AI_SYSTEM_VALUE,
                GEN_AI_OPERATION_NAME: "agent",
            },
        ) as span:
            try:
                # Mode override is documented as "this turn only", so snapshot
                # the prior mode and restore it in the ``finally`` below via a
                # direct assignment (NOT ``set_mode`` — that would schedule a
                # spurious plan->act compaction checkpoint on the next turn).
                prior_mode = self.agent_mode
                mode_overridden = bool(mode and mode != self.agent_mode)
                if mode_overridden:
                    self.set_mode(mode)
                mode_switch = ""
                if self._pending_act_checkpoint and self.agent_mode == "act":
                    self._pending_act_checkpoint = False
                    mode_switch = "plan->act"
                    await self._maybe_checkpoint_before_act()

                # build the user message (vision gate + image-strip).
                try:
                    user_message, _cleaned = self._user_message(message, images)
                except Exception as exc:
                    span.set_status("ERROR", str(exc))
                    yield SessionEvent("error", {"message": str(exc)}), ""
                    return

                # Strip answered image blocks from prior turns so the large
                # base64 payloads aren't re-sent. The new turn's user_message
                # (carrying this turn's images, unstripped) is built above and
                # not touched here.
                cfg = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 75}
                await self._strip_prior_image_blocks(cfg)

                try:
                    payload, cfg_payload = await self._build_run_payload(
                        user_message,
                        active_skills=active_skills,
                        mode_switch=mode_switch,
                    )
                    cfg = cfg_payload
                except Exception as exc:
                    span.set_status("ERROR", str(exc))
                    yield SessionEvent("error", {"message": str(exc)}), ""
                    return

                for queued in self._drain_queue():
                    yield queued, ""

                assistant_text = ""
                cancelled = False
                try:
                    async for ev in self.app.astream_events(
                        payload, config=cfg, version="v2"
                    ):
                        # subagent suppression: when child runs are pending,
                        # drop all events so the caller's spinner isn't fed
                        # spurious assistant/tool stream from the child branch.
                        if self.is_subagent_active():
                            continue
                        # first yield queued events like usage, compact, etc.
                        for queued in self._drain_queue():
                            yield queued, assistant_text
                        # then dispatch the stream events
                        for event, assistant_text in self._dispatch_stream_event(
                            ev, assistant_text
                        ):
                            yield event, assistant_text
                        # cooperative cancel: break out between events so the
                        # post-loop cleanup can flush partial state cleanly.
                        if self._cancel_token.is_set():
                            cancelled = True
                            break
                    # also catch a cancel that arrived during the last
                    # event's downstream processing (between the final yield
                    # and the loop's natural exit): without this, a late
                    # cancel lands silently instead of finalizing.
                    if not cancelled and self._cancel_token.is_set():
                        cancelled = True
                    # finally yield any remaining queued events
                    for queued in self._drain_queue():
                        yield queued, assistant_text
                except asyncio.CancelledError:
                    # Hard-escalation path: the cooperative cancel token failed
                    # to break the stream loop within the backstop window.
                    # Best-effort finalisation before re-raising so the task is
                    # still properly cancelled.
                    try:
                        await asyncio.shield(
                            self._finalize_cancelled_turn(assistant_text, cfg)
                        )
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass
                    final_events: list[SessionEvent] = []
                    try:
                        final_events = self._drain_queue()
                    except Exception:
                        final_events = []
                    for queued in final_events:
                        yield queued, assistant_text
                    raise
                except Exception as exc:
                    span.set_status("ERROR", str(exc))
                    yield SessionEvent("error", {"message": str(exc)}), assistant_text
                    return

                if cancelled:
                    await self._finalize_cancelled_turn(assistant_text, cfg)
                    # drain any events queued during finalize (interrupted,
                    # and any warnings) and yield them after the model stream
                    # has ended.
                    for queued in self._drain_queue():
                        yield queued, assistant_text
                elif self.agent_mode == "plan":
                    # Plan-turn emission: when an adapter hook is installed, it
                    # takes the text directly; otherwise emit a ``plan_turn``
                    # SessionEvent so the caller can still observe the text.
                    if assistant_text.strip():
                        if self.on_plan_turn is not None:
                            try:
                                self.on_plan_turn(assistant_text)
                            except Exception:
                                pass
                        else:
                            yield SessionEvent("plan_turn", {"text": assistant_text}), assistant_text
            finally:
                # Restore the session mode when a one-turn override was applied
                # (see ``mode`` kwarg docstring). Direct assignment avoids
                # ``set_mode``'s plan->act checkpoint side effect.
                if mode_overridden and self.agent_mode != prior_mode:
                    self.agent_mode = prior_mode
                span.set_attribute(TURN_COUNT, self.turn_count)
                last_usage = self._last_usage
                if last_usage is not None:
                    span.set_attribute(INPUT_TOKENS, last_usage.input_tokens)
                    span.set_attribute(OUTPUT_TOKENS, last_usage.output_tokens)
                    span.set_attribute(COST_USD, last_usage.cost_usd or 0)
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
