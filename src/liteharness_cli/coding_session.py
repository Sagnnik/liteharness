from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from liteharness.types import SessionEvent

from liteharness_cli.events import (
    events_to_messages,
    plan_autosave_text,
    restore_cost_from_events,
)
from liteharness_cli.mentions import expand_documents
from liteharness_cli.rollback import (
    create_file_checkpoint,
    restore_paths,
)


# Per-turn suffix appended to partial plan text on an interrupt so the
# autosaved plan file reads naturally.
_INTERRUPTED_SUFFIX = " … [interrupted]"

# ``[Image #N]`` placeholders the TUI's input buffer inserts on image paste.
# Stripped from the user text BEFORE @mention expansion and persistence so
# the durable transcript stays clean and file contents that happen to contain
# the marker survive expansion untouched.
_IMAGE_PLACEHOLDER_RE = re.compile(r"\[Image #\d+\]")


class CodingSession:
    """Coding adapter over :class:`liteharness.Session`.

    Construct via :class:`NessAgent.session` plus this wrapper, or directly:

        from liteharness import NessAgent
        from liteharness_cli import CodingSession

        agent = NessAgent(model=..., prompt=...)
        coding = CodingSession(agent, thread_id="proj-1")
        async for ev in coding.run_turn("add a rate limiter"):
            ...

    The per-Session runtime hooks (``on_plan_turn``, ``on_interrupt``) are
    installed on the underlying :class:`Session` instance rather than the
    shared :class:`~liteharness.NessAgentConfig` so concurrent threads on one
    :class:`NessAgent` never clobber each other's hooks. Rollback mutation
    tracking is adapter-owned: :meth:`run_turn` replays the durable tool log
    after each turn (see :meth:`_record_turn_mutations`).
    """

    def __init__(
        self,
        agent,
        *,
        thread_id: str,
        mode: str = "act",
        vision: bool | None = None,
        git_available: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.agent = agent
        self.cfg = agent.config
        self.thread_id = thread_id
        self._vision = vision

        self.ness_dir = Path(self.cfg.options.ness_dir or Path.cwd() / ".ness")
        self.project_root = Path(self.cfg.options.project_root or Path.cwd())

        self.thread_store = self.cfg.thread_store
        self.cost_tracker = self.cfg.cost_tracker
        self.permission_store = self.cfg.permission_store

        # (/memory, /user, /init, /hooks, /skill, /mcp).
        self.memory_store = self.cfg.memory_store
        self.hook_runner = self.cfg.hook_runner
        self.skill_loader = self.cfg.skill_loader
        self.tool_registry = self.cfg.tool_registry

        # Build the underlying SDK Session and install the per-Session hooks (on_plan_turn and on_interrupt).
        self._session = agent.session(
            thread_id=thread_id,
            mode=mode,
            metadata=metadata,
            git_available=git_available,
            vision=vision,
            on_plan_turn=self._on_plan_turn,
            on_interrupt=self._on_interrupt,
        )

        # Per-turn plan text accumulator
        self._plan_turn_texts: list[str] = []
        # This tracker is process-wide for the coding agent. Durable usage is
        # replayed once per thread, never on every A→B→A switch.
        self._restored_cost_threads: set[str] = {thread_id}

    # ----------------------------------------------------------------------
    # Properties delegating to the underlying SDK Session
    # ----------------------------------------------------------------------

    @property
    def session(self):
        """The underlying domain-agnostic :class:`~liteharness.Session`."""
        return self._session

    @property
    def app(self):
        """The compiled langgraph application for this session."""
        return self._session.app

    @property
    def mode(self) -> str:
        """The current agent mode."""
        return self._session.mode

    @property
    def turn_count(self) -> int:
        """The number of turns run in this session."""
        return self._session.turn_count

    @property
    def context_used(self) -> int:
        """The number of tokens used in this session."""
        return self._session.context_used

    @property
    def context_total(self) -> int:
        """The total number of tokens available in this session."""
        return self._session.context_total

    @property
    def last_usage(self) -> dict[str, Any] | None:
        """The last usage of the active model."""
        usage = self._session._last_usage
        if usage is None:
            return None
        return {
            "model": usage.model,
            "input_tokens": usage.input_tokens,
            "uncached_input_tokens": usage.uncached_input_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": usage.cost_usd,
            "calls": usage.calls,
        }

    @property
    def turn_usage_total(self) -> dict[str, Any] | None:
        """Aggregated token/cost usage for the most recent turn."""
        from liteharness.types import aggregate_usage

        usage = aggregate_usage(self._session._turn_usages)
        if usage is None:
            return None
        return {
            "model": usage.model,
            "input_tokens": usage.input_tokens,
            "uncached_input_tokens": usage.uncached_input_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": usage.cost_usd,
            "calls": usage.calls,
        }

    # ----------------------------------------------------------------------
    # Delegating SDK surface
    # ----------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        self._session.set_mode(mode)

    def toggle_mode(self) -> str:
        return self._session.toggle_mode()

    def active_skills(self, names: list[str]) -> None:
        self._session.active_skills(names)

    def stage_skills(self, names: list[str] | Sequence[str]) -> None:
        self._session.stage_skills(names)

    def request_compact(self) -> None:
        """Request compaction and durable-log it (the CLI's /compact row).

        The SDK only sets the force flag and emits no event, so the adapter —
        the single owner of the durable ``compact`` log — writes the row.
        """
        self.thread_store.append_event(
            self.thread_id, {"kind": "compact", "content": "manual compaction requested"}
        )
        self._session.request_compact()

    def rebuild_graph(self) -> None:
        self._session.rebuild_graph()

    def reload_model(self) -> None:
        """Rebind the chat models after a /config model or reasoning switch.

        ``set_active_model`` / ``set_active_reasoning_effort`` update the
        shared overrides in :mod:`liteharness_cli.chat_model`; the graph
        nodes capture ``cfg.model`` at build time, so the models are
        recreated from the factory and the graph recompiled.
        """
        from liteharness_cli.chat_model import (
            active_model_name,
            create_compaction_model,
            create_model,
            create_reflection_model,
        )
        from liteharness_cli.config import context_window_for, settings
        from liteharness_cli.model_catalog import model_record

        self.cfg.model = create_model(self.thread_id)
        self.cfg.compaction_model = create_compaction_model(self.thread_id)
        self.cfg.reflection_model = create_reflection_model(self.thread_id)
        self._vision = settings.supports_vision
        self._session._vision = self._vision
        window = context_window_for(active_model_name())
        self.cfg.options.context_window = window
        record = model_record(active_model_name())
        if (
            record is not None
            and record.input_price is not None
            and record.output_price is not None
        ):
            self.cost_tracker.pricing[record.id] = (
                record.input_price,
                record.output_price,
                record.cache_read_ratio,
            )
        self._session.rebuild_graph()

    def save_thread(self) -> str:
        """Archive the current thread (the CLI's /save)."""
        return self.thread_store.archive_thread(self.thread_id)

    def cancel(self) -> None:
        self._session.cancel()

    def is_cancelled(self) -> bool:
        return self._session.is_cancelled()

    async def finalize_reflection(self) -> None:
        await self._session.finalize_reflection()

    async def refresh_context_snapshot(self) -> dict[str, Any]:
        return await self._session.refresh_context_snapshot()

    async def get_todos(self) -> list[dict[str, Any]]:
        return await self._session.get_todos()

    async def get_state(self) -> dict[str, Any]:
        return await self._session.get_state()

    async def get_messages(self) -> list[Any]:
        return await self._session.get_messages()

    async def preview_context(self, *, mode: str | None = None):
        """Preview L0–L2 system message + prospective L3 overlay."""
        return await self._session.preview_context(mode=mode)

    def is_subagent_active(self) -> bool:
        """Whether a child subagent run is currently in flight.

        Adapter-owned: used by :meth:`run_turn` to suppress child-branch
        stream noise so the TUI spinner is not fed spurious assistant/tool
        events. Pure :class:`~liteharness.Session` consumers do not filter.
        """
        try:
            from liteharness.tools.subagents import subagent_runs_active
        except ImportError:
            return False
        try:
            return subagent_runs_active() > 0
        except Exception:
            return False

    # ----------------------------------------------------------------------
    # The turn loop
    # ----------------------------------------------------------------------

    async def run_turn(
        self,
        user_text: str,
        *,
        images: list[str] | None = None,
        active_skills: list[str] | None = None,
        mode: str | None = None,
    ) -> AsyncIterator[SessionEvent]:
        """Run one user turn, yielding :class:`SessionEvent` objects.

        Before delegating to :meth:`Session.stream`:

        1. Strip ``[Image #N]`` placeholders from the user text (the TUI's
           paste markers), so the durable transcript stays clean and the
           expansion below cannot eat marker-shaped text inside file bodies.
        2. Expand ``@file`` mentions against current disk. The cleaned,
           still-``@tagged`` text is what gets persisted to the events table;
           the expansion is what the model sees.
        3. Snapshot files (``create_file_checkpoint``) + the per-thread
           session-memory file (``memory_store.read_session_raw``) BEFORE the agent acts.
        4. Persist the user event (``append_event``), then key the rollback
           checkpoint row by that seq (``save_checkpoint``).

        During the stream, ``compaction`` events become durable
        ``append_event({kind: compact})`` rows (caveat-1 relocation). Plan
        autosave is driven by the per-Session hooks (``_on_plan_turn`` /
        ``_on_interrupt``), not by re-consuming events here — each plan text
        is captured exactly once. After the stream, the turn's mutated paths
        are recorded on the checkpoint row from the durable tool log
        (``_record_turn_mutations``).
        """
        self._plan_turn_texts = []
        cleaned = _IMAGE_PLACEHOLDER_RE.sub("", user_text or "").strip()
        expanded = expand_documents(cleaned, self.permission_store)

        # Per-turn rollback checkpoint: snapshot files + the per-thread
        # session-memory file BEFORE the agent acts. ``create_file_checkpoint``
        # is a no-op (returns None) when not in a git repo, so this is safe to
        # run unconditionally.
        git_hash = await asyncio.to_thread(create_file_checkpoint, self.project_root)
        mem_snapshot = await asyncio.to_thread(
            self.memory_store.read_session_raw, self.thread_id
        )

        # Persist the user event and key the checkpoint by its seq. The
        # persisted text is the placeholder-stripped (but still @tagged)
        user_event: dict[str, Any] = {"kind": "user", "content": cleaned}
        if images:
            user_event["images"] = list(images)
        user_seq = self.thread_store.append_event(self.thread_id, user_event)
        if user_seq is None:
            user_seq = 0
        else:
            self.thread_store.save_checkpoint(
                self.thread_id, user_seq, git_hash, mem_snapshot
            )

        try:
            async for ev in self._session.stream(
                expanded,
                images=images,
                active_skills=active_skills,
                mode=mode,
            ):
                # Suppress child-subagent stream noise (spinner hygiene). The
                # underlying Session still runs the graph; we just don't
                # forward those events to the TUI.
                if self.is_subagent_active():
                    continue
                # Adapter-owned side effects run BEFORE yielding so a caller
                # that breaks/errors after receiving the event can't skip
                # durable persistence. The event itself is unaffected. Plan
                # text is NOT captured here — the per-Session hooks own that
                # (success: ``_on_plan_turn``; interrupt: ``_on_interrupt``),
                # so each turn's plan text lands exactly once. ``warning`` /
                # ``interrupted`` / ``plan_turn`` events are pure pass-through
                # for the TUI to render.
                if ev.kind == "compaction":
                    self._log_compaction_event(ev)
                yield ev
        finally:
            # End-of-turn plan autosave from the hook-fed accumulator. Runs
            # even when a caller breaks the stream.
            self._autosave_plan_turn()
            # Rollback bookkeeping: key the turn's mutated paths onto its
            # checkpoint row, replayed from the durable tool log. Best-effort
            # — a miss degrades rollback to a full-tree restore, never worse.
            try:
                self._record_turn_mutations(user_seq)
            except Exception:
                pass
            # Snapshot context for the TUI header.
            try:
                await self.refresh_context_snapshot()
            except Exception:
                pass

    # ----------------------------------------------------------------------
    # Per-Session runtime hooks (installed on the underlying Session)
    # ----------------------------------------------------------------------

    def _on_plan_turn(self, text: str) -> None:
        """Per-Session plan-turn hook (success path): accumulate for autosave."""
        self._handle_plan_turn_text(text)

    def _on_interrupt(self, partial_text: str) -> str:
        """Per-Session interrupt hook: the single owner of interrupted-plan
        capture.

        On a plan-mode interrupt the partial text (plus the convention
        suffix) lands in the plan-text accumulator exactly once — the SDK no
        longer routes interrupts through ``on_plan_turn``, and ``run_turn``
        does not re-consume the ``interrupted`` event for plan text. Returns
        the partial text to surface on the ``interrupted`` SessionEvent.

        The mode check reads the underlying Session's LIVE mode so a one-turn
        ``mode=`` override classifies correctly — the SDK restores its mode
        attribute after an override turn, so the Session is always the source
        of truth for "which mode is this turn running in".
        """
        if self._session.mode == "plan" and partial_text.strip():
            self._handle_plan_turn_text(partial_text + _INTERRUPTED_SUFFIX)
        return partial_text

    def _record_turn_mutations(self, user_seq: int) -> None:
        """Key the turn's mutated paths onto its rollback checkpoint row.

        The SDK durably logs every tool call (``kind == "tool"``) with its
        args and an ``exit`` classification, so mutation tracking needs no
        SDK hook: this replays the log entries appended after the user event
        at ``user_seq`` (events are seq-ordered and contiguous per thread)
        and set-unions the mutated paths onto the checkpoint row written
        before the turn ran. Skips mirror the graph's own gates — plan-mode
        gating (``mode_gated``), permission/hook denies (``denied``), and
        tool exceptions (``Error:`` prefix) never touched the filesystem.
        Failed *shell commands* (``status=error``) ARE recorded: a
        half-failed command may still have mutated the tree. Approval-
        declined calls persist no ``tool`` row at all, so they are skipped
        for free. Idempotent (``add_modified_path`` set-unions and the ``*``
        sentinel short-circuits), so re-draining is harmless.
        """
        events = self.thread_store.load_thread_events(self.thread_id)
        for ev in events[user_seq + 1:]:
            if ev.get("kind") != "tool":
                continue
            if ev.get("exit") in ("denied", "mode_gated"):
                continue
            if str(ev.get("result") or "").startswith("Error:"):
                continue
            for p in _extract_mutated_paths(ev.get("tool"), ev.get("args") or {}):
                self.thread_store.add_modified_path(self.thread_id, user_seq, p)

    # ----------------------------------------------------------------------
    # Plan autosave
    # ----------------------------------------------------------------------

    def _handle_plan_turn_text(self, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        self._plan_turn_texts.append(cleaned)

    def _autosave_plan_turn(self) -> None:
        plan_text = plan_autosave_text(self._plan_turn_texts)
        if plan_text is not None:
            self._save_plan(plan_text)

    def _save_plan(self, text: str) -> Path:
        modes = self.cfg.modes
        if modes and modes.plans_dir is not None:
            plans_dir = Path(modes.plans_dir)
        else:
            plans_dir = self.ness_dir / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        stamp = re.sub(r"[-:.TZ]", "", datetime.now(timezone.utc).isoformat(timespec="seconds")).replace("+0000", "")
        path = plans_dir / f"{stamp}-{self.thread_id}.md"
        path.write_text(text.strip() + "\n", encoding="utf-8")
        return path

    # ----------------------------------------------------------------------
    # Durable compaction log (caveat-1 relocation)
    # ----------------------------------------------------------------------

    def _log_compaction_event(self, ev: SessionEvent) -> None:
        """Write a durable ``compact`` event row from the SDK's SessionEvent.

        The SDK no longer touches ``thread_store`` for compaction (caveat-1);
        it only emits a ``compaction`` SessionEvent. The adapter is the single
        owner of the durable log.
        """
        reason = ev.data.get("reason") or "unknown"
        info = ev.data.get("info") or ""
        forced = bool(ev.data.get("forced"))
        content = f"compaction ({reason}){' [forced]' if forced else ''}: {info}".strip()
        self.thread_store.append_event(
            self.thread_id, {"kind": "compact", "content": content}
        )

    # ----------------------------------------------------------------------
    # Resume / reset / rollback
    # ----------------------------------------------------------------------

    async def resume(
        self,
        thread_id: str,
        *,
        replay_cost: bool = True,
        allow_empty: bool = False,
    ) -> bool:
        """Rebuild the live graph from persisted events for ``thread_id``.

        Mirrors the CLI's ``resume_thread`` semantics:

        * Switching to a *different* thread finalizes the current thread's
          reflection and archives it first. If the target has no persisted
          events, nothing is touched and ``False`` is returned (the TUI shows
          "No saved thread").
        * Resuming the *same* thread (the ``rollback_to`` path) proceeds even
          with zero surviving events — that is the rollback-to-zero case.

        The checkpointer is reset via :meth:`Session.reset_checkpointer`
        before the :meth:`Session.bootstrap` replay so the durable events are
        the single source of truth — reusing the old saver would resurrect
        truncated messages and duplicate the replayed prefix.
        """
        events = self.thread_store.load_thread_events(thread_id)
        if not events and thread_id != self.thread_id and not allow_empty:
            return False
        if thread_id != self.thread_id:
            await self.finalize_reflection()
            self.thread_store.archive_thread(self.thread_id)

        subagents = self.thread_store.list_subagents(thread_id)
        messages = events_to_messages(
            events,
            subagents,
            vision=self._vision,
            permission_store=self.permission_store,
        )
        if replay_cost and thread_id not in self._restored_cost_threads:
            restore_cost_from_events(events, self.cost_tracker)
            self._restored_cost_threads.add(thread_id)

        self.thread_id = thread_id
        self._session.thread_id = thread_id
        self._restored_cost_threads.add(thread_id)
        self._plan_turn_texts = []
        self._session.reset_checkpointer()
        self._session.bootstrap(messages)
        await self.refresh_context_snapshot()
        return True

    async def fork_before(
        self,
        user_seq: int,
    ) -> tuple[str, str, list[dict[str, Any]]]:
        """Fork immediately before a persisted user message and switch to it."""
        selected = next(
            (
                turn
                for turn in self.thread_store.list_user_turns(self.thread_id)
                if int(turn.get("seq", -1)) == user_seq
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"Fork target seq {user_seq} is not a user message")
        checkpoint = self.thread_store.get_checkpoint(self.thread_id, user_seq)
        if checkpoint is None:
            raise ValueError(f"No checkpoint for seq {user_seq} in this thread")
        target = f"session-{uuid.uuid4().hex[:8]}"
        events = self.thread_store.copy_thread_prefix(
            self.thread_id,
            target,
            user_seq,
        )
        await asyncio.to_thread(
            self.memory_store.write_session_raw,
            target,
            str(checkpoint.get("mem_snapshot") or ""),
        )
        await self.resume(target, replay_cost=False, allow_empty=True)
        return target, str(selected.get("content") or ""), events

    async def reset(self, thread_id: str) -> None:
        """Archive the current thread and start fresh from ``thread_id``.

        Mirrors the CLI's ``/reset`` semantics: finalize the abandoned
        thread's reflection, archive it (so its events are no longer listed),
        and rebind the session to the caller-supplied fresh thread id. No
        bootstrap replay — a fresh thread has no events.
        """
        await self.finalize_reflection()
        self.thread_store.archive_thread(self.thread_id)
        self.thread_id = thread_id
        self._session.thread_id = thread_id
        self._restored_cost_threads.add(thread_id)
        self._plan_turn_texts = []
        self._session.reset_checkpointer()
        await self.refresh_context_snapshot()

    async def rollback_to(self, user_seq: int) -> str:
        """Restore the thread to the state captured at checkpoint ``user_seq``.

        Three things happen, in order:

        1. Files: surgically restore the paths the agent's tools mutated at or
           after this turn from the shadow git snapshot.
        2. Session memory: overwrite ``runtime/sessions/mem_<thread_id>.md`` from the
           checkpoint snapshot.
        3. Conversation: hard-truncate the events tail at ``user_seq`` and
           rebuild the live graph from the remaining events. The in-process
           cost_tracker is intentionally left untouched (its tally is
           process-global billing).

        Returns a human-readable status string (non-empty on success).
        """
        if user_seq < 0:
            return "Invalid rollback target."

        checkpoint = self.thread_store.get_checkpoint(self.thread_id, user_seq)
        if checkpoint is None:
            return f"No checkpoint for seq {user_seq} in this thread."

        notes: list[str] = []

        # 1. Files (surgical; full-tree when no paths recorded).
        file_note = await self._restore_rollback_files(checkpoint)
        if file_note:
            notes.append(file_note)

        # 2. Session memory (mem_<thread_id>.md) from the snapshot.
        mem_note = await self._restore_rollback_memory(checkpoint)
        if mem_note:
            notes.append(mem_note)

        # 3. Truncate the durable conversation tail and rebuild the live graph
        #    from the surviving events. cost_tracker is intentionally preserved.
        self.thread_store.truncate_after(self.thread_id, user_seq)
        await self.resume(self.thread_id, replay_cost=False)

        return "\n".join(notes) if notes else f"Rolled back to turn @ seq {user_seq}."

    async def _restore_rollback_files(self, checkpoint: dict) -> str:
        """Surgically restore paths from the checkpoint's git snapshot.

        Returns ``""`` on success or a warning string. Falls back to a notice
        when no git hash exists.
        """
        git_hash = checkpoint.get("git_hash")
        if not git_hash:
            return "No git snapshot for this checkpoint; files not reverted."

        raw = checkpoint.get("modified_paths") or ""
        paths: list[str] = []
        if raw:
            import json as _json

            try:
                paths = list(_json.loads(raw))
            except ValueError:
                paths = []
        try:
            output = await asyncio.to_thread(
                restore_paths, git_hash, paths, self.project_root
            )
            if output and not output.startswith("("):
                return f"File restore note: {output}"
        except Exception as exc:
            return f"File restore failed: {exc}"
        return ""

    async def _restore_rollback_memory(self, checkpoint: dict) -> str:
        """Overwrite the per-thread session-memory file from the snapshot."""
        try:
            snapshot = checkpoint.get("mem_snapshot") or ""
            await asyncio.to_thread(
                self.memory_store.write_session_raw,
                self.thread_id,
                snapshot,
            )
        except Exception as exc:
            return f"Session memory restore failed: {exc}"
        return ""


# ----------------------------------------------------------------------
# Module-scope helpers
# ----------------------------------------------------------------------


def _extract_mutated_paths(name: str, args: dict) -> list[str]:
    """Return the filesystem paths a destructive tool mutated, by tool name.

    ``shell`` is a wildcard (mutated paths cannot be enumerated from the
    command string alone) → returns ``"*"`` which the persistence layer
    collapses to the full-tree-restore sentinel. Read-only tools (``read``,
    ``grep``, ``glob``, ``web_search``, ``skill_view``, ``todo``) return
    ``[]`` and so never trigger the callback in the tools node anyway.
    """
    if not args:
        return []
    n = (name or "").lower()
    if n in ("edit", "write", "delete", "auto_format"):
        p = args.get("path")
        return [str(p)] if p else []
    if n == "shell":
        # Cannot enumerate which paths a shell command touched; mark the
        # turn as full-tree restore.
        return ["*"]
    return []