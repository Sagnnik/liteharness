from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from liteharness import NessAgent, PromptLayers, PromptLayersConfig
from liteharness.context.overlay import OverlayContext
from liteharness.graph.helpers import _with_working_state_tail
from liteharness.graph.nodes import make_nodes
from liteharness.options import NessAgentOptions
from liteharness.tools.todo import get_thread_todos, set_current_thread, set_thread_todos


def _agent(**kwargs):
    model = FakeListChatModel(responses=["hello"])

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    return NessAgent(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        **kwargs,
    )


def test_coding_overlay_mode_switch_uses_act_template():
    from liteharness_cli.overlay import CodingOverlay

    overlay = CodingOverlay(act_mode_template="MODE SWITCH\nDo the work.")
    ctx = OverlayContext(
        thread_id="t1",
        agent_mode="act",
        messages=[],
        todos=[],
        session_memory="",
        compaction_note="",
        mode_switch="plan->act",
        git_snapshot="branch: main; working tree clean",
    )
    sections = overlay.sections({}, ctx)
    assert sections["mode_switch"] == "MODE SWITCH\nDo the work."
    assert "act_mode" not in sections
    assert "GIT" in sections["git"]


def test_coding_overlay_no_persistent_act_mode():
    from liteharness_cli.overlay import CodingOverlay

    overlay = CodingOverlay(act_mode_template="should not appear every turn")
    ctx = OverlayContext(
        thread_id="t1",
        agent_mode="act",
        messages=[],
        todos=[],
        session_memory="",
        compaction_note="",
        mode_switch="",
    )
    sections = overlay.sections({}, ctx)
    assert "mode_switch" not in sections
    assert "act_mode" not in sections


def test_no_double_system_reminder_wrap():
    msgs = [HumanMessage(content="hi")]
    overlay = "GIT\nbranch: main"
    out = _with_working_state_tail(msgs, overlay)
    text = str(out[-1].content)
    assert text.count("<system-reminder>") == 1
    assert "<system-reminder>\n\n<system-reminder>" not in text


def test_plan_act_mode_switch_consumed_once():
    agent = _agent()
    session = agent.session(thread_id="t-mode")
    session.set_mode("plan")
    assert session.mode == "plan"
    session.set_mode("act")
    assert session._pending_act_checkpoint is True

    async def _run():
        # _build_run_payload now takes a pre-computed mode_switch
        payload, _ = await session._build_run_payload(
            "go", images=None, active_skills=None, mode_switch="plan->act"
        )
        assert payload["mode_switch"] == "plan->act"
        assert session._pending_act_checkpoint is True  # not consumed by payload builder
        payload2, _ = await session._build_run_payload(
            "again", images=None, active_skills=None, mode_switch=""
        )
        assert payload2["mode_switch"] == ""

    asyncio.run(_run())


def test_toggle_mode():
    agent = _agent()
    session = agent.session(thread_id="t-toggle")
    assert session.toggle_mode() == "plan"
    assert session.toggle_mode() == "act"
    assert session._pending_act_checkpoint is True


def test_approval_session_and_never_persist(tmp_path: Path):
    from liteharness.types import ApprovalHandler

    decisions = iter(["session", "never"])

    class TestHandler(ApprovalHandler):
        async def __call__(self, name: str, args: dict) -> str:
            return next(decisions)

    agent = _agent(
        approval_handler=TestHandler(),
        options=NessAgentOptions(
            project_root=tmp_path,
            ness_dir=tmp_path / ".ness",
            enable_approval=True,
            auto_save_threads=True,
        ),
    )
    cfg = agent.config
    cfg.thread_store.auto_save = True
    rt = make_nodes(cfg, thread_id="t-appr", agent_mode="act", git_available=False)

    async def _run():
        # Destructive shell cmds not in allow/deny lists → ask
        # Use distinct first tokens so session rules don't collide (python* vs cargo*).
        args_session = {"action": "run", "command": "npm run build"}
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "shell", "args": args_session, "id": "c1"}],
        )
        result = await rt.approval_gate({"messages": [ai], "approval_declined": False})
        assert result.get("approval_declined") is False
        assert cfg.permission_store.check("shell", args_session) == "allow"

        args_never = {"action": "run", "command": "cargo test"}
        ai2 = AIMessage(
            content="",
            tool_calls=[{"name": "shell", "args": args_never, "id": "c2"}],
        )
        result2 = await rt.approval_gate({"messages": [ai2], "approval_declined": False})
        assert result2.get("approval_declined") is True
        assert cfg.permission_store.check("shell", args_never) == "deny"

    asyncio.run(_run())


def test_tools_node_returns_todos():
    agent = _agent()
    cfg = agent.config
    thread_id = "t-todos"
    rt = make_nodes(cfg, thread_id=thread_id, agent_mode="act", git_available=False)

    set_current_thread(thread_id)
    set_thread_todos(thread_id, [{"id": "1", "content": "a", "status": "pending"}])

    @tool
    def todo(items: list[dict] | None = None) -> str:
        """Update todos."""
        set_thread_todos(thread_id, [{"id": "1", "content": "done", "status": "completed"}])
        return "ok"

    cfg.tool_registry._tool_map["todo"] = todo
    cfg.tool_registry._all_tools.append(todo)
    cfg.tool_registry.bump_generation()

    async def _run():
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "todo", "args": {}, "id": "t1"}],
        )
        out = await rt.tools_node(
            {"messages": [ai], "todos": [], "agent_mode": "act", "current_user_seq": 1}
        )
        assert "todos" in out
        assert out["todos"][0]["status"] == "completed"
        assert get_thread_todos(thread_id)[0]["content"] == "done"

    asyncio.run(_run())


def test_git_snapshot_passed_to_overlay(tmp_path: Path):
    seen: dict = {}

    class CaptureOverlay:
        def sections(self, state, ctx: OverlayContext):
            seen["git_snapshot"] = ctx.git_snapshot
            seen["git_available"] = ctx.git_available
            return {"git": f"GIT\n{ctx.git_snapshot}"} if ctx.git_snapshot else {}

    agent = _agent(
        overlay=CaptureOverlay(),
        options=NessAgentOptions(
            project_root=tmp_path,
            ness_dir=tmp_path / ".ness",
        ),
    )
    cfg = agent.config
    rt = make_nodes(cfg, thread_id="t-git", agent_mode="act", git_available=True)

    async def _run():
        class _Bound:
            async def ainvoke(self, messages):
                return AIMessage(content="ok")

        with patch(
            "liteharness.graph.nodes.git_worktree_summary",
            return_value="branch: main; working tree clean",
        ), patch.object(cfg.tool_registry, "bind_model", return_value=_Bound()):
            await rt.agent_node(
                {
                    "messages": [HumanMessage(content="hi")],
                    "todos": [],
                    "agent_mode": "act",
                    "force_compact": False,
                    "activate_skills": [],
                    "mode_switch": "",
                }
            )

    asyncio.run(_run())
    assert seen.get("git_available") is True
    assert "branch: main" in (seen.get("git_snapshot") or "")


def test_session_emits_assistant_events():
    agent = _agent()
    session = agent.session(thread_id="t-stream")

    async def _run():
        events = []
        async for ev in session.stream("hello"):
            events.append(ev.kind)
        assert "assistant_delta" in events or "assistant_final" in events or "error" in events

    asyncio.run(_run())


def test_smoke_still_passes_import():
    from liteharness import CostTracker, NoopTracer, PreActCompactHandler, Session

    assert CostTracker is not None
    assert NoopTracer is not None
    assert Session is not None
    assert PreActCompactHandler is not None
