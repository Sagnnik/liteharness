from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from ness_agent import NessAgent, PromptLayers, PromptLayersConfig
from ness_agent.compaction import CompactionResult, apply_force_floor, summarize_history
from ness_agent.context.layers import AuxPrompts
from ness_agent.graph.nodes import make_nodes
from ness_agent.memory import MemoryStore
from ness_agent.options import MemoryConfig, NessAgentOptions
from ness_agent.persistence import ThreadStore
from ness_agent.reflection import finalize_session_reflection, run_reflection_gate
from ness_agent.tracing.cost import CostTracker


def _agent(tmp_path: Path, **kwargs):
    model = FakeListChatModel(responses=["hello"])

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    return NessAgent(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(ness_dir=tmp_path / ".ness", project_root=tmp_path),
        **kwargs,
    )


def test_apply_force_floor_upgrades_to_summary_when_history_long():
    assert apply_force_floor("none", 0, 11) == ("summary", 10)
    assert apply_force_floor("tool_outputs", 0, 11) == ("summary", 10)
    assert apply_force_floor("summary", 5, 11) == ("summary", 5)


def test_apply_force_floor_tool_outputs_when_history_short():
    assert apply_force_floor("none", 0, 10) == ("tool_outputs", 0)
    assert apply_force_floor("none", 0, 5) == ("tool_outputs", 0)
    assert apply_force_floor("tool_outputs", 3, 5) == ("tool_outputs", 3)


def test_reflection_result_returns_bullets(tmp_path: Path):
    store = ThreadStore(threads_dir=tmp_path / "threads", default_model="m")
    memory = MemoryStore(MemoryConfig(), ness_dir=tmp_path / ".ness")

    class OkModel:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _messages):
            return SimpleNamespace(
                model_dump=lambda: {
                    "new_bullet_points": ["Added rate limiter", "Wired auth"]
                },
                new_bullet_points=["Added rate limiter", "Wired auth"],
            )

    result = asyncio.run(
        run_reflection_gate(
            "session-bullets",
            [HumanMessage(content="hello")],
            OkModel(),
            1,
            memory=memory,
            persistence=store,
            aux_prompts=AuxPrompts(
                reflection=(
                    "t={thread_id} n={user_message_count} msgs={messages} "
                    "bullets={current_session_bullets} todos={todos}"
                )
            ),
        )
    )
    assert result.memory_updated is True
    assert result.bullets == ("Added rate limiter", "Wired auth")
    assert result.error == ""


def test_agent_node_tool_loop_without_overlay(tmp_path: Path):
    """Subagents set overlay=None; tool-loop turns must not crash on render_overlay_delta."""
    store = ThreadStore(threads_dir=tmp_path / "threads", default_model="m")
    agent = _agent(tmp_path)
    cfg = replace(agent.config, overlay=None, thread_store=store)
    rt = make_nodes(cfg, thread_id="subagent-explore-test", mode="act", git_available=False)

    not_compacted = CompactionResult(
        messages=[],
        compacted=False,
        token_count=10,
        action="none",
    )

    async def fake_ainvoke(_msgs):
        return AIMessage(content="ok")

    bind = SimpleNamespace(ainvoke=fake_ainvoke)
    state = {
        "messages": [HumanMessage(content="find routes")],
        "mode": "act",
        "todos": [],
    }

    with (
        patch(
            "ness_agent.graph.nodes.progressive_compact",
            side_effect=lambda conv, **kw: replace(
                not_compacted, messages=list(conv)
            ),
        ),
        patch.object(cfg.tool_registry, "bind_model", return_value=bind),
    ):
        asyncio.run(rt.agent_node(state))
        tool_loop = {
            **state,
            "messages": [
                HumanMessage(content="find routes"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "ping", "args": {}, "id": "c1"}],
                ),
                ToolMessage(content="pong", tool_call_id="c1", name="ping"),
            ],
        }
        updates = asyncio.run(rt.agent_node(tool_loop))

    assert updates["messages"][0].content == "ok"


def test_summarize_history_persists_once_on_success_and_failure(tmp_path: Path):
    store = ThreadStore(threads_dir=tmp_path / "threads", default_model="m")

    class OkModel:
        async def ainvoke(self, _messages):
            return SimpleNamespace(
                content="summary text",
                usage_metadata=None,
                response_metadata={},
            )

    class BoomModel:
        async def ainvoke(self, _messages):
            raise RuntimeError("summarizer down")

    asyncio.run(
        summarize_history(
            [HumanMessage(content="hello")],
            OkModel(),
            thread_id="session-sum-ok",
            persistence=store,
            action="summary",
            kept_recent=4,
        )
    )
    ok_events = [
        e
        for e in store.load_thread_events("session-sum-ok")
        if e.get("kind") == "compaction_llm"
    ]
    assert len(ok_events) == 1
    assert ok_events[0]["response"] == "summary text"

    asyncio.run(
        summarize_history(
            [HumanMessage(content="hello")],
            BoomModel(),
            thread_id="session-sum-fail",
            persistence=store,
            action="summary",
            kept_recent=4,
        )
    )
    fail_events = [
        e
        for e in store.load_thread_events("session-sum-fail")
        if e.get("kind") == "compaction_llm"
    ]
    assert len(fail_events) == 1
    assert "Compaction summary unavailable" in fail_events[0]["response"]


def test_usage_event_always_logged_with_model(tmp_path: Path):
    model = FakeListChatModel(responses=["hello"])
    object.__setattr__(model, "model", "usage-model")
    store = ThreadStore(threads_dir=tmp_path / "threads", default_model="")

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    agent = NessAgent(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(ness_dir=tmp_path / ".ness", project_root=tmp_path),
        cost_tracker=CostTracker(),
    )
    agent.config.thread_store = store
    agent.config.cost_tracker.add = lambda *a, **k: None  # type: ignore[method-assign]
    rt = make_nodes(agent.config, thread_id="session-u", mode="act", git_available=False)

    response = AIMessage(content="ok")
    response.usage_metadata = {"input_tokens": 5, "output_tokens": 1}

    async def fake_ainvoke(_msgs):
        return response

    with (
        patch(
            "ness_agent.graph.nodes.progressive_compact",
            return_value=CompactionResult(
                messages=[HumanMessage(content="hi")],
                compacted=False,
                token_count=5,
            ),
        ),
        patch.object(
            agent.config.tool_registry,
            "bind_model",
            return_value=SimpleNamespace(ainvoke=fake_ainvoke),
        ),
    ):
        asyncio.run(
            rt.agent_node(
                {
                    "messages": [HumanMessage(content="hi")],
                    "mode": "act",
                    "todos": [],
                    "last_input_tokens": 0,
                }
            )
        )

    events = agent.config.thread_store.load_thread_events("session-u")
    usage = next(e for e in events if e.get("kind") == "usage")
    assert usage["model"] == "usage-model"


def test_options_context_window_drives_usable_budget():
    from ness_agent.compaction import resolve_usable_context_budget

    opts = NessAgentOptions(
        context_window=100_000,
        compaction_output_reserve=8_000,
        compaction_input_reserve=2_000,
    )
    assert resolve_usable_context_budget("any-model", opts) == 90_000
    assert resolve_usable_context_budget("any-model", None) == 120_000
    assert resolve_usable_context_budget(
        "any-model",
        NessAgentOptions(context_window=None, compaction_token_budget=50_000),
    ) == 50_000


def test_agent_spec_resolves_backends(tmp_path: Path):
    from ness_agent import AgentSpec, NessAgent
    from ness_agent.tracing.tracer import NoopTracer

    model = FakeListChatModel(responses=["ok"])

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    spec = AgentSpec(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(ness_dir=tmp_path / ".ness", project_root=tmp_path),
    )
    agent = NessAgent.from_spec(spec)
    cfg = agent.config
    assert cfg.memory_store is not None
    assert cfg.thread_store is not None
    assert cfg.permission_store is not None
    assert cfg.tool_registry is not None
    assert cfg.cost_tracker is not None
    assert isinstance(cfg.tracer, NoopTracer)
    assert not hasattr(cfg, "budget")
    assert not hasattr(cfg, "permission_policy")
    assert not hasattr(cfg, "mcp_config")


def test_aggregate_usage_sums_calls_and_costs():
    from ness_agent.types import UsageEvent, aggregate_usage

    assert aggregate_usage([]) is None
    total = aggregate_usage(
        [
            UsageEvent("m", 10, 8, 2, 3, 0.01, calls=1),
            UsageEvent("m", 20, 15, 5, 4, 0.02, calls=1),
        ]
    )
    assert total is not None
    assert total.model == "m"
    assert total.input_tokens == 30
    assert total.uncached_input_tokens == 23
    assert total.cached_input_tokens == 7
    assert total.output_tokens == 7
    assert total.cost_usd == 0.03
    assert total.calls == 2

    mixed = aggregate_usage(
        [
            UsageEvent("a", 1, 1, 0, 1, None),
            UsageEvent("b", 2, 2, 0, 1, 0.5),
        ]
    )
    assert mixed is not None
    assert mixed.model == "*"
    assert mixed.cost_usd == 0.5


def test_run_result_usage_total_accumulates_bridge_events(tmp_path: Path):
    """Session.run exposes usage_total as the sum of per-call usage events."""
    from ness_agent.types import UsageEvent, aggregate_usage

    model = FakeListChatModel(responses=["done"])
    agent = NessAgent(
        model=model,
        tools=[],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(
            ness_dir=tmp_path / ".ness",
            project_root=tmp_path,
            enable_approval=False,
            auto_save_threads=False,
        ),
    )
    session = agent.session(thread_id="t-usage-total")

    # Simulate the usage bridge the agent node would fire mid-turn.
    from ness_agent.session import _active_session
    from ness_agent.session_context import reset_session_context

    async def _run():
        ctx_token = session._install_session_runtime()
        session._last_usage = None
        session._turn_usages = []
        token = _active_session.set(session)
        try:
            bridge = agent.config._usage_bridge
            bridge(UsageEvent("m", 100, 90, 10, 5, 0.1))
            bridge(UsageEvent("m", 200, 150, 50, 8, 0.2))
            assert session._last_usage is not None
            assert session._last_usage.input_tokens == 200
            total = aggregate_usage(session._turn_usages)
            assert total is not None
            assert total.input_tokens == 300
            assert total.calls == 2
            assert total.cost_usd == pytest.approx(0.3)
        finally:
            _active_session.reset(token)
            reset_session_context(ctx_token)

    asyncio.run(_run())
