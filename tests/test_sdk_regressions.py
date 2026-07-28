from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from liteharness import NessAgent, PromptLayers, PromptLayersConfig
from liteharness.compaction import CompactionResult, apply_force_floor, summarize_history
from liteharness.context.layers import AuxPrompts
from liteharness.graph.nodes import make_nodes
from liteharness.memory import MemoryStore
from liteharness.options import MemoryConfig, NessAgentOptions
from liteharness.persistence import ThreadStore
from liteharness.reflection import finalize_session_reflection, run_reflection_gate
from liteharness.tracing.cost import CostTracker


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


def test_make_sdk_cost_tracker_wires_cli_pricing():
    from liteharness_cli.config import MODEL_PRICING, make_sdk_cost_tracker

    tracker = make_sdk_cost_tracker()
    # SDK-side cost tracker exposes the pricing catalog directly so it can
    # estimate cost without the CLI's estimate_cost closure indirection.
    assert tracker.pricing == MODEL_PRICING
    # Pick a known model from MODEL_PRICING and assert the estimate path works.
    # MODEL_PRICING maps case-insensitive substrings; first key here is
    # one shipped in the CLI catalog.
    sample_key = next(iter(MODEL_PRICING))
    usage = tracker.add(
        {"input_tokens": 1_000_000, "output_tokens": 0},
        model_name=sample_key,
        response_metadata={},
    )
    assert usage is not None
    assert usage.cost_source == "estimated"
    expected_input_per_m = MODEL_PRICING[sample_key][0]
    assert usage.cost_usd == expected_input_per_m


def test_thread_store_records_default_model(tmp_path: Path):
    store = ThreadStore(threads_dir=tmp_path / "threads", default_model="gpt-test")
    store.append_event("session-a", {"kind": "user", "content": "hi"})
    rows = store.list_threads()
    assert rows[0]["model"] == "gpt-test"


def test_thread_store_backfills_model_from_usage(tmp_path: Path):
    store = ThreadStore(threads_dir=tmp_path / "threads", default_model="")
    store.append_event("session-b", {"kind": "user", "content": "hi"})
    assert store.list_threads()[0]["model"] == ""
    store.append_event(
        "session-b",
        {"kind": "usage", "model": "deepseek-v4", "input_tokens": 1, "output_tokens": 1},
    )
    assert store.list_threads()[0]["model"] == "deepseek-v4"


def test_ness_agent_wires_thread_store_default_model(tmp_path: Path):
    model = FakeListChatModel(responses=["ok"])
    object.__setattr__(model, "model_name", "wired-model")

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    agent = NessAgent(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(ness_dir=tmp_path / ".ness", project_root=tmp_path),
    )
    assert agent.config.thread_store.default_model == "wired-model"


def test_reflection_error_event_has_full_schema(tmp_path: Path):
    store = ThreadStore(threads_dir=tmp_path / "threads", default_model="m")

    class BoomModel:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _messages):
            raise RuntimeError("boom")

    result = asyncio.run(
        run_reflection_gate(
            "session-r",
            [HumanMessage(content="hello")],
            BoomModel(),
            1,
            persistence=store,
            aux_prompts=AuxPrompts(
                reflection=(
                    "t={thread_id} n={user_message_count} msgs={messages} "
                    "bullets={current_session_bullets} todos={todos}"
                )
            ),
        )
    )
    assert result.error == "boom"
    assert result.bullets == ()
    events = store.load_thread_events("session-r")
    reflection = next(e for e in events if e.get("kind") == "reflection")
    assert set(reflection) >= {
        "kind",
        "prompt",
        "response",
        "message_index",
        "memory_updated",
        "error",
        "t",
    }
    assert reflection["error"] == "boom"
    assert reflection["response"] == {"new_bullet_points": []}
    assert reflection["memory_updated"] is False
    assert "No todos" in reflection["prompt"]


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


def test_finalize_session_reflection_passes_rendered_todos(tmp_path: Path):
    store = ThreadStore(threads_dir=tmp_path / "threads", default_model="m")
    captured: dict = {}

    class CaptureModel:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, messages):
            captured["prompt"] = messages[0].content
            return SimpleNamespace(
                model_dump=lambda: {"new_bullet_points": []},
                new_bullet_points=[],
            )

    app = MagicMock()
    app.aget_state = AsyncMock(
        return_value=SimpleNamespace(
            values={
                "messages": [HumanMessage(content="do work")],
                "todos": [{"id": "1", "content": "Ship it", "status": "pending"}],
                "last_reflection_index": 0,
            }
        )
    )

    asyncio.run(
        finalize_session_reflection(
            app,
            "session-f",
            CaptureModel(),
            persistence=store,
            aux_prompts=AuxPrompts(
                reflection=(
                    "t={thread_id} n={user_message_count} msgs={messages} "
                    "bullets={current_session_bullets} todos={todos}"
                )
            ),
        )
    )
    assert "[pending] 1: Ship it" in captured["prompt"]


def test_agent_node_sets_last_input_tokens_zero_after_compaction(tmp_path: Path):
    store = ThreadStore(threads_dir=tmp_path / "threads", default_model="m")
    agent = _agent(tmp_path)
    agent.config.thread_store = store
    rt = make_nodes(agent.config, thread_id="session-c", mode="act", git_available=False)

    compacted = CompactionResult(
        messages=[HumanMessage(content="hi")],
        compacted=True,
        token_count=10,
        action="summary",
    )
    response = AIMessage(content="ok")

    async def fake_ainvoke(_msgs):
        return response

    with (
        patch(
            "liteharness.graph.nodes.progressive_compact",
            return_value=compacted,
        ),
        patch.object(
            agent.config.tool_registry,
            "bind_model",
            return_value=SimpleNamespace(ainvoke=fake_ainvoke),
        ),
    ):
        updates = asyncio.run(
            rt.agent_node(
                {
                    "messages": [HumanMessage(content="hi")],
                    "mode": "act",
                    "todos": [],
                    "last_input_tokens": 999,
                }
            )
        )

    assert updates["last_input_tokens"] == 0
    assert updates["compaction_message_count"] == 1


def test_agent_node_emits_compaction_bridge_on_compact(tmp_path: Path):
    store = ThreadStore(threads_dir=tmp_path / "threads", default_model="m")
    agent = _agent(tmp_path)
    agent.config.thread_store = store
    seen: list[dict] = []
    agent.config._compaction_bridge = lambda data: seen.append(dict(data))
    rt = make_nodes(agent.config, thread_id="session-cb", mode="act", git_available=False)

    compacted = CompactionResult(
        messages=[HumanMessage(content="hi")],
        compacted=True,
        token_count=10,
        action="tool_outputs",
        kept_recent=0,
    )

    async def fake_ainvoke(_msgs):
        return AIMessage(content="ok")

    with (
        patch(
            "liteharness.graph.nodes.progressive_compact",
            return_value=compacted,
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
                    "force_compact": True,
                    "last_input_tokens": 999,
                }
            )
        )

    assert len(seen) == 1
    assert seen[0]["reason"] == "agent_turn"
    assert seen[0]["action"] == "tool_outputs"
    assert seen[0]["forced"] is True
    assert seen[0]["info"]


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
            "liteharness.graph.nodes.progressive_compact",
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
    from liteharness.compaction import resolve_usable_context_budget

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
    from liteharness import AgentSpec, NessAgent
    from liteharness.tracing.tracer import NoopTracer

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


def test_reflection_schedule_budget_resolve_no_typeerror(tmp_path: Path):
    """reflection_token_ratio > 0 must call resolve_usable_context_budget correctly."""
    from liteharness.graph.nodes import _schedule_reflection_if_due, make_nodes

    model = FakeListChatModel(responses=["hello"])

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    agent = NessAgent(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(
            ness_dir=tmp_path / ".ness",
            project_root=tmp_path,
            reflection_token_ratio=0.5,
            context_window=100_000,
        ),
    )
    rt = make_nodes(agent.config, thread_id="session-ref", mode="act", git_available=False)
    # Should not raise TypeError (previously passed budget as meta).
    _schedule_reflection_if_due(
        rt,
        {"last_reflection_index": 0, "todos": []},
        [HumanMessage(content="hello")],
        "test-model",
    )


def test_aggregate_usage_sums_calls_and_costs():
    from liteharness.types import UsageEvent, aggregate_usage

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


def test_tool_end_data_includes_duration_ms():
    from liteharness.session import _tool_end_data

    msg = ToolMessage(
        tool_call_id="c1",
        name="read",
        content="ok",
        additional_kwargs={"duration_ms": 42},
    )
    data = _tool_end_data(msg)
    assert data["name"] == "read"
    assert data["id"] == "c1"
    assert data["duration_ms"] == 42

    bare = _tool_end_data(ToolMessage(tool_call_id="c2", name="shell", content="x"))
    assert "duration_ms" not in bare


def test_tools_node_stamps_duration_ms_on_tool_message(tmp_path: Path):
    model = FakeListChatModel(responses=["hello"])

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    agent = NessAgent(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(ness_dir=tmp_path / ".ness", project_root=tmp_path),
    )
    rt = make_nodes(agent.config, thread_id="t-dur", mode="act", git_available=False)

    async def _run():
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "ping", "args": {}, "id": "c1"}],
        )
        out = await rt.tools_node({"messages": [ai], "todos": [], "mode": "act"})
        msg = out["messages"][0]
        assert isinstance(msg, ToolMessage)
        assert "duration_ms" in (msg.additional_kwargs or {})
        assert isinstance(msg.additional_kwargs["duration_ms"], int)
        assert msg.additional_kwargs["duration_ms"] >= 0

    asyncio.run(_run())


def test_run_result_usage_total_accumulates_bridge_events(tmp_path: Path):
    """Session.run exposes usage_total as the sum of per-call usage events."""
    from liteharness.types import UsageEvent, aggregate_usage

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
    from liteharness.session import _active_session

    async def _run():
        session._install_session_runtime()
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

    asyncio.run(_run())
