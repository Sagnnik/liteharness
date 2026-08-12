from __future__ import annotations

from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool


def test_import_ness_agent_public_api():
    import ness_agent as lh

    for name in (
        "NessAgent",
        "NessAgentConfig",
        "AgentSpec",
        "Session",
        "ContextPreview",
        "NoopTracer",
        "CostTracker",
        "aggregate_usage",
        "PromptLayers",
        "ToolRegistry",
        "coding_tools",
        "NessAgentOptions",
        "message_to_text",
        "MCPRuntime",
        "MCPServerSpec",
        "MCPServerState",
        "MCPAuthenticationRequired",
    ):
        assert hasattr(lh, name), name


def test_ness_agent_session_builds_graph(tmp_path: Path):
    from ness_agent import NessAgent, NessAgentOptions, PromptLayers, PromptLayersConfig

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    model = FakeListChatModel(responses=["hello"])
    agent = NessAgent(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(
            project_root=tmp_path,
            ness_dir=tmp_path / ".ness",
        ),
    )
    session = agent.session(thread_id="t-smoke")
    assert session.app is not None
    assert session.thread_id == "t-smoke"
    assert agent.config.cost_tracker is not None
    assert agent.config.tracer is not None
