from __future__ import annotations

import sys

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool


def test_import_liteharness_public_api():
    import liteharness as lh

    for name in (
        "NessAgent",
        "NessAgentConfig",
        "AgentSpec",
        "Session",
        "ContextPreview",
        "NoopTracer",
        "CostTracker",
        "PromptLayers",
        "ToolRegistry",
        "coding_tools",
        "NessAgentOptions",
        "message_to_text",
    ):
        assert hasattr(lh, name), name
    assert not hasattr(lh, "ContextBudgetConfig")


def test_tools_import_without_root_config():
    # Ensure SDK tools do not pull in the root harness config/permissions modules.
    before = {k for k in sys.modules if k in {"config", "permissions", "agent", "session"}}
    import liteharness.tools.fs  # noqa: F401
    import liteharness.tools.shell  # noqa: F401
    import liteharness.tools.web  # noqa: F401
    import liteharness.tools.search  # noqa: F401
    import liteharness.tools.subagents  # noqa: F401

    after = {k for k in sys.modules if k in {"config", "permissions", "agent", "session"}}
    assert after == before


def test_prompt_layers_stable_prefix_and_cache():
    from liteharness.context.layers import PromptLayers, PromptLayersConfig

    layers = PromptLayers(PromptLayersConfig(l0="You are a test harness.", persona="Tester."))
    prefix = layers.build_stable_prefix(
        [],
        user_memory="",
        project_memory="",
        skill_catalog="",
        git_available=False,
        metadata={},
    )
    assert "You are a test harness." in prefix
    assert layers._cache.get("content") == prefix
    again = layers.build_stable_prefix(
        [],
        user_memory="",
        project_memory="",
        skill_catalog="",
        git_available=False,
        metadata={},
    )
    assert again == prefix


def test_ness_agent_session_builds_graph():
    from liteharness import NessAgent, PromptLayers, PromptLayersConfig

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    model = FakeListChatModel(responses=["hello"])
    agent = NessAgent(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
    )
    session = agent.session(thread_id="t-smoke")
    assert session.app is not None
    assert session.thread_id == "t-smoke"
    assert agent.config.cost_tracker is not None
    assert agent.config.tracer is not None


def test_instructions_dir_exists_for_cli_prompts():
    # Instruction texts now ship as an importable Python package so users
    # can read/modify them and feed them back into PromptLayersConfig /
    # AuxPrompts / CodingOverlay. The original .md files are gone.
    from liteharness import instructions as I

    for name in (
        "L0_HARNESS", "L1_PROFILE", "PLAN_MODE", "ACT_MODE",
        "COMPACTION", "REFLECTION", "SUBAGENT", "THREAD_SUMMARY", "INIT_MEMORY",
    ):
        const = getattr(I, name)
        assert isinstance(const, str), name
        assert const.strip(), name


def test_default_overlay_is_coding_overlay():
    from liteharness import NessAgent, PromptLayersConfig, CodingOverlay
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    agent = NessAgent(
        model=FakeListChatModel(responses=["x"]),
        prompt=PromptLayersConfig(),
    )
    assert isinstance(agent.config.overlay, CodingOverlay)


def test_no_overlay_opts_out():
    from liteharness import NessAgent, PromptLayersConfig, NoOverlay, CodingOverlay
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    agent = NessAgent(
        model=FakeListChatModel(responses=["x"]),
        prompt=PromptLayersConfig(),
        overlay=NoOverlay(),
    )
    assert isinstance(agent.config.overlay, NoOverlay)
    assert not isinstance(agent.config.overlay, CodingOverlay)
