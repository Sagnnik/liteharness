"""Tests for P0/P1 support-layer wiring: factory, memory, hooks, skills."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool

from liteharness import (
    ApprovalHandler,
    Hook,
    MemoryBackend,
    MemoryConfig,
    MemoryStore,
    NessAgent,
    NessAgentOptions,
    PromptLayers,
    PromptLayersConfig,
    setup_ness_structure,
)
from liteharness.hooks import HookRunner
from liteharness_cli.config import settings
from liteharness_cli.factory import build_coding_agent


def _agent(**kwargs):
    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    return NessAgent(
        model=FakeListChatModel(responses=["ok"]),
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        **kwargs,
    )


def test_factory_wires_hooks_and_skills(tmp_path: Path, monkeypatch):
    ness = tmp_path / ".ness"
    ness.mkdir()
    monkeypatch.setattr(settings, "ness_dir", str(ness))
    monkeypatch.setenv("LITEHARNESS_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("LITEHARNESS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    # Avoid real OpenRouter model construction in unit test.
    from liteharness_cli import factory as fac

    monkeypatch.setattr(fac, "create_model", lambda *_a, **_k: FakeListChatModel(responses=["ok"]))
    monkeypatch.setattr(
        fac, "create_compaction_model", lambda *_a, **_k: FakeListChatModel(responses=["ok"])
    )
    monkeypatch.setattr(
        fac, "create_reflection_model", lambda *_a, **_k: FakeListChatModel(responses=["ok"])
    )
    monkeypatch.setattr(fac, "make_sdk_cost_tracker", lambda: None)

    from liteharness_cli.paths import sanitize_slug

    agent = build_coding_agent(thread_id="session-wire")
    assert agent.config.hooks_config == ness.resolve() / "hooks.json"
    assert agent.config.skills_dir == ness.resolve() / "skills"
    assert agent.config.hook_runner is not None
    assert agent.config.hook_runner.hooks_file == ness.resolve() / "hooks.json"
    assert agent.config.memory_store.user_file == (tmp_path / "cfg" / "USER.md").resolve()
    assert Path(agent.config.modes.plans_dir).resolve() == (
        tmp_path / "cfg" / "plans" / sanitize_slug(tmp_path.name)
    ).resolve()
    assert agent.config.memory_store.session_dir == (
        ness.resolve() / "runtime" / "sessions"
    )


def test_resolve_defaults_hooks_config_to_ness_dir(tmp_path: Path):
    agent = _agent(
        options=NessAgentOptions(project_root=tmp_path, ness_dir=tmp_path / ".ness")
    )
    assert agent.config.hooks_config == (tmp_path / ".ness" / "hooks.json").resolve()


def test_memory_include_resolves_from_project_root(tmp_path: Path):
    ness = tmp_path / ".ness"
    ness.mkdir()
    (tmp_path / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    (ness / "NESS.md").write_text("@AGENTS.md\n", encoding="utf-8")
    store = MemoryStore(MemoryConfig(), ness_dir=ness, project_root=tmp_path)
    text = store.load_project()
    assert "# agents" in text
    assert "missing include" not in text


def test_memory_store_inject_at_spec(tmp_path: Path):
    class Stub(MemoryBackend):
        @property
        def disabled(self) -> bool:
            return False

        def load_project(self):
            return "FROM_STUB"

        def append_project(self, text: str) -> str:
            return "ok"

        def write_project(self, text: str, overwrite: bool = False) -> str:
            return "ok"

        def load_user(self):
            return ""

        def append_user(self, text: str) -> str:
            return "ok"

        def write_user(self, text: str, overwrite: bool = False) -> str:
            return "ok"

        def load_session(self, thread_id: str) -> str:
            return ""

        def append_session_bullets(self, thread_id: str, bullets: list[str]) -> bool:
            return False

        def read_session_raw(self, thread_id: str) -> str:
            return ""

        def write_session_raw(self, thread_id: str, text: str) -> None:
            return None

        def check_health(self):
            return None

    stub = Stub()
    agent = _agent(
        options=NessAgentOptions(project_root=tmp_path, ness_dir=tmp_path / ".ness"),
        memory_store=stub,
    )
    assert agent.config.memory_store is stub
    assert agent.config.memory_store.load_project() == "FROM_STUB"


def test_resolve_rejects_duck_typed_extension_points(tmp_path: Path):
    options = NessAgentOptions(project_root=tmp_path, ness_dir=tmp_path / ".ness")

    class NotAnOverlay:
        def sections(self, state, ctx):
            return {}

    with pytest.raises(TypeError, match="overlay must be an instance of OverlayProvider"):
        _agent(options=options, overlay=NotAnOverlay())

    class NotAMemory:
        disabled = False

        def load_project(self):
            return ""

    with pytest.raises(TypeError, match="memory_store must be an instance of MemoryBackend"):
        _agent(options=options, memory_store=NotAMemory())

    class NotAnApproval:
        async def __call__(self, tool: str, args: dict) -> str:
            return "yes"

    with pytest.raises(TypeError, match="approval_handler must be an instance of ApprovalHandler"):
        _agent(options=options, approval_handler=NotAnApproval())

    class YesApproval(ApprovalHandler):
        async def __call__(self, tool: str, args: dict) -> str:
            return "yes"

    agent = _agent(options=options, approval_handler=YesApproval())
    assert isinstance(agent.config.approval_handler, ApprovalHandler)


def test_build_coding_agent_accepts_render_approval_handler(tmp_path: Path):
    from liteharness_cli.factory import build_coding_agent, prepare_paths
    from liteharness_cli.tui import render

    agent = build_coding_agent(
        thread_id="t",
        approval_handler=render.render_approval_handler,
        paths=prepare_paths(project_root=tmp_path),
    )
    assert isinstance(agent.config.approval_handler, ApprovalHandler)


def test_setup_ness_structure_creates_layout(tmp_path: Path):
    ness = tmp_path / ".ness"
    created = setup_ness_structure(ness)
    assert (ness / "skills").is_dir()
    assert (ness / "runtime" / "sessions").is_dir()
    assert (ness / "runtime" / "shells").is_dir()
    assert (ness / "hooks.json").is_file()
    assert (ness / "permissions.json").is_file()
    explore = (ness / "agents" / "explore.md").read_text(encoding="utf-8")
    exec_body = (ness / "agents" / "exec.md").read_text(encoding="utf-8")
    assert explore.startswith("---")
    assert "findings report" in explore
    assert exec_body.startswith("---")
    assert "execution subagent" in exec_body
    assert not (ness / "plans").exists()
    assert not (ness / "sessions").exists()
    assert any("hooks.json" in c for c in created)
    assert any(c.endswith("explore.md") for c in created)
    assert any(c.endswith("exec.md") for c in created)
    ness_md = ness / "NESS.md"
    assert ness_md.is_file()
    assert ness_md.read_text(encoding="utf-8") == ""
    assert any(c.endswith("NESS.md") for c in created)


def test_setup_ness_structure_does_not_overwrite_ness_md(tmp_path: Path):
    ness = tmp_path / ".ness"
    ness.mkdir()
    ness_md = ness / "NESS.md"
    ness_md.write_text("# keep me\n", encoding="utf-8")
    created = setup_ness_structure(ness)
    assert ness_md.read_text(encoding="utf-8") == "# keep me\n"
    assert not any(c.endswith("NESS.md") for c in created)


def test_setup_ness_structure_does_not_overwrite_agent_profiles(tmp_path: Path):
    ness = tmp_path / ".ness"
    agents = ness / "agents"
    agents.mkdir(parents=True)
    custom = agents / "explore.md"
    custom.write_text("---\ntools: [read]\n---\ncustom explore\n", encoding="utf-8")
    created = setup_ness_structure(ness)
    assert custom.read_text(encoding="utf-8") == "---\ntools: [read]\n---\ncustom explore\n"
    assert not any(c.endswith("explore.md") for c in created)
    assert (agents / "exec.md").is_file()
    assert any(c.endswith("exec.md") for c in created)


def test_session_raw_roundtrip(tmp_path: Path):
    ness = tmp_path / ".ness"
    store = MemoryStore(
        MemoryConfig(session_memory_dir=tmp_path / "custom_sessions"),
        ness_dir=ness,
        project_root=tmp_path,
    )
    store.write_session_raw("t1", "- a\n- b\n")
    assert store.read_session_raw("t1") == "- a\n- b\n"
    assert (tmp_path / "custom_sessions" / "mem_t1.md").exists()
    store.write_session_raw("t1", "")
    assert not (tmp_path / "custom_sessions" / "mem_t1.md").exists()


def test_hook_callable_vetoes_pre_tool_use():
    def deny(payload: dict) -> tuple[bool, str]:
        return False, "blocked"

    runner = HookRunner(hooks_file=None, hooks=[Hook(event="preToolUse", matcher="*", handler=deny)])
    ok, msg = runner.run("preToolUse", {"tool": "shell", "args": {}})
    assert ok is False
    assert "blocked" in msg


def test_hook_register_and_clear():
    runner = HookRunner(hooks_file=None)
    runner.register(
        Hook(event="postToolUse", matcher="*", handler=lambda p: (True, "note"))
    )
    ok, msg = runner.run("postToolUse", {"tool": "read", "args": {}, "result": "x"})
    assert ok is True
    assert "note" in msg
    runner.clear_registered()
    ok2, msg2 = runner.run("postToolUse", {"tool": "read", "args": {}, "result": "x"})
    assert msg2 == ""


def test_cmd_skill_stages_via_coding():
    from types import SimpleNamespace

    from liteharness_cli.tui.commands import cmd_skill
    from tests.test_cli.conftest import FakeCoding

    coding = FakeCoding()
    coding.skill_loader.load = lambda: {
        "alpha": {"name": "alpha", "description": "A", "source": "x"},
        "beta": {"name": "beta", "description": "B", "source": "y"},
    }
    app = SimpleNamespace(coding=coding)

    with patch("liteharness_cli.tui.commands.render") as render:
        render.render_notice = lambda *a, **k: None
        render.render_error = lambda *a, **k: None
        render.render_table = lambda *a, **k: None
        render.render_warning = lambda *a, **k: None
        asyncio.run(cmd_skill(app, "alpha"))
        asyncio.run(cmd_skill(app, "beta"))
        asyncio.run(cmd_skill(app, "alpha"))  # dedupe

    assert coding._pending_skills == ["alpha", "beta"]
