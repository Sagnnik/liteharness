"""Tests for support-layer wiring: factory, memory, hooks, setup."""

from __future__ import annotations

from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ness_ai import (
    ApprovalHandler,
    Hook,
    MemoryConfig,
    MemoryStore,
    setup_ness_structure,
)
from ness_ai.hooks import HookRunner
from ness_cli.config import settings
from ness_cli.factory import build_coding_agent


def test_factory_wires_hooks_and_skills(tmp_path: Path, monkeypatch):
    ness = tmp_path / ".ness"
    ness.mkdir()
    monkeypatch.setattr(settings, "ness_dir", str(ness))
    monkeypatch.setenv("NESS_AI_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("NESS_AI_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    from ness_cli import factory as fac

    monkeypatch.setattr(fac, "create_model", lambda *_a, **_k: FakeListChatModel(responses=["ok"]))
    monkeypatch.setattr(
        fac, "create_compaction_model", lambda *_a, **_k: FakeListChatModel(responses=["ok"])
    )
    monkeypatch.setattr(
        fac, "create_reflection_model", lambda *_a, **_k: FakeListChatModel(responses=["ok"])
    )
    monkeypatch.setattr(fac, "make_sdk_cost_tracker", lambda: None)

    from ness_cli.paths import sanitize_slug

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


def test_memory_include_resolves_from_project_root(tmp_path: Path):
    ness = tmp_path / ".ness"
    ness.mkdir()
    (tmp_path / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
    (ness / "NESS.md").write_text("@AGENTS.md\n", encoding="utf-8")
    store = MemoryStore(MemoryConfig(), ness_dir=ness, project_root=tmp_path)
    text = store.load_project()
    assert "# agents" in text
    assert "missing include" not in text


def test_build_coding_agent_accepts_render_approval_handler(tmp_path: Path, monkeypatch):
    from ness_cli.factory import build_coding_agent, prepare_paths
    from ness_cli.tui import render

    monkeypatch.setenv("NESS_AI_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("NESS_AI_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)

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
    assert explore.startswith("---")
    assert "findings report" in explore
    assert not (ness / "plans").exists()
    assert not (ness / "sessions").exists()
    assert any("hooks.json" in c for c in created)
    assert any(c.endswith("explore.md") for c in created)
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
    assert not any(c.endswith(".md") for c in created if "/agents/" in c)


def test_hook_callable_vetoes_pre_tool_use():
    def deny(payload: dict) -> tuple[bool, str]:
        return False, "blocked"

    runner = HookRunner(hooks_file=None, hooks=[Hook(event="preToolUse", matcher="*", handler=deny)])
    ok, msg = runner.run("preToolUse", {"tool": "shell", "args": {}})
    assert ok is False
    assert "blocked" in msg
