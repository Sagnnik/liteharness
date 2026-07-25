from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from liteharness.context.overlay import OverlayContext, OverlayProvider, render_overlay_delta, wrap_system_reminder
from liteharness.skills import SkillLoader
from liteharness import CodingOverlay


# ---------------------------------------------------------------------------
# SkillLoader parsing tests
# ---------------------------------------------------------------------------

def test_skill_loader_loads_skills_with_all_fields(tmp_path: Path):
    skill_dir = tmp_path / ".ness" / "skills" / "test_skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: test_skill\n"
        "description: A test skill\n"
        "license: MIT\n"
        "compatibility: '>=1.0'\n"
        "metadata:\n"
        "  author: test\n"
        "  version: 2\n"
        "---\n"
        "Skill body content\n"
    )

    loader = SkillLoader(tmp_path / ".ness" / "skills")
    skills = loader.load()
    assert "test_skill" in skills
    s = skills["test_skill"]
    assert s["name"] == "test_skill"
    assert s["description"] == "A test skill"
    assert s["license"] == "MIT"
    assert s["compatibility"] == ">=1.0"
    assert s["metadata"] == {"author": "test", "version": 2}
    assert s["body"] == "Skill body content"
    assert str(s["source"]).endswith("SKILL.md")
    assert "format" not in s


def test_skill_loader_skips_when_name_missing(tmp_path: Path):
    skill_dir = tmp_path / ".ness" / "skills" / "no_name"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\ndescription: No name here\n---\nBody\n"
    )

    loader = SkillLoader(tmp_path / ".ness" / "skills")
    skills = loader.load()
    assert skills == {}


def test_skill_loader_skips_when_name_empty(tmp_path: Path):
    skill_dir = tmp_path / ".ness" / "skills" / "empty_name"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: ''\ndescription: Empty name\n---\nBody\n"
    )

    loader = SkillLoader(tmp_path / ".ness" / "skills")
    skills = loader.load()
    assert skills == {}


def test_skill_loader_skips_when_description_missing(tmp_path: Path):
    skill_dir = tmp_path / ".ness" / "skills" / "no_desc"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: no_desc\n---\nBody\n")

    loader = SkillLoader(tmp_path / ".ness" / "skills")
    skills = loader.load()
    assert skills == {}


def test_skill_loader_skips_when_description_empty(tmp_path: Path):
    skill_dir = tmp_path / ".ness" / "skills" / "empty_desc"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: empty_desc\ndescription: ''\n---\nBody\n")

    loader = SkillLoader(tmp_path / ".ness" / "skills")
    skills = loader.load()
    assert skills == {}


def test_skill_loader_no_fallback_from_dir_name(tmp_path: Path):
    """Dir name must NOT be used as the skill name when frontmatter lacks it."""
    skill_dir = tmp_path / ".ness" / "skills" / "dir_name_skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\ndescription: Dir name should not become name\n---\nBody\n"
    )

    loader = SkillLoader(tmp_path / ".ness" / "skills")
    skills = loader.load()
    assert skills == {}


def test_skill_loader_no_fallback_from_body_for_description(tmp_path: Path):
    """Body heading must NOT be used as description when frontmatter lacks it."""
    skill_dir = tmp_path / ".ness" / "skills" / "skill_x"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: skill_x\n---\n# Heading should not become description\n"
    )

    loader = SkillLoader(tmp_path / ".ness" / "skills")
    skills = loader.load()
    assert skills == {}


def test_skill_loader_optional_fields_default_to_empty(tmp_path: Path):
    skill_dir = tmp_path / ".ness" / "skills" / "minimal"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: minimal\ndescription: Minimal skill\n---\nBody\n"
    )

    loader = SkillLoader(tmp_path / ".ness" / "skills")
    skills = loader.load()
    s = skills["minimal"]
    assert s["license"] == ""
    assert s["compatibility"] == ""
    assert s["metadata"] == {}
    assert s["body"] == "Body"


def test_skill_loader_loads_multiple_skills_skips_invalid(tmp_path: Path):
    skills_root = tmp_path / ".ness" / "skills"
    (skills_root / "valid").mkdir(parents=True)
    (skills_root / "valid" / "SKILL.md").write_text(
        "---\nname: good\ndescription: Good skill\n---\nGood body\n"
    )
    (skills_root / "invalid").mkdir(parents=True)
    (skills_root / "invalid" / "SKILL.md").write_text(
        "---\ndescription: Missing name\n---\nBad body\n"
    )

    loader = SkillLoader(skills_root)
    skills = loader.load()
    assert list(skills) == ["good"]


# ---------------------------------------------------------------------------
# SkillLoader.render_catalog format tests
# ---------------------------------------------------------------------------

def test_render_catalog_includes_path(tmp_path: Path):
    skill_dir = tmp_path / ".ness" / "skills" / "fmt_skill"
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\nname: fmt_skill\ndescription: A formatted skill\n---\nBody\n"
    )

    loader = SkillLoader(tmp_path / ".ness" / "skills")
    skills = loader.load()
    catalog = loader.render_catalog(skills)
    # Should contain name: description: path
    assert "fmt_skill" in catalog
    assert "A formatted skill" in catalog
    assert "SKILL.md" in catalog or str(skill_md) in catalog


def test_render_catalog_empty_when_no_skills():
    loader = SkillLoader()
    assert loader.render_catalog({}) == ""


# ---------------------------------------------------------------------------
# skill_view tool tests
# ---------------------------------------------------------------------------

def test_skill_view_returns_skill_content(tmp_path: Path):
    from liteharness.session_context import SessionContext, set_session_context, reset_session_context

    skill_dir = tmp_path / ".ness" / "skills" / "view_me"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: view_me\ndescription: View this\n---\n# View Me\n\nContent here.\n"
    )

    # Create mock config with skill_loader
    loader = SkillLoader(tmp_path / ".ness" / "skills")
    all_skills = loader.load()

    rt = SessionContext(
        permissions=MagicMock(),
        options=MagicMock(),
        thread_store=MagicMock(),
        ness_dir=tmp_path / ".ness",
        project_root=tmp_path,
        agent_config=MagicMock(),
        all_skills=all_skills,
    )
    token = set_session_context(rt)
    try:
        from liteharness.tools.skill import skill_view

        result = skill_view.invoke({"name": "view_me"})
        data = json.loads(result)
        assert "View Me" in data["content"]
        assert "Content here." in data["content"]
        assert isinstance(data["linked_files"], dict)
        assert data["usage_hint"] == "To view linked files, call read(file_path) tool"
    finally:
        reset_session_context(token)


def test_skill_view_returns_linked_files(tmp_path: Path):
    from liteharness.session_context import SessionContext, set_session_context, reset_session_context

    skill_dir = tmp_path / ".ness" / "skills" / "linked"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: linked\ndescription: Has linked files\n---\nBody\n")

    # Create supporting directories
    ref_dir = skill_dir / "references"
    ref_dir.mkdir()
    (ref_dir / "guide.md").write_text("# Guide")
    (ref_dir / "sources.md").write_text("# Sources")

    script_dir = skill_dir / "scripts"
    script_dir.mkdir()
    (script_dir / "build.sh").write_text("echo build")

    loader = SkillLoader(tmp_path / ".ness" / "skills")
    all_skills = loader.load()

    rt = SessionContext(
        permissions=MagicMock(),
        options=MagicMock(),
        thread_store=MagicMock(),
        ness_dir=tmp_path / ".ness",
        project_root=tmp_path,
        agent_config=MagicMock(),
        all_skills=all_skills,
    )
    token = set_session_context(rt)
    try:
        from liteharness.tools.skill import skill_view

        result = skill_view.invoke({"name": "linked"})
        data = json.loads(result)
        lf = data["linked_files"]
        assert "references" in lf
        assert "scripts" in lf
        assert "templates" not in lf
        assert "assets" not in lf
        assert len(lf["references"]) == 2
        assert any("guide.md" in p for p in lf["references"])
        assert any("sources.md" in p for p in lf["references"])
        assert any("build.sh" in p for p in lf["scripts"])
        # Paths should be absolute
        for p in lf["references"]:
            assert Path(p).is_absolute()
    finally:
        reset_session_context(token)


def test_skill_view_unknown_skill(tmp_path: Path):
    from liteharness.session_context import SessionContext, set_session_context, reset_session_context

    rt = SessionContext(
        permissions=MagicMock(),
        options=MagicMock(),
        thread_store=MagicMock(),
        ness_dir=tmp_path / ".ness",
        project_root=tmp_path,
        agent_config=MagicMock(),
        all_skills={},
    )
    token = set_session_context(rt)
    try:
        from liteharness.tools.skill import skill_view

        result = skill_view.invoke({"name": "nonexistent"})
        assert result.startswith("Error: unknown skill")
    finally:
        reset_session_context(token)


# ---------------------------------------------------------------------------
# Tools registry: skill_view is always available
# ---------------------------------------------------------------------------

def test_skill_view_registered():
    from liteharness.tools import LOCAL_TOOLS, SMALL_ALWAYS_ON, READ_ONLY_TOOLS, TOOL_NAMES
    assert any(t.name == "skill_view" for t in LOCAL_TOOLS)
    assert "skill_view" in SMALL_ALWAYS_ON
    assert "skill_view" in READ_ONLY_TOOLS
    assert "skill_view" in TOOL_NAMES


# ---------------------------------------------------------------------------
# Overlay / L3 sections tests
# ---------------------------------------------------------------------------

def test_overlay_context_has_new_fields():
    ctx = OverlayContext(
        thread_id="t1",
        agent_mode="act",
        messages=[],
        todos=[],
        session_memory="",
        compaction_note="",
        mode_switch="",
    )
    assert ctx.activate_skills == []
    assert ctx.loaded_skills == []


def test_coding_overlay_skill_request_section():
    co = CodingOverlay()
    from liteharness.graph.state import AgentState

    state: AgentState = {}
    ctx = OverlayContext(
        thread_id="t1",
        agent_mode="act",
        messages=[],
        todos=[],
        session_memory="",
        compaction_note="",
        mode_switch="",
        activate_skills=["api_endpoint", "react_component"],
    )
    sections = co.sections(state, ctx)
    assert "skill_request" in sections
    text = sections["skill_request"]
    assert "SKILL REQUEST" in text
    assert '"api_endpoint"' in text
    assert '"react_component"' in text
    assert "skill_view" in text


def test_coding_overlay_skill_request_empty_when_none():
    co = CodingOverlay()
    state: dict = {}
    ctx = OverlayContext(
        thread_id="t1",
        agent_mode="act",
        messages=[],
        todos=[],
        session_memory="",
        compaction_note="",
        mode_switch="",
        activate_skills=[],
    )
    sections = co.sections(state, ctx)
    assert "skill_request" not in sections


def test_coding_overlay_loaded_skills_section():
    co = CodingOverlay()
    state: dict = {}
    ctx = OverlayContext(
        thread_id="t1",
        agent_mode="act",
        messages=[],
        todos=[],
        session_memory="",
        compaction_note="",
        mode_switch="",
        loaded_skills=[
            {"name": "sk1", "description": "desc1", "path": "/path/to/sk1"},
            {"name": "sk2", "description": "desc2", "path": "/path/to/sk2"},
        ],
    )
    sections = co.sections(state, ctx)
    assert "loaded_skills" in sections
    text = sections["loaded_skills"]
    assert "LOADED SKILLS" in text
    assert "- sk1: desc1: /path/to/sk1" in text
    assert "- sk2: desc2: /path/to/sk2" in text


def test_coding_overlay_loaded_skills_empty_when_none():
    co = CodingOverlay()
    state: dict = {}
    ctx = OverlayContext(
        thread_id="t1",
        agent_mode="act",
        messages=[],
        todos=[],
        session_memory="",
        compaction_note="",
        mode_switch="",
        loaded_skills=[],
    )
    sections = co.sections(state, ctx)
    assert "loaded_skills" not in sections


def test_overlay_delta_renders_newly_loaded_skills():
    co = CodingOverlay()
    state: dict = {}

    ctx_t1 = OverlayContext(
        thread_id="t1",
        agent_mode="act",
        messages=[],
        todos=[],
        session_memory="",
        compaction_note="",
        mode_switch="",
        loaded_skills=[{"name": "sk1", "description": "d1", "path": "p1"}],
    )
    sections_t1 = co.sections(state, ctx_t1)

    ctx_t2 = OverlayContext(
        thread_id="t1",
        agent_mode="act",
        messages=[],
        todos=[],
        session_memory="",
        compaction_note="",
        mode_switch="",
        loaded_skills=[
            {"name": "sk1", "description": "d1", "path": "p1"},
            {"name": "sk2", "description": "d2", "path": "p2"},
        ],
    )
    sections_t2 = co.sections(state, ctx_t2)

    # Delta from t1 to t2 includes the entire loaded_skills section
    # because the section content changed (new skill was added)
    delta = render_overlay_delta(sections_t2, sections_t1)
    assert "sk1" in delta
    assert "sk2" in delta
    assert "LOADED SKILLS" in delta


def test_coding_overlay_section_order():
    co = CodingOverlay()
    state: dict = {}
    ctx = OverlayContext(
        thread_id="t1",
        agent_mode="act",
        messages=[],
        todos=[],
        session_memory="mem",
        compaction_note="cp",
        mode_switch="ms",
        activate_skills=["a"],
        git_snapshot="branch: main",
        loaded_skills=[{"name": "x", "description": "y", "path": "z"}],
    )
    sections = co.sections(state, ctx)
    keys = list(sections.keys())
    # skill_request should come first
    assert keys.index("skill_request") < keys.index("mode_switch")
    assert keys.index("loaded_skills") > keys.index("session_memory")
