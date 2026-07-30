from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from liteharness.skills import SkillLoader


def test_skill_loader_skips_invalid_name(tmp_path: Path):
    """Missing or empty frontmatter name is skipped; dir name is not a fallback."""
    skills_root = tmp_path / ".ness" / "skills"
    missing = skills_root / "no_name"
    missing.mkdir(parents=True)
    (missing / "SKILL.md").write_text("---\ndescription: No name here\n---\nBody\n")
    empty = skills_root / "empty_name"
    empty.mkdir(parents=True)
    (empty / "SKILL.md").write_text("---\nname: ''\ndescription: Empty name\n---\nBody\n")
    dir_only = skills_root / "dir_name_skill"
    dir_only.mkdir(parents=True)
    (dir_only / "SKILL.md").write_text(
        "---\ndescription: Dir name should not become name\n---\nBody\n"
    )

    loader = SkillLoader(skills_root)
    assert loader.load() == {}


def test_skill_loader_skips_invalid_description(tmp_path: Path):
    """Missing/empty description is skipped; body heading is not a fallback."""
    skills_root = tmp_path / ".ness" / "skills"
    missing = skills_root / "no_desc"
    missing.mkdir(parents=True)
    (missing / "SKILL.md").write_text("---\nname: no_desc\n---\nBody\n")
    empty = skills_root / "empty_desc"
    empty.mkdir(parents=True)
    (empty / "SKILL.md").write_text("---\nname: empty_desc\ndescription: ''\n---\nBody\n")
    body_heading = skills_root / "skill_x"
    body_heading.mkdir(parents=True)
    (body_heading / "SKILL.md").write_text(
        "---\nname: skill_x\n---\n# Heading should not become description\n"
    )

    loader = SkillLoader(skills_root)
    assert loader.load() == {}


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


def test_render_catalog_empty_when_no_skills():
    loader = SkillLoader()
    assert loader.render_catalog({}) == ""


def test_skill_view_returns_skill_content(tmp_path: Path):
    from liteharness.session_context import SessionContext, set_session_context, reset_session_context

    skill_dir = tmp_path / ".ness" / "skills" / "view_me"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: view_me\ndescription: View this\n---\n# View Me\n\nContent here.\n"
    )

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
        assert data["usage_hint"] == "To view linked files, call read(path=...) tool"
    finally:
        reset_session_context(token)


def test_skill_view_returns_linked_files(tmp_path: Path):
    from liteharness.session_context import SessionContext, set_session_context, reset_session_context

    skill_dir = tmp_path / ".ness" / "skills" / "linked"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: linked\ndescription: Has linked files\n---\nBody\n")

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


def test_skill_view_registered():
    from liteharness.tools import BUILTIN_TOOLS, ALWAYS_ON, READ_ONLY_TOOLS, TOOL_NAMES
    assert any(t.name == "skill_view" for t in BUILTIN_TOOLS)
    assert "skill_view" in ALWAYS_ON
    assert "skill_view" in READ_ONLY_TOOLS
    assert "skill_view" in TOOL_NAMES
