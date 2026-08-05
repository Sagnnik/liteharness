from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from ness_agent.skills import SkillLoader, merge_skill_dirs


def _write_skill(skill_dir: Path, name: str, description: str, body: str = "Body") -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n"
    )


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


def test_skill_loader_disabled_when_no_dirs():
    assert SkillLoader().load() == {}
    assert SkillLoader(skills_dirs=None).load() == {}
    assert SkillLoader(skills_dirs=[]).load() == {}


def test_skill_loader_multi_root(tmp_path: Path):
    ness = tmp_path / ".ness" / "skills"
    agents = tmp_path / ".agents" / "skills"
    _write_skill(ness / "alpha", "alpha", "From ness")
    _write_skill(agents / "beta", "beta", "From agents")

    loader = SkillLoader(skills_dirs=[ness, agents])
    skills = loader.load()
    assert set(skills) == {"alpha", "beta"}
    assert "ness" in skills["alpha"]["source"]
    assert "agents" in skills["beta"]["source"]


def test_skill_loader_nested_category(tmp_path: Path):
    root = tmp_path / ".agents" / "skills"
    _write_skill(root / "product-a" / "skill-one", "skill_one", "Nested one")
    _write_skill(root / "product-a" / "skill-two", "skill_two", "Nested two")
    _write_skill(root / "flat-skill", "flat", "Flat skill")

    skills = SkillLoader(root).load()
    assert set(skills) == {"skill_one", "skill_two", "flat"}


def test_skill_loader_shadowing_does_not_descend(tmp_path: Path):
    root = tmp_path / ".agents" / "skills"
    foo = root / "foo"
    _write_skill(foo, "outer", "Outer skill", body="Outer body")
    _write_skill(foo / "inner", "inner", "Inner skill", body="Inner body")
    (foo / "scripts").mkdir()
    (foo / "scripts" / "run.sh").write_text("echo hi")

    skills = SkillLoader(root).load()
    assert set(skills) == {"outer"}
    assert "Outer body" in skills["outer"]["body"]


def test_skill_loader_user_dir_wins_name_collision(tmp_path: Path):
    ness = tmp_path / ".ness" / "skills"
    agents = tmp_path / ".agents" / "skills"
    _write_skill(ness / "shared", "shared", "Ness wins", body="from-ness")
    _write_skill(agents / "shared", "shared", "Agents loses", body="from-agents")

    skills = SkillLoader(skills_dirs=[ness, agents]).load()
    assert skills["shared"]["body"] == "from-ness"
    assert "ness" in skills["shared"]["source"]


def test_skill_loader_project_wins_over_global(tmp_path: Path, monkeypatch):
    project = tmp_path / "proj"
    home = tmp_path / "home"
    project.mkdir()
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    project_agents = project / ".agents" / "skills"
    global_agents = home / ".agents" / "skills"
    _write_skill(project_agents / "shared", "shared", "Project", body="project")
    _write_skill(global_agents / "shared", "shared", "Global", body="global")
    _write_skill(global_agents / "only_global", "only_global", "Global only")

    dirs = merge_skill_dirs(project, project / ".ness" / "skills")
    skills = SkillLoader(skills_dirs=dirs).load()
    assert skills["shared"]["body"] == "project"
    assert "only_global" in skills


def test_skill_loader_symlink_dedupes_resolved_path(tmp_path: Path):
    canonical = tmp_path / ".agents" / "skills"
    linked_root = tmp_path / ".claude" / "skills"
    _write_skill(canonical / "dup", "dup", "Canonical", body="once")
    linked_root.parent.mkdir(parents=True, exist_ok=True)
    linked_root.symlink_to(canonical)

    skills = SkillLoader(skills_dirs=[canonical, linked_root]).load()
    assert list(skills) == ["dup"]


def test_merge_skill_dirs_order_and_dedupe(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    project = tmp_path / "proj"
    project.mkdir()
    user = project / ".ness" / "skills"

    dirs = merge_skill_dirs(project, user)
    assert dirs[0] == user
    assert dirs[1] == (project / ".agents" / "skills")
    assert (home / ".agents" / "skills") in dirs
    # No duplicate resolved paths
    resolved = [p.resolve() for p in dirs]
    assert len(resolved) == len(set(resolved))


def test_skill_view_returns_skill_content(tmp_path: Path):
    from ness_agent.session_context import SessionContext, set_session_context, reset_session_context

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
        from ness_agent.tools.skill import skill_view

        result = skill_view.invoke({"name": "view_me"})
        data = json.loads(result)
        assert "View Me" in data["content"]
        assert "Content here." in data["content"]
        assert isinstance(data["linked_files"], dict)
        assert data["usage_hint"] == "To view linked files, call read(path=...) tool"
    finally:
        reset_session_context(token)


def test_skill_view_returns_linked_files(tmp_path: Path):
    from ness_agent.session_context import SessionContext, set_session_context, reset_session_context

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
        from ness_agent.tools.skill import skill_view

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
    from ness_agent.session_context import SessionContext, set_session_context, reset_session_context

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
        from ness_agent.tools.skill import skill_view

        result = skill_view.invoke({"name": "nonexistent"})
        assert result.startswith("Error: unknown skill")
    finally:
        reset_session_context(token)


def test_skill_view_registered():
    from ness_agent.tools import BUILTIN_TOOLS, ALWAYS_ON, READ_ONLY_TOOLS, TOOL_NAMES
    assert any(t.name == "skill_view" for t in BUILTIN_TOOLS)
    assert "skill_view" in ALWAYS_ON
    assert "skill_view" in READ_ONLY_TOOLS
    assert "skill_view" in TOOL_NAMES
