"""CLI instruction packaging and global-config loading."""

from __future__ import annotations

from pathlib import Path

from ness_ai.instructions import (
    ACT_MODE,
    COMPACTION,
    INIT_MEMORY,
    L0_HARNESS,
    PLAN_MODE,
    REFLECTION,
    SUBAGENT,
    THREAD_SUMMARY,
)
from ness_cli.instructions import (
    INSTRUCTION_FILES,
    default_instruction_files,
    load_instruction,
    packaged_instruction,
)
from ness_cli.prompts import (
    build_init_memory_prompt,
    default_aux_prompts,
    default_prompt_layers,
    plan_act_modes,
)

_SDK_PARITY = {
    "l0_harness.md": L0_HARNESS,
    "plan_mode.md": PLAN_MODE,
    "act_mode.md": ACT_MODE,
    "compaction.md": COMPACTION,
    "reflection.md": REFLECTION,
    "subagent.md": SUBAGENT,
    "thread_summary.md": THREAD_SUMMARY,
    "init_memory.md": INIT_MEMORY,
}


def test_default_instruction_files_cover_expected_set() -> None:
    files = default_instruction_files()
    assert set(files) == set(INSTRUCTION_FILES)
    for name, content in files.items():
        assert content.strip(), name


def test_packaged_sdk_instruction_parity() -> None:
    for name, expected in _SDK_PARITY.items():
        assert packaged_instruction(name) == expected.strip()


def test_load_instruction_prefers_global_file(tmp_path: Path) -> None:
    from ness_cli.instructions import clear_instruction_cache

    clear_instruction_cache()
    custom = "custom harness rules"
    (tmp_path / "l0_harness.md").write_text(custom + "\n", encoding="utf-8")
    assert load_instruction("l0_harness.md", instructions_dir=tmp_path) == custom
    # Second call hits cache with the same result.
    assert load_instruction("l0_harness.md", instructions_dir=tmp_path) == custom


def test_load_instruction_falls_back_to_packaged(tmp_path: Path) -> None:
    from ness_cli.instructions import clear_instruction_cache

    clear_instruction_cache()
    assert (
        load_instruction("persona.md", instructions_dir=tmp_path)
        == packaged_instruction("persona.md")
    )


def test_default_prompt_layers_reads_global(tmp_path: Path) -> None:
    (tmp_path / "l0_harness.md").write_text("L0 FROM DISK\n", encoding="utf-8")
    (tmp_path / "persona.md").write_text("PERSONA FROM DISK\n", encoding="utf-8")
    layers = default_prompt_layers(instructions_dir=tmp_path)
    assert layers.build_l0() == "L0 FROM DISK"
    assert layers.config.persona == "PERSONA FROM DISK"


def test_default_aux_and_modes_read_global(tmp_path: Path) -> None:
    for name in (
        "compaction.md",
        "reflection.md",
        "subagent.md",
        "thread_summary.md",
        "init_memory.md",
        "plan_mode.md",
        "act_mode.md",
    ):
        (tmp_path / name).write_text(f"BODY:{name}\n", encoding="utf-8")
    aux = default_aux_prompts(instructions_dir=tmp_path)
    assert aux.compaction == "BODY:compaction.md"
    assert aux.reflection == "BODY:reflection.md"
    modes = plan_act_modes(plans_dir=tmp_path / "plans", instructions_dir=tmp_path)
    assert modes.plan_mode_template == "BODY:plan_mode.md"
    assert modes.act_mode_template == "BODY:act_mode.md"
    prompt = build_init_memory_prompt("ctx", instructions_dir=tmp_path)
    assert prompt == "BODY:init_memory.md"


def test_goal_templates_load_from_instructions_dir(tmp_path: Path) -> None:
    from ness_cli.goal import GoalCoordinator

    (tmp_path / "goal_judge.md").write_text(
        "JUDGE {goal} {attempt}/{max_attempts} {validation} {start_seq} {transcript}\n",
        encoding="utf-8",
    )
    (tmp_path / "goal_repair.md").write_text(
        "REPAIR {goal} :: {repair}\n",
        encoding="utf-8",
    )
    (tmp_path / "goal_generic_repair.md").write_text("GENERIC\n", encoding="utf-8")

    store = type(
        "Store",
        (),
        {"load_thread_events_since": staticmethod(lambda *_a, **_k: [])},
    )()
    coding = type("Coding", (), {"thread_store": store, "thread_id": "t1"})()
    coordinator = GoalCoordinator(coding, instructions_dir=tmp_path)
    prompt = coordinator._build_judge_prompt("ship it", 1, 0, "ok")
    assert prompt.startswith("JUDGE ship it 1/")
    assert coordinator._instruction("goal_generic_repair.md") == "GENERIC"
    assert "REPAIR ship it :: fix" in coordinator._instruction("goal_repair.md").format(
        goal="ship it",
        repair="fix",
    )


def test_build_coding_session_threads_instructions_dir(
    tmp_path: Path, monkeypatch
) -> None:
    from ness_cli.factory import build_coding_session
    from ness_cli.paths import resolve_paths, ensure_global_config

    monkeypatch.setenv("NESS_AI_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("NESS_AI_CACHE_DIR", str(tmp_path / "cache"))
    project = tmp_path / "repo"
    project.mkdir()
    monkeypatch.chdir(project)

    # Custom instructions dir distinct from env default would be via NessPaths;
    # resolve_paths uses NESS_AI_CONFIG_DIR, then we pass that NessPaths through.
    paths = resolve_paths(project_root=project)
    ensure_global_config(paths)
    custom = paths.instructions_dir
    (custom / "init_memory.md").write_text(
        "CUSTOM INIT {project_context}\n", encoding="utf-8"
    )
    (custom / "goal_judge.md").write_text(
        "CUSTOM JUDGE {goal} {attempt}/{max_attempts} {validation} {start_seq} {transcript}\n",
        encoding="utf-8",
    )

    coding = build_coding_session(thread_id="t-instr", paths=paths)
    assert coding.instructions_dir == custom
    assert "CUSTOM INIT" in build_init_memory_prompt(
        "ctx", instructions_dir=coding.instructions_dir
    )

    from ness_cli.goal import GoalCoordinator

    coding.thread_store = type(
        "Store",
        (),
        {"load_thread_events_since": staticmethod(lambda *_a, **_k: [])},
    )()
    coordinator = GoalCoordinator(
        coding, instructions_dir=coding.instructions_dir
    )
    assert coordinator._build_judge_prompt("g", 1, 0, "v").startswith("CUSTOM JUDGE")
