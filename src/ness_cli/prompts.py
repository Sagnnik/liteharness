from __future__ import annotations

from pathlib import Path

from ness_ai.context.layers import PromptLayers, PromptLayersConfig, AuxPrompts
from ness_ai.options import ModeConfig
from ness_cli.instructions import load_instruction


def _instr(name: str, *, instructions_dir: Path | None) -> str:
    return load_instruction(name, instructions_dir=instructions_dir)


def default_prompt_layers(
    *,
    instructions_dir: Path | None = None,
    l2_context: str | None = None,
    **overrides,
) -> PromptLayers:
    """Prompt layers from global ``instructions/`` (packaged fallback)."""
    kwargs = {
        "l0": _instr("l0_harness.md", instructions_dir=instructions_dir),
        "persona": _instr("persona.md", instructions_dir=instructions_dir),
        "l2_context": l2_context,
        **overrides,
    }
    return PromptLayers(PromptLayersConfig(**kwargs))


def default_aux_prompts(*, instructions_dir: Path | None = None) -> AuxPrompts:
    """Aux prompts from global ``instructions/`` (packaged fallback)."""
    return AuxPrompts(
        compaction=_instr("compaction.md", instructions_dir=instructions_dir),
        reflection=_instr("reflection.md", instructions_dir=instructions_dir),
        subagent=_instr("subagent.md", instructions_dir=instructions_dir),
        thread_summary=_instr("thread_summary.md", instructions_dir=instructions_dir),
        init_memory=_instr("init_memory.md", instructions_dir=instructions_dir),
    )


def plan_act_modes(
    *,
    plans_dir: Path | None = None,
    instructions_dir: Path | None = None,
) -> ModeConfig:
    """Plan/act mode config with templates from global ``instructions/``."""
    return ModeConfig(
        plans_dir=plans_dir,
        plan_mode_template=_instr("plan_mode.md", instructions_dir=instructions_dir),
        act_mode_template=_instr("act_mode.md", instructions_dir=instructions_dir),
    )


def build_init_memory_prompt(
    project_context: str,
    *,
    instructions_dir: Path | None = None,
) -> str:
    """Format the init-memory template for ``/memory create``."""
    template = _instr("init_memory.md", instructions_dir=instructions_dir)
    return template.format(project_context=project_context)
