from __future__ import annotations

from pathlib import Path

from liteharness.context.layers import PromptLayers, PromptLayersConfig, AuxPrompts
from liteharness.instructions import INIT_MEMORY
from liteharness.options import ModeConfig


def default_prompt_layers(*, l2_context: str | None = None, **overrides) -> PromptLayers:
    """SDK :class:`PromptLayersConfig` defaults, with optional L2 override."""
    return PromptLayers(PromptLayersConfig(l2_context=l2_context, **overrides))


def default_aux_prompts() -> AuxPrompts:
    """SDK :class:`AuxPrompts` (compaction / reflection / subagent / …)."""
    return AuxPrompts()


def plan_act_modes(*, plans_dir: Path | None = None) -> ModeConfig:
    """Plan/act mode config; templates fall through to CodingOverlay defaults."""
    return ModeConfig(plans_dir=plans_dir)


def build_init_memory_prompt(project_context: str) -> str:
    """Format the SDK init-memory template for ``/memory create``."""
    return INIT_MEMORY.format(project_context=project_context)
