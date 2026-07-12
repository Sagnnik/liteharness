from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from liteharness.context.layers import PromptLayers, PromptLayersConfig, TaskPrompts, messages_to_text
from liteharness.options import ModeConfig

INSTRUCTIONS_DIR = Path(__file__).resolve().parents[1] / "liteharness" / "instructions"
DEFAULT_PERSONA = "You are an expert software engineer working inside the user's repository."


@lru_cache(maxsize=None)
def load_instruction(stem: str) -> str:
    ins_path = INSTRUCTIONS_DIR / f"{stem}_instructions.md"
    return ins_path.read_text(encoding="utf-8").strip()


CODING_L0 = load_instruction("l0_harness")


def default_prompt_layers(*, persona: str = DEFAULT_PERSONA, l2_context: str | None = None) -> PromptLayers:
    return PromptLayers(
        PromptLayersConfig(
            l0=CODING_L0,
            persona=persona,
            l1_template=load_instruction("l1_profile"),
            l2_context=l2_context,
            l2_header="PROJECT CONTEXT",
            include_git_flag=True,
        )
    )


def default_task_prompts() -> TaskPrompts:
    return TaskPrompts(
        compaction=load_instruction("compaction"),
        reflection=load_instruction("reflection"),
        subagent=load_instruction("subagent"),
        thread_summary=load_instruction("thread_summary"),
        init_memory=load_instruction("init_memory"),
    )


def plan_act_modes(*, plans_dir: Path | None = None) -> ModeConfig:
    return ModeConfig(
        default="act",
        plans_dir=plans_dir,
        plan_mode_template=load_instruction("plan_mode"),
        act_mode_template=load_instruction("act_mode"),
        reject_state_changing_tools_in_plan=True,
    )


def build_subagent_prompt(agent_name, agent_body, parent_context=""):
    return load_instruction("subagent").format(
        agent_name=agent_name,
        agent_body=agent_body.strip(),
        parent_context=parent_context.strip(),
    )


def build_compaction_prompt(messages):
    return load_instruction("compaction").format(messages=messages)


def build_reflection_prompt(
    thread_id,
    messages,
    user_message_count,
    *,
    current_session_bullets="",
    todos="",
):
    return load_instruction("reflection").format(
        thread_id=thread_id,
        user_message_count=user_message_count,
        messages=messages_to_text(messages),
        current_session_bullets=current_session_bullets.strip() or "(none yet)",
        todos=todos.strip() or "No todos",
    )
