from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NessAgentOptions:
    """Runtime knobs including compaction budget (context window + reserves)."""

    context_window: int | None = None
    compaction_token_budget: int = 120_000
    compaction_output_reserve_tokens: int = 8_192
    compaction_input_reserve_tokens: int = 4_096
    enable_approval: bool = True
    auto_save_threads: bool = True
    reflection_token_ratio: float = 0.0
    session_end_reflection: bool = False
    format_on_write: bool = True
    exa_api_key: str | None = None
    project_root: Path | None = None
    ness_dir: Path | None = None
    # AIMessage cap text injected on a pure-LLM cancel with no partial text or
    # pending tool calls, so the model does not silently resume the abandoned
    # request next turn
    interruption_marker: str = ("… [turn interrupted by user] ")


@dataclass
class MemoryConfig:
    disabled: bool = False
    project_memory: Path | None = None
    user_memory: Path | None = None
    session_memory_dir: Path | None = None


@dataclass
class ModeConfig:
    """Optional plan/act."""
    default: str = "act"
    plans_dir: Path | None = None
    plan_mode_template: str | None = None
    act_mode_template: str | None = None
    reject_state_changing_tools_in_plan: bool = True


@dataclass
class SubagentConfig:
    """Optional parallel sub-agent tasks (spawn_subagent tool uses this)."""
    prompt_template: str | None = None   # default subagent prompt body
    max_parallel: int = 3
    default_tools: tuple[str, ...] = ("read", "grep", "glob", "web_search", "webfetch")
    default_timeout_seconds: int = 300


@dataclass
class PermissionPolicy:
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    ask: list[str] = field(default_factory=lambda: ["*"])
