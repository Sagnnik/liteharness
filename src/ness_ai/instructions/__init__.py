"""Ness AI default prompt instruction texts.

Each constant is the .strip()-equivalent of the original ``_instructions.md``
file. Users can import these to inspect, copy, or modify them and feed the
result back into ``PromptLayersConfig(l0=...)``, ``AuxPrompts(compaction=...)``,
or ``CodingOverlay(plan_mode_template=...)``.
"""

from .l0_harness import L0_HARNESS
from .l1_profile import L1_PROFILE
from .plan_mode import PLAN_MODE
from .act_mode import ACT_MODE
from .compaction import COMPACTION
from .reflection import REFLECTION
from .subagent import SUBAGENT
from .thread_summary import THREAD_SUMMARY
from .init_memory import INIT_MEMORY

ALL_INSTRUCTIONS = {
    "l0_harness": L0_HARNESS,
    "l1_profile": L1_PROFILE,
    "plan_mode": PLAN_MODE,
    "act_mode": ACT_MODE,
    "compaction": COMPACTION,
    "reflection": REFLECTION,
    "subagent": SUBAGENT,
    "thread_summary": THREAD_SUMMARY,
    "init_memory": INIT_MEMORY,
}

__all__ = [
    "L0_HARNESS",
    "L1_PROFILE",
    "PLAN_MODE",
    "ACT_MODE",
    "COMPACTION",
    "REFLECTION",
    "SUBAGENT",
    "THREAD_SUMMARY",
    "INIT_MEMORY",
    "ALL_INSTRUCTIONS",
]