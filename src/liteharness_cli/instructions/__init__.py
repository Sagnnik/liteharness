"""CLI prompt instruction texts packed as Python string constants."""

from .goal_judge import GOAL_JUDGE
from .goal_repair import GOAL_GENERIC_REPAIR, GOAL_REPAIR

__all__ = [
    "GOAL_JUDGE",
    "GOAL_REPAIR",
    "GOAL_GENERIC_REPAIR",
]
