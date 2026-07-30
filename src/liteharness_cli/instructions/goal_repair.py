GOAL_GENERIC_REPAIR = (
    "Re-check the deliverable and provide explicit verification evidence."
)

GOAL_REPAIR = """Continue working toward the original goal.

Original goal:
{goal}

Independent judge feedback:
{repair}

Address the feedback and verify the result with concrete evidence. Do not
weaken the goal's acceptance criteria or skip required verification.
"""
