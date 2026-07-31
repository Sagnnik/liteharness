ACT_MODE = """MODE SWITCH: plan -> act. The user just switched from plan to act mode.

First (mandatory, before any other tool calls):
- Call `todo(todos=[...])` to record the actionable steps you will execute this session (full list replace; one item per numbered plan step, or a single todo for trivial plans).
- Derive steps from the approved plan in the conversation above.
- Do this in a tool-only message before editing files or running state-changing commands.

Then address the user's message:
- If they want the plan implemented, work through the todos in order; mark each completed as you finish.
- If their message redirects (a question, a different task, narrower scope, or they changed their mind), follow the message — do not blindly execute the full plan.
- Do not re-plan or ask for re-approval unless blocked.

Verify each step before moving on."""