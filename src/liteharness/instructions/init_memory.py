INIT_MEMORY = """Draft a concise `.ness/NESS.md` project conventions file from this context.

Treat it like CLAUDE.md or AGENTS.md: durable repo rules a human can review and edit.
Include project purpose, commands, architecture notes, coding conventions, and gotchas.
Do not include session progress, todos, or volatile task state.
Keep it scannable with short sections and bullet lists where helpful.

Project context:
{project_context}"""