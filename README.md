# LiteHarness

LiteHarness is an experimental, hackable coding-agent harness for engineers who want to own the loop. It uses OpenRouter-compatible chat models, LangGraph for orchestration, native tool-calling by default, and filesystem-driven extension points under `.ness/`.

## Quick Start

```bash
uv sync
export OPENAI_API_KEY=...
uv run python main.py
```

Useful environment variables:

- `MODEL_NAME`: model passed to `ChatOpenRouter` (`gpt-4o-mini` by default).
- `MODE`: `json` for native tool-calling or `xml` for fallback XML tool calls.
- `ENABLE_APPROVAL`: require approval for destructive tools.
- `AUTO_SAVE_THREADS`: write thread events to `.ness/threads/`.
- `COMPACTION_TOKEN_BUDGET`: context token budget before compaction triggers (default `120000`).
- `OPENROUTER_SESSION_ID`: optional stable prompt-cache session id. Defaults to the active LiteHarness thread id.
- `NESS_DIR`: project config directory, default `.ness`.

## Architecture

- `main.py`: Rich CLI, slash commands, streaming, image/clipboard handling.
- `agent.py`: LangGraph loop: agent, approval gate, tool executor.
- `prompt.py`: generated native/XML prompts, compaction, memory, subagent prompts.
- `tools/`: local tools for files, search, shell, git, todos, and subagents.
- `permissions.py`: `.ness/permissions.json` allow/deny/ask matching.
- `hooks.py`: `.ness/hooks.json` pre/post/user/session command hooks.
- `mcp_client.py`: stdio MCP startup and namespaced MCP tool wrappers.
- `session.py`: RL-friendly JSONL thread events.
- `skill_loader.py`: `SKILL.md` skill discovery with legacy YAML fallback.

## `.ness/` Layout

```text
.ness/
├── NESS.md              Project memory loaded into prompts
├── permissions.json     Tool allow/deny/ask rules
├── hooks.json           Hook commands
├── mcp.json             MCP stdio servers
├── agents/              Subagent definitions
├── commands/            User slash commands
├── skills/              Project-local SKILL.md skills
├── threads/             Saved JSONL trajectories
├── worktrees/           Optional subagent worktrees
└── shells/              Background shell logs
```

## Skills

The canonical skill format is folder-based:

```text
skills/react_component/SKILL.md
.ness/skills/my_project_skill/SKILL.md
```

Each `SKILL.md` may include YAML frontmatter:

```markdown
---
name: react_component
description: Create React components matching project conventions.
triggers: [react, component, tsx]
---
# React Component

Skill instructions go here.
```

Legacy `skills/*.yaml` and `skills/*.yml` files still load as compatibility fallback.

## Permissions

`.ness/permissions.json` uses glob-style rules:

```json
{
  "allow": ["read_file:*", "grep:*", "bash:git status*"],
  "deny": ["bash:rm -rf*", "bash:sudo*"],
  "ask": ["*"]
}
```

Deny rules win over allow rules. Approval choices are `y`, `n`, `a` always, `N` never, `d` diff, and `s` show args.

## MCP

Configure stdio MCP servers in `.ness/mcp.json`:

```json
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "env": {}
    }
  }
}
```

Tools are exposed as `mcp__<server>__<tool>`. Server startup failures are shown by `/mcp` and do not stop the CLI.

## Subagents

Subagents live in `.ness/agents/<name>.md`:

```markdown
---
tools: [read_file, grep, glob_files, list_files]
worktree: false
---
You are a read-only explorer. Return concise findings with file references.
```

The `spawn_subagent` tool runs a filtered, isolated graph and returns a summary to the parent agent.

## Slash Commands

- `/context`: show project context.
- `/cost`: show token/cost totals.
- `/cache`: show prompt-cache reads/writes and cache hit rate.
- `/skills`: list loaded skills and warnings.
- `/init [force]`: generate `.ness/NESS.md`.
- `/memory` or `/memory add <note>`: read or append memory.
- `/permissions`: list/edit permission rules.
- `/hooks`: list hooks.
- `/mcp`: list MCP server status and tools.
- `/threads`: list saved sessions.
- `/resume <thread_id>`: resume a saved thread.
- `/compact`: mark/manual compaction request.
- `/copy`, `/copy code`, `/copy <n>`: copy assistant output.
- `/image <path>`: attach an image to the next prompt.
- `/save-threads on|off|status`: toggle thread autosave.

Markdown files under `.ness/commands/*.md` become project-local slash commands. Their body is used as a prompt template with `{{args}}` substitution.

## Thread Events

When autosave is on, LiteHarness appends JSONL events to `.ness/threads/<thread_id>.jsonl`:

```json
{"kind": "user", "content": "..."}
{"kind": "assistant", "content": "...", "tool_calls": []}
{"kind": "tool", "tool": "read_file", "args": {}, "result": "...", "duration_ms": 10, "exit": "ok"}
{"kind": "approval", "tool": "edit_file", "decision": "yes"}
{"kind": "usage", "model": "gpt-4o-mini", "input_tokens": 100, "cached_input_tokens": 40, "cache_write_tokens": 10, "output_tokens": 20, "cost_usd": 0.0001, "cost_source": "provider"}
```

This keeps trajectories readable for debugging and usable for later offline analysis.

## Verification

```bash
OPENAI_API_KEY=test uv run python -m compileall -q .
OPENAI_API_KEY=test uv run python -m unittest discover -s tests -v
```
