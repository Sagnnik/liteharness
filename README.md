# LiteHarness

LiteHarness is an experimental, hackable coding-agent harness for engineers who want to own the loop. It uses OpenRouter-compatible chat models, LangGraph for orchestration, native tool-calling by default, and filesystem-driven extension points under `.ness/`.

## Quick Start

```bash
uv sync
export OPENAI_API_KEY=...
uv run python cli/main.py
```

Useful environment variables:

- `MODEL_NAME`: model passed to `ChatOpenRouter` (`gpt-4o-mini` by default).
- `MODE`: `json` for native tool-calling or `xml` for fallback XML tool calls.
- `ENABLE_APPROVAL`: require approval for destructive tools.
- `AUTO_SAVE_THREADS`: write thread events to `.ness/threads/`.
- `REFLECTION_INTERVAL`: user turns between background NESS.md reflection runs (default `5`; set `0` to disable).
- `COMPACTION_TOKEN_BUDGET`: context token budget before compaction triggers (default `120000`).
- `OPENROUTER_SESSION_ID`: optional stable prompt-cache session id. Defaults to the active LiteHarness thread id.
- `OPENAI_BASE_URL`: optional custom OpenAI-compatible base URL.
- `FORMAT_ON_WRITE`: auto-format supported file types after writes (default `true`).
- `NESS_DIR`: project config directory, default `.ness`.

## Architecture

- `cli/main.py`: Rich CLI, slash commands, streaming, image/clipboard handling.
- `agent.py`: LangGraph loop: agent, approval gate, tool executor.
- `context.py`: layered prompt assembly from `instructions/` templates.
- `instructions/`: markdown templates for foundation, modes, compaction, reflection, subagents, and XML fallback.
- `compaction.py`: progressive context compaction by token tier.
- `reflection.py`: background NESS.md maintenance gate with bounded tool calls.
- `memory.py`: NESS.md, USER.md, and LOG.md helpers.
- `tools/`: local tools for files, search, shell, git, todos, and subagents.
- `permissions.py`: `.ness/permissions.json` allow/deny/ask matching.
- `hooks.py`: `.ness/hooks.json` pre/post/user/session command hooks.
- `mcp_client.py`: stdio MCP startup and namespaced MCP tool wrappers.
- `session.py`: JSONL thread events and `index.json` session metadata.
- `skill_loader.py`: `SKILL.md` skill discovery under `.ness/skills/`.
- `config.py`: settings, model pricing, and cost/cache tracking.
- `parsers.py`: native and XML tool-call extraction.

## Prompt Layers

LiteHarness splits context into three layers to keep prompt caching stable:

1. **L1 foundation** (`build_foundation`): identity, universal rules, tool catalog, and `USER.md` preferences. Cached until memory files or tool set change.
2. **L2 project context** (`build_project_context_block`): repo context, `.ness/NESS.md`, git availability, and sticky skill cores.
3. **L3 working state** (`build_working_state_overlay`): appended to the latest user message each turn. Includes agent mode, todos, and reflection nudges.

Skills activate by trigger match and stay sticky for the session once loaded.

## Agent Modes

- **Normal** (`/act`): full tool set. Git read tools appear only inside a git repo.
- **Plan** (`/plan`): read-only tools. Assistant output is saved under `.ness/plans/`. Use `/act` to switch back and execute.

Tool tiers in normal mode:

- Small always-on: `todo_read`, `todo_write`
- L1 core: file, search, shell, and project-context tools
- L2 git read: `git_status`, `git_diff`, `git_log`, `git_show`, `git_blame`
- L3 advanced: git write/worktree tools and `spawn_subagent`
- Dynamic MCP: any `mcp__*` tool registered at startup

## Memory

Three memory files live under `.ness/`:

| File | Purpose |
|------|---------|
| `NESS.md` | Durable project facts: conventions, architecture, commands, gotchas. Loaded into L2 prompts, maintained by the reflection gate, with human overrides via `/memory add`. |
| `USER.md` | Cross-repo user preferences. Human-authored via `/user`; loaded into L1 foundation. |
| `LOG.md` | Episodic per-session notes (helpers present; not yet loaded into prompts). |

Reflection runs in the background every `REFLECTION_INTERVAL` user turns. It uses a bounded tool loop (`read_memory`, `add_to_memory`, `edit_memory`) and keeps `NESS.md` under 12,000 characters.

## Compaction

When context approaches `COMPACTION_TOKEN_BUDGET` (80% safety margin), compaction runs progressively:

| Tier | Token range | Strategy |
|------|-------------|----------|
| 0 | < 8k | No compaction |
| 1 | < 16k | Compact large tool outputs |
| 2 | < 32k | Summarize older history, keep last 10 messages |
| 3 | < 64k | Summarize older history, keep last 6 messages |
| 4 | > 64k | Summarize older history, keep last 4 messages |

Use `/compact` to force compaction on the next model turn.

## `.ness/` Layout

```text
.ness/
├── NESS.md              Project memory loaded into prompts
├── USER.md              Cross-repo user preferences
├── LOG.md               Episodic session notes
├── permissions.json     Tool allow/deny/ask rules
├── hooks.json           Hook commands
├── mcp.json             MCP stdio servers
├── agents/              Subagent definitions
├── commands/            User slash commands
├── skills/              Project-local SKILL.md skills
├── plans/               Saved plan-mode assistant output
├── threads/             Saved JSONL trajectories
│   └── index.json       Thread metadata (cost, turns, summaries)
├── worktrees/           Optional subagent worktrees
└── shells/              Background shell logs
```

## Skills

Skills live under `.ness/skills/<name>/SKILL.md`:

```text
.ness/skills/react_component/SKILL.md
```

Each `SKILL.md` may include YAML frontmatter:

```markdown
---
name: react_component
description: Create React components matching project conventions.
triggers: [react, component, tsx]
references: [examples/Button.tsx]
---
# React Component

Skill instructions go here.
```

Small reference files (≤ 20 lines) are inlined into the prompt. Larger references are listed for on-demand `read_file` fetch.

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

The `spawn_subagent` tool runs a filtered, isolated graph (max depth 2) and returns a summary to the parent agent.

## Slash Commands

- `/plan [prompt]`: switch to read-only plan mode; optionally queue a prompt.
- `/act [prompt]`: switch to normal mode; optionally queue a prompt.
- `/mode`: show current agent mode.
- `/context`: show project context.
- `/cost`: show token/cost totals.
- `/cache`: show prompt-cache reads/writes and cache hit rate.
- `/skills`: list loaded skills and warnings.
- `/init [force]`: generate `.ness/NESS.md`.
- `/memory` or `/memory add <note>`: read or append project memory.
- `/user` or `/user add <note>`: read or append user preferences.
- `/permissions`: list/edit permission rules.
- `/hooks`: list hooks.
- `/mcp`: list MCP server status and tools.
- `/threads`: list saved sessions.
- `/resume <thread_id>`: resume a saved thread.
- `/save`: archive the current thread with a headline summary.
- `/reset`: archive and start a fresh thread.
- `/compact`: mark/manual compaction request.
- `/copy`, `/copy code`, `/copy <n>`: copy assistant output.
- `/image <path>`: attach an image to the next prompt (also `@image:path` inline).
- `/save-threads [on|off]`: toggle thread autosave (prints current status).
- `/exit` or `/quit`: end the session.

Markdown files under `.ness/commands/*.md` become project-local slash commands. Their body is used as a prompt template with `{{args}}` substitution.

## Thread Events

When autosave is on, LiteHarness appends JSONL events to `.ness/threads/<thread_id>.jsonl` and maintains `.ness/threads/index.json`:

```json
{"kind": "user", "content": "..."}
{"kind": "assistant", "content": "...", "tool_calls": []}
{"kind": "tool", "tool": "read_file", "args": {}, "result": "...", "duration_ms": 10, "exit": "ok"}
{"kind": "approval", "tool": "edit_file", "decision": "yes"}
{"kind": "usage", "model": "gpt-4o-mini", "input_tokens": 100, "cached_input_tokens": 40, "cache_write_tokens": 10, "output_tokens": 20, "cost_usd": 0.0001, "cost_source": "provider"}
{"kind": "reflection", "error": "", "over_limit": false}
{"kind": "compact", "content": "manual compaction requested"}
```

Threads are archived on `/save`, `/reset`, `/resume`, and session exit. Archived threads get a headline summary from the first user message.

## Verification

```bash
OPENAI_API_KEY=test uv run python -m compileall -q .
OPENAI_API_KEY=test uv run python -m unittest discover -s tests -v
```
