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
- `REFLECTION_INTERVAL`: user turns between background session-memory reflection runs (default `5`; set `0` to disable).
- `COMPACTION_OUTPUT_RESERVE_TOKENS`: output reserve subtracted from the model context window (default `8192`).
- `COMPACTION_INPUT_RESERVE_TOKENS`: input/system/tool reserve subtracted from the model context window (default `4096`).
- `COMPACTION_TOKEN_BUDGET`: fallback compaction budget when the model context window is unknown (default `120000`).
- `OPENROUTER_SESSION_ID`: optional stable prompt-cache session id. Defaults to the active LiteHarness thread id.
- `OPENAI_BASE_URL`: optional custom OpenAI-compatible base URL.
- `FORMAT_ON_WRITE`: auto-format supported file types after writes (default `true`).
- `NESS_DIR`: project config directory, default `.ness`.
- `EXA_API_KEY`: optional Exa API key for `web_search` and `fetch_url` (get one from [exa.ai](https://exa.ai)).

## Architecture

- `cli/main.py`: Rich CLI, slash commands, streaming, image/clipboard handling.
- `agent.py`: LangGraph loop: agent, approval gate, tool executor.
- `context.py`: layered prompt assembly from `instructions/` templates.
- `instructions/`: markdown templates for L0/L1 prompt layers, modes, compaction, reflection, subagents, and XML fallback.
- `compaction.py`: progressive context compaction by context pressure.
- `reflection.py`: background session-memory reflection with structured output (distillation + loop detection).
- `memory.py`: NESS.md, USER.md, and per-thread session memory helpers.
- `tools/`: local tools for files, search, web (`web_search`, `fetch_url` via Exa), shell, git, todos, and subagents.
- `permissions.py`: `.ness/permissions.json` allow/deny/ask matching.
- `hooks.py`: `.ness/hooks.json` pre/post/user/session command hooks.
- `mcp_client.py`: stdio MCP startup and namespaced MCP tool wrappers.
- `session.py`: JSONL thread events and `index.json` session metadata.
- `skill_loader.py`: `SKILL.md` skill discovery under `.ness/skills/`.
- `config.py`: settings, model pricing, and cost/cache tracking.
- `parsers.py`: native and XML tool-call extraction.

## Prompt Layers

LiteHarness splits context into three layers to keep prompt caching stable:

1. **L0 harness** (`build_l0`): NESS identity, universal rules, output format, and tool-calling protocol.
2. **L1 profile** (`build_l1`): persona, stable tool catalog, `USER.md` preferences, and `.ness/NESS.md` project conventions.
3. **L2 project context** (`build_project_context_block`): repo structure, git availability, sticky skill cores, and current-thread session memory from `.ness/sessions/mem_<thread_id>.md`.
4. **L3 working state** (`build_working_state_overlay`): wrapped in `<working-state>` tags and sent as a dedicated ephemeral `HumanMessage` at the tail of the message list each turn (never persisted to state, never mutating earlier messages, so the cached prefix stays stable through a tool loop). Includes agent mode, environment date/time/cwd/OS, git branch/dirty snapshot (when in a repo), compaction status, todos, and loop-intervention warnings.

Skills activate by trigger match and stay sticky for the session once loaded.

## Agent Modes

- **Normal** (`/act`): execute with the full session tool set. All git tools (read and write) appear only inside a git repo.
- **Plan** (`/plan`): only read-only tool schemas are bound, including `web_search`, `fetch_url`, and read-only `spawn_subagent` for research. Assistant output is saved under `.ness/plans/`. Use `/act` to switch back and execute.

Tool tiers in normal mode:

- Small always-on: `todo_read`, `todo_write`
- L1 core: file, search, syntax checks (`check_syntax`), web (`web_search`, `fetch_url`), shell, and project-context tools
- L2 git read: `git_status`, `git_diff`, `git_log`, `git_show`
- L3 git write: `git_commit`, `git_checkout`, `git_branch`, `git_stash`
- L3 advanced: `spawn_subagent`
- Dynamic MCP: any `mcp__*` tool registered at startup

## Memory

Three memory files live under `.ness/`:

| File | Purpose |
|------|---------|
| `NESS.md` | Durable project conventions (CLAUDE.md / AGENTS.md style). Human-authored via `/init`, `/memory add`, or manual edit. Loaded into L1. |
| `USER.md` | Cross-repo user preferences. Human-authored via `/user`; loaded into L1. |
| `sessions/mem_<thread_id>.md` | Episodic per-session scratchpad. Current thread bullets load into L2. Maintained by the reflection gate. |

Reflection runs in the background every `REFLECTION_INTERVAL` user turns, when a todo is completed, and once more at session exit. It uses structured output to append up to 2 bullets per run to `.ness/sessions/mem_<thread_id>.md` and may inject a one-shot loop warning into the next turn. `NESS.md` remains human-authored; the CLI warns at startup when it exceeds 12,000 characters.

## Compaction

Compaction is model-relative by default. LiteHarness estimates the usable context budget from the model context window minus output and input reserves. If the model window is unknown, `COMPACTION_TOKEN_BUDGET` is used as the fallback (default `120000`). When reserves exceed the window, the full window size is used as the budget.

| Pressure | Action |
|----------|--------|
| < 70% | No compaction |
| 70-85% | Compact large tool outputs |
| >= 85% | Summarize older history; keep `max(4, int(10 * (1 - ratio) / 0.15))` recent messages |

Use `/compact` to force compaction on the next model turn. Manual compaction runs at least a summary that keeps the last 10 messages when there is older history to summarize. When leaving `/plan` with `/act`, LiteHarness shows a pre-execution context checkpoint at 75% pressure and forces compaction without prompting at 92% pressure.

## `.ness/` Layout

```text
.ness/
├── NESS.md              Project conventions loaded into L1
├── USER.md              Cross-repo user preferences
├── sessions/            Per-thread episodic memory (L2 tail)
│   └── mem_<thread_id>.md
├── permissions.json     Tool allow/deny/ask rules
├── hooks.json           Hook commands
├── mcp.json             MCP stdio servers
├── agents/              Subagent definitions
├── commands/            User slash commands
├── skills/              Project-local SKILL.md skills
├── plans/               Saved plan-mode assistant output
├── threads/             Saved JSONL trajectories
│   └── index.json       Thread metadata (cost, turns, summaries)
└── shells/              Background shell job metadata and logs
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
  "allow": ["read_file:*", "grep:*", "shell:run:git status*"],
  "deny": ["shell:run:rm -rf*", "shell:run:sudo*"],
  "ask": ["*"]
}
```

Deny rules win over allow rules. Rules are evaluated in order: persistent deny, session deny, persistent allow, session allow, then ask. Shell command allow/deny rules reject commands with unquoted shell operators (`;`, `&&`, `|`, `>`, `<`, newlines, etc.) so chained or redirect commands fall through to ask instead of matching a prefix rule.

`web_search:*` is allowed by default. `fetch_url` asks for approval per normalized URL; approving one URL does not approve a different path or query, and changing `max_characters` does not require a new approval.

Approval choices are `y` yes, `S` session (allow for this CLI run), `a` always, `n` no, `N` never, `d` diff, and `s` show args.

## MCP

LiteHarness connects to local **stdio** MCP servers at CLI startup. Each server is a child process; LiteHarness discovers its tools and exposes them to the agent.

**Security:** MCP servers run arbitrary commands with your user permissions. Only add servers you trust, same as running `npx @some/mcp-server` directly.

Configure servers in `.ness/mcp.json`. Either `servers` or `mcpServers` works (the latter matches Cursor's config shape):

```json
{
  "servers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "env": {},
      "cwd": ".",
      "startup_timeout": 20
    }
  }
}
```

Per-server fields:

- `command` / `args`: process to spawn. `command` may also be a one-element array like `["npx", "-y", "..."]`.
- `env`: optional environment overrides.
- `cwd`: working directory for the server process (defaults to the project root).
- `startup_timeout`: seconds to wait for connect + tool discovery (default `20`).

Tools are exposed as `mcp__<server>__<tool>`. On boot the CLI prints a one-line MCP summary; use `/mcp` for the full server and tool list. Startup failures do not stop the CLI.

**Approval and permissions:** all `mcp__*` tools require approval when `ENABLE_APPROVAL=true`. You can add explicit rules in `.ness/permissions.json`:

```json
{
  "allow": ["mcp__filesystem__read_file", "mcp__filesystem__list_directory"],
  "deny": ["mcp__filesystem__write_file"],
  "ask": ["*"]
}
```

**Subagents:** subagents are read-only. MCP tools, write tools, shell execution, git write tools, nested subagents, and `todo_write` are rejected even if listed in frontmatter.

```markdown
---
tools: [read_file, grep, glob_files, list_files]
---
```

**Prompt size:** tool descriptions include the full MCP input schema so the model can handle complex arguments. Servers with many tools may increase token usage.

## Subagents

Subagents live in `.ness/agents/<name>.md`:

```markdown
---
tools: [read_file, grep, glob_files, list_files]
---
You are a read-only explorer. Return concise findings with file references.
```

The `spawn_subagent` tool runs one or more filtered, isolated read-only graphs (max depth 2). For one task, pass `name` and `prompt`. For parallel exploration, pass `tasks`, plus optional `num_subagents`, `max_concurrency`, and `timeout`; the parent agent waits until every subagent completes, fails, or times out.

Batch mode validates every task before starting any of them and returns one structured result with each task's status, duration, thread id, label, and output.

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
{"kind": "reflection", "error": "", "memory_updated": true, "stuck_detected": false}
{"kind": "compact", "content": "manual compaction requested"}
```

Threads are archived on `/save`, `/reset`, `/resume`, and session exit. Archived threads get a headline summary from the first user message.

## Verification

```bash
OPENAI_API_KEY=test uv run python -m compileall -q .
OPENAI_API_KEY=test uv run python -m unittest discover -s tests -v
```
