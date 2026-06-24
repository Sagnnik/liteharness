# LiteHarness

LiteHarness is an experimental, hackable coding-agent harness for engineers who want to own the loop. It uses OpenRouter-compatible chat models, LangGraph for orchestration, native tool-calling by default, and filesystem-driven extension points under `.ness/`.

## Quick Start

```bash
uv sync
export OPENAI_API_KEY=...
uv run python cli/main.py
```

Useful environment variables:

- `MODEL_NAME`: model passed to `ChatOpenRouter` (`deepseek-v4-flash` by default).
- `REFLECTION_MODEL_NAME`: model for background session-memory reflection (defaults to `MODEL_NAME`).
- `ENABLE_APPROVAL`: require approval for destructive tools.
- `AUTO_SAVE_THREADS`: write thread events to `.ness/threads/`.
- `REFLECTION_TOKEN_RATIO`: fraction of the usable context budget that must accumulate in new messages before a background reflection run (default `0.4`; set `0` to disable).
- `API_MAX_RETRIES`: retries for chat API calls (default `3`).
- `COMPACTION_OUTPUT_RESERVE_TOKENS`: output reserve subtracted from the model context window (default `8192`).
- `COMPACTION_INPUT_RESERVE_TOKENS`: input/system/tool reserve subtracted from the model context window (default `4096`).
- `COMPACTION_TOKEN_BUDGET`: fallback compaction budget when the model context window is unknown (default `120000`).
- `OPENROUTER_SESSION_ID`: optional stable prompt-cache session id. Defaults to the active LiteHarness thread id.
- `OPENAI_BASE_URL`: optional custom OpenAI-compatible base URL.
- `FORMAT_ON_WRITE`: auto-format supported file types after writes (default `true`).
- `NESS_DIR`: project config directory, default `.ness`.
- `EXA_API_KEY`: optional Exa API key for higher-quality `web_search` and `fetch_url` (get one from [exa.ai](https://exa.ai)). Without it, LiteHarness falls back to DuckDuckGo search and direct HTTP fetch.

CLI flags override env for a single run: `--model`, `--reflection-model`, `--api-key`, `--base-url`, `--openrouter-session-id`. Use `/config` in-session to switch model, keys, approval, and autosave (persisted to `.env`).

## Architecture

- `cli/main.py`: Rich CLI, slash commands, streaming, image/clipboard handling.
- `agent.py`: LangGraph loop: agent, approval gate, tool executor.
- `context.py`: layered prompt assembly from `instructions/` templates.
- `instructions/`: markdown templates for L0/L1 prompt layers, modes, compaction, reflection, and subagents.
- `compaction.py`: progressive context compaction by context pressure.
- `reflection.py`: background session-memory reflection with structured output (semantic distillation).
- `memory.py`: NESS.md, USER.md, and per-thread session memory helpers.
- `tools/`: local tools for files, search, web (`web_search`, `fetch_url` via Exa or DuckDuckGo fallback), shell, git, todos, user clarification (`ask_user`), and subagents.
- `permissions.py`: `.ness/permissions.json` allow/deny/ask matching.
- `hooks.py`: `.ness/hooks.json` pre/post/user/session command hooks.
- `mcp_client.py`: stdio MCP startup and namespaced MCP tool wrappers.
- `session.py`: SQLite thread storage (`threads.db`) for events, metadata, and subagent links.
- `skill_loader.py`: `SKILL.md` skill discovery under `.ness/skills/`.
- `config.py`: settings, model pricing, and cost/cache tracking.
- `parsers.py`: native tool-call extraction.

## Prompt Layers

LiteHarness splits context into four layers to keep prompt caching stable:

1. **L0 harness** (`build_l0`): NESS identity, universal rules, output format, and tool-calling protocol.
2. **L1 profile** (`build_l1`): persona, stable tool catalog, `USER.md` preferences, and `.ness/NESS.md` project conventions.
3. **L2 project context** (`build_project_context_block`): repo structure, git availability, and sticky skill cores.
4. **L3 working state** (`build_working_state_overlay`): wrapped in `<working-state>` tags and sent as a dedicated ephemeral `HumanMessage` at the tail of the message list each turn (never persisted to state, never mutating earlier messages, so the cached prefix stays stable through a tool loop). Includes git branch/dirty snapshot (when in a repo), compaction status, todos, and session memory from `.ness/sessions/mem_<thread_id>.md`. In plan mode, mode instructions are wrapped in an additional ephemeral `<plan-mode path=".ness/plans/">` block inside that overlay (also not cached). L0 documents both tags so the model knows how to interpret them.

Skills activate by trigger match and stay sticky for the session once loaded.

## Agent Modes

LiteHarness binds the **full session tool set in every mode** so the provider prefix cache survives plan ↔ normal switches without a graph rebuild. Plan mode is enforced at **runtime**: state-changing tool calls are rejected in the tool executor (the model sees the rejection in state; the CLI does not surface it). Mode instructions live in the ephemeral L3 overlay, not in the cached system prefix.

- **Normal** (Shift+Tab): execute with the full session tool set. All git tools (read and write) appear only inside a git repo.
- **Plan** (Shift+Tab): read-only planning. The agent researches the codebase, may ask clarifying multiple-choice questions via `ask_user`, drafts a structured plan, and should end with `todo_write` for actionable steps. Assistant output is auto-saved under `.ness/plans/`. Shift+Tab back to normal mode to execute.

Plan-mode workflow (when requirements are ambiguous):

1. **Clarify** — call `ask_user` with MCQ options (mark the recommended choice; the user may add a note per question).
2. **Research** — read-only tools and `spawn_subagent` for parallel investigation.
3. **Plan** — numbered steps with file paths, verification, and risks.
4. **Todos** — call `todo_write` at the end of every plan.

Session tool tiers (same set bound in both modes):

- Small always-on: `todo_read`, `todo_write`, `ask_user`
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
| `sessions/mem_<thread_id>.md` | Episodic per-session scratchpad. Current thread bullets load into L3. Maintained by the reflection gate. |

Reflection runs in the background when new messages since the last run exceed `REFLECTION_TOKEN_RATIO` of the usable context budget, and once more at session exit. It uses structured output (via `REFLECTION_MODEL_NAME`) to append up to 2 bullets per run to `.ness/sessions/mem_<thread_id>.md`. Bullets appear in the L3 working-state overlay on subsequent turns. `NESS.md` remains human-authored; the CLI warns at startup when it exceeds 20,000 characters.

## Compaction

Compaction is model-relative by default. LiteHarness estimates the usable context budget from the model context window minus output and input reserves. If the model window is unknown, `COMPACTION_TOKEN_BUDGET` is used as the fallback (default `120000`). When reserves exceed the window, the full window size is used as the budget.

| Pressure | Action |
|----------|--------|
| < 70% | No compaction |
| 70-85% | Compact large tool outputs |
| >= 85% | Summarize older history; keep `max(4, int(10 * (1 - ratio) / 0.15))` recent messages |

Use `/compact` to force compaction on the next model turn. Manual compaction runs at least a summary that keeps the last 10 messages when there is older history to summarize. When leaving plan mode (Shift+Tab to normal), LiteHarness shows a pre-execution context checkpoint at 75% pressure and forces compaction without prompting at 92% pressure.

## `.ness/` Layout

```text
.ness/
├── NESS.md              Project conventions loaded into L1
├── USER.md              Cross-repo user preferences
├── sessions/            Per-thread episodic memory (L3 overlay)
│   └── mem_<thread_id>.md
├── permissions.json     Tool allow/deny/ask rules
├── hooks.json           Hook commands
├── mcp.json             MCP stdio servers
├── agents/              Subagent definitions
├── commands/            User slash commands
├── skills/              Project-local SKILL.md skills
├── plans/               Saved plan-mode assistant output
├── threads/             Saved session trajectories (SQLite)
│   └── threads.db       Thread metadata, events, and subagent links
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

### Web search providers

`web_search` and `fetch_url` pick a provider automatically:

| Provider | When used | Notes |
|----------|-----------|-------|
| **Exa** | `EXA_API_KEY` is set | Semantic search, content highlights, reliable fetch |
| **DuckDuckGo fallback** | No Exa key | Keyword search via DuckDuckGo HTML; direct HTTP fetch with `trafilatura` / BeautifulSoup extraction |

The fallback requires no API key but is less capable: no neural search, weaker snippets, no JavaScript rendering on fetch, and occasional DuckDuckGo rate limits or CAPTCHAs. Set `EXA_API_KEY` when you need more reliable web access.

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

The `spawn_subagent` tool runs one or more filtered, isolated read-only graphs. Only the parent agent can spawn subagents; nested spawning is blocked by the read-only tool filter. For one task, pass `name` and `prompt`. For parallel exploration, pass `tasks`, plus optional `max_concurrency` and `timeout`; the parent agent waits until every subagent completes, fails, or times out.

Batch mode validates every task before starting any of them and returns one structured result with each task's status, duration, thread id, label, and output.

## Slash Commands

Shift+Tab toggles plan/normal mode without rebuilding the graph or invalidating the prompt cache. Current mode appears in the prompt prefix and bottom toolbar. Use `/menu` for a searchable command picker or `/help` for the full list.

**General**

- `/menu`: searchable command picker.
- `/help`: show the command reference.
- `/config`: switch model, set API keys, toggle approval/autosave (persisted to `.env`).
- `/exit` or `/quit`: end the session.

**Session**

- `/cost`: show token/cost totals.
- `/cache`: show prompt-cache reads/writes and cache hit rate.
- `/threads`: list saved sessions.
- `/resume <thread_id>`: resume a saved thread.
- `/save`: archive the current thread with a headline summary.
- `/reset`: archive and start a fresh thread.
- `/compact`: mark/manual compaction request.

**Context & memory**

- `/skills`: list loaded skills and warnings.
- `/init [force]`: generate `.ness/NESS.md`.
- `/memory` or `/memory add <note>`: read or append project memory.
- `/user` or `/user add <note>`: read or append user preferences.

**Tools & policy**

- `/permissions`: list/edit permission rules.
- `/hooks`: list hooks.
- `/mcp`: list MCP server status and tools.

**Input**

- `/copy`, `/copy code`, `/copy <n>`: copy assistant output.
- `/image <path>`: attach an image to the next prompt (also `@image:path` inline).

Markdown files under `.ness/commands/*.md` become project-local slash commands. Their body is used as a prompt template with `{{args}}` substitution.

## Thread Events

When autosave is on, LiteHarness stores events in `.ness/threads/threads.db`:

- **`threads`**: user `session-*` metadata (cost, turns, summaries, archive state)
- **`events`**: append-only JSON payloads for user sessions only
- **`subagents`**: subagent run metadata (status, output, duration) linked to a parent `session-*` thread

Event kinds stored in `events.payload` (session threads only):

```json
{"kind": "user", "content": "...", "t": "..."}
{"kind": "assistant", "content": "...", "tool_calls": [], "t": "..."}
{"kind": "tool", "tool": "read_file", "args": {}, "result": "...", "call_id": "...", "duration_ms": 10, "exit": "ok", "t": "..."}
{"kind": "approval", "tool": "edit_file", "decision": "yes", "t": "..."}
{"kind": "usage", "model": "deepseek-v4-flash", "input_tokens": 100, "cached_input_tokens": 40, "cache_write_tokens": 10, "output_tokens": 20, "cost_usd": 0.0001, "cost_source": "provider", "t": "..."}
{"kind": "reflection", "prompt": "...", "response": {"new_bullet_points": []}, "message_index": 12, "memory_updated": true, "error": "", "t": "..."}
{"kind": "compaction_llm", "prompt": "...", "response": "...", "action": "summary", "kept_recent": 10, "t": "..."}
{"kind": "compact", "content": "manual compaction requested", "t": "..."}
```

`/threads` lists user `session-*` threads only. Subagent trajectories are not stored in `events`; subagent LLM usage rolls up into the parent session's `threads` aggregates. Subagent outputs are stored in the `subagents` table.

`/resume` rebuilds user messages, assistant tool-call turns, and tool results from saved events. `spawn_subagent` tool output is supplemented from linked subagent outputs when available.

Threads are archived on `/save`, `/reset`, `/resume`, and session exit. Archived threads get a headline summary from the first user message.

## Verification

```bash
OPENAI_API_KEY=test uv run python -m compileall -q .
OPENAI_API_KEY=test uv run python -m unittest discover -s tests -v
```
