<p align="center">
  <img src="assets/banner-light-geo.svg" alt="LiteHarness — hackable coding-agent harness" width="100%">
</p>

# LiteHarness

LiteHarness is an experimental, hackable coding-agent harness for engineers who want to own the loop. It uses OpenRouter-compatible chat models, LangGraph for orchestration, native tool-calling by default, and filesystem-driven extension points under `.ness/`.

## Quick Start

```bash
uv sync
export OPENAI_API_KEY=...
uv run ness
```

Or skip the env var and set the key in-session: `uv run ness`, then `/config` > Provider > Provider API key. Settings are stored globally in `configs.json` (non-secrets) and `secrets.json` under the global config root — see Config layout. Process env vars still override the JSON values for a run; on first start, known keys from an existing project `.env` are imported into the JSON files once (the `.env` is left untouched).

Useful settings (env var shown; all except `NESS_DIR` are also editable via `/config`):

- `MODEL_NAME`: model passed to `ChatOpenRouter` (`deepseek/deepseek-v4-flash` by default).
- `REFLECTION_MODEL_NAME`: model for background session-memory reflection (defaults to `MODEL_NAME`).
- `ENABLE_APPROVAL`: require approval for destructive tools.
- `AUTO_SAVE_THREADS`: write thread events to `.ness/threads/`.
- `SESSION_END_REFLECTION`: run a final reflection pass when a session ends (default off). Mid-session reflection is still controlled by `REFLECTION_TOKEN_RATIO`.
- `REFLECTION_TOKEN_RATIO`: fraction of the usable context budget that must accumulate in new messages before a background reflection run (default `0.4`; set `0` to disable).
- `API_MAX_RETRIES`: retries for chat API calls (default `3`).
- `COMPACTION_OUTPUT_RESERVE`: output reserve subtracted from the model context window (default `8192`).
- `COMPACTION_INPUT_RESERVE`: input/system/tool reserve subtracted from the model context window (default `4096`).
- `COMPACTION_TOKEN_BUDGET`: fallback compaction budget when the model context window is unknown (default `120000`).
- `OPENROUTER_SESSION_ID`: optional stable prompt-cache session id. Defaults to the active LiteHarness thread id.
- `OPENROUTER_CACHE_TTL`: Anthropic prompt-cache lifetime (`5m` by default; `1h` is supported).
- `OPENROUTER_ANTHROPIC_MESSAGES`: use OpenRouter's Messages API for Anthropic models, including deferred MCP tool loading (default `true`).
- `GOAL_JUDGE_MODEL`: optional model used by the independent `/goal` judge (defaults to `REFLECTION_MODEL_NAME`).
- `GOAL_MAX_ATTEMPTS`: maximum worker/judge attempts for `/goal` (default `3`).
- `OPENAI_BASE_URL`: optional custom OpenAI-compatible base URL.
- `FORMAT_ON_WRITE`: auto-format supported file types after writes (default `true`).
- `NESS_DIR`: project config directory, default `.ness`.
- `LITEHARNESS_CONFIG_DIR`: override global config root (`USER.md`, `configs.json`, `secrets.json`, `instructions/`, `plans/`).
- `LITEHARNESS_CACHE_DIR`: override cache root (global OpenRouter catalog plus per-project `cli_history`).
- `EXA_API_KEY`: optional Exa API key for higher-quality `web_search` and `fetch_url` (get one from [exa.ai](https://exa.ai)). Without it, LiteHarness falls back to DuckDuckGo search and direct HTTP fetch.

CLI flags override env for a single run: `--model`, `--reflection-model`, `--api-key`, `--base-url`, `--openrouter-session-id`, `--reasoning-effort`, `--worktree` / `-w`, `--print` / `-p`, and `--yolo`. Yolo is session-only and bypasses approval prompts and persisted permission denials in act mode; hook vetoes and plan-mode read-only rules still apply. Use `/config` in-session to edit every adapter setting: provider keys and endpoints, model and reasoning, approval/autosave/reflection behavior, compaction budgets, and more (persisted to global `configs.json` / `secrets.json`).

### Headless one-shot queries (`-p` / `--print`)

Run a single query without opening the TUI; the final response goes to stdout and the process exits:

```bash
uv run ness -p "what does the auth module do?"
cat build-error.txt | uv run ness -p "explain the root cause" > diagnosis.txt
uv run ness -p --yolo "run the test suite and fix any failures"
```

Approvals are deny-by-default in print mode: tools already allowed by `.ness/permissions.json` run normally, anything that would prompt is auto-denied and the denial is fed back to the model, and `--yolo` bypasses the gate. `-p` composes with the other flags — `--worktree`, `--resume`, `--model`, etc. stdout carries only the final response; diagnostics and the `ness --resume <thread_id>` hint go to stderr. Exit codes: `0` success, `1` turn error, `2` usage error, `130` interrupted.

The `/config` model picker lazily refreshes text-output, tool-capable LLMs and VLMs from OpenRouter. Its global disk cache is reused for 24 hours, stale data remains available while refreshing, and the packaged model list is the offline fallback. Type in the model picker to search by model, display name, or provider. Reasoning choices are shown only when literal provider values are available; LiteHarness does not rename or rank-map them.

### Parallel sessions (git worktrees)

Run a second agent in an isolated checkout and branch without touching your main working tree:

```bash
# Terminal 1 — main checkout
uv run ness

# Terminal 2 — isolated agent (creates .ness/worktrees/auth on first launch)
uv run ness --worktree auth
```

Each worktree gets its own branch (`worktree-<name>`), file edits, and runtime data (`.ness/threads/`, `.ness/runtime/sessions`, shell jobs). Tracked `.ness` files (agents, skills, permissions, NESS.md) inherit from git. Config and secrets are global (see Config layout), so worktrees need no per-checkout setup. Re-launching with the same `--worktree` name reuses the existing checkout. Merge back with normal git when done (`git merge worktree-auth`, etc.).

## Architecture

- `src/liteharness/`: SDK — LangGraph agent loop, tools (files, search, web, shell, todos, `question`, subagents), permissions, memory, persistence, prompt layers/overlays, MCP, skills, hooks, compaction, reflection, and tracing.
- `src/liteharness_cli/`: coding adapter — `build_coding_agent` / `CodingSession`, path resolver, chat model factory, settings/pricing, rollback, and git worktree bootstrap.
- `src/liteharness_cli/tui/`: Ness TUI entry (`ness` / `liteharness_cli.tui.main`), streaming, slash commands, and clipboard handling.

## Prompt Layers

LiteHarness splits context into four layers to keep prompt caching stable:

1. **L0 harness** (`PromptLayers` / `L0_HARNESS`): NESS identity, universal rules, output format, and tool-calling protocol.
2. **L1 profile** (`build_l1`): persona, stable tool catalog, an always-on one-line skill catalog, `USER.md` preferences, and `.ness/NESS.md` project conventions.
3. **L2 project context**: app-supplied domain/repo structure (`PromptLayersConfig.l2_context`); not auto-loaded by bare `Session`.
4. **L3 working state** (`CodingOverlay` / `render_overlay_delta`): wrapped in `<system-reminder>` tags and injected ephemerally each turn (never persisted to state). On a fresh user turn the **full overlay** is appended to the latest human message; during a tool loop only the **per-section delta** (sections that changed since the last model invocation) is sent as a separate tail `HumanMessage` — if nothing changed, no tail is appended at all. The static `<plan-mode>` block is injected once on the fresh user message and never re-injected mid-turn (it would re-prime planning). After compaction the full overlay is re-injected because the model's context was rewritten. Includes git branch/dirty snapshot (when in a repo), compaction status, todos, session memory, skill-request hints, and loaded-skill summaries. In **plan** mode only, instructions are wrapped in an additional ephemeral `<plan-mode>` block (path points at the global plans dir for the CLI) (also not cached). Act mode omits a mode block. L0 documents `<plan-mode>` and `<system-reminder>`.

The L1 skill catalog lists every available skill with its path; full skill bodies enter the conversation when the model calls `skill_view` (or `read`s the path). `/skill <name>` stages a one-shot L3 hint for the next turn — it does not inject the body itself (see Skills below).

## Agent Modes

LiteHarness binds the **full session tool set in every mode** so the provider prefix cache survives plan ↔ act switches without a graph rebuild. Plan mode is enforced at **runtime**: state-changing tool calls are rejected in the tool executor (the model sees the rejection in state; the CLI does not surface it). **Plan** mode instructions live in the ephemeral L3 `<plan-mode>` overlay; **act** mode has no mode block (like OpenCode build — L0 + tools + dynamic L3 state only).

- **Act** (Shift+Tab): default execution / build mode — full tool set via L0 and permissions. L3 carries git, todos, compaction, and session memory when present. On the first act turn after a plan→act toggle, L3 prepends a one-shot `MODE SWITCH` note (inside the existing `<system-reminder>`) telling the model to call `todo` first, then address the user's message; it is cleared from state after that single model call so it never repeats.
- **Plan** (Shift+Tab): read-only planning. The agent researches the codebase, may ask clarifying multiple-choice questions via `question` (before any plan prose), then delivers exactly one final plan. Only the terminal plan message is auto-saved under the global `plans/<project-slug>/` directory. Shift+Tab back to act mode to execute.

Plan-mode workflow:

1. **Clarify** — if a decision materially changes the plan, call `question` with MCQ options before drafting (mark the recommended choice; never ask in prose).
2. **Research** — read-only tools first; use `spawn_subagent` only when a few targeted reads are insufficient (see L0 subagents rule).
3. **Plan** — one final message: numbered steps with file paths, verification, and risks; no tool calls in that message.
4. **Act** — Shift+Tab to act/build mode; on the first act turn the agent records todos from the plan via `todo`, then executes (or follows the user's message if they redirect); do not re-plan unless blocked or the user redirects.

Session tool tiers (same set bound in both modes):

- Always-on: `todo`, `question`, `skill_view`
- Core: file (`read`, `write`, `delete`, `edit`), search, web (`web_search`, `fetch_url`), and shell
- Tool discovery: `search_tools`, `add_tools` for loading deferred MCP tools on demand
- Advanced: `spawn_subagent`
- Loaded MCP tools: any `mcp__*` tool activated this session (deferred by default; load via `search_tools`/`add_tools` or `/mcp <server> [tool]`)

## Memory

| File | Purpose |
|------|---------|
| `.ness/NESS.md` | Durable project conventions (CLAUDE.md / AGENTS.md style). Human-authored via `/memory add`, manual edit, or agent edit when asked; optional LLM draft via `/memory create`. `/init` creates an empty file. Loaded into L1. May inline existing `@AGENTS.md` / `@CLAUDE.md` files (see below). |
| Global `USER.md` | Cross-repo user preferences (see Config layout). Human-authored via `/user`; loaded into L1. |
| `.ness/runtime/sessions/mem_<thread_id>.md` | Episodic per-session scratchpad. Current thread bullets load into L3. Maintained by the reflection gate. |

Reflection runs in the background when new messages since the last run exceed `REFLECTION_TOKEN_RATIO` of the usable context budget. An optional final pass at session exit is controlled by `SESSION_END_REFLECTION` (default off). It uses structured output (via `REFLECTION_MODEL_NAME`) to append up to 2 bullets per run to `.ness/runtime/sessions/mem_<thread_id>.md`. Bullets appear in the L3 system-reminder overlay on subsequent turns. `NESS.md` remains human-authored; the CLI warns at startup when its resolved size exceeds 20,000 characters.

### NESS.md includes

A standalone line in `NESS.md` of the form `@<path>` inlines that file's contents in place at runtime, so a repo that already ships an `AGENTS.md` or `CLAUDE.md` is picked up without duplication:

```markdown
@AGENTS.md
@CLAUDE.md

<extra LiteHarness-specific conventions here>
```

Includes resolve relative to the project root, reject paths that escape it, skip missing files (leaving a `# (missing include: ...)` marker), guard against cycles, and are size-capped. Changes to an included file invalidate the L1 prompt cache. The CLI also warns at startup when the assembled static prefix (L0+L1+L2) exceeds ~7,000 tokens.

## Compaction

Compaction is model-relative by default. LiteHarness estimates the usable context budget from the model context window minus output and input reserves. If the model window is unknown, `COMPACTION_TOKEN_BUDGET` is used as the fallback (default `120000`). When reserves exceed the window, the full window size is used as the budget.

| Pressure | Action |
|----------|--------|
| < 70% | No compaction |
| 70-80% | Compact large tool outputs |
| >= 80% | Summarize older history; keep `max(4, min(10, int(10 * (1 - ratio) / 0.20)))` recent messages |

Summary compaction triggers at 80% (not at the context ceiling): past that point the summarizing model is already degraded by context rot, so LiteHarness compacts before the summary itself would degrade. Use `/compact` to force compaction on the next model turn. Manual compaction runs at least a summary that keeps the last 10 messages when there is older history to summarize. When leaving plan mode (Shift+Tab to act), LiteHarness shows a pre-execution context checkpoint at 75% pressure and forces compaction without prompting at 92% pressure.

## Config layout

Ness splits **global** user data, **project** config, and **runtime** cache:

```text
# Global config (platformdirs user_config_dir("liteharness"))
# Linux: ~/.config/liteharness/
# macOS: ~/Library/Application Support/liteharness/
# Windows: %APPDATA%\liteharness\
USER.md                  Cross-repo user preferences
configs.json             Non-secret adapter settings (only values you changed)
secrets.json             API keys and other secrets (mode 0600)
instructions/            Editable prompt templates (L0, persona, plan/act, aux, goal)
plans/<project-slug>/    Saved plan-mode output for this project

# Per-project cache (platformdirs user_cache_dir("liteharness")/<hash>/)
cli_history              Prompt history for this project root

# Per-project .ness/ (NESS_DIR, default ".ness")
.ness/
├── NESS.md              Project conventions loaded into L1
├── permissions.json     Tool allow/deny/ask rules
├── hooks.json           Hook commands
├── mcp.json             MCP stdio servers
├── agents/              Subagent definitions
├── commands/            User slash commands
├── skills/              Project-local SKILL.md skills
├── threads/             Saved session trajectories (SQLite)
│   └── threads.db
└── runtime/
    ├── sessions/        Per-thread episodic memory (L3)
    │   └── mem_<thread_id>.md
    └── shells/          Background shell job metadata and logs
```

Override roots with `LITEHARNESS_CONFIG_DIR`, `LITEHARNESS_CACHE_DIR`, and `NESS_DIR`.

Settings resolve in this order (highest wins): CLI flags > process env vars > `secrets.json` / `configs.json` > built-in defaults. `configs.json` is written lazily — it only contains values you changed via `/config` (defaults stay in code and evolve with upgrades).

## Skills

Skills live under `.ness/skills/<name>/SKILL.md` (wired via `skills_dir` on the coding agent):

```text
.ness/skills/react_component/SKILL.md
```

Each `SKILL.md` may include YAML frontmatter:

```markdown
---
name: react_component
description: Create React components matching project conventions.
---
# React Component

Skill instructions go here.
```

Skill loading is two-stage. A one-line catalog of every available skill (`name: description`, plus path) is always present in L1. Full `SKILL.md` bodies load when the model calls the `skill_view` tool (or `read`s the catalog path); that content stays in the conversation as a tool message. `/skill <name>` stages a one-shot L3 `skill_request` hint for the next user turn. Successfully viewed skills accumulate in L3 as a `loaded_skills` summary (metadata only — the body remains in tool history).

## Permissions

`.ness/permissions.json` uses glob-style rules:

```json
{
  "allow": ["read:*", "grep:*", "shell:run:git status*"],
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

Tools are exposed as `mcp__<server>__<tool>`. The startup header shows connected MCP servers in Add-ons; use `/mcp` for the full server and tool list. Connection failures are shown as a startup warning. Startup failures do not stop the CLI.

**Approval and permissions:** all `mcp__*` tools require approval when `ENABLE_APPROVAL=true`. You can add explicit rules in `.ness/permissions.json`:

```json
{
  "allow": ["mcp__filesystem__read_file", "mcp__filesystem__list_directory"],
  "deny": ["mcp__filesystem__write_file"],
  "ask": ["*"]
}
```

**Subagents:** subagents are read-only. MCP tools, write tools, shell execution, nested subagents, and `todo` are rejected even if listed in frontmatter.

```markdown
---
tools: [read, grep, glob, web_search, fetch_url]
---
```

**Prompt size:** tool descriptions include the full MCP input schema so the model can handle complex arguments. Servers with many tools may increase token usage.

## Subagents

Subagents live in `.ness/agents/<name>.md`:

```markdown
---
tools: [read, grep, glob, web_search, fetch_url]
---
You are a read-only explorer. Return concise findings with file references.
```

The `spawn_subagent` tool runs one or more filtered, isolated read-only graphs. Only the parent agent can spawn subagents; nested spawning is blocked by the read-only tool filter. Always pass `tasks` — a non-empty list of `{name, prompt, label?}` — plus optional `max_concurrency` and `timeout`. Never call with bare top-level `name`/`prompt`. A single investigation still uses a one-item list:

```python
spawn_subagent(tasks=[{"name": "explore", "prompt": "Find route handlers"}])
```

The parent agent waits until every subagent completes, fails, or times out.

Batch mode validates every task before starting any of them and returns one structured result with each task's status, duration, thread id, label, and output.

## Slash Commands

Shift+Tab toggles plan/act mode without rebuilding the graph or invalidating the prompt cache. Current mode appears in the prompt prefix and footer. Type `/` for the command picker or `/help` for the full list.

**General**

- `/help`: show the command reference.
- `/config`: edit provider keys/endpoints, model/reasoning, behavior toggles, compaction budgets, and advanced options (persisted to global `configs.json` / `secrets.json`).
- `/exit` or `/quit`: end the session.

**Session**

- `/status`: show session, model, token, cost, and cache stats.
- `/threads`: open a scrollable saved-thread picker and switch the transcript in place.
- `/fork`: choose a human message, copy the conversation state before it into a child thread, and prefill that message for editing. Forking copies session memory/checkpoints but leaves current working-tree files unchanged.
- `/goal <objective>`: run up to three worker attempts, each followed by an isolated read-only judge. Failed verdicts become repair instructions for the next attempt.
- `/save`: archive the current thread with a headline summary.
- `/new`: archive and start a fresh thread.
- `/compact`: mark/manual compaction request.

**Context & memory**

- `/skill [<name>]`: list skills, or stage a skill for the next message (model loads via `skill_view`).
- `/init`: initialize project `.ness/` (dirs, permissions, hooks, mcp, default agent profiles, empty `NESS.md`) and ensure global config (`USER.md`, `instructions/`, `plans/<slug>/`).
- `/memory` or `/memory add <note>`: read or append project memory.
- `/memory create [force]`: opt-in LLM draft of `NESS.md` from project context (`force` overwrites non-empty content).
- `/user` or `/user add <note>`: read or append user preferences.

**Tools & policy**

- `/permissions`: list/edit permission rules.
- `/hooks`: list hooks.
- `/mcp`: list MCP server status and tools.

**Input**

- `/copy`, `/copy code`, `/copy <n>`: copy assistant output.
- `Ctrl+G`: paste an image from the clipboard into the prompt as `[Image #N]`. The image is resized (max 2000px long edge, max 5 MB) and sent to vision-capable models.
- `@path/to/file`: attach a file's contents to the next prompt — its current contents are inlined as a `<document>` block above your text. Type `@` to see suggestions from the repo's tracked paths; ↑/↓ to pick, Enter or Tab to complete, Esc to dismiss. Mention tokens persist on resume/rollback and re-expand from disk.

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
{"kind": "tool", "tool": "read", "args": {}, "result": "...", "call_id": "...", "duration_ms": 10, "exit": "ok", "t": "..."}
{"kind": "approval", "tool": "edit", "decision": "yes", "t": "..."}
{"kind": "usage", "model": "deepseek/deepseek-v4-flash", "input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 20, "cost_usd": 0.0001, "cost_source": "provider", "t": "..."}
{"kind": "reflection", "prompt": "...", "response": {"new_bullet_points": []}, "message_index": 12, "memory_updated": true, "error": "", "t": "..."}
{"kind": "compaction_llm", "prompt": "...", "response": "...", "action": "summary", "kept_recent": 10, "t": "..."}
{"kind": "compact", "content": "manual compaction requested", "t": "..."}
```

`/threads` lists user `session-*` threads only. The original conversation shows `×N` when it has forks; each fork shows `fork #k` in creation order. Fork lineage is stored explicitly on the thread row; inherited usage remains in the copied event history but is excluded from the child thread's cost totals. Subagent trajectories are not stored in `events`; subagent LLM usage rolls up into the parent session's `threads` aggregates. Subagent outputs are stored in the `subagents` table.

Selecting a thread rebuilds user messages, assistant tool-call turns, and tool results from saved events. The startup `--resume <thread_id>` flag remains available for automation. `spawn_subagent` tool output is supplemented from linked subagent outputs when available.

Threads are archived on `/save`, `/new`, thread switching/forking, and session exit. Archived threads get a headline summary from the first user message.

## Verification

```bash
OPENAI_API_KEY=test uv run python -m compileall -q .
OPENAI_API_KEY=test uv run pytest -q
# Optional paid provider smoke test:
OPENROUTER_LIVE_TEST=1 OPENAI_API_KEY=... uv run pytest -q -m live
```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
