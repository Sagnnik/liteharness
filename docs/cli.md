# Ness CLI

**Ness** is the interactive coding-agent CLI shipped with Ness AI. It uses OpenRouter-compatible chat models, plan/act modes, filesystem-driven extension points under `.ness/`, and a full TUI for approvals, thread history, and configuration.

See also: [Configuration](configuration.md) · [Architecture](architecture.md) · [SDK](sdk.md)

---

## Getting started

```bash
pip install ness-ai
export OPENAI_API_KEY=...    # or set via /config on first launch
ness
```

Initialize a project and global config:

```bash
ness
/init
```

`/init` creates project `.ness/` (dirs, permissions, hooks, mcp, default agent profiles, empty `NESS.md`) and ensures global config (`USER.md`, `instructions/`, `plans/<slug>/`).

Or skip the env var and set the key in-session: `/config` → Provider → Provider API key.

---

## Headless one-shot queries (`-p` / `--print`)

Run a single query without opening the TUI; the final response goes to stdout and the process exits:

```bash
ness -p "what does the auth module do?"
cat build-error.txt | ness -p "explain the root cause" > diagnosis.txt
ness -p --yolo "run the test suite and fix any failures"
```

Approvals are deny-by-default in print mode: tools already allowed by `.ness/permissions.json` run normally, anything that would prompt is auto-denied and the denial is fed back to the model, and `--yolo` bypasses the gate. `-p` composes with the other flags — `--worktree`, `--resume`, `--model`, etc.

- **stdout** — final response only
- **stderr** — diagnostics and the `ness --resume <thread_id>` hint
- **Exit codes:** `0` success, `1` turn error, `2` usage error, `130` interrupted

---

## Parallel sessions (git worktrees)

Run a second agent in an isolated checkout and branch without touching your main working tree:

```bash
# Terminal 1 — main checkout
ness

# Terminal 2 — isolated agent (creates .ness/worktrees/auth on first launch)
ness --worktree auth
```

Each worktree gets its own branch (`worktree-<name>`), file edits, and runtime data (`.ness/threads/`, `.ness/runtime/sessions`, shell jobs). Tracked `.ness` files (agents, skills, permissions, NESS.md) inherit from git. Config and secrets are global (see [Configuration](configuration.md)), so worktrees need no per-checkout setup. Re-launching with the same `--worktree` name reuses the existing checkout. Merge back with normal git when done (`git merge worktree-auth`, etc.).

---

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
---
# React Component

Skill instructions go here.
```

Skill loading is two-stage. A one-line catalog of every available skill (`name: description`, plus path) is always present in L1. Full `SKILL.md` bodies load when the model calls the `skill_view` tool (or `read`s the catalog path); that content stays in the conversation as a tool message. `/skill <name>` stages a one-shot L3 `skill_request` hint for the next user turn. Successfully viewed skills accumulate in L3 as a `loaded_skills` summary (metadata only — the body remains in tool history).

---

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

Approval choices are `y` yes, `S` session (allow for this CLI run), `a` always, `n` no, `N` never, `d` diff, and `s` show args.

### Web search providers

`web_search` and `fetch_url` pick a provider automatically:

| Provider | When used | Notes |
|----------|-----------|-------|
| **Exa** | `EXA_API_KEY` is set | Semantic search, content highlights, reliable fetch |
| **DuckDuckGo fallback** | No Exa key | Keyword search via DuckDuckGo HTML; direct HTTP fetch with `trafilatura` / BeautifulSoup extraction |

The fallback requires no API key but is less capable: no neural search, weaker snippets, no JavaScript rendering on fetch, and occasional DuckDuckGo rate limits or CAPTCHAs. Set `EXA_API_KEY` when you need more reliable web access.

---

## MCP

Ness AI connects to local **stdio** MCP servers at CLI startup. Each server is a child process; Ness AI discovers its tools and exposes them to the agent.

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

**Prompt size:** tool descriptions include the full MCP input schema so the model can handle complex arguments. Servers with many tools may increase token usage.

---

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

---

## Slash commands

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
- `/init`: initialize project `.ness/` and ensure global config.
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

---

## Thread events

When autosave is on, Ness AI stores events in `.ness/threads/threads.db`:

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
