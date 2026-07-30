# Architecture

LiteHarness is split into a reusable **SDK** and a **coding CLI adapter** (Ness).

| Path | Role |
|------|------|
| `src/liteharness/` | SDK — LangGraph agent loop, tools (files, search, web, shell, todos, `question`, subagents), permissions, memory, persistence, prompt layers/overlays, MCP, skills, hooks, compaction, reflection, and tracing |
| `src/liteharness_cli/` | Coding adapter — `build_coding_agent` / `CodingSession`, path resolver, chat model factory, settings/pricing, rollback, and git worktree bootstrap |
| `src/liteharness_cli/tui/` | Ness TUI entry (`ness` / `liteharness_cli.tui.main`), streaming, slash commands, and clipboard handling |

**LiteHarness** is the project and Python package. **Ness** is the interactive CLI (`ness` command).

See also: [SDK guide](sdk.md) · [CLI guide](cli.md) · [Configuration](configuration.md)

---

## Prompt layers

LiteHarness splits context into four layers to keep prompt caching stable:

1. **L0 harness** (`PromptLayers` / `L0_HARNESS`): NESS identity, universal rules, output format, and tool-calling protocol.
2. **L1 profile** (`build_l1`): persona, stable tool catalog, an always-on one-line skill catalog, `USER.md` preferences, and `.ness/NESS.md` project conventions.
3. **L2 project context**: app-supplied domain/repo structure (`PromptLayersConfig.l2_context`); not auto-loaded by bare `Session`.
4. **L3 working state** (`CodingOverlay` / `render_overlay_delta`): wrapped in `<system-reminder>` tags and injected ephemerally each turn (never persisted to state). On a fresh user turn the **full overlay** is appended to the latest human message; during a tool loop only the **per-section delta** (sections that changed since the last model invocation) is sent as a separate tail `HumanMessage` — if nothing changed, no tail is appended at all. The static `<plan-mode>` block is injected once on the fresh user message and never re-injected mid-turn (it would re-prime planning). After compaction the full overlay is re-injected because the model's context was rewritten. Includes git branch/dirty snapshot (when in a repo), compaction status, todos, session memory, skill-request hints, and loaded-skill summaries. In **plan** mode only, instructions are wrapped in an additional ephemeral `<plan-mode>` block (path points at the global plans dir for the CLI) (also not cached). Act mode omits a mode block. L0 documents `<plan-mode>` and `<system-reminder>`.

The L1 skill catalog lists every available skill with its path; full skill bodies enter the conversation when the model calls `skill_view` (or `read`s the path). `/skill <name>` stages a one-shot L3 hint for the next turn — it does not inject the body itself (see [Skills in the CLI guide](cli.md#skills)).

---

## Agent modes

LiteHarness binds the **full session tool set in every mode** so the provider prefix cache survives plan ↔ act switches without a graph rebuild. Plan mode is enforced at **runtime**: state-changing tool calls are rejected in the tool executor (the model sees the rejection in state; the CLI does not surface it). **Plan** mode instructions live in the ephemeral L3 `<plan-mode>` overlay; **act** mode has no mode block (L0 + tools + dynamic L3 state only).

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

---

## Memory

| File | Purpose |
|------|---------|
| `.ness/NESS.md` | Durable project conventions (CLAUDE.md / AGENTS.md style). Human-authored via `/memory add`, manual edit, or agent edit when asked; optional LLM draft via `/memory create`. `/init` creates an empty file. Loaded into L1. May inline existing `@AGENTS.md` / `@CLAUDE.md` files (see below). |
| Global `USER.md` | Cross-repo user preferences (see [Configuration](configuration.md)). Human-authored via `/user`; loaded into L1. |
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

---

## Compaction

Compaction is model-relative by default. LiteHarness estimates the usable context budget from the model context window minus output and input reserves. If the model window is unknown, `COMPACTION_TOKEN_BUDGET` is used as the fallback (default `120000`). When reserves exceed the window, the full window size is used as the budget.

| Pressure | Action |
|----------|--------|
| < 70% | No compaction |
| 70-80% | Compact large tool outputs |
| >= 80% | Summarize older history; keep `max(4, min(10, int(10 * (1 - ratio) / 0.20)))` recent messages |

Summary compaction triggers at 80% (not at the context ceiling): past that point the summarizing model is already degraded by context rot, so LiteHarness compacts before the summary itself would degrade. Use `/compact` to force compaction on the next model turn. Manual compaction runs at least a summary that keeps the last 10 messages when there is older history to summarize. When leaving plan mode (Shift+Tab to act), LiteHarness shows a pre-execution context checkpoint at 75% pressure and forces compaction without prompting at 92% pressure.
