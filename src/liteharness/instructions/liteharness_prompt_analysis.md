# Detailed Analysis: Prompt Instructions from Open-Source Coding Agents
## Ranked by Relevance to LiteHarness Architecture

---

## Executive Summary

LiteHarness uses a **4-layer prompt architecture** (L0-L3) with:
- L0: Harness identity, universal rules, output format, tool-calling protocol
- L1: Persona, stable tool catalog, USER.md preferences, NESS.md project conventions
- L2: Repo structure, git availability, sticky skill cores
- L3: Ephemeral working state overlay (git snapshot, compaction status, todos, session memory)

This analysis maps how other agents structure their prompts and what LiteHarness can adopt.

---

# TIER 1: Must-Study — Direct Architectural Parallels

---

## 1. Claude Code (Anthropic) — The Gold Standard

### Architecture Mapping to LiteHarness

| Claude Code Component | LiteHarness Equivalent | Notes |
|----------------------|----------------------|-------|
| `<functions>` block (tool definitions) | L0 harness + L1 tool catalog | Claude Code injects full tool schemas with descriptions directly into system prompt |
| Core instructions (tone, task management, git safety, tool policies) | L0 harness rules + L1 profile | Split across L0 (universal) and L1 (persona-specific) |
| `<env>` block (working dir, git repo, platform, date) | L3 working state overlay | Claude Code puts this in system prompt; LiteHarness puts in L3 ephemeral overlay |
| Model identification line | L0 harness identity | "You are powered by model X" |
| Git status snapshot | L3 working state overlay | Current branch, status, recent commits |
| Mode-dependent instructions (plan vs build) | L3 plan-mode overlay | Claude Code uses conditional injection; LiteHarness uses ephemeral L3 overlay |
| `CLAUDE.md` loading | `NESS.md` + `USER.md` | Project conventions loaded into L1 |
| Skills system | `.ness/skills/` | Both use trigger-based activation with YAML frontmatter |

### Key Prompt Structure (from leaked v2.1.195 Opus 4.8)

```
1. Identity: "You are Claude Code, Anthropic's official CLI for Claude."
2. Safety preamble: Defensive security only, URL generation rules
3. Help/feedback pointers: /help, GitHub issues link
4. Self-reference rule: When asked about Claude Code, use WebFetch tool
5. TONE AND STYLE section (~15 rules)
   - Concise, direct, <4 lines unless detail requested
   - No preambles/postambles
   - One-word answers best
   - No emojis unless requested
   - Use tools for actions, text for communication only
   - Code references: `file_path:line_number` format
6. PROACTIVENESS section
   - Balance: do the right thing when asked, don't surprise user
7. FOLLOWING CONVENTIONS section
   - Check existing code style before changes
   - Verify library availability (check package.json, neighbors)
   - Security best practices
8. CODE STYLE section
   - DO NOT ADD ANY COMMENTS unless asked
9. TASK MANAGEMENT section (extensive)
   - TodoWrite tool usage mandatory for planning
   - Mark todos completed immediately
   - Detailed examples of todo workflow
10. DOING TASKS section
    - Recommended steps: TodoWrite → search → implement → verify → lint/typecheck
    - NEVER commit unless explicitly asked
11. TOOL USAGE POLICY section
    - Use Task tool for file search (reduce context usage)
    - Parallel tool calls for independent operations
    - WebFetch redirect handling
12. `<env>` block with template variables
13. Model identification
14. Git status snapshot
```

### What LiteHarness Should Adopt

**A. Tone & Style Rules (L0)**
Claude Code's tone section is extremely prescriptive and effective. Key rules:
- "You MUST answer concisely with fewer than 4 lines"
- "Only address the specific query or task at hand"
- "Do not add additional code explanation summary unless requested"
- "After working on a file, just stop"
- "Never use tools like Bash or code comments as means to communicate"

**B. Task Management Integration (L0/L3)**
Claude Code makes TodoWrite usage *mandatory* with extensive examples. This is more forceful than LiteHarness's current approach. Consider adding:
- Explicit requirement to use `todo` tool for any multi-step task
- Examples of good vs bad todo usage in L0
- Rule: "Mark todos as completed as soon as you are done with a task. Do not batch up multiple tasks before marking as completed."

**C. Tool Usage Policy (L0)**
- "When doing file search, prefer to use the Task tool" → LiteHarness could recommend `spawn_subagent` for searches
- "You can call multiple tools in a single response" → Already in LiteHarness, but could be emphasized
- "When making multiple bash tool calls, send a single message with multiple tool calls to run in parallel"

**D. Git Safety Rules (L1/L3)**
Claude Code has explicit git workflow rules:
- Run `git status` to see untracked files (never use `-uall`)
- Run `git diff` to see staged and unstaged changes
- Check if current branch tracks remote
- Run `git log` and `git diff [base-branch]...HEAD` for PR context
- NEVER commit unless explicitly asked

**E. Code Reference Format (L0)**
Claude Code mandates: `file_path:line_number` format when referencing code. This is a simple but powerful convention.

### What LiteHarness Should NOT Adopt

- Claude Code's system prompt is **~40k tokens** (Opus 4.8). This is massive and causes context pressure issues. LiteHarness's layered approach with L3 as ephemeral is architecturally superior.
- Claude Code puts environment info in the system prompt (cached). LiteHarness correctly puts git snapshot in L3 (ephemeral, not cached), which is better for cache stability.

---

## 2. OpenCode — Closest Architectural Sibling

### Architecture Mapping to LiteHarness

| OpenCode Component | LiteHarness Equivalent | Notes |
|-------------------|----------------------|-------|
| `SystemPrompt.provider(model.modelID)` | L0 harness (provider-specific) | Different prompts per provider (Anthropic, OpenAI, Gemini) |
| `SystemPrompt.environment()` | L3 working state overlay | Working directory, platform, date |
| `SystemPrompt.custom()` | L1 profile (NESS.md, USER.md, AGENTS.md) | Custom instructions from AGENTS.md files |
| Agent-specific prompt (build.md, plan.md) | L1 profile + L3 mode overlay | Each agent has its own system prompt |
| `BUILD_SWITCH` reminder | L3 plan-mode overlay | System reminder when switching plan→build |
| Tool registry with wildcard filtering | L1 tool catalog + permissions | Tools filtered by agent permissions |
| Auto-summarization at 90% context | Compaction (L0/L3) | LiteHarness has more sophisticated tiered compaction |
| Todo tools per session | L3 working state (todos) | Same concept |
| Subagent task tool | `spawn_subagent` | Same concept, OpenCode calls it `task` |
| MCP tool integration | MCP client | Both support deferred MCP loading |

### System Prompt Hierarchy (from source)

```
1. Provider Header (optional identity spoofing)
   "You are Claude, a large language model trained by Anthropic."
   ↓
2. Provider-Specific Prompt
   Anthropic: "You are an expert coding assistant. Use precise, technical language..."
   OpenAI: "You are an advanced AI coding assistant. Write production-quality code..."
   Gemini: "You are a helpful coding AI. Focus on practical solutions..."
   ↓
3. Environment Information
   <env>
   Working directory: /Users/me/projects/my-app
   Is directory a git repo: yes
   Platform: darwin
   Today's date: Saturday, January 15, 2025
   </env>
   <project>
   src/
     auth/
       login.ts
       middleware.ts
   </project>
   ↓
4. Custom Instructions (AGENTS.md files discovered)
   ↓
5. Agent-Specific Prompt (if using custom agent)
   ↓
6. User Override (--system flag)
```

### Build Agent Prompt (from source)

```
You are opencode, an interactive CLI agent specializing in software engineering tasks.
Your primary goal is to help users safely and efficiently.

# Operational Guidelines
## Tone and Style (CLI Interaction)
- Concise & Direct: professional, direct, concise tone for CLI
- Minimal Output: fewer than 3 lines per response (excluding tool use/code)
- Clarity over Brevity: prioritize clarity for essential explanations
- No Chitchat: avoid filler, preambles, postambles
- Formatting: GitHub-flavored Markdown, monospace rendering
- Tools vs Text: use tools for actions, text ONLY for communication
- Handling Inability: state briefly (1-2 sentences), offer alternatives

## Security and Safety Rules
- Explain Critical Commands: before executing bash that modifies filesystem,
  provide brief explanation of purpose and impact
- Security First: never introduce code exposing secrets/API keys

## Tool Usage
- File Paths: ALWAYS use absolute paths
- Parallelism: execute multiple independent tool calls in parallel
- Command Execution: use bash tool, explain modifying commands first
- Background Processes: use & for commands unlikely to stop
- Interactive Commands: avoid commands requiring user interaction
- Respect User Confirmations: if user cancels tool call, do NOT retry
```

### Plan Agent Prompt

Same as Build but with restrictions:
- Cannot use `edit` tool
- Must ask permission for `bash`
- Focus on analysis and planning without making changes

### What LiteHarness Should Adopt

**A. Provider-Specific Prompts (L0 enhancement)**
OpenCode has different base prompts per provider. LiteHarness currently uses a single L0 for all providers. Consider:
- `instructions/l0_anthropic.md` vs `instructions/l0_openai.md`
- Or detect provider and inject provider-specific tone/tool-calling guidance

**B. AGENTS.md Discovery Pattern (L1 enhancement)**
OpenCode discovers and concatenates AGENTS.md from:
- `~/.config/opencode/AGENTS.md` (global)
- Parent directories (walking up from cwd)
- Current directory

LiteHarness already has `NESS.md` at project root. Consider:
- Walking up parent directories for `NESS.md` (useful in monorepos)
- Global `~/.ness/USER.md` (already planned)

**C. BUILD_SWITCH Reminder (L3 enhancement)**
OpenCode injects a synthetic system message when switching from plan to build:
```
<system-reminder>
Your operational mode has changed from plan to build.
You are no longer in read-only mode.
You are permitted to make file changes, run shell commands, and utilize your arsenal of tools as needed.
</system-reminder>
```

LiteHarness already has this concept but could make the reminder more explicit and structured.

**D. Tool Description Best Practices (L1)**
OpenCode's tool descriptions are extremely detailed. Example from `read` tool:
```
Reads a file from the local filesystem. You can access any file directly.
Usage:
- The filePath parameter must be an absolute path
- By default, reads up to 2000 lines
- You can optionally specify line offset and limit
- Any lines longer than 2000 characters will be truncated
- Results returned using cat -n format with line numbers starting at 1
- Cannot read binary files or images
- Call multiple tools in parallel for speculative reads
- If file exists but is empty, you will receive a system reminder warning
```

**E. Agent Definition Format (L1)**
OpenCode uses YAML frontmatter in markdown for agent definitions:
```markdown
---
description: Reviews code for quality and best practices
mode: subagent
model: anthropic/claude-sonnet-4-20250514
temperature: 0.1
tools:
  write: false
  edit: false
  bash: false
---
You are in code review mode. Focus on:
- Code quality and best practices
- Potential bugs and edge cases
- Performance implications
- Security considerations
```

LiteHarness already uses this pattern for skills. Consider using it for subagents too.

---

## 3. Pi (pi-coding-agent) — Minimalist Philosophy

### Architecture Mapping to LiteHarness

| Pi Component | LiteHarness Equivalent | Notes |
|-------------|----------------------|-------|
| `~/.pi/agent/SYSTEM.md` | L0 harness (replace) | Full system prompt override |
| `APPEND_SYSTEM.md` | L1 profile append | Extend without replacing |
| `AGENTS.md` / `CLAUDE.md` | `NESS.md` + `USER.md` | Project conventions, loaded from cwd + parents |
| `.pi/skills/` | `.ness/skills/` | Same concept, lazy loading |
| `.pi/extensions/` | `.ness/` extensions | Plugin system |
| Tree-structured sessions | Thread events in SQLite | Both persist sessions |
| Custom compaction | Compaction (L0/L3) | Pi uses extensions for custom compaction |
| `--tools` allowlist | Permissions system | Pi uses CLI flags; LiteHarness uses `.ness/permissions.json` |

### Key Philosophy: Minimal System Prompt

Pi's core insight: **"The system prompt should be under 1k tokens. Everything else is context engineering."**

Pi's default system prompt is extremely short:
```
You are a helpful coding assistant.
```

The heavy lifting is done by:
1. **AGENTS.md** — project conventions (like NESS.md)
2. **Skills** — loaded on demand via triggers
3. **Context files** — appended to user messages, not system prompt
4. **Lazy loading** — skills only load their full instructions when triggered

### What LiteHarness Should Adopt

**A. Skill Lazy Loading (L1 enhancement)**
Pi's skills have a one-line description in the system prompt:
```
- react_component: Create React components matching project conventions
```

The full SKILL.md is only loaded when the skill is triggered. LiteHarness already does this but could:
- Keep skill descriptions minimal in L1
- Load full skill content only when triggered (already done, but verify)
- Unload skills when no longer relevant (not currently done)

**B. Context File Concatenation (L1 enhancement)**
Pi concatenates all discovered AGENTS.md files:
- `~/.pi/agent/AGENTS.md` (global)
- Parent dirs + cwd (walking up)

LiteHarness currently only loads `.ness/NESS.md` at project root. Consider:
```
.ness/NESS.md (project)
~/.ness/USER.md (global user preferences)
parent_dirs/NESS.md (for monorepos)
```

**C. SYSTEM.md vs APPEND_SYSTEM.md (L0/L1)**
Pi distinguishes between:
- `SYSTEM.md`: Replace the default system prompt entirely
- `APPEND_SYSTEM.md`: Extend the default without replacing

LiteHarness could offer:
- `.ness/SYSTEM.md` — override L0 entirely (power user mode)
- `.ness/APPEND.md` — append to L1 (current NESS.md behavior)

**D. Tree-Structured Sessions**
Pi's sessions are tree-structured (not linear), allowing branching and forking:
```
pi -c  # Continue last session
pi --tree  # Jump to any point and branch
```

LiteHarness's SQLite thread storage could support this with parent message references.

---

## 4. Roo Code (VS Code Extension) — Mode-Specific Prompts

### Architecture Mapping to LiteHarness

| Roo Code Component | LiteHarness Equivalent | Notes |
|-------------------|----------------------|-------|
| Global `.roorules` | `~/.ness/USER.md` | Cross-repo user preferences |
| Workspace `.roorules` | `.ness/NESS.md` | Project-specific rules |
| Mode-specific `.roo/rules-{mode}/` | `.ness/skills/` + mode instructions | Rules per agent mode |
| `AGENTS.md` support | `NESS.md` | Same concept |
| Custom instructions | L1 profile | User-defined behavior rules |
| Prompt preview | — | Shows assembled system prompt |
| Mode switching | Plan/Normal mode | Roo has more modes (Code, Ask, Debug, Architect) |

### Prompt Hierarchy

```
1. Base system prompt (Roo Code core)
2. Global rules (~/.roorules)
3. Workspace rules (.roorules in workspace)
4. Mode-specific rules (.roo/rules-{mode}/)
5. AGENTS.md (if present)
6. Custom instructions (from settings)
```

### Mode Definitions

Roo Code has predefined modes with specific tool access:
- **Code**: Full tool access, coding focus
- **Ask**: Read-only, Q&A focus
- **Debug**: Read + bash, debugging focus
- **Architect**: Planning and design, limited write

Each mode has:
- Custom system prompt section
- Restricted tool set
- Specific behavior rules

### What LiteHarness Should Adopt

**A. Mode-Specific Rule Directories (L1 enhancement)**
Roo Code's `.roo/rules-{mode}/` pattern is powerful. LiteHarness could use:
```
.ness/
  rules/
    normal.md     # Normal mode rules
    plan.md       # Plan mode rules
    review.md     # Review subagent rules
```

**B. Prompt Preview Feature**
Roo Code has a "Prompt Preview" that shows the assembled system prompt. LiteHarness could add a `/debug prompt` command showing the full assembled L0-L3 prompt.

**C. Custom Instructions Integration (L1)**
Roo Code allows users to add custom instructions via VS Code settings. LiteHarness's `/config` and `/user` commands already do this, but could be more explicit about where instructions land in the prompt hierarchy.

---

## 5. Cline (VS Code Extension) — Exhaustive Tool Documentation

### Architecture Mapping to LiteHarness

| Cline Component | LiteHarness Equivalent | Notes |
|----------------|----------------------|-------|
| System prompt (~10k tokens) | L0 + L1 combined | Very large, includes all tool docs |
| CAPABILITIES section | L0 harness | What the agent can do |
| RULES section | L0 harness rules | Behavioral constraints |
| Tool definitions with XML examples | L1 tool catalog | Detailed usage examples |
| Act Mode vs Plan Mode | Normal vs Plan mode | Same concept |
| `attempt_completion` tool | — | Explicit task completion signal |
| One tool per message restriction | — | LiteHarness allows multiple |

### System Prompt Structure

```
1. ROLE DEFINITION
   "You are Cline, a highly skilled software engineer..."

2. TOOL USE FORMATTING
   - XML-style tool calls
   - One tool per message
   - Wait for user confirmation after each tool

3. CAPABILITIES
   - Execute CLI commands
   - File management (read, write, list)
   - Search (regex, file search)
   - Web interaction (fetch, search)
   - Image analysis

4. RULES
   - Stay in working directory
   - Don't ask unnecessary questions
   - Check file contents before editing
   - Use write_to_file for new files, replace_in_file for edits
   - NEVER overwrite .clinerules or .cursorrules
   - One tool per message

5. SYSTEM INFORMATION
   - OS, shell, home directory, working directory

6. OBJECTIVE
   - Analyze task → use tools sequentially → think critically → finalize with attempt_completion

7. ACT MODE vs PLAN MODE
   - Act: execute with tools
   - Plan: discuss before executing
```

### What LiteHarness Should Adopt

**A. Explicit Tool Use Format (L0)**
Cline's XML-style tool calls are very explicit:
```xml
<read_file>
<args>
  <file_path>src/main.ts</file_path>
</args>
</read_file>
```

LiteHarness uses native tool-calling (function calling). This is better for modern models, but Cline's explicit format could be useful for:
- Documentation examples in L0
- Fallback parsing in `parsers.py`

**B. `attempt_completion` Tool (L0/L3)**
Cline requires the agent to call `attempt_completion` when done:
```xml
<attempt_completion>
<result>
I've implemented the feature by...
</result>
</attempt_completion>
```

This is a clean pattern. LiteHarness could:
- Add an `attempt_completion` or `done` tool
- Use it to signal task completion explicitly
- Trigger reflection/compaction on completion

**C. File Edit Strategy Rules (L0)**
Cline has explicit rules for when to use which edit tool:
- `write_to_file`: For new files or complete rewrites
- `replace_in_file`: For targeted edits to existing files
- Never use `write_to_file` to modify existing files

LiteHarness's `write_file` vs `edit` distinction could be clarified:
```
- write_file: Create new files or completely replace existing files
- edit: Make targeted changes to existing files using SEARCH/REPLACE blocks
- delete_file: Remove files
```

---

# TIER 2: Highly Relevant — Specific Patterns to Adopt

---

## 6. Aider — Edit Format Mastery

### Architecture Mapping to LiteHarness

| Aider Component | LiteHarness Equivalent | Notes |
|----------------|----------------------|-------|
| `.aiderCode.md` | `NESS.md` + L1 profile | Terms of reference for agent |
| `.aiderContext.md` | — | Context file loader |
| Architect/Editor split | Reflection model | Different models for planning vs execution |
| Chat modes (code/ask/architect) | Plan/Normal modes | Aider has more modes |
| Edit formats (wholefile, diff, udiff) | `edit` tool format | Multiple code change formats |
| Repo map | L2 project context | Codebase structure summary |
| Git integration | Git tool | Both use git for undo/safety |

### Edit Format Prompts

Aider has **separate system prompt sections** for each edit format:

**Whole-file format:**
```
When editing files, return the entire content of the file.
Use this format:
```python
# filename.py
<entire file content>
```
```

**SEARCH/REPLACE format:**
```
When editing files, use SEARCH/REPLACE blocks:
<<<<<<< SEARCH
old code
=======
new code
>>>>>>> REPLACE
```

**Unified diff format:**
```
When editing files, output unified diff format:
--- a/filename.py
+++ b/filename.py
@@ -1,5 +1,5 @@
-old line
+new line
```

### What LiteHarness Should Adopt

**A. Edit Format Instructions (L0)**
LiteHarness currently uses a single `edit` tool. Consider supporting multiple edit formats with format-specific instructions:
```
# In L0 or as a skill
## Edit Format: SEARCH/REPLACE
When making changes to existing files, use SEARCH/REPLACE blocks:

<<<<<<< SEARCH
exact old code to find
=======
new code to replace with
>>>>>>> REPLACE

Rules:
- SEARCH must match exactly (including whitespace)
- Include enough context lines (3-5) to make SEARCH unique
- One block per logical change
```

**B. Precision Prompting Methodology (L1)**
Aider's "Precision Prompting" approach with structured context files:
- `.aiderCode.md`: Terms of reference (system prompt)
- `projectOverview.md`: Goals, user needs, requirements
- `systemDesign.md`: Architecture, patterns, tech decisions
- `techEnvironment.md`: Dev environment, dependencies, setup
- `activeDevelopment.md`: Current status, tasks, progress

LiteHarness could recommend a similar structure:
```
.ness/
  NESS.md          # Project conventions (already exists)
  USER.md          # User preferences (already exists)
  OVERVIEW.md      # Project overview (new)
  DESIGN.md        # System design (new)
  ENV.md           # Tech environment (new)
  ACTIVE.md        # Active development status (new)
```

**C. Architect/Editor Model Split**
Aider's `--architect` mode uses one model for planning and another for editing:
```bash
aider --architect --model gpt-5 --editor-model gpt-5-mini
```

LiteHarness already has `REFLECTION_MODEL_NAME`. Consider extending this pattern:
- `PLAN_MODEL_NAME`: Model for plan mode
- `EXECUTE_MODEL_NAME`: Model for normal mode
- `EDIT_MODEL_NAME`: Model specifically for code edits

---

## 7. Hermes — SOUL.md Identity Layer

### Architecture Mapping to LiteHarness

| Hermes Component | LiteHarness Equivalent | Notes |
|-----------------|----------------------|-------|
| `SOUL.md` | L0 harness identity | Static persona at slot #1 |
| Self-improving skill loop | Reflection (background) | Agent evaluates and improves itself |
| Cron-scheduled jobs | — | LiteHarness doesn't have this |
| Profile-based agents | `.ness/agents/` | Agent definitions |
| `~/.hermes/profiles/` | `.ness/agents/` | User-defined agent profiles |

### SOUL.md Pattern

Hermes uses a `SOUL.md` file that sits at the **very first position** in the system prompt:
```markdown
# SOUL

I am Hermes, a coding agent harness.
My purpose is to assist with software engineering tasks.
I value: correctness, efficiency, clarity, and user autonomy.
```

This is **hand-authored and never auto-generated**. It provides:
- Identity anchor (who am I)
- Value system (what do I prioritize)
- Behavioral north star (how do I make decisions)

### What LiteHarness Should Adopt

**A. Identity Anchor in L0**
LiteHarness's L0 could start with a concise identity statement:
```markdown
# LiteHarness Identity

You are LiteHarness, an experimental coding agent harness.
Your purpose: help engineers own the agent loop.
Core values: transparency, hackability, user control.
```

**B. Self-Improving Loop (Reflection enhancement)**
Hermes's skill improvement loop:
1. Agent completes task
2. Evaluates its own performance
3. Extracts lessons learned
4. Writes improvements back to skill files

LiteHarness's reflection already distills bullets. Consider:
- Reflection also suggests skill improvements
- Background job to update `.ness/skills/` based on session patterns
- `SKILL.md` auto-generation from repeated task patterns

---

## 8. Codex CLI (OpenAI) — Protocol & Hooks

### Architecture Mapping to LiteHarness

| Codex Component | LiteHarness Equivalent | Notes |
|----------------|----------------------|-------|
| Item/Turn/Thread protocol | Session events (SQLite) | Structured conversation persistence |
| Lifecycle hooks | `.ness/hooks.json` | Both have hook systems |
| Plan.md / Implement.md / Documentation.md | `.ness/plans/` | Structured planning artifacts |
| Versioned Skill bundles | `.ness/skills/` | Skill manifest with routing |
| `/responses/compact` endpoint | `/compact` command | Explicit compaction API |
| Managed shell container | Shell tool | Both execute shell commands |

### Planning Artifacts

Codex uses three structured files for long-horizon tasks:
```
Plan.md          # High-level plan with milestones
Implement.md     # Current implementation status
Documentation.md # Auto-generated docs
```

### What LiteHarness Should Adopt

**A. Structured Planning Artifacts (L3 enhancement)**
When in plan mode, LiteHarness could generate:
```
.ness/plans/
  current.md       # Current plan (loaded into L3)
  archive/         # Completed plans
```

**B. Skill Manifest with Negative Examples (L1)**
Codex's skill routing improved from 73% to 85% by adding negative examples:
```yaml
---
name: react_component
triggers: [react, component, tsx]
negative_triggers: [vue, angular, svelte]  # Don't trigger for these
---
```

---

# TIER 3: Reference & Inspiration

---

## 9. Augment Code — Role/Identity/Output Formatting

Augment's system prompt has explicit sections:
1. **Role**: "You are the Augment Agent, an agentic coding AI assistant"
2. **Identity**: "Based on GPT-5" (or Claude Sonnet 4)
3. **Output Formatting**: Structured response templates
4. **Preliminary Tasks**: What to do first
5. **Planning and Task Management**: How to plan
6. **Editing Instructions**: Code change rules
7. **Package Management**: Dependency rules
8. **Testing and Validation**: Verification requirements
9. **Communication**: How to interact with user
10. **Cost Balancing**: Token efficiency rules

### What to Adopt

LiteHarness's L0 could include a "Cost Balancing" section:
```markdown
## Token Efficiency
- Prefer targeted edits over full file rewrites
- Use grep/search to narrow scope before reading files
- Batch independent tool calls in parallel
- Summarize long outputs before returning to user
```

---

## 10. Devin — Phase-Gated Planning

Devin's planning module:
1. **Requirement Clarification** → ask_user with MCQ
2. **Specification** → structured spec document
3. **Implementation Plan** → numbered steps with file paths
4. **Task Execution** → step-by-step with verification

### What to Adopt

LiteHarness's plan mode already follows this. Enhance with:
- Explicit requirement clarification phase before research
- Structured spec output format
- Risk assessment per step

---

## 11. Kiro — Mode Classifier

Kiro uses a mode classifier to determine which phase to enter:
```
Mode Classifier → Specification → Implementation Plan → 
Requirement Clarification → Task Execution
```

### What to Adopt

LiteHarness could auto-detect when to enter plan mode:
- User asks "how should I..." → suggest plan mode
- User asks "implement..." → normal mode
- Complex multi-file task → auto-suggest plan mode

---

## 12. Cursor — Memory Rating System

Cursor's agent prompt includes a memory rating system:
- Rate how important each piece of context is
- Prioritize high-rated memories in context
- Evict low-rated memories when under pressure

### What to Adopt

LiteHarness's reflection could include memory importance scoring:
```markdown
Reflection output:
- [HIGH] User prefers functional components over class components
- [MEDIUM] Project uses pnpm, not npm
- [LOW] User likes dark theme
```

---

# SYNTHESIS: Recommended Prompt Structure for LiteHarness

Based on analysis of all agents, here is the recommended enhanced structure:

## L0: Harness Identity & Universal Rules (~2k tokens)

```markdown
# LiteHarness

You are LiteHarness, an experimental coding agent harness for engineers who want to own the loop.

## Identity
- Purpose: Help with software engineering tasks through tool use
- Values: Transparency, hackability, user control, correctness
- Tone: Concise, direct, professional. No filler, no emojis unless requested.

## Output Rules
- Answer in fewer than 4 lines unless detail is requested
- No preambles ("Okay, I will...") or postambles ("I have finished...")
- One-word answers are best when sufficient
- Use tools for actions; text output is for communication only
- Never use tool calls or code comments to communicate with user
- Reference code as `file_path:line_number`

## Tool Calling Protocol
- You can call multiple tools in parallel for independent operations
- Wait for tool results before proceeding with dependent operations
- Batch file reads, searches, and bash commands together
- Always use absolute paths for file operations

## Security
- Assist with defensive security only
- Never introduce code that exposes secrets, API keys, or credentials
- Never commit secrets to the repository
- Explain the purpose and impact of destructive bash commands before running

## Task Management
- Use the `todo` tool to plan any multi-step task
- Mark todos as completed immediately when done
- Do not batch multiple completions
- Break complex tasks into steps under 15 minutes each

## Git Safety
- Run `git status` before making changes (never use `-uall`)
- Run `git diff` to review changes before committing
- Check if current branch tracks remote
- NEVER commit unless explicitly asked by user

## Code Style
- Follow existing project conventions
- Verify library availability before using (check package.json, neighbors)
- DO NOT ADD COMMENTS unless explicitly requested
- Prefer editing existing files over creating new ones
- Use SEARCH/REPLACE format for targeted edits
```

## L1: Profile & Project Context (~3-5k tokens)

```markdown
# L1: Profile

## Persona
[Loaded from model-specific or user-defined persona]

## Tool Catalog
[Stable tool descriptions with usage guidelines]

## User Preferences (from USER.md)
[Cross-repo preferences]

## Project Conventions (from NESS.md)
[Project-specific rules, build commands, architecture]

## Skills (sticky, triggered)
[One-line descriptions of active skills]
```

## L2: Project Context Block (~1-2k tokens)

```markdown
# L2: Project Context

## Repository Structure
[Tree view of key directories]

## Git Availability
[Yes/No, branch, remote status]

## Sticky Skill Cores
[Full content of triggered skills]
```

## L3: Working State Overlay (ephemeral, ~1-2k tokens)

```markdown
<working-state>
## Git Snapshot
Branch: main | Dirty: 3 files | Remote: up to date

## Compaction Status
Pressure: 45% | Last compacted: turn 12 | Messages: 24

## Active Todos
- [x] Research auth patterns
- [ ] Implement JWT middleware
- [ ] Add tests

## Session Memory
- User prefers Zod over Joi for validation
- Project uses Bun runtime
- Last worked on: src/auth/middleware.ts
</working-state>

<plan-mode path=".ness/plans/">
## Current Plan
[When in plan mode, show plan instructions here]
</plan-mode>
```

---

# APPENDIX: Curated Collections for Reference

| Collection | URL | Best For |
|-----------|-----|----------|
| Piebald-AI/claude-code-system-prompts | github.com/Piebald-AI/claude-code-system-prompts | Complete Claude Code prompt history (515 prompts, 221 versions) |
| asgeirtj/system_prompts_leaks | github.com/asgeirtj/system_prompts_leaks | Multi-agent collection (Claude, GPT, Gemini, Grok, Cursor, etc.) |
| EliFuzz/awesome-system-prompts | github.com/EliFuzz/awesome-system-prompts | Largest collection with tools and modes |
| tallesborges/agentic-system-prompts | github.com/tallesborges/agentic-system-prompts | Side-by-side agent comparison |
| ai-boost/awesome-harness-engineering | github.com/ai-boost/awesome-harness-engineering | Academic papers on harness design |
| jwadow/agentic-prompts | github.com/jwadow/agentic-prompts | Reusable prompt templates |

---

# Key Takeaways

1. **Claude Code** has the most mature prompt engineering but suffers from prompt bloat (~40k tokens). LiteHarness's layered architecture is architecturally superior — keep it.

2. **OpenCode** is the closest sibling — study their provider-specific prompts, BUILD_SWITCH pattern, and AGENTS.md discovery.

3. **Pi** proves that minimal system prompts work if context engineering is strong. Consider making L0 even more concise.

4. **Roo Code** shows the power of mode-specific rule directories. Consider `.ness/rules/{mode}.md`.

5. **Cline** demonstrates explicit tool usage examples in prompts. Enhance tool descriptions with XML/JSON examples.

6. **Aider** has the best edit format documentation. Add SEARCH/REPLACE format instructions to L0.

7. **Hermes** shows the value of a static identity anchor (SOUL.md). Add a concise identity statement to L0.

8. **Codex CLI** validates structured planning artifacts (Plan.md, Implement.md). Enhance `.ness/plans/` with templates.
