# LiteHarness SDK examples

Domain-agnostic agent harness in `src/liteharness/`. The coding CLI adapter lives in `src/liteharness_cli/` (tools, overlays, pricing, OpenRouter wiring).

Construct agents with `NessAgent(...)` kwargs (builds an `AgentSpec` internally) or `NessAgent.from_spec(AgentSpec(...))`. Compaction budget knobs live on `NessAgentOptions` — there is no separate budget config.

**App responsibilities (not done by bare `Session.run`):**

- Supply `l2_context` in the prompt when the model needs project/domain structure (SDK does not auto-load repo context).
- Append user events / call `thread_store.save_checkpoint` if you want resumable threads (the coding CLI does this around the graph).
- Pass `cost_tracker=make_sdk_cost_tracker()` from `liteharness_cli.config` when you want estimated USD for non-provider-cost models.

---

## Minimal coding agent (zero boilerplate)

`tools=` and `overlay=` are both optional. The SDK ships with a fully wired `CodingOverlay` (plan/act blocks, git snapshot, compaction note, todos, session memory, loaded skills) and defaults `TaskPrompts` (compaction / reflection / subagent / thread_summary / init_memory) to internal instruction texts — so a bare agent is a working coding agent.

```python
from liteharness import NessAgent, PromptLayersConfig
from langchain_openai import ChatOpenAI

agent = NessAgent(
    model=ChatOpenAI(model="gpt-4o"),
    prompt=PromptLayersConfig(),   # default L0 from liteharness.instructions.L0_HARNESS
    # tools=      omitted -> all SDK built-in tools (LOCAL_TOOLS)
    # overlay=    omitted -> CodingOverlay (plan/act, git, todos, compaction, ...)
    # task_prompts= omitted -> defaults to liteharness.instructions.{COMPACTION,REFLECTION,...}
)
session = agent.session(thread_id="proj-1")
# session.toggle_mode() flips plan <-> act using the SDK's default plan/act instruction texts
await session.run("Plan then implement: add a rate limiter on /api/login")
```

### Default overlay

- If `overlay=` is omitted, the agent is configured with `CodingOverlay` (`from liteharness import CodingOverlay`). It renders:
  - `<plan-mode path=".ness/plans/">...</plan-mode>` when `session.agent_mode == "plan"`, using `liteharness.instructions.PLAN_MODE` (or `modes.plan_mode_template` if you supply a `ModeConfig`)
  - `mode_switch` on the first act turn after a plan->act toggle, using `liteharness.instructions.ACT_MODE` (or `modes.act_mode_template`)
  - `git`, `compaction`, `todos`, `session_memory`, `loaded_skills`, and `skill_request` sections from the `OverlayContext`
- To **opt out of L3 entirely** pass `overlay=NoOverlay()` (apps that need no working-state overlay, or want to drive everything from the model alone).
- To use a **custom L3**, pass your own `OverlayProvider` (see the four examples below).

### Instruction texts are Python-importable

The default instruction bodies live in the `liteharness.instructions` package, not as opaque `.md` files:

```python
from liteharness.instructions import L0_HARNESS, COMPACTION, REFLECTION, SUBAGENT, THREAD_SUMMARY, INIT_MEMORY, PLAN_MODE, ACT_MODE, L1_PROFILE

# Copy and modify, then feed the modified text back in:
my_l0 = L0_HARNESS.replace("NESS", "Acme Assistant")
agent = NessAgent(
    model=...,
    prompt=PromptLayersConfig(l0=my_l0, persona="..."),
    # or override only one task prompt:
    # task_prompts=TaskPrompts(compaction=my_compaction_template),
)
```

### Tools: `BaseTool`, callable, or built-in name

`tools=` accepts a mix of `BaseTool` instances, plain callables (auto-wrapped with `StructuredTool.from_function`), and strings naming SDK built-ins (`"read"`, `"grep"`, `"glob"`, `"shell"`, ...):

```python
from liteharness import NessAgent, PromptLayersConfig

agent = NessAgent(
    model=...,
    prompt=PromptLayersConfig(),
    tools=["read", "grep", "glob", my_custom_fn],   # mixed list
)
```

---

## RAG Application

```python
from pathlib import Path
from liteharness import (
    NessAgent, NessAgentOptions, MemoryConfig,
    OverlayProvider, OverlayContext,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def vector_search(query: str, top_k: int = 5) -> str:
    """Semantic search over the indexed knowledge base."""
    ...

@tool
def fetch_chunk(doc_id: str, chunk_id: str) -> str:
    """Fetch a full chunk by id for deeper inspection."""
    ...

@tool
def cite_sources(sources: list[str]) -> str:
    """Record the sources used in the answer."""
    ...


class RAGOverlay(OverlayProvider):
    def sections(self, state, ctx: OverlayContext) -> dict[str, str]:
        sections = {}
        retrieval = ctx.metadata.get("retrieval_summary", "")
        if retrieval:
            sections["retrieval_context"] = f"RETRIEVED THIS TURN\n{retrieval}"
        score = ctx.metadata.get("grounding_score", "")
        if score:
            sections["confidence"] = f"Grounding score: {score}"
        open_q = ctx.metadata.get("open_questions", "")
        if open_q:
            sections["open_questions"] = f"Unanswered:\n{open_q}"
        return sections


agent = NessAgent(
    model=ChatOpenAI(model="gpt-4o", temperature=0),
    tools=[vector_search, fetch_chunk, cite_sources],
    prompt={
        "l0": (
            "You are a knowledge assistant. Answer ONLY from retrieved sources. "
            "Always cite doc_id. If context is insufficient, say so — do not invent facts."
        ),
        "persona": "Precise, citation-first research assistant for Acme internal docs.",
        "l2_context": kb_catalog.describe(),     # app-supplied; not auto-loaded
        "l2_header": "KNOWLEDGE BASE",
        "include_git_flag": False,
        "include_skill_catalog": False,
    },
    skills_dir=None,
    memory=MemoryConfig(
        project_memory=Path("./kb/POLICIES.md"),
        user_memory=Path("./users/u-42.md"),
        session_memory_dir=Path("./sessions"),
    ),
    overlay=RAGOverlay(),
    options=NessAgentOptions(
        enable_approval=False,
        context_window=128_000,               # drives compaction usable budget
        reflection_token_ratio=0.3,
    ),
)


async def answer(user_id: str, question: str) -> str:
    session = agent.session(thread_id=f"user-{user_id}")
    results = retriever.retrieve(question)
    session.metadata["retrieval_summary"] = results.summary
    session.metadata["grounding_score"] = results.score
    result = await session.run(question)
    await session.finalize_reflection()
    return result.assistant_message
```

## Deep Research Application

```python
from pathlib import Path
from liteharness import (
    NessAgent, NessAgentOptions, SubagentConfig,
    OverlayProvider, OverlayContext,
)
from langchain_openai import ChatOpenAI
from my_tools import web_search, webfetch, save_note, spawn_subagent, build_report


class ResearchOverlay(OverlayProvider):
    """L3: outstanding questions, sources collected, report outline."""
    def sections(self, state, ctx: OverlayContext) -> dict[str, str]:
        sections = {}
        outline = ctx.metadata.get("outline", "")
        if outline:
            sections["outline"] = f"REPORT OUTLINE\n{outline}"
        sources = ctx.metadata.get("sources_collected", [])
        if sources:
            sections["sources"] = "SOURCES\n" + "\n".join(f"- {s}" for s in sources[-15:])
        todos = ctx.todos
        if todos:
            lines = "\n".join(f"- [{t.get('status')}] {t.get('content')}" for t in todos
                              if t.get("status") != "completed")
            if lines: sections["plan"] = f"RESEARCH PLAN\n{lines}"
        return sections


agent = NessAgent(
    model=ChatOpenAI(model="gpt-4o"),
    compaction_model=ChatOpenAI(model="gpt-4o-mini"),
    reflection_model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[web_search, webfetch, save_note, spawn_subagent, build_report],
    prompt={
        "l0": (
            "You are a research analyst. Work in phases: scope → gather → synthesize. "
            "Cite every claim with URL and access date. Prefer primary sources. "
            "Use spawn_subagent to parallelize independent research threads."
        ),
        "persona": "Thorough, skeptical analyst producing structured reports.",
        "l2_context": f"Research brief: {brief}\nDeadline: {deadline}\nOutput format: markdown",
        "l2_header": "RESEARCH BRIEF",
        "include_git_flag": False,
    },
    skills_dir=Path("./skills/research"),
    subagents=SubagentConfig(
        prompt_template=RESEARCH_SUBAGENT_PROMPT,
        max_parallel=3,
        default_tools=("web_search", "webfetch", "save_note"),
        default_timeout_seconds=600,
    ),
    overlay=ResearchOverlay(),
    options=NessAgentOptions(
        enable_approval=True,
        context_window=200_000,
        reflection_token_ratio=0.25,
        auto_save_threads=True,
    ),
)


async def research(topic: str) -> str:
    session = agent.session(thread_id=f"research-{slugify(topic)}")
    session.metadata["outline"] = initial_outline(topic)
    session.metadata["sources_collected"] = []
    # Optional: persist a user event for resumable threads
    # agent.config.thread_store.append_event(session.thread_id, {"kind": "user", "content": topic})
    result = await session.run(topic, mode="act")
    await session.finalize_reflection()
    return result.assistant_message
```

## Video Generation Application

```python
from liteharness import (
    NessAgent, NessAgentOptions, SubagentConfig,
    OverlayProvider, OverlayContext,
)
from langchain_openai import ChatOpenAI
from my_tools import (
    generate_scene, render_clip, stitch_clips, upload_to_bucket,
    review_storyboard, get_asset_duration,
)


class VideoOverlay(OverlayProvider):
    """L3: storyboard progress, render queue, asset inventory."""
    def sections(self, state, ctx: OverlayContext) -> dict[str, str]:
        sections = {}
        queue = ctx.metadata.get("render_queue", [])
        if queue:
            sections["render_queue"] = "RENDER QUEUE\n" + "\n".join(
                f"- {job['clip_id']}: {job['status']}" for job in queue[:10])
        assets = ctx.metadata.get("assets", {})
        if assets:
            lines = "\n".join(f"- {name}: {info['duration']:.1f}s, {info['resolution']}"
                              for name, info in assets.items())
            sections["assets"] = f"ASSET INVENTORY\n{lines}"
        budget = ctx.metadata.get("render_budget_seconds", 0)
        spent = ctx.metadata.get("render_seconds_used", 0)
        if budget:
            sections["budget"] = f"Render budget: {spent:.1f}/{budget:.1f}s used"
        todos = ctx.todos
        if todos:
            lines = "\n".join(f"- [{t.get('status')}] Scene {t.get('id')}: {t.get('content')}"
                              for t in todos if t.get("status") != "completed")
            if lines: sections["storyboard"] = f"STORYBOARD\n{lines}"
        return sections


agent = NessAgent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[generate_scene, render_clip, stitch_clips, upload_to_bucket,
           review_storyboard, get_asset_duration],
    prompt={
        "l0": (
            "You are a video director. Break the brief into scenes, generate each, "
            "review, stitch, and upload. Respect the render-second budget. "
            "If a scene fails review, regenerate before stitching. "
            "Never upload until all clips pass quality check."
        ),
        "persona": "Efficient video director optimizing for quality within time budget.",
        "l2_context": f"Project: {project_name}\nStyle: {style_guide}\n"
                      f"Duration target: {target_duration}s\nResolution: {resolution}",
        "l2_header": "VIDEO BRIEF",
        "include_git_flag": False,
    },
    modes=None,
    subagents=SubagentConfig(
        prompt_template=VIDEO_REVIEW_SUBAGENT_PROMPT,
        max_parallel=2,
        default_tools=("render_clip", "review_storyboard", "get_asset_duration"),
        default_timeout_seconds=900,
    ),
    overlay=VideoOverlay(),
    options=NessAgentOptions(
        enable_approval=True,
        context_window=200_000,
        reflection_token_ratio=0.0,
        auto_save_threads=True,
    ),
)


async def produce_video(brief: str) -> str:
    session = agent.session(thread_id=f"video-{project_id}")
    session.metadata["render_budget_seconds"] = 300
    session.metadata["render_seconds_used"] = 0
    session.metadata["assets"] = {}
    session.metadata["render_queue"] = []
    result = await session.run(brief)
    return result.assistant_message
```

## Customer Support Application

```python
from pathlib import Path
from liteharness import (
    AgentSpec, NessAgent, NessAgentOptions, MemoryConfig,
    OverlayProvider, OverlayContext, ApprovalHandler,
)
from langchain_openai import ChatOpenAI
from my_tools import (
    lookup_order, search_kb, create_ticket, escalate_to_human,
    update_account, send_email, fetch_conversation_history,
)


class SupportOverlay(OverlayProvider):
    """L3: customer context, SLA timer, escalation state."""
    def sections(self, state, ctx: OverlayContext) -> dict[str, str]:
        sections = {}
        customer = ctx.metadata.get("customer", {})
        if customer:
            lines = [f"Customer: {customer.get('name', 'unknown')}",
                     f"Tier: {customer.get('tier', 'standard')}",
                     f"Account age: {customer.get('account_age_days', 0)} days"]
            sections["customer"] = "CUSTOMER\n" + "\n".join(lines)
        history = ctx.metadata.get("recent_conversations", "")
        if history:
            sections["history"] = f"RECENT CONVERSATIONS\n{history}"
        sla = ctx.metadata.get("sla_minutes_remaining")
        if sla is not None:
            sections["sla"] = f"SLA: {sla:.0f} minutes remaining"
        escalated = ctx.metadata.get("escalated", False)
        if escalated:
            sections["escalation"] = "ESCALATED to human — follow their guidance only."
        sentiment = ctx.metadata.get("sentiment", "")
        if sentiment:
            sections["sentiment"] = f"Detected sentiment: {sentiment}"
        return sections


class SupportApproval(ApprovalHandler):
    """Auto-approve safe actions; escalate destructive ones to a human queue."""
    SAFE_ACTIONS = {"lookup_order", "search_kb", "fetch_conversation_history", "send_email"}
    async def __call__(self, tool: str, args: dict) -> str:
        if tool in self.SAFE_ACTIONS:
            return "yes"
        await enqueue_for_human_review(tool, args)
        return "no"


# Equivalent to NessAgent(...): build an AgentSpec then resolve
agent = NessAgent.from_spec(AgentSpec(
    model=ChatOpenAI(model="gpt-4o", temperature=0.2),
    tools=[lookup_order, search_kb, create_ticket, escalate_to_human,
           update_account, send_email, fetch_conversation_history],
    prompt={
        "l0": (
            "You are a customer support assistant for Acme Corp. "
            "Be concise, empathetic, and action-oriented. "
            "Always look up the order/account before making changes. "
            "Escalate to human when: customer is upset, issue is billing-related, "
            "or you've attempted 2 fixes without resolution. "
            "Never make account changes without customer confirmation."
        ),
        "persona": "Empathetic, efficient support specialist. Concise responses.",
        "l2_context": "Product: Acme SaaS\nSupport hours: 24/7\nEscalation channel: #support-escalations",
        "l2_header": "SUPPORT CONTEXT",
        "include_git_flag": False,
        "include_skill_catalog": True,
    },
    skills_dir=Path("./skills/support"),
    memory=MemoryConfig(
        project_memory=Path("./support/POLICIES.md"),
        user_memory=Path(f"./customers/{customer_id}.md"),
        session_memory_dir=Path("./support/sessions"),
    ),
    overlay=SupportOverlay(),
    approval_handler=SupportApproval(),
    options=NessAgentOptions(
        enable_approval=True,
        context_window=64_000,
        reflection_token_ratio=0.5,
        session_end_reflection=True,
        auto_save_threads=True,
    ),
))


async def handle_message(customer_id: str, message: str) -> str:
    customer = await load_customer(customer_id)
    history = await load_recent_conversations(customer_id, limit=5)
    session = agent.session(thread_id=f"support-{customer_id}-{ticket_id}")
    session.metadata["customer"] = customer
    session.metadata["recent_conversations"] = history
    session.metadata["sla_minutes_remaining"] = await sla_remaining(ticket_id)
    session.metadata["sentiment"] = analyze_sentiment(message)
    result = await session.run(message)
    await session.finalize_reflection()
    return result.assistant_message
```

---

Same harness across all four: turn loop, compaction, reflection, permissions, skills, tool execution. Apps differ only in tools, prompts, overlay, memory paths, approval, modes, options (including `context_window`), and subagent config.
