# SDK guide

The **Ness Agent SDK** (`ness-agent` on PyPI) is a LangGraph-based agent harness you can embed in your own apps, scripts, and internal tools. It provides the agent loop, built-in tools, permissions, memory, skills, hooks, compaction, reflection, and optional tracing.

The **Ness CLI** is a reference coding adapter built on top of this SDK (`ness_cli`).

See also: [Architecture](architecture.md) · [Configuration](configuration.md) · [CLI guide](cli.md)

---

## Installation

```bash
pip install ness-agent
```

Optional OpenTelemetry tracing:

```bash
pip install ness-agent[tracing]
```

Requires **Python 3.12+**.

---

## Quick start

Minimal agent with a custom tool:

```python
import asyncio

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool

from ness_agent import NessAgent, PromptLayers, PromptLayersConfig


@tool
def ping() -> str:
    """Return pong."""
    return "pong"


async def main() -> None:
    model = FakeListChatModel(responses=["hello"])
    agent = NessAgent(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="You are a helpful agent.", persona="Be concise.")),
    )
    session = agent.session(thread_id="demo-1")
    result = await session.run("say hello")
    print(result.text)


asyncio.run(main())
```

Replace `FakeListChatModel` with any LangChain `BaseChatModel` (OpenAI-compatible, OpenRouter, etc.).

---

## Coding adapter (optional)

If you want the same wiring as the Ness CLI — OpenRouter models, `.ness/` paths, plan/act overlays, pricing — use the coding adapter:

```python
from ness_cli.factory import build_coding_session

coding = build_coding_session(thread_id="session-abc123")
async for event in coding.run_turn("add a rate limiter"):
    ...
```

This module is included in the same `ness-agent` package; the CLI entry point is `ness`.

---

## Public API

Core exports from `ness_agent`:

| Symbol | Purpose |
|--------|---------|
| `NessAgent`, `AgentSpec`, `NessAgentConfig` | Agent configuration and construction |
| `Session` | Run turns, stream events, manage thread state |
| `PromptLayers`, `PromptLayersConfig` | L0–L2 prompt assembly |
| `NessAgentOptions`, `MemoryConfig`, `ModeConfig` | Behavior toggles |
| `ToolRegistry`, `coding_tools` | Built-in and custom tools |
| `PermissionStore`, `HookRunner`, `SkillLoader` | Policy and extension points |
| `ThreadStore`, `MemoryStore` | Persistence backends |
| `CostTracker`, `TracingConfig`, `Tracer` | Usage and observability |
| `CodingOverlay`, `OverlayProvider` | Ephemeral L3 context (used by CLI) |

Import smoke test: `tests/test_sdk_smoke.py`.

---

## Prompt layers

The SDK splits prompts into L0–L3 layers for stable prefix caching. See [Architecture → Prompt layers](architecture.md#prompt-layers) for the full model.

When using the SDK directly, you supply L0–L2 via `PromptLayers` / `PromptLayersConfig`. L3 overlays are optional via `OverlayProvider` implementations such as `CodingOverlay`.

---

## Tracing

Install the tracing extra, then pass `TracingConfig` through `NessAgentOptions`:

```python
from ness_agent import NessAgent, NessAgentOptions
from ness_agent.tracing import TracingConfig

agent = NessAgent(
    model=model,
    prompt=prompt,
    options=NessAgentOptions(tracing=TracingConfig(...)),
)
```

See `tests/tracing/` for integration examples.

---

## Stability

Ness Agent is **0.x experimental**. Public APIs may change until 1.0. Pin versions in production and watch [CHANGELOG](../CHANGELOG.md).
