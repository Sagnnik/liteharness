<p align="center">
  <img src="https://raw.githubusercontent.com/Sagnnik/ness-agent/main/assets/banner-light-geo.svg" alt="Ness Agent — hackable coding-agent harness" width="100%">
</p>

# Ness Agent

[![CI](https://github.com/Sagnnik/ness-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Sagnnik/ness-agent/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/Sagnnik/ness-agent)](LICENSE)

Ness Agent is an experimental, hackable coding-agent harness for engineers who want to own the loop. It ships as a **Python SDK** you can embed in your own tools and **Ness**, an interactive CLI for day-to-day coding sessions.

> **0.x experimental** — APIs may change until 1.0. See [CHANGELOG](https://github.com/Sagnnik/ness-agent/blob/main/CHANGELOG.md).

## Table of contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick start — CLI (Ness)](#quick-start--cli-ness)
- [Quick start — SDK](#quick-start--sdk)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview

| Component | What it is |
|-----------|------------|
| **Ness Agent SDK** | LangGraph agent loop, built-in tools, permissions, memory, skills, hooks, MCP, compaction, reflection, tracing |
| **Ness CLI** | Terminal UI (`ness`), plan/act modes, git worktrees, global config, `.ness/` project layout |

Both are included in the `ness-agent` PyPI package. OpenRouter-compatible chat models, native tool-calling, and filesystem-driven extension points under `.ness/`.

## Installation

Requires **Python 3.12+**.

```bash
pip install ness-agent
```

Optional tracing support:

```bash
pip install ness-agent[tracing]
```

**From source:**

Contributors (run tests, use project venv):

```bash
git clone https://github.com/Sagnnik/ness-agent.git
cd ness-agent
uv sync
uv run ness
```

Install `ness` on your PATH from a local clone (editable):

```bash
git clone https://github.com/Sagnnik/ness-agent.git
cd ness-agent
uv tool install -e .
ness
```

Or with pip: `pip install -e .` (then run `ness` from that environment).

## Quick start — CLI (Ness)

```bash
export OPENAI_API_KEY=...   # or set via /config on first launch
ness
/init                       # create .ness/ and global config
```

Headless one-shot:

```bash
ness -p "what does the auth module do?"
```

Parallel isolated session:

```bash
ness --worktree feature-x
```

Full CLI reference: [docs/cli.md](https://github.com/Sagnnik/ness-agent/blob/main/docs/cli.md) · Configuration: [docs/configuration.md](https://github.com/Sagnnik/ness-agent/blob/main/docs/configuration.md)

## Quick start — SDK

```python
import asyncio
from langchain_core.tools import tool
from ness_agent import NessAgent, PromptLayers, PromptLayersConfig

@tool
def ping() -> str:
    """Return pong."""
    return "pong"

async def main() -> None:
    agent = NessAgent(
        model=your_chat_model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="You are a helpful agent.", persona="Be concise.")),
    )
    session = agent.session(thread_id="demo-1")
    result = await session.run("say hello")
    print(result.text)

asyncio.run(main())
```

Full SDK guide: [docs/sdk.md](https://github.com/Sagnnik/ness-agent/blob/main/docs/sdk.md) · Architecture: [docs/architecture.md](https://github.com/Sagnnik/ness-agent/blob/main/docs/architecture.md)

## Documentation

| Guide | Description |
|-------|-------------|
| [docs/README.md](https://github.com/Sagnnik/ness-agent/blob/main/docs/README.md) | Documentation index |
| [docs/sdk.md](https://github.com/Sagnnik/ness-agent/blob/main/docs/sdk.md) | SDK usage, public API, tracing |
| [docs/cli.md](https://github.com/Sagnnik/ness-agent/blob/main/docs/cli.md) | Ness TUI, slash commands, MCP, permissions |
| [docs/configuration.md](https://github.com/Sagnnik/ness-agent/blob/main/docs/configuration.md) | Global config, `.ness/` layout, env vars |
| [docs/architecture.md](https://github.com/Sagnnik/ness-agent/blob/main/docs/architecture.md) | Prompt layers, modes, memory, compaction |

## Contributing

Contributions welcome. See [CONTRIBUTING.md](https://github.com/Sagnnik/ness-agent/blob/main/CONTRIBUTING.md) for dev setup and PR guidelines.

## License

Licensed under the Apache License 2.0. See [LICENSE](https://github.com/Sagnnik/ness-agent/blob/main/LICENSE).
