<p align="center">
  <img src="assets/banner-light-geo.svg" alt="Ness AI — hackable coding-agent harness" width="100%">
</p>

# Ness AI

[![CI](https://github.com/Sagnnik/ness-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Sagnnik/ness-agent/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/ness-ai)](https://pypi.org/project/ness-ai/)
[![License](https://img.shields.io/github/license/Sagnnik/ness-agent)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ness-ai)](https://pypi.org/project/ness-ai/)

Ness AI is an experimental, hackable coding-agent harness for engineers who want to own the loop. It ships as a **Python SDK** you can embed in your own tools and **Ness**, an interactive CLI for day-to-day coding sessions.

> **0.x experimental** — APIs may change until 1.0. See [CHANGELOG](CHANGELOG.md).

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
| **Ness AI SDK** | LangGraph agent loop, built-in tools, permissions, memory, skills, hooks, MCP, compaction, reflection, tracing |
| **Ness CLI** | Terminal UI (`ness`), plan/act modes, git worktrees, global config, `.ness/` project layout |

Both are included in the `ness-ai` PyPI package. OpenRouter-compatible chat models, native tool-calling, and filesystem-driven extension points under `.ness/`.

## Installation

Requires **Python 3.12+**.

```bash
pip install ness-ai
```

Optional tracing support:

```bash
pip install ness-ai[tracing]
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

Full CLI reference: [docs/cli.md](docs/cli.md) · Configuration: [docs/configuration.md](docs/configuration.md)

## Quick start — SDK

```python
import asyncio
from langchain_core.tools import tool
from ness_ai import NessAgent, PromptLayers, PromptLayersConfig

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

Full SDK guide: [docs/sdk.md](docs/sdk.md) · Architecture: [docs/architecture.md](docs/architecture.md)

## Documentation

| Guide | Description |
|-------|-------------|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/sdk.md](docs/sdk.md) | SDK usage, public API, tracing |
| [docs/cli.md](docs/cli.md) | Ness TUI, slash commands, MCP, permissions |
| [docs/configuration.md](docs/configuration.md) | Global config, `.ness/` layout, env vars |
| [docs/architecture.md](docs/architecture.md) | Prompt layers, modes, memory, compaction |

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup and PR guidelines.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
