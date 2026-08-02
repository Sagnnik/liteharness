---
title: "Harness Engineering: A Ness Agent Intro"
date: "2026-08-01"
description: "An opening field note on keeping the coding-agent loop inspectable, replaceable, and close to the engineer."
slug: harness-engineering-a-ness-agent-intro
---

## The loop is a surface, not a black box

Coding agents are often presented as a finished experience: type a request, wait for an answer, accept the change. That is useful right up until the work needs a different tool policy, a better memory boundary, or a project-specific instruction layer.

Ness Agent starts one level lower. It is an experimental harness for the loop itself: the messages, tools, permissions, persistence, context, and the thin operator interface that connects them. The goal is not to hide those pieces. It is to make their seams legible enough to inspect and change.

## SDK or CLI: choose the surface that fits the work

The Python SDK is the embedding surface. Use it when an internal tool, workflow, or product needs an agent loop with its own model, tools, prompt layers, and storage choices.

The `ness` CLI is the everyday coding surface. It wires the same system into an interactive terminal session, including plan/act modes, thread history, and git worktrees. A headless turn is available when the surrounding automation only needs a final response:

```bash
ness -p "map the authentication module"
```

The useful distinction is not beginner versus advanced. It is whether the surrounding application is the operator, or you are.

## Keep extensions local

The `.ness/` directory gives a repository a place to say how the harness should behave. Project conventions live in `NESS.md`; permissions, hooks, MCP servers, skills, custom commands, and saved thread state have visible homes beside it.

That makes extension work reviewable. A new skill can be versioned with the project. A permission change can be diffed. A hook can be traced to a file instead of an invisible account setting. The default is not that every project needs every extension; the default is that a project can own the ones it does need.

## The first useful question

Before adding capability, ask which part of the loop needs to change. It may be an operator workflow in the CLI, an overlay that supplies current repository state, a tool registry, or an application-specific SDK integration. Naming the layer turns a vague request into an engineering decision.

Ness is early software, and its 0.x interfaces will move. The more important promise is simpler: the loop stays close enough to be yours.
