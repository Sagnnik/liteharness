# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Cache-safe `ness_agent.summarize()` API and durable summary checkpoints for resume/rollback.
- Canonical model-facing history that preserves ordinary request prefixes while keeping L3 reminders out of durable transcripts.

### Changed

- Compaction now uses the main bound model, identical tools/session/system prefix, a human tail instruction, and a boundary-safe graph node.
- `/compact` retains the active user/tool turn verbatim and summarizes completed history only.
- Compaction checkpoints atomically retain the active semantic suffix, preventing SDK resume from dropping an unlogged user turn.
- Cache-safe forks retain the last successful model/tool binding; canonical image blocks are no longer stripped before compaction, and session pressure includes the stable system prefix.

### Removed

- Progressive tool-output compaction, the 40-message summary limit, `compaction_model`, `progressive_compact`, and `summarize_history`.
- `COMPACTION_INPUT_RESERVE` and `COMPACTION_OUTPUT_RESERVE`; use `COMPACTION_BUFFER_TOKENS` and `COMPACTION_SUMMARY_MAX_TOKENS`.

## [0.1.0] - 2026-07-31 — Released

### Added

- Initial public release of **Ness Agent** (SDK) and **Ness** (CLI).
- SDK: LangGraph agent loop, built-in tools, permissions, memory, skills, hooks, MCP, compaction, reflection, and tracing.
- CLI: interactive TUI (`ness`), headless print mode (`-p`), plan/act modes, git worktrees, global config, and `.ness/` project layout.

[Unreleased]: https://github.com/Sagnnik/ness-agent/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Sagnnik/ness-agent/releases/tag/v0.1.0
