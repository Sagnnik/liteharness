# Configuration

Ness splits **global** user data, **project** config, and **runtime** cache.

See also: [CLI guide](cli.md) · [Architecture](architecture.md)

---

## Directory layout

```text
# Global config (platformdirs user_config_dir("ness-agent"))
# Linux: ~/.config/ness-agent/
# macOS: ~/Library/Application Support/ness-agent/
# Windows: %APPDATA%\ness-agent\
USER.md                  Cross-repo user preferences
configs.json             Non-secret adapter settings (only values you changed)
secrets.json             API keys and other secrets (mode 0600)
instructions/            Editable prompt templates (L0, persona, plan/act, aux, goal)
plans/<project-slug>/    Saved plan-mode output for this project

# Per-project cache (platformdirs user_cache_dir("ness-agent")/<hash>/)
cli_history              Prompt history for this project root

# Per-project .ness/ (NESS_DIR, default ".ness")
.ness/
├── NESS.md              Project conventions loaded into L1
├── permissions.json     Tool allow/deny/ask rules
├── hooks.json           Hook commands
├── mcp.json             MCP stdio servers
├── agents/              Subagent definitions
├── commands/            User slash commands
├── skills/              Project-local SKILL.md skills
├── threads/             Saved session trajectories (SQLite)
│   └── threads.db
└── runtime/
    ├── sessions/        Per-thread episodic memory (L3)
    │   └── mem_<thread_id>.md
    └── shells/          Background shell job metadata and logs
```

Override roots with `NESS_AGENT_CONFIG_DIR`, `NESS_AGENT_CACHE_DIR`, and `NESS_DIR`.

---

## Settings resolution

Settings resolve in this order (highest wins):

1. CLI flags
2. Process environment variables
3. `secrets.json` / `configs.json`
4. Built-in defaults

`configs.json` is written lazily — it only contains values you changed via `/config` (defaults stay in code and evolve with upgrades).

On first start, known keys from an existing project `.env` are imported into the JSON files once (the `.env` is left untouched). You can also set the API key in-session: `/config` → Provider → Provider API key.

---

## Environment variables

All except `NESS_DIR` are also editable via `/config` in the Ness TUI.

| Variable | Description |
|----------|-------------|
| `MODEL_NAME` | Model passed to `ChatOpenRouter` (`deepseek/deepseek-v4-flash` by default) |
| `REFLECTION_MODEL_NAME` | Model for background session-memory reflection (defaults to `MODEL_NAME`) |
| `ENABLE_APPROVAL` | Require approval for destructive tools |
| `AUTO_SAVE_THREADS` | Write thread events to `.ness/threads/` |
| `SESSION_END_REFLECTION` | Run a final reflection pass when a session ends (default off) |
| `REFLECTION_TOKEN_RATIO` | Fraction of usable context that must accumulate before reflection (default `0.4`; set `0` to disable) |
| `API_MAX_RETRIES` | Retries for chat API calls (default `3`) |
| `COMPACTION_OUTPUT_RESERVE` | Output reserve subtracted from model context window (default `8192`) |
| `COMPACTION_INPUT_RESERVE` | Input/system/tool reserve subtracted from model context window (default `4096`) |
| `COMPACTION_TOKEN_BUDGET` | Fallback compaction budget when model context window is unknown (default `120000`) |
| `OPENROUTER_SESSION_ID` | Optional stable prompt-cache session id (defaults to active thread id) |
| `OPENROUTER_CACHE_TTL` | Anthropic prompt-cache lifetime (`5m` by default; `1h` supported) |
| `OPENROUTER_ANTHROPIC_MESSAGES` | Use OpenRouter Messages API for Anthropic models (default `true`) |
| `GOAL_JUDGE_MODEL` | Model for independent `/goal` judge (defaults to `REFLECTION_MODEL_NAME`) |
| `GOAL_MAX_ATTEMPTS` | Maximum worker/judge attempts for `/goal` (default `3`) |
| `OPENAI_BASE_URL` | Optional custom OpenAI-compatible base URL |
| `OPENAI_API_KEY` | Provider API key (also stored in `secrets.json` via `/config`) |
| `FORMAT_ON_WRITE` | Auto-format supported file types after writes (default `true`) |
| `NESS_DIR` | Project config directory (default `.ness`) |
| `NESS_AGENT_CONFIG_DIR` | Override global config root |
| `NESS_AGENT_CACHE_DIR` | Override cache root (OpenRouter catalog + per-project `cli_history`) |
| `EXA_API_KEY` | Optional Exa API key for higher-quality `web_search` and `fetch_url` ([exa.ai](https://exa.ai)) |

### CLI flags

Flags override env for a single run: `--model`, `--reflection-model`, `--api-key`, `--base-url`, `--openrouter-session-id`, `--reasoning-effort`, `--worktree` / `-w`, `--print` / `-p`, and `--yolo`.

`--yolo` is session-only and bypasses approval prompts and persisted permission denials in act mode; hook vetoes and plan-mode read-only rules still apply.

Use `/config` in-session to edit provider keys/endpoints, model and reasoning, approval/autosave/reflection behavior, compaction budgets, and more.

The `/config` model picker lazily refreshes text-output, tool-capable LLMs and VLMs from OpenRouter. Its global disk cache is reused for 24 hours, stale data remains available while refreshing, and the packaged model list is the offline fallback.
