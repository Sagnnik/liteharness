"""Semantic-convention attribute and span-name constants.

Uses the OpenTelemetry GenAI semantic conventions where they exist
(``gen_ai.*``) so OTel-compatible backends (Tempo, Jaeger, Langfuse, Grafana)
chart token usage natively. Ness AI-specific extras use a ``ness_ai.*``
prefix so they remain clearly custom.

Span names are dotted lowercase identifiers. ``TOOL_EXEC`` is a format string
— pass the tool name through ``.format(name=...)``.
"""

from __future__ import annotations

# --- span names ---------------------------------------------------------
TURN = "session.turn"
LLM_CALL = "agent.llm_call"
TOOL_EXEC = "tool.{name}"
COMPACTION_SUMMARIZE = "compaction.summarize"
REFLECTION = "reflection.gate"
APPROVAL = "approval.check"

# --- OTel GenAI semantic conventions ------------------------------------
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_CACHE_READ_TOKENS = "gen_ai.usage.cache_read_tokens"
GEN_AI_USAGE_COST_USD = "gen_ai.usage.cost_usd"
GEN_AI_USAGE_CACHE_HIT_RATE = "gen_ai.usage.cache_hit_rate"

# Message-content attributes (deprecated-but-widely-supported GenAI convention).
# Newer ``gen_ai.input.messages`` / ``gen_ai.output.messages`` are still in development
GEN_AI_PROMPT = "gen_ai.prompt"
GEN_AI_COMPLETION = "gen_ai.completion"
GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"

# Convenience aliases for instrumentation call sites (shorter names).
MODEL_NAME = GEN_AI_REQUEST_MODEL
INPUT_TOKENS = GEN_AI_USAGE_INPUT_TOKENS
OUTPUT_TOKENS = GEN_AI_USAGE_OUTPUT_TOKENS
CACHE_READ_TOKENS = GEN_AI_USAGE_CACHE_READ_TOKENS
COST_USD = GEN_AI_USAGE_COST_USD
CACHE_HIT_RATE = GEN_AI_USAGE_CACHE_HIT_RATE

# --- tool attributes ----------------------------------------------------
TOOL_NAME = "tool.name"
TOOL_DURATION_MS = "tool.duration_ms"
TOOL_ERROR = "tool.error"
TOOL_ARGS = "tool.args"
TOOL_EXIT_STATUS = "tool.exit_status"

# --- session attributes ------------------------------------------------
THREAD_ID = "session.thread_id"
AGENT_MODE = "session.mode"
TURN_COUNT = "session.turn_count"

# --- span kinds --------------------------------------------------------
KIND_INTERNAL = "internal"
KIND_CLIENT = "client"

# --- fixed values ------------------------------------------------------
GEN_AI_SYSTEM_VALUE = "ness-ai"