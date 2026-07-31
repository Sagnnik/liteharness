"""Dynamic MCP tool discovery: a lightweight BM25 search over the MCP catalog
plus an activation tool. Both are always bound so the agent can find and load
deferred MCP tools on demand without bloating the prefix until needed.

Catalog and activation are owned by the session :class:`ToolRegistry`
(resolved via :func:`~ness_ai.session_context.get_session_context`).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

from langchain_core.tools import tool

if TYPE_CHECKING:
    from ness_ai.tools import ToolRegistry

# Loaded-tool soft cap: warn the model/user once the bound set gets large enough
# to hurt tool-selection accuracy.
TOOL_COUNT_WARN_THRESHOLD = 35

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


def _tool_registry() -> ToolRegistry:
    """Resolve the active session's ToolRegistry (sole MCP source of truth)."""
    from ness_ai.session_context import get_session_context

    ctx = get_session_context()
    reg = getattr(ctx.agent_config, "tool_registry", None) if ctx.agent_config else None
    if reg is None:
        raise RuntimeError(
            "No ToolRegistry on agent_config. Create a Session (or set_session_context "
            "with a wired NessAgentConfig) before invoking search_tools / add_tools."
        )
    return reg


def _catalog_documents() -> list[dict[str, Any]]:
    """Flatten the MCP catalog into searchable documents (one per deferred tool)."""
    reg = _tool_registry()
    active = reg.active_mcp_tools

    docs: list[dict[str, Any]] = []
    for server, info in reg.mcp_catalog().items():
        server_desc = str(info.get("description") or "")
        for entry in info.get("tools", []):
            name = str(entry.get("name") or "")
            if not name or name in active:
                continue
            short = str(entry.get("tool") or "")
            description = str(entry.get("description") or "")
            arg_names = " ".join(str(a) for a in entry.get("arg_names", []))
            blob = " ".join([short, name, server, server_desc, description, arg_names])
            docs.append(
                {
                    "name": name,
                    "description": description,
                    "tokens": _tokenize(blob),
                }
            )
    return docs


def _bm25_rank(query: str, docs: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """
    BM25 ranking algorithm for document retrieval.
    it ranks docs on the basis of these 3 factors:
    1. Term Frequency (TF): How often a term appears in the document. Frequency is rated higher.
    2. Inverse Document Frequency (IDF): How rare a term is across all documents. Rarity is rated higher.
    3. Length Normalization: Adjust for document length. Penalizes longer documents.
    The score is calculated by the formula:
    score = sum(IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * length / avg_len)))
    for each term in the query.
    
    where:
    - IDF = log(1 + (n - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
    - TF = counts.get(term, 0)
    - k1 = 1.5
    - b = 0.75
    - avg_len = sum(len(doc["tokens"]) for doc in docs) / n
    - length = len(tokens) or 1
    - counts = Counter(tokens)
    - doc_freq = Counter()
    
    The final score is the sum of the scores for all terms.
    The document with the highest score is ranked highest.
    """
    query_terms = _tokenize(query)
    if not query_terms or not docs:
        return []

    n = len(docs)
    doc_freq: Counter[str] = Counter()
    for doc in docs:
        for term in set(doc["tokens"]):
            doc_freq[term] += 1

    avg_len = sum(len(doc["tokens"]) for doc in docs) / n
    k1, b = 1.5, 0.75

    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in docs:
        tokens = doc["tokens"]
        length = len(tokens) or 1
        counts = Counter(tokens)
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            idf = math.log(1 + (n - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = tf + k1 * (1 - b + b * length / avg_len)
            score += idf * (tf * (k1 + 1)) / denom
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:limit]]


@tool
def search_tools(query: str, limit: int = 5) -> str:
    """Search the catalog of available (not-yet-loaded) MCP tools by capability.

    Use this when a task needs an external capability that is not in your current
    tools. It returns the most relevant MCP tool names with short descriptions;
    load the ones you need with add_tools before calling them.
    """
    limit = max(1, min(int(limit or 5), 10))
    docs = _catalog_documents()
    if not docs:
        return "No additional MCP tools are available to load."

    matches = _bm25_rank(query, docs, limit)
    if not matches:
        return f"No MCP tools matched '{query}'. Try different keywords."

    lines = [f"Found {len(matches)} tool(s) (load with add_tools):"]
    for doc in matches:
        desc = doc["description"].strip().replace("\n", " ")
        if len(desc) > 160:
            desc = desc[:157] + "..."
        lines.append(f"- {doc['name']}" + (f": {desc}" if desc else ""))
    return "\n".join(lines)


@tool
def add_tools(names: list[str]) -> str:
    """Load one or more MCP tools (by full name, e.g. mcp__server__tool) so they
    become callable. Find names first with search_tools. Loading is sticky for the
    rest of the session.
    """
    if not names:
        return "Error: add_tools requires at least one tool name"

    reg = _tool_registry()
    added, unknown = reg.activate_mcp(names)

    parts: list[str] = []
    if added:
        parts.append(f"Loaded {len(added)} tool(s): {', '.join(sorted(added))}")
    already = [n for n in names if n not in added and n not in unknown]
    if already:
        parts.append(f"Already loaded: {', '.join(sorted(set(already)))}")
    if unknown:
        parts.append(f"Unknown (not in catalog): {', '.join(sorted(set(unknown)))}")

    total = len(reg.tool_names())
    if total > TOOL_COUNT_WARN_THRESHOLD:
        parts.append(
            f"Warning: {total} tools now loaded (> {TOOL_COUNT_WARN_THRESHOLD}); "
            "tool-selection accuracy may degrade."
        )

    if not added and not already and not unknown:
        return "No tools loaded."
    return "\n".join(parts)
