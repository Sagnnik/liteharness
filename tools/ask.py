from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Awaitable, Callable

from langchain_core.tools import tool

# Async handler injected by the runtime (the CLI) that floats MCQ questions to the user and returns the chosen answers. 
# QuestionHandler is a callable that takes a list of questions and returns an awaitable list of answers.
QuestionHandler = Callable[[list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]]

_question_handler: ContextVar[QuestionHandler | None] = ContextVar(
    "question_handler", default=None
)


def set_question_runtime(handler: QuestionHandler | None) -> None:
    """Set the interactive question handler for the current run."""
    _question_handler.set(handler)


@tool
async def ask_user(questions: list[dict] | None = None) -> str:
    """Ask the user multiple-choice clarification questions and wait for answers.

    Use this when requirements are ambiguous or several valid approaches exist,
    before committing to a plan or implementation. Each question is a dict:
      {
        "id": "approach",                      # optional stable id
        "prompt": "Which storage backend?",    # required question text
        "options": [                            # 2+ options
          {"id": "redis", "label": "Redis", "recommended": true},
          {"id": "memory", "label": "In-memory"}
        ],
        "allow_note": true                      # optional free-form note (default true)
      }
    Mark the best option with "recommended": true. The user picks one option per
    question and may add a note. Returns the questions with the chosen answers.
    """
    normalized = _normalize_questions(questions)
    if isinstance(normalized, str):
        return f"Error: {normalized}"

    handler = _question_handler.get()
    if handler is None:
        return (
            "Error: no interactive question handler available in this runtime; "
            "proceed using your best judgment and state your assumptions."
        )

    try:
        answers = await handler(normalized)
    except Exception as exc:  # pragma: no cover - defensive
        return f"Error: question handler failed: {exc}"

    return _format_answers(normalized, answers)


def _normalize_questions(questions: list[dict] | None) -> list[dict[str, Any]] | str:
    if not questions or not isinstance(questions, list):
        return "ask_user requires a non-empty list of questions"

    normalized: list[dict[str, Any]] = []
    for idx, raw in enumerate(questions, 1):
        if not isinstance(raw, dict):
            return f"question {idx} must be an object"
        prompt = str(raw.get("prompt") or "").strip()
        if not prompt:
            return f"question {idx} requires a prompt"

        raw_options = raw.get("options") or []
        if not isinstance(raw_options, list) or len(raw_options) < 2:
            return f"question {idx} requires at least 2 options"

        options: list[dict[str, Any]] = []
        for opt_idx, opt in enumerate(raw_options, 1):
            if isinstance(opt, dict):
                label = str(opt.get("label") or "").strip()
                opt_id = str(opt.get("id") or opt_idx)
                recommended = bool(opt.get("recommended"))
            else:
                label = str(opt).strip()
                opt_id = str(opt_idx)
                recommended = False
            if not label:
                return f"question {idx} option {opt_idx} requires a label"
            options.append({"id": opt_id, "label": label, "recommended": recommended})

        normalized.append(
            {
                "id": str(raw.get("id") or idx),
                "prompt": prompt,
                "options": options,
                "allow_note": bool(raw.get("allow_note", True)),
            }
        )
    return normalized


def _format_answers(
    questions: list[dict[str, Any]],
    answers: list[dict[str, Any]] | None,
) -> str:
    answers = answers or []
    by_id = {str(answer.get("id")): answer for answer in answers if isinstance(answer, dict)}

    lines: list[str] = ["User clarifications:"]
    for question in questions:
        qid = str(question["id"])
        answer = by_id.get(qid, {})
        selected = answer.get("selected") if isinstance(answer, dict) else None
        if isinstance(selected, dict):
            chosen = str(selected.get("label") or selected.get("id") or "(no selection)")
        elif selected:
            chosen = str(selected)
        else:
            chosen = "(no selection)"
        note = str((answer.get("note") if isinstance(answer, dict) else "") or "").strip()
        lines.append(f"- Q: {question['prompt']}")
        lines.append(f"  A: {chosen}")
        if note:
            lines.append(f"  Note: {note}")
    return "\n".join(lines)
