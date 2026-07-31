from __future__ import annotations

from contextvars import ContextVar
from typing import Annotated, Any, Awaitable, Callable

from langchain_core.tools import tool
from pydantic import BaseModel, Field

QuestionHandler = Callable[[list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]]

_question_handler: ContextVar[QuestionHandler | None] = ContextVar(
    "question_handler", default=None
)


class QuestionOption(BaseModel):
    """One multiple-choice option."""

    id: str | None = Field(default=None, description="Stable option id; auto-assigned when omitted.")
    label: str = Field(description="Option text shown to the user.")
    recommended: bool = Field(default=False, description="Mark the preferred option.")


class QuestionItem(BaseModel):
    """One clarification question with at least two options."""

    id: str | None = Field(default=None, description="Stable question id; auto-assigned when omitted.")
    prompt: str = Field(description="Question text.")
    options: Annotated[list[QuestionOption], Field(min_length=2)] = Field(
        description="At least two options for the user to choose from."
    )
    allow_note: bool = Field(default=True, description="Allow a free-form note with the answer.")


def set_question_runtime(handler: QuestionHandler | None) -> None:
    """Set the interactive question handler for the current run."""
    _question_handler.set(handler)


@tool
async def question(
    questions: Annotated[list[QuestionItem], Field(min_length=1)],
) -> str:
    """Ask the user for input.

    Use this when requirements are ambiguous or several valid approaches exist,
    before committing to a plan or implementation. Each question needs a prompt
    and at least two options with labels. Mark the best option with
    ``recommended: true``. Returns the questions with the chosen answers.
    """
    normalized = _question_payload(questions)

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


def _question_payload(questions: list[QuestionItem]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(questions, 1):
        options = []
        for opt_idx, opt in enumerate(item.options, 1):
            options.append(
                {
                    "id": str(opt.id or opt_idx),
                    "label": opt.label.strip(),
                    "recommended": bool(opt.recommended),
                }
            )
        normalized.append(
            {
                "id": str(item.id or idx),
                "prompt": item.prompt.strip(),
                "options": options,
                "allow_note": bool(item.allow_note),
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
        note = str((answer.get("note") if isinstance(answer, dict) else "") or "").strip()
        if isinstance(selected, dict):
            chosen = str(selected.get("label") or selected.get("id") or "(no selection)")
        elif selected:
            chosen = str(selected)
        elif note == "cancelled by user":
            chosen = "(cancelled)"
        else:
            chosen = "(no selection)"
        lines.append(f"- Q: {question['prompt']}")
        lines.append(f"  A: {chosen}")
        if note:
            lines.append(f"  Note: {note}")
    return "\n".join(lines)
