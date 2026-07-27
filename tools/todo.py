from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

_current_thread_id: ContextVar[str] = ContextVar("current_thread_id", default="default")
_todo_store: dict[str, list[dict]] = {}
VALID_STATUSES = {"pending", "in_progress", "completed"}
TodoStatus = Literal["pending", "in_progress", "completed"]


class TodoItem(BaseModel):
    """One task list entry."""

    id: str | None = Field(default=None, description="Stable id; auto-assigned when omitted.")
    content: str = Field(description="Task description.")
    status: TodoStatus = Field(default="pending", description="pending, in_progress, or completed.")


def set_current_thread(thread_id: str) -> None:
    _current_thread_id.set(thread_id)


def set_thread_todos(thread_id: str, todos: list[dict]) -> None:
    _todo_store[thread_id] = _normalize_todos(todos)


def get_thread_todos(thread_id: str | None = None) -> list[dict]:
    return [dict(todo) for todo in _todo_store.get(thread_id or _current_thread_id.get(), [])]


@tool
def todo(todos: list[TodoItem]) -> str:
    """Replace the task list with the provided todos.

    Todo shape: {id?, content, status} — status is pending, in_progress, or completed.
    Pass the full list on every call (mark items completed by updating their status).
    Keep only one item in_progress at a time.
    """
    thread_id = _current_thread_id.get()
    try:
        normalized = _normalize_todos(todos)
    except ValueError as exc:
        return f"Error: {exc}"
    _todo_store[thread_id] = normalized
    return f"Updated {len(normalized)} todos"


def _normalize_todos(todos: list[Any]) -> list[dict]:
    normalized = []
    for idx, todo in enumerate(todos, 1):
        if isinstance(todo, TodoItem):
            raw = todo.model_dump()
        elif isinstance(todo, dict):
            # Graph/session state still stores plain dicts.
            raw = todo
        else:
            raise ValueError(f"todo {idx} must be an object")
        status = _normalize_status(raw.get("status", "pending"), f"todo {idx}")
        normalized.append(
            {
                "id": str(raw.get("id") or idx),
                "content": "" if raw.get("content") is None else str(raw.get("content")),
                "status": status,
            }
        )
    return normalized


def _normalize_status(status: str, label: str) -> str:
    normalized = str(status)
    if normalized not in VALID_STATUSES:
        raise ValueError(f"{label} has invalid status {normalized}")
    return normalized
