from __future__ import annotations

from contextvars import ContextVar

from langchain_core.tools import tool

_current_thread_id: ContextVar[str] = ContextVar("current_thread_id", default="default") # access via getters and setters
_todo_store: dict[str, list[dict]] = {}
VALID_STATUSES = {"pending", "in_progress", "completed"}
VALID_ACTIONS = {"replace", "insert", "update", "delete", "clear"}

# AgentState has todos: list[dict] which are indirectly stored in the _todo_store
def set_current_thread(thread_id: str) -> None:
    _current_thread_id.set(thread_id)


def set_thread_todos(thread_id: str, todos: list[dict]) -> None:
    _todo_store[thread_id] = _normalize_todos(todos)


def get_thread_todos(thread_id: str | None = None) -> list[dict]:
    return [dict(todo) for todo in _todo_store.get(thread_id or _current_thread_id.get(), [])]


@tool
def todo(
    todos: list[dict] | None = None,
    action: str = "replace",
    content: str | None = None,
    status: str | None = None,
    id: str | None = None,
    index: int | None = None,
) -> str:
    """Manage a task list.

    Actions: replace (full list), insert, update, delete, clear.
    Todo shape: {id, content, status} — status is pending, in_progress, or completed.
    Indexes are one-based; omit index on insert to append.
    """
    action = str(action or "replace")
    if action not in VALID_ACTIONS:
        return f"Error: invalid action {action}"

    thread_id = _current_thread_id.get()
    try:
        if action == "replace":
            if todos is None:
                return "Error: replace requires todos"
            normalized = _normalize_todos(todos)
            _todo_store[thread_id] = normalized
            return f"Updated {len(normalized)} todos"

        current = get_thread_todos(thread_id)

        if action == "clear":
            _todo_store[thread_id] = []
            return "Cleared todos"

        if action == "insert":
            if content is None:
                return "Error: insert requires content"
            normalized_status = _normalize_status(status or "pending", "new todo")
            insert_at = _resolve_insert_index(index, len(current))
            todo_id = str(id or _next_id(current))
            if any(todo["id"] == todo_id for todo in current):
                return f"Error: todo id {todo_id} already exists"
            current.insert(
                insert_at,
                {
                    "id": todo_id,
                    "content": "" if content is None else str(content),
                    "status": normalized_status,
                },
            )
            _todo_store[thread_id] = current
            return f"Inserted todo {todo_id} at index {insert_at + 1}"

        if action == "update":
            if not id:
                return "Error: update requires id"
            todo_pos = _find_todo_index(current, str(id))
            if todo_pos is None:
                return f"Error: no todo with id {id}"
            updated = dict(current[todo_pos])
            if content is not None:
                updated["content"] = str(content)
            if status is not None:
                updated["status"] = _normalize_status(status, f"todo {id}")
            del current[todo_pos]
            move_to = todo_pos if index is None else _resolve_move_index(index, len(current))
            current.insert(move_to, updated)
            _todo_store[thread_id] = current
            return f"Updated todo {id}"

        if action == "delete":
            if not id:
                return "Error: delete requires id"
            todo_pos = _find_todo_index(current, str(id))
            if todo_pos is None:
                return f"Error: no todo with id {id}"
            del current[todo_pos]
            _todo_store[thread_id] = current
            return f"Deleted todo {id}"
    except ValueError as exc:
        return f"Error: {exc}"

    return f"Error: unsupported action {action}"


def _normalize_todos(todos: list[dict]) -> list[dict]:
    normalized = []
    for idx, todo in enumerate(todos, 1):
        if not isinstance(todo, dict):
            raise ValueError(f"todo {idx} must be an object")
        status = _normalize_status(todo.get("status", "pending"), f"todo {idx}")
        normalized.append(
            {
                "id": str(todo.get("id") or idx),
                "content": "" if todo.get("content") is None else str(todo.get("content")),
                "status": status,
            }
        )
    return normalized


def _normalize_status(status: str, label: str) -> str:
    normalized = str(status)
    if normalized not in VALID_STATUSES:
        raise ValueError(f"{label} has invalid status {normalized}")
    return normalized


def _resolve_insert_index(index: int | None, length: int) -> int:
    if index is None:
        return length
    if index < 1 or index > length + 1:
        raise ValueError(f"index must be between 1 and {length + 1}")
    return index - 1


def _resolve_move_index(index: int, length_after_removal: int) -> int:
    if index < 1 or index > length_after_removal + 1:
        raise ValueError(f"index must be between 1 and {length_after_removal + 1}")
    return index - 1


def _find_todo_index(todos: list[dict], todo_id: str) -> int | None:
    for idx, todo in enumerate(todos):
        if todo.get("id") == todo_id:
            return idx
    return None


def _next_id(todos: list[dict]) -> str:
    numeric_ids = [int(todo["id"]) for todo in todos if str(todo.get("id", "")).isdigit()]
    return str(max(numeric_ids, default=0) + 1)
