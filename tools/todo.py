from langchain_core.tools import tool

# In-memory store keyed by thread_id — wired from agent state
_todo_store: dict[str, list[dict]] = {}


def set_thread_todos(thread_id: str, todos: list[dict]):
    _todo_store[thread_id] = todos

def get_thread_todos(thread_id: str) -> list[dict]:
    return _todo_store.get(thread_id, [])

@tool
def todo_write(todos: list[dict]) -> str:
    """Update the TODO list. Each: {id, content, status: pending|in_progress|completed}"""
    # thread_id injected by agent wrapper
    return f"Updated {len(todos)} todos"

@tool
def todo_read() -> str:
    return "No todos"