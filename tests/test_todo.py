from __future__ import annotations

import os
import unittest
import uuid

os.environ.setdefault("OPENAI_API_KEY", "test")

from langchain_core.messages import AIMessage, HumanMessage

from agent import build_graph
from tools import get_tools_for_names
from context import build_working_state_sections, render_todos
from tools.todo import get_thread_todos, set_current_thread, set_thread_todos, todo


class TodoToolTests(unittest.TestCase):
    def setUp(self):
        self.thread_id = f"todo-{uuid.uuid4().hex}"
        set_current_thread(self.thread_id)
        set_thread_todos(self.thread_id, [])

    def test_replace_keeps_existing_contract(self):
        result = todo.invoke(
            {
                "todos": [
                    {"content": "Plan change", "status": "completed"},
                    {"id": "custom", "content": "Run tests", "status": "pending"},
                ]
            }
        )

        self.assertEqual(result, "Updated 2 todos")
        self.assertEqual(
            get_thread_todos(self.thread_id),
            [
                {"id": "1", "content": "Plan change", "status": "completed"},
                {"id": "custom", "content": "Run tests", "status": "pending"},
            ],
        )

    def test_replace_updates_status_in_full_list(self):
        todo.invoke(
            {
                "todos": [
                    {"id": "1", "content": "First", "status": "pending"},
                    {"id": "2", "content": "Second", "status": "pending"},
                ]
            }
        )
        result = todo.invoke(
            {
                "todos": [
                    {"id": "2", "content": "Second", "status": "completed"},
                    {"id": "1", "content": "First", "status": "in_progress"},
                ]
            }
        )
        self.assertEqual(result, "Updated 2 todos")
        self.assertEqual(
            [(t["id"], t["status"]) for t in get_thread_todos(self.thread_id)],
            [("2", "completed"), ("1", "in_progress")],
        )

    def test_invalid_status_rejected_by_schema(self):
        with self.assertRaises(Exception):
            todo.invoke({"todos": [{"content": "x", "status": "blocked"}]})

    def test_get_thread_todos_returns_copies(self):
        set_thread_todos(self.thread_id, [{"id": "1", "content": "Stable", "status": "pending"}])

        todos = get_thread_todos(self.thread_id)
        todos[0]["status"] = "completed"

        self.assertEqual(get_thread_todos(self.thread_id)[0]["status"], "pending")

    def test_schema_requires_todos(self):
        schema = todo.args_schema.model_json_schema()
        self.assertIn("todos", schema.get("required", []))
        self.assertNotIn("action", schema.get("properties", {}))

    def test_missing_todos_fails_schema(self):
        with self.assertRaises(Exception):
            todo.invoke({})


class TodoGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_todo_updates_graph_state_and_next_context_overlay(self):
        class TodoModel:
            def __init__(self):
                self.calls = 0
                self.bound_tools = []
                self.seen_messages = []

            def bind_tools(self, tools):
                self.bound_tools = list(tools)
                return self

            async def ainvoke(self, messages):
                self.calls += 1
                self.seen_messages.append(list(messages))
                if self.calls == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "todo",
                                "args": {
                                    "todos": [
                                        {"id": "2", "content": "Inserted first", "status": "pending"},
                                        {"id": "1", "content": "Existing", "status": "pending"},
                                    ]
                                },
                                "id": "call-1",
                            }
                        ],
                    )
                return AIMessage(content="done")

        thread_id = f"todo-graph-{uuid.uuid4().hex}"
        model = TodoModel()
        app = build_graph(model, tools=get_tools_for_names(["todo"]), thread_id=thread_id)
        result = await app.ainvoke(
            {
                "messages": [HumanMessage(content="track this")],
                "approval_declined": False,
                "todos": [{"id": "1", "content": "Existing", "status": "pending"}],
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        self.assertEqual(
            result["todos"],
            [
                {"id": "2", "content": "Inserted first", "status": "pending"},
                {"id": "1", "content": "Existing", "status": "pending"},
            ],
        )
        working_state_tail = model.seen_messages[1][-1]
        self.assertEqual(working_state_tail.type, "human")
        self.assertIn("<system-reminder>", working_state_tail.content)
        self.assertIn("- [pending] 2: Inserted first", working_state_tail.content)
        self.assertIn("- [pending] 1: Existing", working_state_tail.content)
        self.assertNotIn("TODOS\nNo todos", working_state_tail.content)


class RenderTodosTests(unittest.TestCase):
    def test_render_todos_omits_completed(self):
        todos = [
            {"id": "1", "content": "Done", "status": "completed"},
            {"id": "2", "content": "Active", "status": "in_progress"},
        ]
        rendered = render_todos(todos)
        self.assertIn("- [in_progress] 2: Active", rendered)
        self.assertNotIn("Done", rendered)

    def test_render_todos_empty_when_all_completed(self):
        todos = [{"id": "1", "content": "Done", "status": "completed"}]
        self.assertEqual(render_todos(todos), "")
        self.assertEqual(render_todos([]), "")
        self.assertEqual(render_todos(None), "")

    def test_working_state_overlay_omits_todos_section_when_empty(self):
        sections = build_working_state_sections("act", todos=render_todos([]))
        overlay = "\n\n".join(sections.values())
        self.assertNotIn("TODOS", overlay)

    def test_working_state_overlay_includes_active_todos(self):
        sections = build_working_state_sections(
            "act",
            todos=render_todos([{"id": "1", "content": "Ship it", "status": "pending"}]),
        )
        overlay = "\n\n".join(sections.values())
        self.assertIn("TODOS\n- [pending] 1: Ship it", overlay)


if __name__ == "__main__":
    unittest.main()
