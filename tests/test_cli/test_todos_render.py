from __future__ import annotations

import unittest
from unittest import mock

from langchain_core.messages import ToolMessage

from cli import render
from cli.session_app import SessionApp
from tests.test_cli.helpers import make_app


class TodosRenderTests(unittest.TestCase):
    def test_append_todos_replaces_in_place(self) -> None:
        app = make_app()
        app.append_todos([{"id": "1", "content": "First", "status": "pending"}])
        start = app._todos_block_start
        self.assertIsNotNone(start)

        app.append_todos(
            [
                {"id": "1", "content": "First", "status": "in_progress"},
                {"id": "2", "content": "Second", "status": "pending"},
            ]
        )
        self.assertEqual(app._todos_block_start, start)
        plain = app._transcript_plain_text()
        self.assertEqual(plain.count("todos"), 1)
        self.assertIn("Second", plain)

    def test_append_todos_clears_block_when_empty(self) -> None:
        app = make_app()
        app.append_todos([{"id": "1", "content": "Ship it", "status": "pending"}])
        self.assertIsNotNone(app._todos_block_start)

        app.append_todos([])
        self.assertIsNone(app._todos_block_start)
        self.assertEqual(app._todos_block_count, 0)
        self.assertNotIn("Ship it", app._transcript_plain_text())

    def test_render_tool_results_refreshes_todos(self) -> None:
        session = mock.Mock(spec=SessionApp)
        session.thread_id = "todo-render-thread"
        todos = [{"id": "1", "content": "Ship it", "status": "pending"}]
        render_calls: list[list[dict]] = []

        with mock.patch("cli.session_app.get_thread_todos", return_value=todos):
            with mock.patch.object(render, "render_todos", side_effect=lambda t: render_calls.append(list(t))):
                with mock.patch.object(render, "render_tool_result"):
                    event = {
                        "data": {
                            "output": {
                                "messages": [
                                    ToolMessage(
                                        content="Updated 1 todos",
                                        name="todo",
                                        tool_call_id="call-1",
                                    )
                                ]
                            }
                        }
                    }
                    SessionApp._render_tool_results(session, event)

        self.assertEqual(render_calls, [todos])


if __name__ == "__main__":
    unittest.main()
