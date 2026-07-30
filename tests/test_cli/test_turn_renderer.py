from __future__ import annotations

from liteharness.types import SessionEvent
from liteharness_cli.tui import render
from liteharness_cli.tui.turn_renderer import TurnRenderer


class _SpySink(render._NullSink):
    def __init__(self) -> None:
        self.subagent_outputs: list[str] = []
        self.tool_results: list[tuple[str, str, str | None]] = []
        self.diffs: list[tuple[str, str]] = []
        self.shell_outputs: list[str] = []

    def append_subagent_output(self, content: str) -> None:
        self.subagent_outputs.append(content)

    def append_tool_result(
        self, name: str, content: str, *, exit_status: str | None = None
    ) -> None:
        self.tool_results.append((name, content, exit_status))

    def append_diff(self, diff_text: str, *, title: str = "diff") -> None:
        self.diffs.append((title, diff_text))

    def append_shell_output(self, content: str) -> None:
        self.shell_outputs.append(content)


def test_turn_renderer_routes_spawn_subagent_to_subagent_panel() -> None:
    sink = _SpySink()
    render.set_sink(sink)
    try:
        renderer = TurnRenderer()
        renderer.feed(
            SessionEvent(
                "tool_end",
                {"name": "spawn_subagent", "content": "Found routes in src/api.py"},
            )
        )
    finally:
        render.set_sink(None)

    assert sink.subagent_outputs == ["Found routes in src/api.py"]
    assert sink.tool_results == []


def test_turn_renderer_todo_still_uses_tool_result_path() -> None:
    sink = _SpySink()
    render.set_sink(sink)
    try:
        renderer = TurnRenderer()
        renderer.feed(
            SessionEvent("tool_end", {"name": "todo", "content": "Updated 2 todos"})
        )
    finally:
        render.set_sink(None)

    assert sink.tool_results == [("todo", "Updated 2 todos", None)]
    assert sink.subagent_outputs == []


def test_turn_renderer_routes_edit_diff_and_shell() -> None:
    sink = _SpySink()
    render.set_sink(sink)
    try:
        renderer = TurnRenderer()
        renderer.feed(
            SessionEvent(
                "tool_end",
                {
                    "name": "write",
                    "content": "Wrote x.py\ndiff:\n@@ -0,0 +1 @@\n+x\n",
                },
            )
        )
        renderer.feed(
            SessionEvent(
                "tool_end",
                {"name": "shell", "content": "status=ok\noutput:\nok\n"},
            )
        )
    finally:
        render.set_sink(None)

    assert sink.diffs and sink.diffs[0][0] == "diff write"
    assert "+x" in sink.diffs[0][1]
    assert sink.shell_outputs == ["status=ok\noutput:\nok\n"]
