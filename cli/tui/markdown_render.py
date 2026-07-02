from __future__ import annotations

from typing import Any

from cli.tui.models import TranscriptLine

_STATUS_MARKER: dict[str, str] = {
    "completed": "[x]",
    "in_progress": "[~]",
    "pending": "[ ]",
    "cancelled": "[-]",
}


def todos_transcript_lines(todos: list[dict[str, Any]], *, width: int) -> list[TranscriptLine]:
    del width
    if not todos:
        return []

    status_width = max(len("status"), *(len(str(todo.get("status", ""))) for todo in todos))
    lines: list[TranscriptLine] = [
        TranscriptLine("class:transcript.todo.title", "todos"),
        TranscriptLine(
            "class:transcript.muted",
            f"  {'status'.ljust(status_width)}  task",
        ),
        TranscriptLine("class:transcript.muted", f"  {'-' * status_width}  {'-' * 24}"),
    ]

    for todo in todos:
        status = str(todo.get("status", ""))
        marker = _STATUS_MARKER.get(status, "[ ]")
        content = str(todo.get("content", ""))
        lines.append(
            TranscriptLine(
                "class:transcript.panel",
                f"  {marker} {status.ljust(status_width)}  {content}",
            )
        )
    return lines
