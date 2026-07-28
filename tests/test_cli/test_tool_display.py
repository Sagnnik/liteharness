from __future__ import annotations

from liteharness_cli.tui.tool_display import format_shell_output, format_tool_args


def test_format_shell_output_marks_validation_errors_as_error() -> None:
    content = (
        "Error: 1 validation error for shell\n"
        "action\n"
        "  Field required [type=missing, input_value={'command': 'git status'}, "
        "input_type=dict]"
    )
    header, body = format_shell_output(content)
    assert header == "status=error"
    assert body.startswith("Error: 1 validation error")


def test_format_shell_output_keeps_structured_ok_status() -> None:
    content = "status=ok\nexit_code=0\nduration_ms=12\ncwd=/tmp\noutput_truncated=false\noutput:\nok"
    header, body = format_shell_output(content)
    assert header == "status=ok"
    assert body == "ok"


def test_format_tool_args_defaults_missing_shell_action_to_run() -> None:
    token = format_tool_args("shell", {"command": "git status --short", "timeout": 10})
    assert "run" in token
    assert "git status --short" in token
