from __future__ import annotations

import asyncio
from unittest.mock import patch

from cli import render
from cli.commands import dispatch
from config import settings
from tests.test_cli.helpers import make_app


async def _dispatch_with_sink(app, command: str) -> None:
    render.set_sink(app)
    try:
        await dispatch(app.session, command)
    finally:
        render.set_sink(None)


def test_help_command_lists_supported_commands():
    app = make_app()
    asyncio.run(_dispatch_with_sink(app, "/help"))
    text = "\n".join(line.text for line in app._lines)
    assert "commands" in text
    assert "/config" in text
    assert "/status" in text
    assert "/menu" not in text
    assert "/cost" not in text
    assert "/cache" not in text
    assert "/skills" not in text
    assert "/image" not in text


def test_dispatch_exit_sets_session_flag():
    app = make_app()
    asyncio.run(_dispatch_with_sink(app, "/exit"))
    assert app.session.should_exit is True


def test_status_command_shows_session_summary():
    app = make_app()
    asyncio.run(_dispatch_with_sink(app, "/status"))
    text = "\n".join(line.text for line in app._lines)
    assert "session status" in text
    assert "cache read" in text


def test_config_action_can_update_persisted_setting():
    app = make_app()
    previous = settings.enable_approval
    settings.enable_approval = True
    try:
        app._open_picker("config_action", "/config", index=0)
        items = app._config_action_items()
        app._menu_index = next(i for i, item in enumerate(items) if item.key == "approval")
        with patch("cli.tui.config_flow.write_env"):
            app._apply_picker_selection()
        assert settings.enable_approval is False
        assert app._menu_kind is None
    finally:
        settings.enable_approval = previous
