from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from liteharness_cli.config import settings

from liteharness_cli.tui import render
from liteharness_cli.tui.commands import dispatch
from liteharness_cli.tui.config_flow import ConfigResult


async def _dispatch_with_sink(app, command: str) -> None:
    render.set_sink(app)
    try:
        await dispatch(app, command)
    finally:
        render.set_sink(None)


def test_help_command_lists_supported_commands(make_app):
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


def test_dispatch_exit_sets_session_flag(make_app):
    app = make_app()
    asyncio.run(_dispatch_with_sink(app, "/exit"))
    assert app.should_exit is True


def test_status_command_shows_session_summary(make_app):
    app = make_app()
    asyncio.run(_dispatch_with_sink(app, "/status"))
    text = "\n".join(line.text for line in app._lines)
    assert "session status" in text
    assert "cache read" in text


def test_config_session_toggles_update_active_runtime(make_app):
    app = make_app()
    options = SimpleNamespace(
        enable_approval=True,
        yolo_mode=False,
        auto_save_threads=True,
        session_end_reflection=False,
    )
    app.coding.cfg = SimpleNamespace(options=options)
    app.coding.thread_store.auto_save = True
    old = (
        settings.enable_approval,
        settings.auto_save_threads,
        settings.session_end_reflection,
    )
    settings.enable_approval = False
    settings.auto_save_threads = False
    settings.session_end_reflection = True
    try:
        with patch(
            "liteharness_cli.tui.commands.run_config",
            new_callable=AsyncMock,
            return_value=ConfigResult(session_update=True),
        ):
            asyncio.run(_dispatch_with_sink(app, "/config"))
    finally:
        (
            settings.enable_approval,
            settings.auto_save_threads,
            settings.session_end_reflection,
        ) = old

    assert options.enable_approval is False
    assert options.auto_save_threads is False
    assert options.session_end_reflection is True
    assert app.coding.thread_store.auto_save is False


def test_config_action_can_update_persisted_setting(make_app):
    app = make_app()
    previous = settings.enable_approval
    settings.enable_approval = True
    try:
        app._open_picker("config_action", "/config", index=0)
        items = app._config_action_items()
        app._menu_index = next(i for i, item in enumerate(items) if item.key == "approval")
        with patch("liteharness_cli.tui.config_flow.write_env"):
            app._apply_picker_selection()
        assert settings.enable_approval is False
        assert app._menu_kind is None
    finally:
        settings.enable_approval = previous


def test_rollback_command_with_numeric_arg_calls_rollback_to(make_app):
    app = make_app()
    asyncio.run(_dispatch_with_sink(app, "/rollback 5"))
    assert app.coding.rolled_back_seq == 5


def test_rollback_command_no_turns_warns(make_app):
    app = make_app()
    with patch.object(app.coding.thread_store, "list_user_turns", return_value=[]):
        asyncio.run(_dispatch_with_sink(app, "/rollback"))
    assert app.coding.rolled_back_seq is None
