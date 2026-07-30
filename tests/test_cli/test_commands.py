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


def test_config_action_can_update_persisted_setting(make_app, tmp_path, monkeypatch):
    monkeypatch.setenv("LITEHARNESS_CONFIG_DIR", str(tmp_path / "cfg"))
    from liteharness_cli.config_store import load_configs

    app = make_app()
    previous = settings.enable_approval
    settings.enable_approval = True
    try:
        app._config_section = "behavior"
        app._open_picker("config_section", "/config", index=0)
        items = app._visible_menu_items()
        app._menu_index = next(i for i, item in enumerate(items) if item.key == "enable_approval")
        app._apply_picker_selection()
        assert settings.enable_approval is False
        # Bool toggles stay in the section menu and persist to configs.json.
        assert app._menu_kind == "config_section"
        assert load_configs()["enable_approval"] is False
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


async def _dispatch_busy(app, command: str) -> None:
    render.set_sink(app)
    try:
        await dispatch(app, command, busy=True)
    finally:
        render.set_sink(None)


def test_memory_create_refused_while_busy(make_app):
    app = make_app()
    invoke = AsyncMock(return_value=SimpleNamespace(content="# Project"))
    app.coding.agent.config.model = SimpleNamespace(ainvoke=invoke)
    asyncio.run(_dispatch_busy(app, "/memory create"))
    text = "\n".join(line.text for line in app._lines)
    assert "/memory create is not available while a task is running" in text
    invoke.assert_not_called()


def test_memory_read_and_add_allowed_while_busy(make_app):
    app = make_app()
    with patch.object(app.coding.memory_store, "load_project", return_value="existing notes"):
        asyncio.run(_dispatch_busy(app, "/memory"))
    text = "\n".join(line.text for line in app._lines)
    assert "existing notes" in text

    with patch.object(
        app.coding.memory_store, "append_project", return_value="Updated .ness/NESS.md"
    ) as append:
        asyncio.run(_dispatch_busy(app, "/memory add remember this"))
        append.assert_called_once_with("remember this")
