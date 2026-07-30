"""Approval prompt cancel: Esc / Ctrl+C deny the call (soft cancel → \"no\")."""

from __future__ import annotations

import asyncio


def test_cancel_open_prompt_resolves_approval_to_empty_string(make_app) -> None:
    app = make_app()

    async def _exercise() -> object:
        app._prompt_future = asyncio.get_running_loop().create_future()
        app._prompt_kind = "approval"
        assert app._cancel_open_prompt() is True
        return app._prompt_future.result()

    assert asyncio.run(_exercise()) == ""
    assert app._prompt_kind is None


def test_ask_approval_cancel_menu_returns_no(make_app) -> None:
    app = make_app()

    async def _exercise() -> str:
        async def esc_soon() -> None:
            await asyncio.sleep(0)
            app._cancel_menu()

        task = asyncio.create_task(esc_soon())
        decision = await app.ask_approval("shell", {"command": "rm -rf /"})
        await task
        return decision

    assert asyncio.run(_exercise()) == "no"
    assert app._prompt_kind is None
    assert app._menu_kind is None


def test_ask_approval_cancel_open_prompt_returns_no(make_app) -> None:
    app = make_app()

    async def _exercise() -> str:
        async def ctrl_c_soon() -> None:
            await asyncio.sleep(0)
            assert app._cancel_open_prompt() is True

        task = asyncio.create_task(ctrl_c_soon())
        decision = await app.ask_approval("shell", {"command": "npm run build"})
        await task
        return decision

    assert asyncio.run(_exercise()) == "no"
    assert app._prompt_kind is None
    assert app._menu_kind is None


def test_ask_approval_explicit_deny_once(make_app) -> None:
    app = make_app()

    async def _exercise() -> str:
        async def deny_soon() -> None:
            await asyncio.sleep(0)
            app._apply_approval_selection("no")

        task = asyncio.create_task(deny_soon())
        decision = await app.ask_approval("write", {"path": "x.py", "content": "x"})
        await task
        return decision

    assert asyncio.run(_exercise()) == "no"
