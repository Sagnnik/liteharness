"""Tests for headless print mode (``ness -p "query"``).

The turn runner is exercised against a REAL CodingSession (bindable fake
chat model over a temp project root) so the durable-event/checkpoint path is
covered end-to-end; ``run_headless`` itself is tested with the session
factory and MCP manager monkeypatched out; CLI arg wiring uses typer's
CliRunner with ``run_headless`` faked.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from typer.testing import CliRunner

from liteharness import NessAgent, NessAgentOptions, PromptLayers, PromptLayersConfig, SessionEvent
from liteharness_cli import CodingSession, headless
from liteharness_cli.tui import main as tui_main


class _BindableFakeModel:
    """bind_tools-capable fake that cycles fixed AIMessage responses and
    records the messages of every call (vs FakeListChatModel)."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.received: list[list] = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages, **kwargs):
        resp = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        self.received.append(list(messages))
        return resp

    @property
    def model(self):
        return "bindfake"


def _tool_call(call_id: str = "call-1") -> dict:
    return {"name": "ping", "args": {}, "id": call_id, "type": "tool_call"}


def _make_coding(tmp_path: Path, model, *, yolo: bool = False) -> CodingSession:
    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    agent = NessAgent(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(
            project_root=tmp_path,
            ness_dir=tmp_path / ".ness",
            auto_save_threads=True,
            enable_approval=not yolo,
            yolo_mode=yolo,
        ),
    )
    agent.config.thread_store.auto_save = True
    # Custom user-supplied tools are filtered out of the active set by
    # ToolRegistry unless they are built-ins or explicitly included. Activate
    # the custom tool so the tools_node can resolve and invoke it.
    reg = agent.config.tool_registry
    reg._include = {"ping"}  # type: ignore[attr-defined]
    reg.bump_generation()
    return CodingSession(agent, thread_id="t-print", mode="act", vision=False)


class _ScriptedCoding:
    """Minimal CodingSession stand-in yielding a fixed event list."""

    def __init__(self, events: list[SessionEvent]) -> None:
        self._events = events

    async def run_turn(self, prompt: str):
        for ev in self._events:
            yield ev


class _FakeMCP:
    def __init__(self, **kwargs) -> None:
        self.tools: dict = {}
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def catalog(self) -> dict:
        return {}

    def startup_summary(self) -> tuple[str, str]:
        return ("", "info")


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ----------------------------------------------------------------------
# merge_prompt_parts
# ----------------------------------------------------------------------


def test_merge_prompt_parts_both():
    assert headless.merge_prompt_parts(["explain", "this"], "log body\n") == (
        "log body\n\nexplain this"
    )


def test_merge_prompt_parts_either():
    assert headless.merge_prompt_parts(["query"], "") == "query"
    assert headless.merge_prompt_parts([], "piped") == "piped"
    assert headless.merge_prompt_parts(None, "piped") == "piped"


def test_merge_prompt_parts_empty():
    assert headless.merge_prompt_parts([], "") is None
    assert headless.merge_prompt_parts(None, "  \n ") is None


# ----------------------------------------------------------------------
# run_headless_turn (real CodingSession)
# ----------------------------------------------------------------------


def test_headless_turn_returns_last_final_text(tmp_path: Path):
    model = _BindableFakeModel(
        [
            AIMessage(content="let me check", tool_calls=[_tool_call()]),
            AIMessage(content="world"),
        ]
    )
    coding = _make_coding(tmp_path, model)

    text, code = _run(headless.run_headless_turn(coding, "hello"))

    assert (text, code) == ("world", 0)
    assert model.calls == 2
    # Durable parity: the user event was persisted like a TUI turn.
    durable = coding.thread_store.load_thread_events("t-print")
    assert any(e.get("kind") == "user" and "hello" in str(e.get("content")) for e in durable)


def test_headless_turn_denies_unapproved_tool(tmp_path: Path):
    """approval_handler=None + gated tool -> denial fed back, turn completes."""
    model = _BindableFakeModel(
        [
            AIMessage(content="", tool_calls=[_tool_call()]),
            AIMessage(content="cannot run that"),
        ]
    )
    coding = _make_coding(tmp_path, model)
    # Force the (normally read-only) ping tool through the approval gate.
    coding.tool_registry.is_destructive = lambda name, args: True

    text, code = _run(headless.run_headless_turn(coding, "run ping"))

    assert (text, code) == ("cannot run that", 0)
    # The denial was persisted as a durable approval event...
    durable = coding.thread_store.load_thread_events("t-print")
    assert any(
        e.get("kind") == "approval" and e.get("decision") == "no" for e in durable
    ), durable
    # ...and fed back to the model as a ToolMessage.
    second_call = str(model.received[1])
    assert "Approval required but no handler configured" in second_call
    assert "pong" not in second_call


def test_headless_turn_yolo_executes_tool(tmp_path: Path):
    model = _BindableFakeModel(
        [
            AIMessage(content="", tool_calls=[_tool_call()]),
            AIMessage(content="done"),
        ]
    )
    coding = _make_coding(tmp_path, model, yolo=True)
    coding.tool_registry.is_destructive = lambda name, args: True

    text, code = _run(headless.run_headless_turn(coding, "run ping"))

    assert (text, code) == ("done", 0)
    assert "pong" in str(model.received[1])


def test_headless_turn_error_event_maps_to_exit_1(capsys):
    coding = _ScriptedCoding(
        [SessionEvent(kind="error", data={"message": "boom"})]
    )
    text, code = _run(headless.run_headless_turn(coding, "q"))
    assert (text, code) == ("", 1)
    assert "boom" in capsys.readouterr().err


def test_headless_turn_interrupted_maps_to_exit_130():
    coding = _ScriptedCoding(
        [SessionEvent(kind="interrupted", data={"partial_text": "half an answer"})]
    )
    text, code = _run(headless.run_headless_turn(coding, "q"))
    assert (text, code) == ("half an answer", 130)


# ----------------------------------------------------------------------
# run_headless (session factory + MCP monkeypatched out)
# ----------------------------------------------------------------------


def _patch_headless_infra(monkeypatch, tmp_path: Path, coding: CodingSession) -> _FakeMCP:
    fake_mcp = _FakeMCP()
    monkeypatch.setattr(headless, "is_git_repo", lambda: False)
    monkeypatch.setattr(
        headless, "prepare_paths", lambda: SimpleNamespace(project_root=tmp_path)
    )
    monkeypatch.setattr(headless, "MCPManager", lambda **kw: fake_mcp)
    monkeypatch.setattr(headless, "build_coding_session", lambda **kw: coding)
    monkeypatch.setattr(headless, "provider_key_missing", lambda: False)
    monkeypatch.setattr(headless.settings, "auto_save_threads", True)
    return fake_mcp


def test_run_headless_missing_api_key(tmp_path: Path, monkeypatch, capsys):
    coding = _make_coding(tmp_path, _BindableFakeModel([AIMessage(content="x")]))
    _patch_headless_infra(monkeypatch, tmp_path, coding)
    monkeypatch.setattr(headless, "provider_key_missing", lambda: True)

    code = _run(headless.run_headless("hello"))

    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no provider API key configured" in captured.err


def test_run_headless_end_to_end(tmp_path: Path, monkeypatch, capsys):
    model = _BindableFakeModel([AIMessage(content="world")])
    coding = _make_coding(tmp_path, model)
    fake_mcp = _patch_headless_infra(monkeypatch, tmp_path, coding)

    code = _run(headless.run_headless("hello"))

    assert code == 0
    captured = capsys.readouterr()
    # stdout carries ONLY the final response; the resume hint goes to stderr.
    assert captured.out == "world\n"
    assert "Resume: ness --resume t-print" in captured.err
    assert fake_mcp.started and fake_mcp.stopped
    durable = coding.thread_store.load_thread_events("t-print")
    assert any(e.get("kind") == "user" for e in durable)


def test_run_headless_resume_missing_thread(tmp_path: Path, monkeypatch, capsys):
    coding = _make_coding(tmp_path, _BindableFakeModel([AIMessage(content="x")]))
    _patch_headless_infra(monkeypatch, tmp_path, coding)

    code = _run(headless.run_headless("hello", resume_thread_id="no-such-thread"))

    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no saved thread" in captured.err


# ----------------------------------------------------------------------
# CLI wiring (typer)
# ----------------------------------------------------------------------


def _fake_run_headless_factory(captured: dict):
    async def _fake(prompt: str, *, resume_thread_id=None, yolo=False) -> int:
        captured.update(prompt=prompt, resume=resume_thread_id, yolo=yolo)
        return 0

    return _fake


def test_cli_print_flag_joins_prompt_args(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(tui_main, "run_headless", _fake_run_headless_factory(captured))
    result = CliRunner().invoke(tui_main.app, ["-p", "explain", "this"])
    assert result.exit_code == 0, result.output
    assert captured["prompt"] == "explain this"


def test_cli_print_flag_reads_piped_stdin(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(tui_main, "run_headless", _fake_run_headless_factory(captured))
    result = CliRunner().invoke(tui_main.app, ["-p", "summarize"], input="log body\n")
    assert result.exit_code == 0, result.output
    assert captured["prompt"] == "log body\n\nsummarize"


def test_cli_print_flag_forwards_resume_and_yolo(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(tui_main, "run_headless", _fake_run_headless_factory(captured))
    result = CliRunner().invoke(
        tui_main.app, ["-p", "--yolo", "--resume", "t-1", "q"]
    )
    assert result.exit_code == 0, result.output
    assert captured["resume"] == "t-1"
    assert captured["yolo"] is True


def test_cli_print_flag_requires_prompt(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(tui_main, "run_headless", _fake_run_headless_factory(captured))
    result = CliRunner().invoke(tui_main.app, ["-p"], input="  \n")
    assert result.exit_code == 2
    assert captured == {}


def test_cli_positional_prompt_without_print_fails(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(tui_main, "run_headless", _fake_run_headless_factory(captured))
    result = CliRunner().invoke(tui_main.app, ["hello"])
    assert result.exit_code == 2
    assert "requires --print" in result.output
    assert captured == {}
