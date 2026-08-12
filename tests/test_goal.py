from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from ness_cli.goal import (
    GoalCoordinator,
    JudgeStructuredOutput,
    JudgeVerdict,
    _format_goal_conversation,
    _repair_text,
    _verdict_from_structured,
)


def test_verdict_from_structured() -> None:
    verdict = _verdict_from_structured(
        JudgeStructuredOutput(
            passed=True,
            unmet=[],
            evidence=["observable result"],
            repair_instruction="",
        )
    )
    assert verdict.passed is True
    assert verdict.evidence == ("observable result",)


def test_repair_text_prefers_unmet_over_generic_or_empty() -> None:
    generic = "Re-check the deliverable and provide explicit verification evidence."
    assert "finish the remaining step" in _repair_text(
        JudgeVerdict(False, ("finish the remaining step",), (), ""),
        generic_repair=generic,
    )
    assert "finish the remaining step" in _repair_text(
        JudgeVerdict(
            False,
            ("finish the remaining step",),
            (),
            generic,
        ),
        generic_repair=generic,
    )


def test_format_goal_conversation_renders_core_kinds() -> None:
    transcript = _format_goal_conversation(
        [
            {"seq": 3, "kind": "goal", "phase": "start", "goal": "ship feature"},
            {"seq": 4, "kind": "user", "content": "ship feature"},
            {
                "seq": 5,
                "kind": "assistant",
                "content": "working",
                "tool_calls": [{"name": "write"}],
            },
            {
                "seq": 6,
                "kind": "tool",
                "tool": "write",
                "exit": "ok",
                "args": {"path": "a.py"},
                "result": "wrote a.py",
            },
            {"seq": 7, "kind": "usage", "input_tokens": 1},
        ]
    )
    assert "goal start" in transcript
    assert "user:\nship feature" in transcript
    assert "assistant:" in transcript
    assert "tool write" in transcript
    assert "wrote a.py" in transcript
    assert "usage" not in transcript


def test_format_goal_conversation_drops_oldest_when_over_budget(
    monkeypatch,
) -> None:
    import ness_cli.goal as goal_mod

    monkeypatch.setattr(goal_mod, "_TRANSCRIPT_CHAR_BUDGET", 80)
    monkeypatch.setattr(goal_mod, "_EVENT_CHAR_BUDGET", 40)
    transcript = _format_goal_conversation(
        [
            {"seq": 1, "kind": "user", "content": "aaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
            {"seq": 2, "kind": "user", "content": "bbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
            {"seq": 3, "kind": "user", "content": "cccccccccccccccccccccccccccc"},
        ]
    )
    assert "omitted" in transcript
    assert "cccccccc" in transcript


class _Store:
    def __init__(self) -> None:
        self.events = []
        self._next_seq = 0

    def load_thread_events(self, thread_id: str):
        return list(self.events)

    def load_thread_events_since(self, thread_id: str, start_seq: int):
        return [
            event
            for event in self.events
            if int(event.get("seq", -1)) >= int(start_seq)
        ]

    def append_event(self, thread_id: str, event):
        payload = dict(event)
        payload["seq"] = self._next_seq
        self._next_seq += 1
        self.events.append(payload)
        return payload["seq"]


class _Hooks:
    def load(self):
        return []

    def run(self, event, payload):
        return True, ""


class _FailingHooks:
    def load(self):
        return [SimpleNamespace(event="goalValidate")]

    def run(self, event, payload):
        return False, "validation failed"


class _Coordinator(GoalCoordinator):
    def __init__(self, coding, verdicts):
        super().__init__(coding, max_attempts=3)
        self.verdicts = iter(verdicts)

    async def _judge(self, *args, **kwargs):
        return next(self.verdicts)


def test_goal_repair_loop_stops_after_pass() -> None:
    coding = SimpleNamespace(
        thread_store=_Store(),
        thread_id="session-test",
        hook_runner=_Hooks(),
        is_cancelled=lambda: False,
    )
    coordinator = _Coordinator(
        coding,
        [
            JudgeVerdict(False, ("missing piece",), (), "add the missing piece"),
            JudgeVerdict(True, (), ("complete",), ""),
        ],
    )
    prompts: list[str] = []
    statuses: list[tuple[str, str]] = []

    async def worker(prompt: str) -> None:
        prompts.append(prompt)

    result = asyncio.run(
        coordinator.run(
            "ship it",
            worker_turn=worker,
            on_status=lambda role, message: statuses.append((role, message)),
        )
    )

    assert result.passed is True
    assert result.attempts == 2
    assert prompts[0] == "ship it"
    assert "add the missing piece" in prompts[1]
    assert "acceptance criteria" in prompts[1]
    assert "skip tests" not in prompts[1].lower()
    assert [role for role, _ in statuses] == ["worker", "judge", "worker", "judge"]


def test_goal_repair_uses_unmet_when_repair_instruction_empty() -> None:
    coding = SimpleNamespace(
        thread_store=_Store(),
        thread_id="session-test",
        hook_runner=_Hooks(),
        is_cancelled=lambda: False,
    )
    coordinator = _Coordinator(
        coding,
        [
            JudgeVerdict(False, ("Provide verification evidence from the run.",), (), ""),
            JudgeVerdict(True, (), ("verified",), ""),
        ],
    )
    prompts: list[str] = []

    async def worker(prompt: str) -> None:
        prompts.append(prompt)

    result = asyncio.run(
        coordinator.run(
            "complete the objective",
            worker_turn=worker,
            on_status=lambda *_: None,
        )
    )

    assert result.passed is True
    assert "Provide verification evidence" in prompts[1]


def test_validation_failure_overrides_passing_judge() -> None:
    coding = SimpleNamespace(
        thread_store=_Store(),
        thread_id="session-test",
        hook_runner=_FailingHooks(),
        is_cancelled=lambda: False,
    )
    coordinator = _Coordinator(
        coding,
        [JudgeVerdict(True, (), ("looks good",), "")],
    )
    coordinator.max_attempts = 1

    async def worker(prompt: str) -> None:
        return None

    result = asyncio.run(
        coordinator.run(
            "ship it",
            worker_turn=worker,
            on_status=lambda *_: None,
        )
    )

    assert result.passed is False
    assert "validation failed" in result.verdict.repair_instruction
    assert any("Deterministic validation failed" in item for item in result.verdict.unmet)


def test_judge_prompt_uses_conversation_slice_not_digests() -> None:
    store = _Store()
    store.append_event("session-test", {"kind": "user", "content": "earlier work"})
    start_seq = store.append_event(
        "session-test",
        {"kind": "goal", "phase": "start", "goal": "implement feature X"},
    )
    store.append_event(
        "session-test",
        {"kind": "user", "content": "implement feature X"},
    )
    store.append_event(
        "session-test",
        {
            "kind": "tool",
            "tool": "write",
            "exit": "ok",
            "args": {"path": "x.py"},
            "result": "wrote x.py",
        },
    )

    coding = SimpleNamespace(
        thread_store=store,
        thread_id="session-test",
    )
    ainvoke = AsyncMock(
        return_value=JudgeStructuredOutput(
            passed=True,
            unmet=[],
            evidence=["feature present"],
            repair_instruction="",
        )
    )
    structured = SimpleNamespace(ainvoke=ainvoke)
    model = SimpleNamespace(with_structured_output=lambda _schema: structured)
    coordinator = GoalCoordinator(coding, max_attempts=1)
    coordinator._judge_model = model

    verdict = asyncio.run(
        coordinator._judge(
            "implement feature X",
            1,
            start_seq,
            "(no deterministic validation configured)",
        )
    )

    prompt = ainvoke.await_args.args[0][0].content
    assert "Conversation since goal start" in prompt
    assert "implement feature X" in prompt
    assert "wrote x.py" in prompt
    assert "earlier work" not in prompt
    assert "Worker shell commands" not in prompt
    assert "Current git diff" not in prompt
    assert "Current todos" not in prompt
    assert "do not invent missing shell" not in prompt.lower()
    assert verdict.passed is True


def test_judge_structured_output_failure_is_fail_verdict() -> None:
    coding = SimpleNamespace(
        thread_store=_Store(),
        thread_id="session-test",
    )

    class Boom:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _messages):
            raise RuntimeError("provider boom")

    coordinator = GoalCoordinator(coding, max_attempts=1)
    coordinator._judge_model = Boom()

    verdict = asyncio.run(
        coordinator._judge("goal", 1, 0, "(no deterministic validation configured)")
    )
    assert verdict.passed is False
    assert "structured output failed" in verdict.unmet[0]


def test_unset_goal_judge_uses_active_provider_reflection_model(monkeypatch) -> None:
    coding = SimpleNamespace(thread_store=_Store(), thread_id="session-test")
    expected = SimpleNamespace()
    calls: list[str] = []

    def create_reflection(thread_id: str):
        calls.append(thread_id)
        return expected

    monkeypatch.setattr("ness_cli.goal.create_reflection_model", create_reflection)
    monkeypatch.setattr("ness_cli.goal.settings.goal_judge_model", None)

    result = GoalCoordinator(coding, max_attempts=1)._build_judge_model()

    assert result is expected
    assert calls[0].startswith("judge-")
