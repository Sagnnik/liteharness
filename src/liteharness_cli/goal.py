from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from liteharness_cli.chat_model import build_chat_model
from liteharness_cli.config import settings
from liteharness_cli.instructions import GOAL_GENERIC_REPAIR, GOAL_JUDGE, GOAL_REPAIR

WorkerTurn = Callable[[str], Awaitable[None]]
StatusCallback = Callable[[str, str], None]

_EVENT_CHAR_BUDGET = 2_000
_TRANSCRIPT_CHAR_BUDGET = 100_000


class JudgeStructuredOutput(BaseModel):
    """Schema enforced via ``model.with_structured_output``."""

    passed: bool = Field(description="Whether the worker met the user's exact goal.")
    unmet: list[str] = Field(
        default_factory=list,
        description="Concrete goal criteria that are still unmet.",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Observable evidence from the conversation supporting the verdict.",
    )
    repair_instruction: str = Field(
        default="",
        description="Actionable next step for the worker if passed is false.",
    )


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    passed: bool
    unmet: tuple[str, ...]
    evidence: tuple[str, ...]
    repair_instruction: str


@dataclass(frozen=True, slots=True)
class GoalResult:
    passed: bool
    attempts: int
    verdict: JudgeVerdict


def _verdict_from_structured(out: JudgeStructuredOutput) -> JudgeVerdict:
    return JudgeVerdict(
        passed=bool(out.passed),
        unmet=tuple(str(item) for item in out.unmet or ()),
        evidence=tuple(str(item) for item in out.evidence or ()),
        repair_instruction=str(out.repair_instruction or "").strip(),
    )


def _clip(text: str, limit: int) -> str:
    body = text.strip()
    if len(body) <= limit:
        return body
    return body[: max(0, limit - 14)] + "\n...[truncated]"


def _format_event(event: dict) -> str:
    kind = str(event.get("kind") or "event")
    seq = event.get("seq")
    prefix = f"[{seq}] " if seq is not None else ""

    if kind == "user":
        return f"{prefix}user:\n{_clip(str(event.get('content') or ''), _EVENT_CHAR_BUDGET)}"

    if kind == "assistant":
        parts = [f"{prefix}assistant:"]
        content = str(event.get("content") or "").strip()
        if content:
            parts.append(_clip(content, _EVENT_CHAR_BUDGET))
        tool_calls = event.get("tool_calls") or []
        if tool_calls:
            names = [
                str(tc.get("name") or "unknown")
                for tc in tool_calls
                if isinstance(tc, dict)
            ]
            if names:
                parts.append("tool_calls: " + ", ".join(names))
        return "\n".join(parts)

    if kind == "tool":
        tool = str(event.get("tool") or "tool")
        exit_status = str(event.get("exit") or "")
        args = event.get("args") or {}
        args_text = _clip(
            json.dumps(args, ensure_ascii=False, default=str),
            min(400, _EVENT_CHAR_BUDGET // 2),
        )
        result = _clip(str(event.get("result") or ""), _EVENT_CHAR_BUDGET)
        header = f"{prefix}tool {tool}"
        if exit_status:
            header += f" exit={exit_status}"
        return f"{header}\nargs={args_text}\n{result}"

    if kind == "goal":
        phase = str(event.get("phase") or "")
        payload = {
            key: value
            for key, value in event.items()
            if key not in {"kind", "t", "seq"}
        }
        return (
            f"{prefix}goal {phase}:\n"
            f"{_clip(json.dumps(payload, ensure_ascii=False, default=str), _EVENT_CHAR_BUDGET)}"
        )

    if kind in {"usage", "compact", "reflection"}:
        return f"{prefix}{kind}"

    payload = {
        key: value for key, value in event.items() if key not in {"t"}
    }
    return f"{prefix}{kind}:\n{_clip(json.dumps(payload, ensure_ascii=False, default=str), _EVENT_CHAR_BUDGET)}"


def _format_goal_conversation(events: list[dict]) -> str:
    """Render the goal-slice thread as a compact, domain-agnostic transcript.

    Drops noisy bookkeeping kinds. If the transcript exceeds the budget,
    keeps the newest events (oldest dropped first).
    """
    blocks: list[str] = []
    for event in events:
        kind = str(event.get("kind") or "")
        if kind in {"usage", "compact", "reflection"}:
            continue
        blocks.append(_format_event(event))

    if not blocks:
        return "(no conversation events in goal slice)"

    total = 0
    kept: list[str] = []
    for block in reversed(blocks):
        size = len(block) + (1 if kept else 0)
        if kept and total + size > _TRANSCRIPT_CHAR_BUDGET:
            break
        kept.append(block)
        total += size
    kept.reverse()
    dropped = len(blocks) - len(kept)
    body = "\n\n".join(kept)
    if dropped:
        return f"(omitted {dropped} older events to fit budget)\n\n{body}"
    return body


def _repair_text(verdict: JudgeVerdict) -> str:
    repair = verdict.repair_instruction.strip()
    if repair and repair != GOAL_GENERIC_REPAIR:
        return repair
    unmet = "\n".join(item.strip() for item in verdict.unmet if item.strip())
    return unmet or repair or GOAL_GENERIC_REPAIR


class GoalCoordinator:
    """Bounded maker-verifier loop using an isolated structured-output judge."""

    def __init__(self, coding, *, max_attempts: int | None = None) -> None:
        self.coding = coding
        self.max_attempts = max(1, max_attempts or settings.goal_max_attempts)
        self._judge_model: BaseChatModel | None = None

    def _build_judge_model(self) -> BaseChatModel:
        judge_id = f"judge-{uuid.uuid4().hex[:8]}"
        judge_model_name = settings.goal_judge_model or settings.reflection_model_name
        return build_chat_model(
            judge_id,
            model_name=judge_model_name,
            session_suffix="goal-judge",
        )

    def _build_judge_prompt(
        self,
        goal: str,
        attempt: int,
        start_seq: int,
        validation: str,
    ) -> str:
        events = self.coding.thread_store.load_thread_events_since(
            self.coding.thread_id,
            start_seq,
        )
        transcript = _format_goal_conversation(events)
        return GOAL_JUDGE.format(
            goal=goal,
            attempt=attempt,
            max_attempts=self.max_attempts,
            validation=validation,
            start_seq=start_seq,
            transcript=transcript,
        )

    async def _judge(
        self,
        goal: str,
        attempt: int,
        start_seq: int,
        validation: str,
    ) -> JudgeVerdict:
        if self._judge_model is None:
            self._judge_model = self._build_judge_model()
        prompt = self._build_judge_prompt(goal, attempt, start_seq, validation)
        try:
            structured = self._judge_model.with_structured_output(JudgeStructuredOutput)
            out: JudgeStructuredOutput = await structured.ainvoke(
                [HumanMessage(content=prompt)]
            )
        except Exception as exc:
            return JudgeVerdict(
                False,
                (f"Judge structured output failed: {exc}",),
                (),
                "",
            )
        return _verdict_from_structured(out)

    async def run(
        self,
        goal: str,
        *,
        worker_turn: WorkerTurn,
        on_status: StatusCallback,
    ) -> GoalResult:
        start_seq = self.coding.thread_store.append_event(
            self.coding.thread_id,
            {
                "kind": "goal",
                "phase": "start",
                "goal": goal,
                "max_attempts": self.max_attempts,
            },
        )
        if start_seq is None:
            # Autosave off: fall back to judging an empty slice rather than crashing.
            start_seq = 0
        instruction = goal
        last_verdict = JudgeVerdict(
            False,
            ("No attempt completed.",),
            (),
            goal,
        )
        for attempt in range(1, self.max_attempts + 1):
            on_status("worker", f"attempt {attempt}/{self.max_attempts}")
            await worker_turn(instruction)
            if self.coding.is_cancelled():
                return GoalResult(False, attempt, last_verdict)
            has_validation = any(
                hook.event == "goalValidate"
                for hook in self.coding.hook_runner.load()
            )
            if has_validation:
                hook_ok, hook_message = self.coding.hook_runner.run(
                    "goalValidate",
                    {
                        "goal": goal,
                        "attempt": attempt,
                        "thread_id": self.coding.thread_id,
                    },
                )
                validation = (
                    f"PASS: {hook_message or 'configured validation passed'}"
                    if hook_ok
                    else f"FAIL: {hook_message or 'configured validation failed'}"
                )
            else:
                hook_ok = True
                validation = "(no deterministic validation configured)"
            on_status("judge", f"verifying attempt {attempt}")
            last_verdict = await self._judge(
                goal,
                attempt,
                start_seq,
                validation,
            )
            if not hook_ok:
                repair = (
                    f"Deterministic validation failed: {validation}. "
                    + last_verdict.repair_instruction
                ).strip()
                last_verdict = JudgeVerdict(
                    passed=False,
                    unmet=(
                        *last_verdict.unmet,
                        f"Deterministic validation failed: {validation}",
                    ),
                    evidence=(*last_verdict.evidence, validation),
                    repair_instruction=repair,
                )
            self.coding.thread_store.append_event(
                self.coding.thread_id,
                {
                    "kind": "goal",
                    "phase": "judge",
                    "attempt": attempt,
                    "pass": last_verdict.passed and hook_ok,
                    "unmet": list(last_verdict.unmet),
                    "evidence": list(last_verdict.evidence),
                    "repair_instruction": last_verdict.repair_instruction,
                    "validation": validation,
                },
            )
            if last_verdict.passed and hook_ok:
                return GoalResult(True, attempt, last_verdict)
            if attempt < self.max_attempts:
                repair = _repair_text(last_verdict)
                instruction = GOAL_REPAIR.format(goal=goal, repair=repair)
        return GoalResult(False, self.max_attempts, last_verdict)
