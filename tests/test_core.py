import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("OPENAI_API_KEY", "test")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import _with_working_state_tail, build_graph
from context import load_instruction
from compaction import (
    COMPACTION_HARD_RATIO,
    ContextPressure,
    calculate_context_pressure,
    compact_messages_progressively,
    compaction_action_for_ratio,
    resolve_usable_context_budget,
    summarize_history,
)
from config import CostTracker, settings
from parsers import extract_tool_calls
from reflection import run_reflection_gate
from skill_loader import render_skill_catalog, select_sticky_skills
from tools import get_tools_for_names, select_tools_for_session


class EchoModel:
    def __init__(self, text: str = "done") -> None:
        self.text = text
        self.bound_tools = []
        self.seen_messages = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    async def ainvoke(self, messages):
        self.seen_messages.append(list(messages))
        return AIMessage(content=self.text)


class FakeToolCallModel(EchoModel):
    def __init__(self, tool_call: dict) -> None:
        super().__init__()
        self.tool_call = tool_call
        self.calls = 0

    async def ainvoke(self, messages):
        self.seen_messages.append(list(messages))
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="", tool_calls=[self.tool_call])
        return AIMessage(content="done")


class CoreRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._settings = {
            "enable_approval": settings.enable_approval,
            "auto_save_threads": settings.auto_save_threads,
            "session_end_reflection": settings.session_end_reflection,
            "reflection_token_ratio": settings.reflection_token_ratio,
            "compaction_token_budget": settings.compaction_token_budget,
            "model_name": settings.model_name,
            "compaction_output_reserve_tokens": settings.compaction_output_reserve_tokens,
            "compaction_input_reserve_tokens": settings.compaction_input_reserve_tokens,
        }
        settings.enable_approval = False
        settings.auto_save_threads = False

    def tearDown(self) -> None:
        for name, value in self._settings.items():
            setattr(settings, name, value)

    def test_native_tool_args_keep_dicts(self) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "agent.py"}, "id": "1"}],
        )
        self.assertEqual(extract_tool_calls(msg), [("read_file", {"path": "agent.py"}, "1")])

    def test_session_tools_include_deployment_surface_and_git_gate(self) -> None:
        with_git = {tool.name for tool in select_tools_for_session(git_repo=True)}
        for name in (
            "read_file",
            "write_file",
            "delete_file",
            "edit",
            "check_syntax",
            "web_search",
            "fetch_url",
            "shell",
            "git",
            "todo",
            "search_tools",
            "add_tools",
            "spawn_subagent",
            "ask_user",
        ):
            self.assertIn(name, with_git)

        without_git = {tool.name for tool in select_tools_for_session(git_repo=False)}
        self.assertNotIn("git", without_git)
        self.assertIn("write_file", without_git)

    async def test_graph_executes_native_tool_call(self) -> None:
        model = FakeToolCallModel(
            {"name": "read_file", "args": {"path": "README.md"}, "id": "call-1"}
        )
        app = build_graph(model, tools=get_tools_for_names(["read_file"]), thread_id="unit")
        result = await app.ainvoke(
            {
                "messages": [HumanMessage(content="read README.md")],
                "approval_declined": False,
                "todos": [],
            },
            config={"configurable": {"thread_id": "unit"}},
        )

        self.assertEqual(result["messages"][-1].content, "done")
        self.assertTrue(any(message.type == "tool" for message in result["messages"]))

    async def test_plan_and_act_bind_same_tool_schema(self) -> None:
        plan_model = EchoModel()
        act_model = EchoModel()
        build_graph(plan_model, agent_mode="plan", git_available=True)
        build_graph(act_model, agent_mode="act", git_available=True)

        plan_names = sorted(tool.name for tool in plan_model.bound_tools)
        act_names = sorted(tool.name for tool in act_model.bound_tools)
        self.assertEqual(plan_names, act_names)
        for name in ("read_file", "write_file", "git", "shell", "ask_user"):
            self.assertIn(name, plan_names)

    async def test_plan_mode_blocks_write_tools_before_permissions(self) -> None:
        model = FakeToolCallModel(
            {
                "name": "write_file",
                "args": {"path": "blocked.txt", "content": "nope"},
                "id": "call-1",
            }
        )
        app = build_graph(
            model,
            tools=get_tools_for_names(["write_file"]),
            thread_id="plan-gate",
            agent_mode="plan",
        )

        with mock.patch("agent.check_with_rule", side_effect=AssertionError("permission should not run")):
            result = await app.ainvoke(
                {
                    "messages": [HumanMessage(content="write a file")],
                    "approval_declined": False,
                    "todos": [],
                    "agent_mode": "plan",
                },
                config={"configurable": {"thread_id": "plan-gate"}},
            )

        tool_messages = [message for message in result["messages"] if message.type == "tool"]
        self.assertTrue(tool_messages)
        self.assertIn("Unavailable in plan mode", tool_messages[-1].content)

    async def test_plan_mode_allows_read_only_shell_inspection(self) -> None:
        model = FakeToolCallModel(
            {"name": "shell", "args": {"action": "read", "job_id": "missing"}, "id": "call-1"}
        )
        app = build_graph(
            model,
            tools=get_tools_for_names(["shell"]),
            thread_id="plan-shell-read",
            agent_mode="plan",
        )

        with mock.patch("agent.check_with_rule", return_value=("allow", "shell:read:*")):
            result = await app.ainvoke(
                {
                    "messages": [HumanMessage(content="inspect a job")],
                    "approval_declined": False,
                    "todos": [],
                    "agent_mode": "plan",
                },
                config={"configurable": {"thread_id": "plan-shell-read"}},
            )

        tool_messages = [message for message in result["messages"] if message.type == "tool"]
        self.assertTrue(tool_messages)
        self.assertIn("Unknown shell job: missing", tool_messages[0].content)

    async def test_raw_state_keeps_user_message_while_model_sees_overlay(self) -> None:
        model = EchoModel()
        app = build_graph(model, tools=[], thread_id="raw-overlay")
        result = await app.ainvoke(
            {
                "messages": [HumanMessage(content="keep this raw")],
                "approval_declined": False,
                "todos": [],
            },
            config={"configurable": {"thread_id": "raw-overlay"}},
        )

        raw_humans = [message for message in result["messages"] if message.type == "human"]
        self.assertEqual(len(raw_humans), 1)
        self.assertEqual(raw_humans[0].content, "keep this raw")

        seen_humans = [message for message in model.seen_messages[0] if message.type == "human"]
        self.assertEqual(len(seen_humans), 1)
        seen = seen_humans[0].content
        self.assertIn("keep this raw", seen)
        self.assertIn("<system-reminder>", seen)
        self.assertIn("</system-reminder>", seen)

    async def test_mode_switch_note_one_shot_on_first_act_turn(self) -> None:
        model = EchoModel()
        app = build_graph(model, tools=[], thread_id="mode-switch", agent_mode="act")

        await app.ainvoke(
            {
                "messages": [HumanMessage(content="go")],
                "approval_declined": False,
                "todos": [],
                "agent_mode": "act",
                "mode_switch": "plan->act",
            },
            config={"configurable": {"thread_id": "mode-switch"}},
        )

        first_seen = [m for m in model.seen_messages[0] if m.type == "human"]
        first_overlay = first_seen[-1].content
        self.assertIn("MODE SWITCH", first_overlay)
        self.assertIn("action: replace", first_overlay)
        self.assertIn("do not blindly execute", first_overlay)

        # second turn on the same thread, no mode_switch in the payload
        await app.ainvoke(
            {
                "messages": [HumanMessage(content="next step")],
                "approval_declined": False,
                "todos": [],
                "agent_mode": "act",
                "mode_switch": "",
            },
            config={"configurable": {"thread_id": "mode-switch"}},
        )

        second_seen = [m for m in model.seen_messages[1] if m.type == "human"]
        second_overlay = second_seen[-1].content
        self.assertNotIn("MODE SWITCH", second_overlay)

    def test_plan_mode_instructions_defer_todo_to_act_switch(self) -> None:
        text = load_instruction("plan_mode")
        # Plan mode no longer tells the agent to call `todo` — that's
        # deferred to the first act turn after plan->act switch.
        self.assertNotIn("Immediately call `todo`", text)
        self.assertNotIn("action: replace", text)
        self.assertIn("Do not include tool calls in this message", text)

    async def test_plan_mode_block_only_on_fresh_turn_not_tool_loop(self) -> None:
        model = FakeToolCallModel(
            {"name": "read_file", "args": {"path": "README.md"}, "id": "call-1"}
        )
        app = build_graph(
            model,
            tools=get_tools_for_names(["read_file"]),
            thread_id="plan-delta",
            agent_mode="plan",
            git_available=False,
        )
        await app.ainvoke(
            {
                "messages": [HumanMessage(content="plan something")],
                "approval_declined": False,
                "todos": [],
                "agent_mode": "plan",
            },
            config={"configurable": {"thread_id": "plan-delta"}},
        )

        # fresh turn: full overlay on the user message, includes <plan-mode>
        fresh_user = [m for m in model.seen_messages[0] if m.type == "human"][0]
        self.assertIn("<plan-mode", fresh_user.content)

        # tool loop: read_file doesn't change any section -> empty delta -> no overlay tail
        tool_loop_msgs = model.seen_messages[1]
        overlay_tails = [
            m for m in tool_loop_msgs
            if m.type == "human" and "<system-reminder>" in (m.content or "")
        ]
        self.assertEqual(len(overlay_tails), 0)

    async def test_tool_loop_no_tail_when_nothing_changed(self) -> None:
        model = FakeToolCallModel(
            {"name": "read_file", "args": {"path": "README.md"}, "id": "call-1"}
        )
        app = build_graph(
            model,
            tools=get_tools_for_names(["read_file"]),
            thread_id="no-delta",
            agent_mode="act",
            git_available=False,
        )
        await app.ainvoke(
            {
                "messages": [HumanMessage(content="read it")],
                "approval_declined": False,
                "todos": [],
                "agent_mode": "act",
            },
            config={"configurable": {"thread_id": "no-delta"}},
        )

        # tool loop: nothing changed -> empty delta -> no tail HumanMessage
        tool_loop_msgs = model.seen_messages[1]
        last = tool_loop_msgs[-1]
        self.assertNotEqual(last.type, "human")  # last is the ToolMessage

    async def test_tool_loop_delta_only_changed_todos(self) -> None:
        model = FakeToolCallModel(
            {
                "name": "todo",
                "args": {
                    "action": "insert",
                    "index": 1,
                    "content": "New task",
                    "status": "pending",
                },
                "id": "call-1",
            }
        )
        app = build_graph(
            model,
            tools=get_tools_for_names(["todo"]),
            thread_id="todos-delta",
            agent_mode="plan",
            git_available=False,
        )
        await app.ainvoke(
            {
                "messages": [HumanMessage(content="plan this")],
                "approval_declined": False,
                "todos": [{"id": "1", "content": "Existing", "status": "pending"}],
                "agent_mode": "plan",
            },
            config={"configurable": {"thread_id": "todos-delta"}},
        )

        # tool loop: todos changed -> delta tail with TODOS, but no <plan-mode>, no GIT
        tool_loop_msgs = model.seen_messages[1]
        tail = tool_loop_msgs[-1]
        self.assertEqual(tail.type, "human")  # delta tail HumanMessage
        self.assertIn("<system-reminder>", tail.content)
        self.assertIn("TODOS", tail.content)
        self.assertIn("New task", tail.content)
        self.assertNotIn("<plan-mode", tail.content)
        self.assertNotIn("GIT", tail.content)

    def test_cost_tracker_accounts_for_cache_and_provider_costs(self) -> None:
        tracker = CostTracker()
        estimated = tracker.add(
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_token_details": {"cache_read": 40},
            },
            "gpt-4o-mini",
        )
        provider = tracker.add(
            {"input_tokens": 100, "output_tokens": 1},
            "gpt-4o-mini",
            {"cost": 0.123},
        )

        self.assertIsNotNone(estimated)
        self.assertEqual(estimated["cached_input_tokens"], 40)
        self.assertEqual(estimated["uncached_input_tokens"], 60)
        self.assertEqual(estimated["cost_source"], "estimated")
        self.assertIsNotNone(provider)
        self.assertEqual(provider["cost_source"], "provider")
        self.assertEqual(provider["cost_usd"], 0.123)

    async def test_reflection_structured_output_appends_session_memory(self) -> None:
        import memory
        import reflection

        class StructuredModel:
            def with_structured_output(self, _schema):
                return self

            async def ainvoke(self, _messages):
                return reflection.ReflectionStructuredOutput(
                    new_bullet_points=["Added retry wrapper to fetch_url"],
                )

        with tempfile.TemporaryDirectory() as tmp:
            ness = Path(tmp) / ".ness"
            ness.mkdir()
            with (
                mock.patch.object(memory, "NESS_DIR", ness),
                mock.patch.object(memory, "SESSIONS_DIR", ness / "sessions"),
            ):
                result = await run_reflection_gate(
                    "thread-1",
                    [HumanMessage(content="implemented retry")],
                    StructuredModel(),
                    5,
                )
                loaded = memory.load_session_memory("thread-1")

        self.assertTrue(result.memory_updated)
        self.assertIn("retry wrapper", loaded)

    def test_ness_includes_inline_files_without_escaping_project_root(self) -> None:
        import memory

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("Run pytest before committing.\n", encoding="utf-8")
            original = os.getcwd()
            os.chdir(root)
            try:
                expanded, includes = memory._expand_ness_includes("@AGENTS.md\n\nProject extras.")
                escaped, escaped_includes = memory._expand_ness_includes("@../secret.md")
            finally:
                os.chdir(original)

        self.assertIn("Run pytest before committing.", expanded)
        self.assertIn("Project extras.", expanded)
        self.assertEqual([path.name for path in includes], ["AGENTS.md"])
        self.assertIn("missing include", escaped)
        self.assertEqual(escaped_includes, [])

    def test_skill_catalog_and_sticky_selection(self) -> None:
        skills = {
            "alpha": {"name": "alpha", "description": "Do alpha things", "triggers": ["alpha"]},
            "beta": {"name": "beta", "description": "Do beta things", "triggers": ["beta"]},
        }
        catalog = render_skill_catalog(skills)
        sticky: set[str] = set()
        selected = {skill["name"] for skill in select_sticky_skills("please use alpha", skills, sticky)}

        self.assertIn("/skill <name>", catalog)
        self.assertIn("- alpha: Do alpha things", catalog)
        self.assertIn("- beta: Do beta things", catalog)
        self.assertNotIn("read_file", catalog)
        self.assertEqual(selected, {"alpha"})
        self.assertEqual(
            {skill["name"] for skill in select_sticky_skills("later turn", skills, sticky)},
            {"alpha"},
        )

    def test_compaction_policy_uses_thresholds_and_bounded_keep_counts(self) -> None:
        self.assertEqual(compaction_action_for_ratio(0.69), ("none", 0))
        self.assertEqual(compaction_action_for_ratio(0.70), ("tool_outputs", 0))
        self.assertEqual(compaction_action_for_ratio(0.79), ("tool_outputs", 0))

        summary80 = compaction_action_for_ratio(0.80)
        summary90 = compaction_action_for_ratio(0.90)
        summary97 = compaction_action_for_ratio(0.97)
        self.assertEqual(summary80[0], "summary")
        self.assertEqual(summary90[0], "summary")
        self.assertEqual(summary97[0], "summary")
        self.assertGreaterEqual(summary80[1], summary90[1])
        self.assertGreaterEqual(summary90[1], summary97[1])
        self.assertGreaterEqual(summary97[1], 1)

        pressure = calculate_context_pressure(
            [HumanMessage(content="hello")],
            max_tokens=100,
            known_input_tokens=int(100 * COMPACTION_HARD_RATIO),
        )
        self.assertIsInstance(pressure, ContextPressure)
        self.assertTrue(pressure.hard_threshold_reached)
        self.assertEqual(pressure.action, "summary")
        self.assertGreaterEqual(pressure.keep_recent, 1)

    async def test_progressive_compaction_switches_from_tools_to_summary(self) -> None:
        messages = [
            HumanMessage(content=f"request {index}") if index % 2 == 0 else AIMessage(content=f"response {index}")
            for index in range(14)
        ]
        messages.insert(2, ToolMessage(tool_call_id="0", name="grep", content="config.py\nagent.py\n" * 40))

        class FakeSummaryModel:
            async def ainvoke(self, _messages):
                return AIMessage(content="Earlier work edited config.py and preserved constraints.")

        tool_only = await compact_messages_progressively(
            messages,
            max_tokens=10_000,
            known_input_tokens=7_500,
            summary_model=FakeSummaryModel(),
        )
        summarized = await compact_messages_progressively(
            messages,
            max_tokens=10_000,
            known_input_tokens=8_500,
            summary_model=FakeSummaryModel(),
        )

        self.assertTrue(tool_only.compacted)
        self.assertEqual(tool_only.action, "tool_outputs")
        self.assertFalse(
            any(message.type == "system" and "COMPACTED HISTORY" in message.content for message in tool_only.messages)
        )
        self.assertTrue(summarized.compacted)
        self.assertEqual(summarized.action, "summary")
        self.assertGreaterEqual(summarized.kept_recent, 1)
        self.assertLessEqual(summarized.kept_recent, 10)
        self.assertTrue(
            any(message.type == "system" and "COMPACTED HISTORY" in message.content for message in summarized.messages)
        )

    async def test_summarize_history_falls_back_without_model(self) -> None:
        summary = await summarize_history([HumanMessage(content="hello")], None)
        self.assertIn("[Compaction summary unavailable]", summary)
        self.assertIn("human: hello", summary)

    def test_usable_budget_prefers_model_relative_defaults(self) -> None:
        settings.model_name = "gpt-4o-mini"
        settings.compaction_output_reserve_tokens = 8_000
        settings.compaction_input_reserve_tokens = 2_000
        self.assertEqual(resolve_usable_context_budget(), 118_000)


class WorkingStateTailTests(unittest.TestCase):
    def test_appends_overlay_to_fresh_human_turn(self) -> None:
        result = _with_working_state_tail([HumanMessage(content="hello")], "TODOS\n- [pending] 1: Example")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].type, "human")
        self.assertIn("hello", result[0].content)
        self.assertIn("<system-reminder>", result[0].content)
        self.assertIn("TODOS", result[0].content)

    def test_adds_tail_message_after_tool_result(self) -> None:
        messages = [
            HumanMessage(content="hi"),
            AIMessage(content="ok"),
            ToolMessage(tool_call_id="call-1", name="read_file", content="file contents"),
        ]
        result = _with_working_state_tail(messages, "TODOS\n- [pending] 1: Example")

        self.assertEqual(len(result), 4)
        self.assertEqual(result[-1].type, "human")
        self.assertIn("<system-reminder>", result[-1].content)
        self.assertIn("TODOS", result[-1].content)
        self.assertEqual(result[0].content, "hi")


if __name__ == "__main__":
    unittest.main()
