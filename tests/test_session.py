from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from liteharness.persistence import ThreadStore
from liteharness_cli.events import _maybe_enrich_spawn_subagent_result, events_to_messages


class SessionStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.ness_dir = Path(self._tmpdir.name)
        self.store = ThreadStore(threads_dir=self.ness_dir / "threads", auto_save=True)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_append_and_load_roundtrip(self) -> None:
        self.store.append_event("session-abc", {"kind": "user", "content": "hello"})
        self.store.append_event(
            "session-abc",
            {
                "kind": "assistant",
                "content": "",
                "tool_calls": [{"name": "read", "args": {"path": "a"}, "id": "call-1"}],
            },
        )
        self.store.append_event(
            "session-abc",
            {
                "kind": "tool",
                "tool": "read",
                "args": {"path": "a"},
                "result": "file contents",
                "call_id": "call-1",
                "duration_ms": 1,
                "exit": "ok",
            },
        )

        events = self.store.load_thread_events("session-abc")
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0]["kind"], "user")
        self.assertIn("t", events[0])
        self.assertEqual(events[1]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(events[2]["result"], "file contents")

    def test_usage_updates_thread_aggregates(self) -> None:
        self.store.append_event("session-cost", {"kind": "user", "content": "hi"})
        self.store.append_event(
            "session-cost",
            {
                "kind": "usage",
                "model": "gpt-4o-mini",
                "input_tokens": 100,
                "cached_input_tokens": 20,
                "output_tokens": 10,
                "cost_usd": 0.01,
            },
        )

        rows = self.store.list_threads(5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["turn_count"], 1)
        self.assertEqual(rows[0]["input_tokens"], 100)
        self.assertEqual(rows[0]["cached_input_tokens"], 20)
        self.assertEqual(rows[0]["output_tokens"], 10)
        self.assertAlmostEqual(rows[0]["total_cost_usd"], 0.01)

    def test_list_threads_filters_to_session_prefix(self) -> None:
        self.store.append_event("session-visible", {"kind": "user", "content": "a"})
        self.store.append_event("subagent-explore-deadbeef", {"kind": "user", "content": "b"})
        self.store.append_event("todo-graph-abc", {"kind": "user", "content": "c"})

        rows = self.store.list_threads(10)
        self.assertEqual([row["thread_id"] for row in rows], ["session-visible"])
        self.assertIsNone(self._thread_row("subagent-explore-deadbeef"))
        self.assertEqual(self.store.load_thread_events("subagent-explore-deadbeef"), [])

    def test_archive_thread_idempotent(self) -> None:
        self.store.append_event("session-archive", {"kind": "user", "content": "archive me please"})
        first = self.store.archive_thread("session-archive")
        second = self.store.archive_thread("session-archive")
        self.assertIn("Archived thread", first)
        self.assertIn("already archived", second)

    def test_subagent_registration(self) -> None:
        self.store.append_event("session-parent", {"kind": "user", "content": "run subagents"})
        self.store.register_subagent(
            "session-parent",
            "subagent-explore-111",
            agent_name="explore",
            label="scan repo",
        )
        self.store.complete_subagent(
            "subagent-explore-111",
            status="ok",
            output="found main.py",
            duration_ms=42,
        )

        self.assertIsNotNone(self._thread_row("session-parent"))
        self.assertIsNone(self._thread_row("subagent-explore-111"))
        rows = self.store.list_subagents("session-parent")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["agent_name"], "explore")
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["output"], "found main.py")

    def test_subagent_usage_rolls_up_to_parent(self) -> None:
        self.store.append_event("session-parent-cost", {"kind": "user", "content": "run subagents"})
        self.store.register_subagent(
            "session-parent-cost",
            "subagent-explore-cost",
            agent_name="explore",
            label="scan",
        )
        self.store.append_event(
            "subagent-explore-cost",
            {
                "kind": "usage",
                "model": "gpt-4o-mini",
                "input_tokens": 50,
                "cached_input_tokens": 10,
                "output_tokens": 5,
                "cost_usd": 0.005,
            },
        )

        self.assertEqual(self.store.load_thread_events("subagent-explore-cost"), [])
        rows = self.store.list_threads(5)
        parent = next(row for row in rows if row["thread_id"] == "session-parent-cost")
        self.assertEqual(parent["input_tokens"], 50)
        self.assertEqual(parent["cached_input_tokens"], 10)
        self.assertEqual(parent["output_tokens"], 5)
        self.assertAlmostEqual(parent["total_cost_usd"], 0.005)

    def _thread_row(self, thread_id: str) -> dict | None:
        import sqlite3

        with sqlite3.connect(self.store.threads_db) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT thread_id FROM threads WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        return dict(row) if row else None

    def test_concurrent_append_same_thread(self) -> None:
        errors: list[Exception] = []

        def worker(index: int) -> None:
            try:
                self.store.append_event(
                    "session-concurrent",
                    {"kind": "usage", "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0},
                )
            except Exception as exc:
                errors.append(exc)

        self.store.append_event("session-concurrent", {"kind": "user", "content": "start"})
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        events = self.store.load_thread_events("session-concurrent")
        self.assertEqual(len(events), 9)

    def test_autosave_disabled_skips_writes(self) -> None:
        store = ThreadStore(threads_dir=self.ness_dir / "threads-off", auto_save=False)
        store.append_event("session-off", {"kind": "user", "content": "nope"})
        self.assertFalse(store.threads_db.exists())


class ResumeReplayTests(unittest.TestCase):
    def test_events_to_messages_replays_tool_chain(self) -> None:
        events = [
            {"kind": "user", "content": "read file"},
            {
                "kind": "assistant",
                "content": "",
                "tool_calls": [
                    {"name": "read", "args": {"path": "a"}, "id": "call-1", "type": "tool_call"}
                ],
            },
            {
                "kind": "tool",
                "tool": "read",
                "args": {"path": "a"},
                "result": "contents",
                "call_id": "call-1",
            },
            {"kind": "assistant", "content": "done"},
        ]
        messages = events_to_messages(events)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1].tool_calls[0]["id"], "call-1")
        self.assertEqual(messages[2].tool_call_id, "call-1")
        self.assertEqual(messages[2].content, "contents")

    def test_spawn_subagent_result_enrichment(self) -> None:
        short = "status=ok"
        subagents = [
            {
                "subagent_thread_id": "subagent-explore-111",
                "agent_name": "explore",
                "label": "",
                "status": "ok",
                "duration_ms": 10,
                "output": "detailed findings from subagent run",
            }
        ]
        enriched = _maybe_enrich_spawn_subagent_result("spawn_subagent", short, subagents)
        self.assertIn("detailed findings", enriched)
        self.assertGreater(len(enriched), len(short))


if __name__ == "__main__":
    unittest.main()
