from __future__ import annotations

from ness_agent.persistence import ThreadStore


def test_copy_thread_prefix_preserves_lineage_without_double_cost(tmp_path) -> None:
    store = ThreadStore(tmp_path / "threads", default_model="test-model")
    source = "session-source"
    first = store.append_event(source, {"kind": "user", "content": "first"})
    store.save_checkpoint(source, first, "git-a", "memory-a")
    store.append_event(source, {"kind": "assistant", "content": "answer"})
    store.append_event(
        source,
        {
            "kind": "usage",
            "model": "test-model",
            "input_tokens": 100,
            "output_tokens": 10,
            "cost_usd": 1.0,
        },
    )
    second = store.append_event(source, {"kind": "user", "content": "second"})
    store.save_checkpoint(source, second, "git-b", "memory-b")

    copied = store.copy_thread_prefix(source, "session-child", second)

    assert [event["kind"] for event in copied] == ["user", "assistant", "usage"]
    assert copied[-1]["inherited"] is True
    child = next(
        item for item in store.list_threads(10) if item["thread_id"] == "session-child"
    )
    assert child["fork_root_id"] == source
    assert child["fork_parent_id"] == source
    assert child["forked_from_seq"] == second
    assert child["turn_count"] == 1
    assert child["total_cost_usd"] == 0.0
    root = next(
        item for item in store.list_threads(10) if item["thread_id"] == source
    )
    assert root["fork_count"] == 1
    assert root["fork_index"] == 0
    assert child["fork_count"] == 1
    assert child["fork_index"] == 1


def test_fork_of_fork_keeps_root_lineage(tmp_path) -> None:
    store = ThreadStore(tmp_path / "threads")
    source = "session-root"
    boundary = store.append_event(source, {"kind": "user", "content": "root"})
    store.save_checkpoint(source, boundary, None, "")
    store.copy_thread_prefix(source, "session-child", boundary)
    child_boundary = store.append_event(
        "session-child", {"kind": "user", "content": "branch"}
    )
    store.save_checkpoint("session-child", child_boundary, None, "")

    store.copy_thread_prefix(
        "session-child",
        "session-grandchild",
        child_boundary,
    )

    rows = {row["thread_id"]: row for row in store.list_threads(10)}
    assert rows["session-grandchild"]["fork_root_id"] == source
    assert rows[source]["fork_count"] == 2
    assert rows[source]["fork_index"] == 0
    assert rows["session-child"]["fork_index"] == 1
    assert rows["session-grandchild"]["fork_index"] == 2
