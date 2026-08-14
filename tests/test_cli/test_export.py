from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from ness_agent.persistence import ThreadStore
from ness_cli.export import (
    ExportError,
    export_thread_html,
    normalize_events,
    resolve_export_path,
)


def _store(tmp_path: Path) -> ThreadStore:
    return ThreadStore(threads_dir=tmp_path / ".ness" / "threads", auto_save=True)


def test_export_contains_full_pre_compaction_history_and_normalized_jsonl(tmp_path):
    store = _store(tmp_path)
    thread_id = "session-export"
    store.set_thread_name(thread_id, "Audit export flow")
    store.append_event(
        thread_id,
        {
            "kind": "user",
            "content": "old request before compaction",
            "images": ["data:image/png;base64,SECRET_IMAGE_DATA"],
        },
    )
    store.append_event(
        thread_id,
        {
            "kind": "assistant",
            "content": "old answer </script><script>alert('x')</script> __JSONL_NAME__",
            "tool_calls": [
                {"name": "read", "args": {"path": "README.md"}, "id": "call-1"}
            ],
        },
    )
    store.append_event(
        thread_id,
        {
            "kind": "tool",
            "tool": "read",
            "args": {"path": "README.md"},
            "result": "file contents",
            "call_id": "call-1",
            "duration_ms": 8,
            "exit": "ok",
        },
    )
    store.append_event(
        thread_id,
        {
            "kind": "usage",
            "model": "test-model",
            "input_tokens": 100,
            "cached_input_tokens": 40,
            "output_tokens": 20,
        },
    )
    store.append_compaction_checkpoint(
        thread_id,
        {
            "response": "Earlier work was summarized.",
            "trigger": "automatic",
            "before_tokens": 1000,
            "after_tokens": 200,
            "active_suffix": [
                {"type": "human", "data": {"content": "DUPLICATE_ACTIVE_SUFFIX"}}
            ],
        },
        active_turn=False,
    )
    store.append_event(thread_id, {"kind": "user", "content": "new request"})
    store.append_event(thread_id, {"kind": "assistant", "content": "new answer"})

    destination = tmp_path / "exports" / "session.html"
    result = export_thread_html(
        thread_store=store,
        thread_id=thread_id,
        project_root=tmp_path,
        destination=destination,
        generated_at=datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc),
    )

    assert result.path == destination
    assert result.event_count == 7
    document = destination.read_text(encoding="utf-8")
    assert "old request before compaction" in document
    assert "new answer" in document
    assert "__JSONL_NAME__" in document
    assert "SECRET_IMAGE_DATA" not in document
    assert "DUPLICATE_ACTIVE_SUFFIX" not in document

    soup = BeautifulSoup(document, "html.parser")
    assert soup.title is not None
    assert "Audit export flow" in soup.title.get_text()
    assert len(soup.select("article.entry")) == 7
    assert len(soup.select("nav .nav-item")) == 7
    assert soup.select_one("meta[http-equiv='Content-Security-Policy']") is not None
    assert not soup.select("script[src], link[href]")
    scripts = soup.find_all("script")
    assert len(scripts) == 2
    assert all("alert('x')" not in (script.string or "") for script in scripts[1:])

    payload_node = soup.select_one("#export-jsonl")
    assert payload_node is not None
    jsonl = json.loads(payload_node.get_text())
    exported = [json.loads(line) for line in jsonl.splitlines()]
    assert [item["seq"] for item in exported] == list(range(7))
    assert exported[0]["details"]["attachments"] == [
        {"type": "image", "omitted": True, "label": "Image 1"}
    ]
    assert exported[4]["kind"] == "compaction_llm"
    assert exported[4]["content"] == "Earlier work was summarized."
    assert "active_suffix" not in exported[4]["details"]
    assert set(exported[0]) == {
        "seq",
        "timestamp",
        "kind",
        "category",
        "title",
        "preview",
        "content",
        "details",
    }


def test_normalize_event_types_and_subagent_summary():
    events = [
        {"kind": "approval", "tool": "shell", "decision": "yes"},
        {"kind": "compact", "content": "manual compaction requested"},
        {
            "kind": "reflection",
            "response": {"new_bullet_points": ["Remember this"]},
            "memory_updated": True,
        },
        {"kind": "goal", "phase": "start", "goal": "Ship it"},
        {"kind": "tool", "tool": "spawn_subagent", "result": "done"},
    ]
    records = normalize_events(
        events,
        subagents=[
            {
                "subagent_thread_id": "subagent-review-1",
                "agent_name": "review",
                "status": "ok",
                "output": "Looks good",
            }
        ],
    )

    assert [record.kind for record in records] == [
        "approval",
        "compact",
        "reflection",
        "goal",
        "tool",
    ]
    assert records[2].content == "• Remember this"
    assert records[3].title == "Goal · start"
    assert records[4].details["subagents"][0]["output"] == "Looks good"


def test_resolve_export_path_supports_quotes_and_requires_html(tmp_path):
    assert resolve_export_path('"reports/My session.html"', tmp_path) == (
        tmp_path / "reports" / "My session.html"
    ).resolve()
    absolute = tmp_path / "absolute.HTML"
    assert resolve_export_path(str(absolute), tmp_path) == absolute.resolve()

    with pytest.raises(ExportError, match="Usage"):
        resolve_export_path("", tmp_path)
    with pytest.raises(ExportError, match="must end in .html"):
        resolve_export_path("report.json", tmp_path)


def test_export_refuses_overwrite_disabled_autosave_and_empty_session(tmp_path):
    store = _store(tmp_path)
    store.append_event("session-one", {"kind": "user", "content": "hello"})
    destination = tmp_path / "session.html"
    destination.write_text("keep me", encoding="utf-8")
    with pytest.raises(ExportError, match="Refusing to overwrite"):
        export_thread_html(
            thread_store=store,
            thread_id="session-one",
            project_root=tmp_path,
            destination=destination,
        )
    assert destination.read_text(encoding="utf-8") == "keep me"

    store.auto_save = False
    with pytest.raises(ExportError, match="autosave is disabled"):
        export_thread_html(
            thread_store=store,
            thread_id="session-one",
            project_root=tmp_path,
            destination=tmp_path / "disabled.html",
        )

    empty = _store(tmp_path / "empty")
    with pytest.raises(ExportError, match="no durable events"):
        export_thread_html(
            thread_store=empty,
            thread_id="session-empty",
            project_root=tmp_path,
            destination=tmp_path / "empty.html",
        )
