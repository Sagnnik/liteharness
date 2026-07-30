from __future__ import annotations

import liteharness_cli.tui.transcript as transcript_module
from liteharness_cli.tui.header import header_lines
from liteharness_cli.tui.models import TranscriptLine
from liteharness_cli.tui.transcript import TranscriptMixin


def _all_text(lines: list[TranscriptLine]) -> str:
    return "\n".join(line.text for line in lines)


def _assert_fragments_join(lines: list[TranscriptLine]) -> None:
    for line in lines:
        if not line.fragments:
            continue
        joined = "".join(text for _, text in line.fragments)
        assert joined == line.text, f"fragment mismatch: {joined!r} != {line.text!r}"


def _common_kwargs(**over: object) -> dict:
    base: dict = dict(
        mode="act",
        model="deepseek-v4-flash",
        approval=True,
        project="~/projects/liteharness",
        addons_summary="2 MCPs (echo, fs), 3 Skills",
        version="0.1.0",
    )
    base.update(over)
    return base


def test_wide_layout_has_logo_title_dashboard_and_hints():
    lines = header_lines(**_common_kwargs(width=120, show_logo=True))
    assert len(lines) >= 6
    text = _all_text(lines)
    assert "Lite" in text
    assert "Harness" in text
    assert "Session" in text and "deepseek-v4-flash" in text
    assert "Project" in text and "~/projects/liteharness" in text
    assert "Mode" in text and "Act (auto-approval)" in text
    assert "Add-ons" in text and "3 Skills" in text
    assert "Hints:" in text
    assert "/help" in text and "/config" in text
    # rounded panel borders
    assert "╭" in text and "╮" in text and "╰" in text and "╯" in text
    assert "Hints:" in lines[-1].text
    _assert_fragments_join(lines)


def test_all_rows_exact_width_wide():
    width = 120
    lines = header_lines(**_common_kwargs(width=width, show_logo=True))
    for line in lines:
        assert len(line.text) == width, (
            f"row width {len(line.text)} != {width}: {line.text!r}"
        )


def test_narrow_layout_drops_logo():
    width = 80
    lines = header_lines(**_common_kwargs(width=width, show_logo=False))
    text = _all_text(lines)
    # no braille pattern char (U+2800..U+28FF) should appear
    assert not any("\u2800" <= ch <= "\u28ff" for ch in text), (
        "braille logo leaked into narrow path"
    )
    assert "Lite" in text and "Harness" in text
    assert "Session" in text and "Mode" in text
    assert "Hints:" in lines[-1].text
    _assert_fragments_join(lines)


def test_too_narrow_returns_empty():
    lines = header_lines(**_common_kwargs(width=20, show_logo=True))
    assert lines == []


def test_plan_mode_label():
    lines = header_lines(
        **_common_kwargs(mode="plan", approval=False, width=110, show_logo=True)
    )
    text = _all_text(lines)
    assert "Plan" in text
    assert "auto-approval" not in text


def test_yolo_mode_label():
    lines = header_lines(
        **_common_kwargs(
            mode="act",
            approval=False,
            yolo=True,
            width=110,
            show_logo=True,
        )
    )
    assert "Act (yolo)" in _all_text(lines)


def test_title_uses_gradient_fragments():
    lines = header_lines(**_common_kwargs(width=100, show_logo=False))
    title_line = lines[0]
    assert title_line.fragments is not None
    # find the "Harness" portion: should be 7 fragments, each with unique fg
    frag_texts = [t for _, t in title_line.fragments]
    assert "H" in frag_texts and "s" in frag_texts
    harness_styles = [s for s, t in title_line.fragments if t and t in "Harness"]
    assert len(harness_styles) == 7
    # gradient: first harness char ~= cyan, last ~= purple
    assert "39c5cf" in harness_styles[0].lower()
    assert "a78bfa" in harness_styles[-1].lower()


def test_long_project_is_truncated_with_ellipsis():
    long_project = "~/projects/" + ("x" * 100)
    lines = header_lines(
        **_common_kwargs(project=long_project, width=110, show_logo=True)
    )
    text = _all_text(lines)
    assert "…" in text


# --- in-place replace + resize reflow (TranscriptMixin behavior) ------------


class _FakeStore:
    """Minimal TranscriptStore stand-in for append_header / reflow tests."""

    def __init__(self, width: int = 120) -> None:
        self.lines: list[TranscriptLine] = []
        self.revision = 0
        self.width = width

    def set_width(self, width: int) -> bool:
        if width == self.width:
            return False
        self.width = width
        return True

    def append(self, lines: list[TranscriptLine]) -> None:
        self.lines.extend(lines)
        self.revision += 1

    def replace(self, start: int, count: int, lines: list[TranscriptLine]) -> None:
        self.lines[start : start + count] = lines
        self.revision += 1


class _HeaderHarness(TranscriptMixin):
    """Bare TranscriptMixin instance with stubbed dependencies."""

    def __init__(self, *, width: int = 120) -> None:
        self._transcript_store = _FakeStore(width=width)
        self._transcript_revision = 0
        self._transcript_render_width = width
        self._header_block: dict | None = None
        self._follow_transcript = True
        self._transcript_pane = None
        self._transcript_ready = __import__("asyncio").Event()
        self._transcript_viewport_height = 40

    def _scroll_transcript_to_bottom(self) -> None: ...

    def _transcript_viewport_lines(self) -> int:
        return 40

    def invalidate(self) -> None: ...

    def append_notice(self, title: str, *lines: str) -> None: ...

    # keep pytest from collecting this as a test class
    __test__ = False


def _patch_addon_helpers() -> None:
    """Pin the lazy project/addons/version helpers to deterministic values."""
    transcript_module._header_project = lambda: "~/projects/liteharness"
    transcript_module._header_addons_summary = lambda *args: "0 MCPs, 1 Skills"
    transcript_module._header_version = lambda: "0.1.0"


def test_append_header_replaces_block_in_place_instead_of_duplicating():
    _patch_addon_helpers()
    m = _HeaderHarness(width=120)
    m.append_header(
        mode="act",
        model="m1",
        approval=True,
        autosave=True,
        session_end_reflection=True,
    )
    first_count = len(m._transcript_store.lines)
    assert m._header_block["start"] == 0
    assert m._header_block["count"] == first_count
    assert m._header_block["source"]["mode"] == "act"

    # second call (simulates /config refresh after a mode/model change)
    m.append_header(
        mode="plan",
        model="m2",
        approval=False,
        autosave=True,
        session_end_reflection=True,
    )
    second_count = len(m._transcript_store.lines)
    # no duplicate banner: the line count is unchanged and the block still sits at index 0
    assert second_count == first_count
    assert m._header_block["start"] == 0
    assert m._header_block["count"] == second_count
    assert m._header_block["source"]["mode"] == "plan"
    text = "\n".join(line.text for line in m._transcript_store.lines)
    assert "Plan" in text and "m2" in text
    # the refreshed mode should NOT be accompanied by the stale one
    assert text.count("LiteHarness") == 1


def test_resize_reflows_header_block_to_new_width():
    _patch_addon_helpers()
    m = _HeaderHarness(width=120)
    m.append_header(
        mode="act",
        model="m1",
        approval=True,
        autosave=True,
        session_end_reflection=True,
    )
    # shrink
    m._on_transcript_render_size(90, 40)
    assert m._header_block["width"] == 90
    for line in m._transcript_store.lines:
        if line.text:
            assert len(line.text) == 90, f"line not reflowed to 90: {line.text[:30]!r}"
    # grow back
    m._on_transcript_render_size(140, 40)
    assert m._header_block["width"] == 140
    for line in m._transcript_store.lines:
        if line.text:
            assert len(line.text) == 140, (
                f"line not reflowed to 140: {line.text[:30]!r}"
            )


def test_resize_is_noop_when_width_unchanged():
    _patch_addon_helpers()
    m = _HeaderHarness(width=120)
    m.append_header(
        mode="act",
        model="m1",
        approval=True,
        autosave=True,
        session_end_reflection=True,
    )
    before_revision = m._transcript_store.revision
    m._on_transcript_render_size(120, 40)  # same width
    assert m._transcript_store.revision == before_revision
