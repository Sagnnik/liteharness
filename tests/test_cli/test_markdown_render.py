from __future__ import annotations

from cli.tui.markdown_render import markdown_transcript_lines
from cli.tui.models import TranscriptLine


def _all_text(lines: list[TranscriptLine]) -> str:
    """Concatenate every line's text (the visible, escape-stripped transcript)."""
    return "\n".join(line.text for line in lines)


def _assert_fragments_join(lines: list[TranscriptLine]) -> None:
    """TranscriptStore invariant: fragment text concatenation == line.text."""
    for line in lines:
        if line.fragments is None:
            continue
        joined = "".join(text for _, text in line.fragments)
        assert joined == line.text, f"fragment mismatch: {joined!r} != {line.text!r}"


def _has_styled_fragment(lines: list[TranscriptLine], *, containing: str = "") -> bool:
    for line in lines:
        for style, text in line.fragments or []:
            if containing in text and style:
                return True
    return False


def test_empty_returns_single_blank_line():
    lines = markdown_transcript_lines("", width=80)
    assert len(lines) == 1
    assert lines[0].text == ""


def test_plain_text_is_present_and_invariant_holds():
    lines = markdown_transcript_lines("just some words", width=80)
    assert "just some words" in _all_text(lines)
    _assert_fragments_join(lines)


def test_headings_are_deep_blue_and_bold():
    lines = markdown_transcript_lines("# Title\n\n## Sub", width=80)
    assert "Title" in _all_text(lines)
    assert "Sub" in _all_text(lines)
    assert any("#4aa3df" in (s or "") and "bold" in (s or "") for line in lines for s, _ in line.fragments or [])
    _assert_fragments_join(lines)


def test_bold_italic_inline_code_and_link_styles():
    lines = markdown_transcript_lines("Some **bold** and *italic* and `code` and [a link](http://x.com).", width=80)
    text = _all_text(lines)
    for needle in ("bold", "italic", "code", "a link"):
        assert needle in text
    # inline code is branded yellow on the code background
    assert _has_styled_fragment(lines, containing="code") and any(
        "#d29922" in (s or "") and "bg:#252531" in (s or "")
        for line in lines
        for s, _ in line.fragments or []
    )
    # link URL must not leak into the visible transcript
    assert "http" not in text
    _assert_fragments_join(lines)


def test_unordered_and_ordered_lists():
    lines = markdown_transcript_lines("- alpha\n- beta\n1. first\n2. second", width=80)
    text = _all_text(lines)
    for needle in ("alpha", "beta", "first", "second"):
        assert needle in text
    assert "•" in text  # rich bullet glyph
    # bullet/number markers are colored deep blue
    assert any("#4aa3df" in (s or "") for line in lines for s, _ in line.fragments or [])
    _assert_fragments_join(lines)


def test_fenced_code_block_has_background_and_highlight():
    lines = markdown_transcript_lines("```python\nprint(1)\n```", width=80)
    text = _all_text(lines)
    assert "print(1)" in text
    # code rows carry a background style and the print builtin is syntax-colored
    assert any("bg:#" in (s or "") for line in lines for s, _ in line.fragments or [])
    assert any("print" in t for line in lines for _, t in line.fragments or [])
    _assert_fragments_join(lines)


def test_unknown_fence_still_has_background():
    lines = markdown_transcript_lines("```nonsense-lang\nhello\n```", width=80)
    assert "hello" in _all_text(lines)
    assert any("bg:#" in (s or "") for line in lines for s, _ in line.fragments or [])
    _assert_fragments_join(lines)


def test_blockquote_has_bar():
    lines = markdown_transcript_lines("> a quoted line", width=80)
    text = _all_text(lines)
    assert "a quoted line" in text
    assert "▌" in text  # rich blockquote bar
    _assert_fragments_join(lines)


def test_horizontal_rule():
    lines = markdown_transcript_lines("---", width=80)
    assert len(lines) == 1
    assert set(lines[0].text.strip()) == {"-"}
    _assert_fragments_join(lines)


def test_blocks_are_separated_by_blank_lines():
    # Rich inserts blank spacing lines between blocks; this is the readability fix.
    lines = markdown_transcript_lines("# H\n\npara one.\n\npara two.", width=80)
    texts = [line.text for line in lines]
    assert "" in texts  # at least one blank separator line


def test_complex_document_invariant_holds():
    text = (
        "# Heading\n\n"
        "Paragraph with **bold**, *italic*, `code` and [link](http://x).\n"
        "second line of paragraph.\n\n"
        "## Subheading\n\n"
        "- item one\n- item two\n\n"
        "1. first\n2. second\n\n"
        "```python\nx = 1\nprint(x)\n```\n\n"
        "> a quote\n> more quote\n\n"
        "---\n\n"
        "final plain text"
    )
    lines = markdown_transcript_lines(text, width=100)
    _assert_fragments_join(lines)
    joined = _all_text(lines)
    assert "http" not in joined  # no link leakage
    for needle in ("Heading", "bold", "italic", "code", "link", "item one", "print", "final plain text"):
        assert needle in joined


def test_live_stream_helper_remains_plain():
    # The streaming path must NOT use markdown so incremental paint stays smooth;
    # only finalize swaps in styled markdown.
    from cli.tui.transcript import TranscriptMixin

    lines = TranscriptMixin._assistant_stream_lines("not **markdown**")
    assert len(lines) == 1
    assert lines[0].text == "not **markdown**"
    assert lines[0].fragments is None
