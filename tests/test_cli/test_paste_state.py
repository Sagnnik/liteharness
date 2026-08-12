from __future__ import annotations


def test_multiline_paste_preserves_existing_prompt_and_expands_on_submit(make_app):
    app = make_app()
    app._buffer.insert_text("Explain this: ")

    app._buffer.insert_text("first line\nsecond line\nthird line")

    assert app._buffer.text == "Explain this: [pasted 3 lines]"
    assert app._expand_paste(app._buffer.text) == (
        "Explain this: first line\nsecond line\nthird line"
    )

    app._buffer.insert_text(" and summarize it")
    assert app._buffer.text == "Explain this: [pasted 3 lines] and summarize it"
    assert app._expand_paste(app._buffer.text) == (
        "Explain this: first line\nsecond line\nthird line and summarize it"
    )


def test_multiline_paste_in_middle_preserves_following_text_and_cursor(make_app):
    app = make_app()
    app._buffer.insert_text("before after")
    app._buffer.cursor_position = len("before ")

    app._buffer.insert_text("one\ntwo")

    assert app._buffer.text == "before [pasted 2 lines]after"
    assert app._buffer.cursor_position == len("before [pasted 2 lines]")
    assert app._expand_paste(app._buffer.text) == "before one\ntwoafter"


def test_deleting_paste_marker_discards_only_pasted_payload(make_app):
    app = make_app()
    app._buffer.insert_text("keep ")
    app._buffer.insert_text("remove\nthis")

    app._buffer.text = app._buffer.text.replace("[pasted 2 lines]", "")

    assert app._buffer.text == "keep "
    assert app._pending_paste is None


def test_deleting_image_marker_discards_matching_image_only(make_app):
    app = make_app()
    app._pending_images[1] = "data:image/png;base64,first"
    app._buffer.insert_text("[Image #1] ")
    app._pending_images[2] = "data:image/png;base64,second"
    app._buffer.insert_text("[Image #2] ")

    app._buffer.text = app._buffer.text.replace("[Image #1] ", "")

    assert app._pending_images == {2: "data:image/png;base64,second"}
    assert app._images_for_text(app._buffer.text) == ["data:image/png;base64,second"]
