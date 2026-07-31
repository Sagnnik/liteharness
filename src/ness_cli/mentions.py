"""@-mention expansion for the coding adapter.

Adapted from ``cli/mentions.py`` to live on the SDK side of the world so the
:mod:`ness_cli.coding_session` adapter can expand ``@file`` mentions
without importing the CLI's ``permissions`` module.

Typing ``@`` in the TUI's input buffer inserts a visible ``@<relative/path>``
token; on submit, :func:`expand_documents` reads each mentioned file
(validated through :class:`ness_ai.permissions.PermissionStore`) and
prepends one ``<document>`` block per file before the user's prose::

    <document>
      <document_content>
      [file contents]
      </document_content>
      <source>relative/path</source>
    </document>

The raw ``@``-tagged text is what gets persisted to the events table; the
expansion happens fresh on every ``run_turn`` and again on resume/rollback
replay so file content always reflects current disk (see the symmetric branch
in :mod:`ness_cli.events`'s :func:`events_to_messages`).

The menu support (``index_files``, ``filter_files``) stays on the TUI side;
this module only ships the expansion primitive.
"""

from __future__ import annotations

import re
from pathlib import Path

from ness_ai.permissions import PermissionStore


# A mention is ``@`` not preceded by a word char, followed by one or more
# path-safe chars.
_MENTION_TOKEN_RE = re.compile(r"(?<![\w])@([\w./\-]+)")

# Files larger than this are not inlined as a <document> block; an inline note
# tells the model the file is large and to use the read tool with an offset.
MAX_INLINE_FILE_BYTES = 256 * 1024


def extract_mentions(text: str) -> tuple[str, list[str]]:
    """Return ``(text_unchanged, list_of_mention_paths)``.

    The text is returned as-is so callers (buffer rendering, transcript
    persistence) keep the visible ``@token``; the path list is in the order
    the mentions appear, duplicates preserved.
    """
    paths: list[str] = []
    for match in _MENTION_TOKEN_RE.finditer(text or ""):
        paths.append(match.group(1))
    return text or "", paths


def expand_documents(text: str, permission_store: PermissionStore) -> str:
    """Prepend ``<document>`` blocks for each @mention and return the augmented text.

    The user's original text (with its ``@tokens``) is preserved verbatim
    after the block preamble. Failed reads (missing, outside root, binary,
    oversized) become inline ``Error: ...`` notes — the run still proceeds
    so the rest of the prompt reaches the model.
    """
    if not text:
        return text
    mentions = extract_mentions(text)[1]
    if not mentions:
        return text

    blocks: list[str] = []
    for rel in mentions:
        blocks.append(_render_document_block(rel, permission_store))

    preamble = "\n\n".join(blocks)
    if not preamble.strip():
        return text
    return preamble + "\n\n" + text


def _render_document_block(rel: str, permission_store: PermissionStore) -> str:
    """Wrap one mentioned file in the <document> XML block, or an error note."""
    try:
        abs_path = permission_store.validate_path(rel)
    except Exception as exc:
        return (
            f"<document>\n  <document_content>\n  Error: {exc}\n  </document_content>\n"
            f"  <source>{rel}</source>\n</document>"
        )

    p = Path(abs_path)
    try:
        if not p.exists():
            return (
                f"<document>\n  <document_content>\n  Error: {rel} does not exist\n"
                f"  </document_content>\n  <source>{rel}</source>\n</document>"
            )
        if p.is_dir():
            return (
                f"<document>\n  <document_content>\n  Error: {rel} is a directory\n"
                f"  </document_content>\n  <source>{rel}</source>\n</document>"
            )
        size = p.stat().st_size
        if size > MAX_INLINE_FILE_BYTES:
            return (
                f"<document>\n  <document_content>\n  Error: {rel} is too large ({size} bytes) to inline "
                f"as a mention; use the read tool with an offset to inspect it.\n"
                f"  </document_content>\n  <source>{rel}</source>\n</document>"
            )
        content = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return (
            f"<document>\n  <document_content>\n  Error: {rel} is not valid UTF-8 (likely binary)\n"
            f"  </document_content>\n  <source>{rel}</source>\n</document>"
        )
    except Exception as exc:
        return (
            f"<document>\n  <document_content>\n  Error: {exc}\n  </document_content>\n"
            f"  <source>{rel}</source>\n</document>"
        )

    return (
        f"<document>\n  <document_content>\n  {content}\n  </document_content>\n"
        f"  <source>{rel}</source>\n</document>"
    )