"""Central color palette and styling for the LiteHarness CLI."""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

# --- raw palette ------------------------------------------------------------
GRAY = "#8a909c"
GRAY_DIM = "#5c626d"
GRAY_BRIGHT = "#c4c9d4"
USER_BOX_BG = "#2d2d38"
CYAN = "#39c5cf"
CYAN_DIM = "#2a8f97"
NAVY = "#1f3a63"
NAVY_BRIGHT = "#2f5a96"
GREEN = "#3fb950"
RED = "#f85149"
PURPLE = "#a78bfa"
YELLOW = "#d29922"
MINT = "#87C095"
LAVENDER = "#C6A0F6"
# expanded accents so cyan is not the only hue (deep blue + yellow identity)
BLUE = "#4aa3df"
#CODE_BG = "#252531"
CODE_BG = "#1A1B26"

# --- rich theme -------------------------------------------------------------
# Terminal fallback styles. The active TUI uses PTK_STYLE_RULES below.
RICH_THEME = Theme(
    {
        # message sections
        "user": f"bold {GRAY_BRIGHT}",
        "user.frame": f"{GRAY}",
        "assistant": "default",
        "assistant.frame": NAVY_BRIGHT,
        "assistant.label": f"bold {CYAN}",
        # tools
        "tool": f"bold {BLUE}",
        "tool.args": f"{GRAY_DIM}",
        "tool.result": f"{GRAY}",
        # status / notices
        "notice": YELLOW,
        "notice.frame": NAVY,
        "warning": YELLOW,
        "error": f"bold {RED}",
        "muted": GRAY_DIM,
        "accent": CYAN,
        "accent.dim": CYAN_DIM,
        # diffs
        "diff.add": GREEN,
        "diff.del": RED,
        "diff.meta": CYAN_DIM,
        "diff.hunk": NAVY_BRIGHT,
        # chrome
        "header": f"bold {CYAN}",
        "header.frame": NAVY_BRIGHT,
        "panel.frame": NAVY,
        "usage": f"{GRAY_DIM}",
        "usage.value": GRAY,
        # tables
        "table.header": f"bold {BLUE}",
        "table.dim": GRAY_DIM,
        # markdown (rendered by Rich into the TUI via the ANSI bridge in
        # cli.markdown; these keys brand Rich's Markdown element styles)
        "markdown.h1": f"bold {LAVENDER}",
        "markdown.h2": f"bold {LAVENDER}",
        "markdown.h3": f"bold {LAVENDER}",
        "markdown.h4": f"bold {LAVENDER}",
        "markdown.h5": f"bold {LAVENDER}",
        "markdown.h6": f"bold {LAVENDER}",
        "markdown.code": f"bold {MINT}",
        "markdown.block_quote": f"italic {GRAY}",
        "markdown.item.bullet": BLUE,
        "markdown.item.number": BLUE,
        "markdown.link": f"underline {PURPLE}",
        "markdown.hr": GRAY_DIM,
        "markdown.strong": "bold",
        "markdown.em": "italic",
    }
)


# --- prompt_toolkit style ---------------------------------------------------
# Full-screen TUI styles.
PTK_STYLE_RULES: dict[str, str] = {
    # input line
    "prompt": f"bold {CYAN}",
    "prompt.mode": f"bold {GREEN}",
    "prompt.mode.plan": f"bold {CYAN}",
    "": GRAY_BRIGHT,  # default input text
    "screen": "",
    "transcript.header": f"bold {CYAN}",
    "transcript.muted": GRAY_DIM,
    "transcript.user": f"bg:{USER_BOX_BG} {GRAY_BRIGHT}",
    "transcript.notice": f"bold {YELLOW}",
    "transcript.panel": GRAY,
    "transcript.warning": YELLOW,
    "transcript.error": f"bold {RED}",
    "transcript.assistant": "default",
    "transcript.reasoning.collapsed": f"bold {CYAN_DIM}",
    "transcript.reasoning": GRAY_DIM,
    "transcript.todo.title": f"bold {BLUE}",
    "transcript.tool": f"bold {BLUE}",
    "transcript.tool.args": GRAY,
    "transcript.tool.result": GRAY,
    "transcript.subagent.summary": GRAY_DIM,
    "transcript.diff.add": GREEN,
    "transcript.diff.del": RED,
    "transcript.diff.meta": CYAN_DIM,
    "transcript.diff.hunk": NAVY_BRIGHT,
    "transcript.diff.context": GRAY_DIM,
    "transcript.selection": "reverse",
    "transcript.tag.session": f"bold {CYAN}",
    "transcript.tag.mcp": f"bold {GREEN}",
    "transcript.tag.notice": f"bold {YELLOW}",
    "transcript.tag.skill": f"bold {PURPLE}",
    "transcript.tag.init": f"bold {BLUE}",
    "transcript.tag.save": f"bold {GRAY_BRIGHT}",
    "transcript.tag.body": GRAY_BRIGHT,
    "chrome.rule": PURPLE,
    "chrome.working.spinner": PURPLE,
    "chrome.worked": GRAY,
    "chrome.queue": f"bold {YELLOW}",
    "chrome.queue.arrow": YELLOW,
    "chrome.queue.preview": GRAY,
    "chrome.stats.key": GRAY_DIM,
    "chrome.stats.value": GRAY,
    "chrome.stats.accent": BLUE,
    "chrome.path": GRAY,
    "chrome.menu.header": CYAN,
    "chrome.menu.hint": GRAY_DIM,
    "chrome.menu.row": GRAY,
    "chrome.menu.row.current": f"bg:{NAVY}",
    "chrome.menu.arrow": f"bg:{NAVY} {CYAN}",
    "chrome.menu.label.current": f"bg:{NAVY} bold {GRAY_BRIGHT}",
    "chrome.menu.desc.current": f"bg:{NAVY} {GRAY}",
    "chrome.menu.suffix": f"bg:{NAVY} {CYAN}",
    "chrome.form.label": f"bold {BLUE}",
    "chrome.form.hint": GRAY_DIM,
    "chrome.input.box": GRAY_DIM,
    "chrome.input.field": GRAY_BRIGHT,
}


def build_console(**kwargs) -> Console:
    """Create a Console bound to the LiteHarness theme."""
    return Console(theme=RICH_THEME, **kwargs)


# Shared console used across the CLI.
console = build_console()
