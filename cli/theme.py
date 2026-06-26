"""Central color palette and styling for the LiteHarness CLI.

Palette (only these colors are used): black, gray, cyan, navy, green, red.
There is intentionally no yellow; warnings map to cyan and errors map to red.
"""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

# --- raw palette ------------------------------------------------------------
BLACK = "#0b0e14"
GRAY = "#8a909c"
GRAY_DIM = "#5c626d"
GRAY_BRIGHT = "#c4c9d4"
GRAY_DARK = "#1e222a"  # modal panel background (menu / config overlays)
CYAN = "#39c5cf"
CYAN_DIM = "#2a8f97"
NAVY = "#1f3a63"
NAVY_BRIGHT = "#2f5a96"
GREEN = "#3fb950"
RED = "#f85149"


# --- rich theme -------------------------------------------------------------
# Style names are referenced across render.py / commands.py / menu.py.
RICH_THEME = Theme(
    {
        # message sections
        "user": f"bold {GRAY_BRIGHT}",
        "user.frame": f"{GRAY}",
        "assistant": "default",
        "assistant.frame": NAVY_BRIGHT,
        "assistant.label": f"bold {CYAN}",
        # tools
        "tool": f"bold {CYAN}",
        "tool.args": f"{GRAY_DIM}",
        "tool.result": f"{GRAY}",
        # status / notices
        "notice": CYAN,
        "notice.frame": NAVY,
        "warning": CYAN,
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
        "table.header": f"bold {CYAN}",
        "table.dim": GRAY_DIM,
        "mode.normal": f"bold {GREEN}",
        "mode.plan": f"bold {CYAN}",
    }
)


# --- prompt_toolkit style ---------------------------------------------------
# Mirrors the palette for the input prompt, completer, bottom toolbar and the
# /menu overlay.
PTK_STYLE_RULES: dict[str, str] = {
    # input line
    "prompt": f"bold {CYAN}",
    "prompt.mode": f"bold {GREEN}",
    "prompt.mode.plan": f"bold {CYAN}",
    "": GRAY_BRIGHT,  # default input text
    # bottom toolbar
    "bottom-toolbar": f"bg:{NAVY} {GRAY_BRIGHT}",
    "bottom-toolbar.key": f"bg:{NAVY} bold {CYAN}",
    "bottom-toolbar.sep": f"bg:{NAVY} {GRAY_DIM}",
    # completion menu
    "completion-menu": f"bg:{BLACK} {GRAY}",
    "completion-menu.completion": f"bg:{BLACK} {GRAY}",
    "completion-menu.completion.current": f"bg:{NAVY} {GRAY_BRIGHT}",
    "completion-menu.meta.completion": f"bg:{BLACK} {GRAY_DIM}",
    "completion-menu.meta.completion.current": f"bg:{NAVY} {GRAY}",
    # /menu and /config overlays
    "menu.screen": f"bg:{BLACK}",
    "menu.bar": f"bg:{GRAY_DARK}",
    "menu.body": f"bg:{GRAY_DARK}",
    "menu.title": f"bold {CYAN}",
    "menu.frame": f"bg:{GRAY_DARK} bold {CYAN}",
    "menu.item": f"bg:{GRAY_DARK} {GRAY_BRIGHT}",
    "menu.item.current": f"bg:{NAVY} bold {GRAY_BRIGHT}",
    "menu.group": f"bg:{GRAY_DARK} bold {CYAN_DIM}",
    "menu.filter": f"bg:{GRAY_DARK} bold {CYAN}",
    "menu.hint": f"bg:{GRAY_DARK} {GRAY_DIM}",
}


def build_console(**kwargs) -> Console:
    """Create a Console bound to the LiteHarness theme."""
    return Console(theme=RICH_THEME, **kwargs)


# Shared console used across the CLI.
console = build_console()
