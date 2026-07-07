"""LiteHarness CLI startup header dashboard.

Renders a concept-faithful header (gradient loop logo + LiteHarness title +
rounded dashboard panel + hints) as a list of ``TranscriptLine`` rows
that the prompt_toolkit transcript pane prints before the first prompt.

The module is pure: every helper takes its inputs as arguments and returns
``TranscriptLine`` rows, so it is fully testable without a live TUI. The live
``TranscriptMixin.append_header`` (cli/transcript.py) is the only caller and
computes ``project`` / ``addons_summary`` / ``version`` there.
"""

from __future__ import annotations

import math
from functools import lru_cache

from cli.models import TranscriptLine
from cli.theme import (
    BLUE,
    CYAN,
    GRAY_BRIGHT,
    GRAY_DIM,
    PURPLE,
    YELLOW,
)

# --- palette (mirrors assets/cli-header-concept.svg) ------------------------
_LOOP_STOPS = (CYAN, BLUE, PURPLE)  # cyan -> blue -> purple

# --- braille (2x4 dot matrix) -----------------------------------------------
# Braille base codepoint U+2800; each char encodes an 8-bit dot pattern:
#   bit 0  top-left      bit 4  mid-left
#   bit 1  top-mid-left  bit 5  bot-left
#   bit 2  top-right     bit 6  mid-right
#   bit 3  mid-right     bit 7  bot-right   (only right col actually differs)
# We only need the per-dot offset map below.
_BRAILLE_BASE = 0x2800
_BRAILLE_DOTS = (
    (0, 0),
    (0, 1),
    (0, 2),  # left column dots 1,2,3
    (1, 0),
    (1, 1),
    (1, 2),  # right column dots 4,5,6
    (0, 3),
    (1, 3),  # bottom row dots 7,8
)


def _interp_hex(stops: tuple[str, ...], t: float) -> str:
    """Return the interpolated hex color along ``stops`` at position ``t`` in [0,1]."""
    if t <= 0.0:
        return stops[0]
    if t >= 1.0:
        return stops[-1]
    seg = t * (len(stops) - 1)
    i = int(seg)
    frac = seg - i
    a, b = stops[i], stops[i + 1]
    ar, ag, ab = int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)
    br, bg, bb = int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)
    r = round(ar + (br - ar) * frac)
    g = round(ag + (bg - ag) * frac)
    bl = round(ab + (bb - ab) * frac)
    return f"#{r:02x}{g:02x}{bl:02x}"


def _char_style(hex_color: str, bold: bool = False) -> str:
    """Build a prompt_toolkit style string: 'fg:#xxxxxx bold'."""
    return f"fg:{hex_color}" + (" bold" if bold else "")


# --- braille loop logo ------------------------------------------------------
# Each braille cell covers a 2x4 dot matrix. The SVG concept (assets/cli-header-
# concept.svg) uses a gradient ring plus three inner chord lines and three nodes.
_GRID_COLS = 16
_GRID_ROWS = 6
_DOT_W = _GRID_COLS * 2
_DOT_H = _GRID_ROWS * 4
_CX = (_DOT_W - 1) / 2.0
_CY = (_DOT_H - 1) / 2.0
_RING_R = min(_CX, _CY) - 1.0  # SVG ring radius ~60 at center 88
_RING_STROKE = 2.5
_CHORD_GRAY = "#4b5563"


def _ring_color_for_angle(angle: float) -> str:
    """Map an angle to a gradient color (matches the SVG loop-gradient)."""
    t = math.sin(angle) * 0.5 + 0.5
    return _interp_hex(_LOOP_STOPS, t)


def _svg_node_positions(cx: float, cy: float, ring_r: float) -> dict[str, tuple[float, float]]:
    """Node centers from cli-header-concept.svg (ring radius 60, center 88,88)."""
    return {
        "top": (cx, cy - ring_r),
        "br": (cx + ring_r * 52 / 60, cy + ring_r * 30 / 60),
        "bl": (cx - ring_r * 52 / 60, cy + ring_r * 30 / 60),
    }


def _svg_chord_anchor_top(cx: float, cy: float, ring_r: float) -> tuple[float, float]:
    """Inner chord origin slightly below the top node (SVG y=35 vs node y=28)."""
    return (cx, cy - ring_r * 53 / 60)


def _canvas_set(canvas: dict[tuple[int, int], str], x: int, y: int, color: str) -> None:
    if 0 <= x < _DOT_W and 0 <= y < _DOT_H:
        canvas[(x, y)] = color


def _iter_line(x0: float, y0: float, x1: float, y1: float):
    """Yield integer pixel coordinates along a line (Bresenham)."""
    x0i, y0i = int(round(x0)), int(round(y0))
    x1i, y1i = int(round(x1)), int(round(y1))
    dx = abs(x1i - x0i)
    dy = -abs(y1i - y0i)
    sx = 1 if x0i < x1i else -1
    sy = 1 if y0i < y1i else -1
    err = dx + dy
    x, y = x0i, y0i
    while True:
        yield x, y
        if x == x1i and y == y1i:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _draw_line(
    canvas: dict[tuple[int, int], str],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str,
    *,
    dashed: bool = False,
) -> None:
    for i, (x, y) in enumerate(_iter_line(x0, y0, x1, y1)):
        if dashed and i % 4 >= 2:
            continue
        _canvas_set(canvas, x, y, color)


def _draw_disc(
    canvas: dict[tuple[int, int], str],
    cx: float,
    cy: float,
    radius: float,
    color: str,
) -> None:
    r2 = radius * radius
    r_max = int(radius + 0.75)
    for dy in range(-r_max, r_max + 1):
        for dx in range(-r_max, r_max + 1):
            if dx * dx + dy * dy <= r2:
                _canvas_set(canvas, int(round(cx + dx)), int(round(cy + dy)), color)


def _canvas_to_rows(canvas: dict[tuple[int, int], str]) -> list[TranscriptLine]:
    rows: list[TranscriptLine] = []
    for gy in range(_GRID_ROWS):
        fragments: list[tuple[str, str]] = []
        for gx in range(_GRID_COLS):
            char = 0
            color: str | None = None
            for bit, (dx, dy) in enumerate(_BRAILLE_DOTS):
                px = gx * 2 + dx
                py = gy * 4 + dy
                dot_color = canvas.get((px, py))
                if dot_color:
                    char |= 1 << bit
                    color = dot_color
            if char:
                fragments.append((_char_style(color or CYAN), chr(_BRAILLE_BASE + char)))
            else:
                fragments.append(("", " "))
        rows.append(
            TranscriptLine(
                style="",
                text="".join(t for _, t in fragments),
                fragments=fragments,
            )
        )
    return rows


@lru_cache(maxsize=1)
def _logo_rows() -> list[TranscriptLine]:
    """Pre-render the braille loop logo once (deterministic, <1ms)."""
    canvas: dict[tuple[int, int], str] = {}
    cx, cy = _CX, _CY
    r_out = _RING_R
    r_in = r_out - _RING_STROKE
    nodes = _svg_node_positions(cx, cy, r_out)
    top_anchor = _svg_chord_anchor_top(cx, cy, r_out)

    # Gradient ring (SVG active loop path).
    for py in range(_DOT_H):
        for px in range(_DOT_W):
            vx = px - cx
            vy = py - cy
            dist = math.hypot(vx, vy)
            if (r_in - 0.35) <= dist <= (r_out + 0.35):
                angle = math.atan2(vy, vx)
                _canvas_set(canvas, px, py, _ring_color_for_angle(angle))

    # Inner chord lines (SVG lines between the three nodes).
    _draw_line(canvas, *top_anchor, *nodes["br"], PURPLE)
    _draw_line(canvas, *top_anchor, *nodes["bl"], CYAN)
    _draw_line(canvas, *nodes["br"], *nodes["bl"], _CHORD_GRAY, dashed=True)

    # Node discs on top of ring/chords.
    _draw_disc(canvas, *nodes["top"], 1.6, CYAN)
    _draw_disc(canvas, *nodes["br"], 1.4, PURPLE)
    _draw_disc(canvas, *nodes["bl"], 1.4, YELLOW)
    return _canvas_to_rows(canvas)


def _logo_width() -> int:
    return _GRID_COLS  # chars wide


def _logo_height() -> int:  # noqa: D401 (tiny accessor)
    return _GRID_ROWS


def _gradientized_title(text: str) -> list[tuple[str, str]]:
    """Return fragments for a gradientized title string (cyan -> blue -> purple)."""
    if not text:
        return []
    out: list[tuple[str, str]] = []
    last = len(text) - 1
    for i, ch in enumerate(text):
        color = _interp_hex(_LOOP_STOPS, i / max(1, last))
        out.append((_char_style(color, bold=True), ch))
    return out


# --- rounded banner rows ----------------------------------------------------
# Rounded box drawing for a left-aligned body inside a fixed width.
# columns: total available width; margins: chars to indent the body from left/right.


def _panel_rows(
    *,
    row_specs: list[tuple[str, str, str, str]],
    width: int,
    inner_pad: int = 1,
) -> list[TranscriptLine]:
    """Build a rounded box wrapping rows of `[(label, value, label, value), ...]`.

    Each row becomes one body line inside the panel, with two cells separated
    by a vertical divider `│`. The label appears in dim gray, the value in
    bright gray.
    """
    box_style = "class:chrome.header.box"
    key_style = "class:chrome.header.key"
    val_style = "class:chrome.header.value"
    div_style = "class:chrome.header.box"

    inner_w = width - 2  # outer borders
    # column split: leave ~middle column for the divider
    left_half = (inner_w - 1) // 2
    right_half = inner_w - 1 - left_half

    lines: list[TranscriptLine] = []
    # top border
    lines.append(
        TranscriptLine(
            style=box_style,
            text="╭" + "─" * (inner_w) + "╮",
            fragments=[(box_style, "╭" + "─" * inner_w + "╮")],
        )
    )
    for left_label, left_val, right_label, right_val in row_specs:
        left_str = f"{left_label} {left_val}"
        right_str = f"{right_label} {right_val}"
        if len(left_str) > left_half - inner_pad * 2:
            left_str = left_str[: left_half - inner_pad * 2 - 1] + "…"
        if len(right_str) > right_half - inner_pad * 2:
            right_str = right_str[: right_half - inner_pad * 2 - 1] + "…"

        # build line: │ pad left pad │ pad right pad │
        gap_left = " " * (left_half - len(left_str) - inner_pad)
        gap_right = " " * (right_half - len(right_str) - inner_pad)
        text = (
            f"│{' ' * inner_pad}{left_str}{gap_left}"
            f"│{' ' * inner_pad}{right_str}{gap_right}│"
        )
        fragments: list[tuple[str, str]] = [
            (box_style, "│"),
            ("", " " * inner_pad),
            (key_style, f"{left_label} "),
            (val_style, left_val),
            ("", gap_left),
            (div_style, "│"),
            ("", " " * inner_pad),
            (key_style, f"{right_label} "),
            (val_style, right_val),
            ("", gap_right),
            (box_style, "│"),
        ]
        lines.append(TranscriptLine(style=box_style, text=text, fragments=fragments))
    # bottom border
    lines.append(
        TranscriptLine(
            style=box_style,
            text="╰" + "─" * inner_w + "╯",
            fragments=[(box_style, "╰" + "─" * inner_w + "╯")],
        )
    )
    return lines


def _title_line(*, version: str, width: int) -> TranscriptLine:
    """Build the Lite + Harness (gradient) + version row. Title bold."""
    left = "Lite"
    title_style = _char_style(GRAY_BRIGHT, bold=True)
    title_fragments: list[tuple[str, str]] = [(title_style, left)]
    title_fragments.extend(_gradientized_title("Harness"))
    # version appended with a dim style and one space gap
    ver_text = f" v{version}"
    title_fragments.append((_char_style(GRAY_DIM), ver_text))

    # right-pad with spaces to fill `width` so subsequent rows align
    used = len(left) + len("Harness") + len(ver_text)
    pad = max(0, width - used)
    title_fragments.append(("", " " * pad))

    text = "Lite" + "Harness" + ver_text + (" " * pad)
    return TranscriptLine(style="", text=text, fragments=title_fragments)


def _hints_line(*, width: int) -> TranscriptLine:
    """Build the hints row from the concept SVG."""
    key = "class:chrome.header.hint.key"
    dim = "class:chrome.header.hint"
    accent = "class:chrome.header.hint.accent"
    fragments: list[tuple[str, str]] = []
    fragments.append((dim, "Hints: "))
    fragments.append((key, "↑/↓"))
    fragments.append((dim, " select · "))
    fragments.append((key, "Enter"))
    fragments.append((dim, " run · "))
    fragments.append((key, "Tab"))
    fragments.append((dim, " complete · "))
    fragments.append((key, "Shift+Tab"))
    fragments.append((dim, " toggle "))
    fragments.append((key, "Act/Plan"))
    fragments.append((dim, "  "))
    fragments.append((accent, "/help"))
    fragments.append((dim, " · "))
    fragments.append((accent, "/config"))
    text = "".join(t for _, t in fragments)
    if len(text) > width:
        # truncate (rarely: very narrow terminals): drop the tail cleanly
        cut = max(8, width)
        text = text[:cut]
        fragments = [(dim, text)]
    if len(text) < width:
        fragments.append(("", " " * (width - len(text))))
        text += " " * (width - len(text))
    return TranscriptLine(style=dim, text=text, fragments=fragments)


def _mode_label(mode: str, approval: bool) -> str:
    mode = (mode or "").lower()
    if mode == "act" and approval:
        return "Act (auto-approval)"
    return mode.capitalize() or "Act"


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    if n <= 1:
        return "…"
    return s[: n - 1] + "…"


def header_lines(
    *,
    mode: str,
    model: str,
    approval: bool,
    project: str,
    addons_summary: str,
    version: str,
    width: int,
    show_logo: bool,
) -> list[TranscriptLine]:
    """Render the LiteHarness startup header.

    Returns ordered ``TranscriptLine`` rows (top to bottom), each exactly
    ``width`` columns wide. Empty list is returned when ``width < 40`` so
    very-narrow terminals do not get a garbled banner.
    """
    width = max(0, int(width))
    if width < 40:
        return []

    lines: list[TranscriptLine] = []

    if not show_logo:
        body_w = width
        title = _title_line(version=version, width=body_w)
        lines.append(title)
        lines.extend(
            _panel_rows(
                row_specs=[
                    (
                        "Session :",
                        model,
                        "Project :",
                        _truncate(project, max(20, body_w // 2 - 18)),
                    ),
                    (
                        "Mode    :",
                        _mode_label(mode, approval),
                        "Add-ons :",
                        _truncate(addons_summary, max(20, body_w // 2 - 18)),
                    ),
                ],
                width=body_w,
            )
        )
        lines.append(_hints_line(width=body_w))
        return lines

    # --- with logo: left cell = logo, right cell = stacked body ---
    logo = _logo_rows()
    logo_w = _logo_width()
    logo_h = _logo_height()
    body_w = max(40, width - logo_w - 2)  # 2-char gutter
    gutter = "  "

    # Build body rows (title + blank + panel 3 rows + blank + hints)
    body: list[TranscriptLine] = []
    body.append(_title_line(version=version, width=body_w))
    body.extend(
        _panel_rows(
            row_specs=[
                (
                    "Session :",
                    model,
                    "Project :",
                    _truncate(project, max(20, body_w // 2 - 18)),
                ),
                (
                    "Mode    :",
                    _mode_label(mode, approval),
                    "Add-ons :",
                    _truncate(addons_summary, max(20, body_w // 2 - 18)),
                ),
            ],
            width=body_w,
        )
    )
    body.append(_hints_line(width=body_w))

    # Total composed height = max(logo_h, len(body))
    rows_total = max(logo_h, len(body))
    for i in range(rows_total):
        # left cell
        if i < logo_h:
            l_line = logo[i]
        else:
            l_line = TranscriptLine("", " " * logo_w, fragments=[("", " " * logo_w)])
        # right cell
        if i < len(body):
            r_line = body[i]
        else:
            r_line = TranscriptLine("", " " * body_w, fragments=[("", " " * body_w)])
        # normalize fragments (TranscriptLine.fragments may be None) and pad to body_w
        r_frag = (
            list(r_line.fragments)
            if r_line.fragments is not None
            else [(r_line.style, r_line.text)]
        )
        r_text = r_line.text
        short = body_w - len(r_text)
        if short > 0:
            r_frag = r_frag + [("", " " * short)]
            r_text = r_text + " " * short
        l_frag = (
            list(l_line.fragments)
            if l_line.fragments is not None
            else [(l_line.style, l_line.text)]
        )
        fragments: list[tuple[str, str]] = l_frag + [("", gutter)] + r_frag
        text = l_line.text + gutter + r_text
        lines.append(TranscriptLine(style="", text=text, fragments=fragments))

    return lines
