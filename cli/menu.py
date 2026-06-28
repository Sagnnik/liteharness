"""Command catalog metadata and the transient /menu overlay.

The catalog is the single source of truth for which slash commands exist; it is
consumed by both the dispatcher (commands.py) and /help. The overlay is a
prompt_toolkit Application with arrow-key navigation and type-to-filter that
closes back into the normal scrollback.
"""

from __future__ import annotations

from dataclasses import dataclass

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.screen import Point
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame

from cli.theme import PTK_STYLE_RULES


@dataclass(frozen=True)
class CommandSpec:
    name: str
    summary: str
    group: str
    usage: str = ""


# Order matters for /help grouping; only kept commands are listed.
COMMAND_CATALOG: tuple[CommandSpec, ...] = (
    # general
    CommandSpec("menu", "Open this command menu", "General", "/menu"),
    CommandSpec("help", "Show the command reference", "General", "/help"),
    CommandSpec("config", "Set API keys, switch model, toggle options", "General", "/config"),
    CommandSpec("exit", "End the session", "General", "/exit"),
    # session
    CommandSpec("cost", "Show token and cost totals", "Session", "/cost"),
    CommandSpec("cache", "Show prompt-cache read/write stats", "Session", "/cache"),
    CommandSpec("threads", "List saved sessions", "Session", "/threads"),
    CommandSpec("resume", "Resume a saved thread", "Session", "/resume <id>"),
    CommandSpec("save", "Archive the current thread", "Session", "/save"),
    CommandSpec("reset", "Archive and start a fresh thread", "Session", "/reset"),
    CommandSpec("compact", "Force compaction on the next turn", "Session", "/compact"),
    # context & memory
    CommandSpec("skills", "List loaded skills and warnings", "Context", "/skills"),
    CommandSpec("skill", "Load a skill's full instructions next turn", "Context", "/skill [<name>]"),
    CommandSpec("init", "Generate .ness/NESS.md", "Context", "/init [force]"),
    CommandSpec("memory", "Read or append project memory", "Context", "/memory [add <note>]"),
    CommandSpec("user", "Read or append user preferences", "Context", "/user [add <note>]"),
    # tools & policy
    CommandSpec("permissions", "View or edit permission rules", "Tools", "/permissions"),
    CommandSpec("hooks", "List configured hooks", "Tools", "/hooks"),
    CommandSpec("mcp", "Show MCP server and tool status", "Tools", "/mcp"),
    # input
    CommandSpec("copy", "Copy assistant output", "Input", "/copy [code|<n>]"),
    CommandSpec("image", "Attach an image to the next prompt", "Input", "/image <path>"),
)

COMMAND_NAMES: tuple[str, ...] = tuple(spec.name for spec in COMMAND_CATALOG)
_BY_NAME: dict[str, CommandSpec] = {spec.name: spec for spec in COMMAND_CATALOG}

_GROUP_ORDER = ("General", "Session", "Context", "Tools", "Input")


def get_command(name: str) -> CommandSpec | None:
    return _BY_NAME.get(name)


def _sorted_specs() -> list[CommandSpec]:
    return sorted(
        COMMAND_CATALOG,
        key=lambda spec: (_GROUP_ORDER.index(spec.group) if spec.group in _GROUP_ORDER else 99, spec.name),
    )


def _matches(spec: CommandSpec, query: str) -> bool:
    if not query:
        return True
    q = query.lower()
    return q in spec.name.lower() or q in spec.summary.lower() or q in spec.group.lower()


async def open_menu() -> str | None:
    """Run the overlay; return the chosen command name or None if cancelled."""
    state = {"index": 0, "filter": ""}

    def visible() -> list[CommandSpec]:
        return [spec for spec in _sorted_specs() if _matches(spec, state["filter"])]

    def clamp() -> None:
        items = visible()
        if not items:
            state["index"] = 0
        else:
            state["index"] = max(0, min(state["index"], len(items) - 1))

    def list_fragments():
        items = visible()
        fragments = []
        if not items:
            fragments.append(("class:menu.hint", "  no matching commands"))
            return fragments
        last_group = None
        line = 0
        for i, spec in enumerate(items):
            if spec.group != last_group:
                fragments.append(("class:menu.group", f"  {spec.group}\n"))
                last_group = spec.group
                line += 1
            current = i == state["index"]
            style = "class:menu.item.current" if current else "class:menu.item"
            prefix = "› " if current else "  "
            text = f"{prefix}/{spec.name:<12} {spec.summary}"
            fragments.append((style, text + "\n"))
            line += 1
        return fragments

    def cursor_position() -> Point:
        # Keep the selected row in view by pointing the cursor at it.
        items = visible()
        if not items:
            return Point(0, 0)
        last_group = None
        line = 0
        for i, spec in enumerate(items):
            if spec.group != last_group:
                last_group = spec.group
                line += 1
            if i == state["index"]:
                return Point(0, line)
            line += 1
        return Point(0, 0)

    def filter_fragments():
        frags = [("class:menu.hint", "  filter: ")]
        frags.append(("class:menu.filter", state["filter"] or " "))
        return frags

    def hint_fragments():
        return [("class:menu.hint", "  ↑/↓ move   ⏎ select   esc cancel   type to filter")]

    body = Window(
        content=FormattedTextControl(list_fragments, focusable=True, get_cursor_position=cursor_position),
        always_hide_cursor=True,
        wrap_lines=False,
        height=Dimension(min=6, max=18),
        style="class:menu.body",
    )

    layout = Layout(
        HSplit(
            [
                Window(FormattedTextControl(filter_fragments), height=1, style="class:menu.bar"),
                Frame(body, title="menu", style="class:menu.frame"),
                Window(FormattedTextControl(hint_fragments), height=1, style="class:menu.bar"),
            ],
            style="class:menu.screen",
        )
    )

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("c-p")
    def _up(event) -> None:
        state["index"] -= 1
        clamp()

    @kb.add("down")
    @kb.add("c-n")
    def _down(event) -> None:
        state["index"] += 1
        clamp()

    @kb.add("enter")
    def _select(event) -> None:
        items = visible()
        if items:
            event.app.exit(result=items[state["index"]].name)
        else:
            event.app.exit(result=None)

    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event) -> None:
        event.app.exit(result=None)

    @kb.add("backspace")
    def _backspace(event) -> None:
        state["filter"] = state["filter"][:-1]
        state["index"] = 0
        clamp()

    @kb.add("<any>")
    def _typed(event) -> None:
        text = event.data
        if text and text.isprintable():
            state["filter"] += text
            state["index"] = 0
            clamp()

    app: Application = Application(
        layout=layout,
        key_bindings=kb,
        style=Style.from_dict(PTK_STYLE_RULES),
        full_screen=True,
        mouse_support=False,
    )
    return await app.run_async()
