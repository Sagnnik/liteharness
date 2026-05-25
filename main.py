import asyncio
import os
import json
from datetime import datetime
from typing import List

from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph.state import Runnable
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich import box

from config import settings, cost_tracker
from agent import build_graph
from utils import is_complex_request
from prompt import PLAN_PROMPT, STEP_PROMPT

console = Console()

async def generate_plan(user_input: str, model: ChatOpenRouter) -> List[str] | None:
    """Ask the model to generate a step-by-step plan for the user's request"""
    prompt = PLAN_PROMPT.format(user_input=user_input)
    response = await model.ainvoke([HumanMessage(content=prompt)])
    text = response.content.strip()
    if text.upper().startswith("NO_PLAN_NEEDED"):
        return None

    steps = []
    for line in text.split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            step = line.split(".", 1)[1].strip() if "." in line else line
            steps.append(step)

    return steps if steps else None


async def run_turn(app: Runnable, user_input:str, thread_id:str):
    """Run a single turn of the conversation"""
    seen = 0
    async for event in app.astream(
        {"messages": [HumanMessage(content=user_input)]},
        config = {"configurable": {"thread_id": thread_id}},
        stream_mode="values",
    ):
        msgs = event.get("messages", [])
        if not msgs:
            continue

        new_msgs = msgs[seen:]
        seen = len(msgs)

        for msg in new_msgs:
            if msg.type in ("ai", "assistant"):
                tcs = getattr(msg, "tool_calls", None)
                if tcs:
                    for tc in tcs:
                        name = tc.get("name", "unknown")
                        args = tc.get("args", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)

                            except Exception:
                                pass

                        # Pretty tool call
                        tool_text = Text()
                        tool_text.append("🔧 ", style="bold yellow")
                        tool_text.append(f"{name}", style="bold cyan")
                        tool_text.append("(", style="dim")
                        arg_str = json.dumps(args, indent=None)[:120]
                        tool_text.append(arg_str, style="dim")
                        tool_text.append(")", style="dim")
                        console.print(tool_text)

                else:
                    content = msg.content
                    if content:
                        console.print()
                        # Typewriter effect
                        typed = Text()
                        typed.append("📝 ", style="bold green")
                        console.print(typed, end="")
                        for ch in content:
                            console.print(ch, end="", style="white")
                            await asyncio.sleep(0.0003)
                        console.print()

            elif msg.type == "tool":
                preview = msg.content.replace("\n", " ")[:220]
                tool_res = Text()
                tool_res.append("  ✅ ", style="bold green")
                tool_res.append(preview, style="dim")
                if len(msg.content) > 220:
                    tool_res.append("...", style="dim italic")
                console.print(tool_res)

async def main():
    # TODO: Add a cool logo here
    # Header
    header = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    header.add_column(style="bold cyan", justify="center")
    header.add_row("LiteHarness")
    header.add_row(f"Mode: [bold]{settings.mode}[/bold] | Model: [bold]{settings.model_name}[/bold]")
    header.add_row(
        f"Approval: [bold]{'ON' if settings.enable_approval else 'OFF'}[/bold] | "
        f"Planning: [bold]{'ON' if settings.enable_planning else 'OFF'}[/bold]"
    )
    console.print(Panel(header, border_style="bright_blue", title="Agent", subtitle="v2.0"))

    # Commands Table
    cmd_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    cmd_table.add_column("Command", style="cyan")
    cmd_table.add_column("Action", style="white")
    cmd_table.add_row("exit / quit", "Leave the session")
    cmd_table.add_row("/reset", "Clear conversation history")
    cmd_table.add_row("/context", "Show project structure")
    cmd_table.add_row("/cost", "Show session cost")
    cmd_table.add_row("/skills", "List loaded skills")
    console.print(cmd_table)
    console.print()

    model = ChatOpenRouter(
        model=settings.model_name,
        api_key=settings.openai_api_key,
    )

    app = build_graph(model)

    thread_id = "session-1"

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        # Empty input
        if not user_input.strip():
            continue
        
        # Commands
        if user_input.startswith("/"):
            
            cmd = user_input[1:].lower()

            # Break conditions and empty input
            if cmd in ("exit", "quit"):
                break
            
            # Clear History
            elif cmd == "reset":
                thread_id = f"session-{int(asyncio.get_event_loop().time())}"
                console.print("   [yellow] History Cleared[/yellow]\n")
                continue
            
            # Get Project Context
            elif cmd == "context":
                from utils import get_project_context
                ctx = get_project_context()
                console.print(Panel(Syntax(ctx, "text", theme="monokai", line_numbers=False), title="Project Context", border_style="green"))
                continue

            # Get Session Cost
            elif cmd == "cost":
                console.print(Panel(cost_tracker.report(), title="Session Cost", border_style="yellow"))
                continue

            # List Loaded Skills
            elif cmd == "skills":
                from skill_loader import load_skills
                skills = load_skills()
                if not skills:
                    console.print("[dim]No skills loaded.[/dim]")
                else:
                    skill_table = Table(box=box.SIMPLE, show_header=True)
                    skill_table.add_column("Skill", style="cyan")
                    skill_table.add_column("Triggers", style="white")
                    for name, skill in skills.items():
                        triggers = ", ".join(skill.get("triggers", []))
                        skill_table.add_row(name, triggers)
                    console.print(skill_table)
                continue
            else:
                console.print(f"Unknown command: [bold red]{cmd}[/bold red]")
                continue

        # Plan and Execute
        if settings.enable_planning and is_complex_request(user_input):
            console.print("[dim italic] Analyzing request for planning...[/dim italic]")
            plan = await generate_plan(user_input, model)

            if plan:

                # Display the plan in a table
                plan_table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
                plan_table.add_column("#", style="cyan", justify="right")
                plan_table.add_column("Step", style="white")
                for i, step in enumerate(plan, 1):
                    plan_table.add_row(str(i), step)

                console.print(Panel(plan_table, title="Plan", border_style="magenta"))
                console.print()

                # Execute the plan step by step
                for i, step in enumerate(plan, 1):
                    console.print(Rule(f"[bold cyan]Step {i}/{len(plan)}[/bold cyan]", style="cyan"))
                    step_prompt = STEP_PROMPT.format(step=step)
                    await run_turn(app, step_prompt, thread_id)

                console.print(Rule("[bold green]Final Summary[/bold green]", style="green"))
                await run_turn(app, "Summarize what was accomplished in this session.", thread_id)
                continue

        # Normal Single Turn Execution
        console.print("[dim italic]🤔 Agent thinking...[/dim italic]\n")
        await run_turn(app, user_input, thread_id)

    # Footer
    console.print()
    console.print(Rule(style="bright_blue"))
    cost_panel = Panel(
        cost_tracker.report(),
        title="Session Summary",
        border_style="bright_blue"
    )
    console.print(cost_panel)
    console.print("[bold green]Goodbye![/bold green]")

if __name__ == "__main__":
    asyncio.run(main())
