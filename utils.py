import subprocess
import os
from pathlib import Path
from langchain_core.messages import SystemMessage, ToolMessage
import difflib
from config import settings
from permissions import check
from tools import DESTRUCTIVE_TOOLS

def get_project_context(max_files: int = 60) -> str:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            timeout=10,
        )
        files = result.stdout.strip().split("\n") if result.returncode == 0 else []

    except Exception:
        files = []

    if not files:
        ignore = {
            ".git", "node_modules", "__pycache__", ".venv", "venv",
            "dist", "build", ".next", ".turbo", ".idea", ".vscode",
        }
        files = []
        for root, dirs, filenames in os.walk(os.getcwd()):
            dirs[:] = [d for d in dirs if d not in ignore]
            for fname in filenames:
                fp = os.path.join(root, fname)
                if any(part.startswith(".") for part in Path(fp).parts):
                    continue
                files.append(fp)
                if len(files) >= max_files:
                    break
            if len(files) >= max_files:
                break

    tree = {}
    for f in files:
        parts = Path(f).parts
        current = tree
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current.setdefault("__files__", []).append(parts[-1])

    def render(d, prefix=""):
        lines = []
        items = sorted(d.items(), key=lambda x: (x[0] == "__files__", x[0]))
        for k, v in items:
            if k == "__files__":
                for fname in sorted(v):
                    lines.append(f"{prefix}├──{fname}")

            else:
                lines.append(f"{prefix}├── {k}/")
                lines.extend(render(v, prefix + "│   "))
        return lines

    tree_str = "\n".join(render(tree))

    key_files = [
        "package.json", "requirements.txt", "pyproject.toml",
        "Cargo.toml", "go.mod", "README.md", "tsconfig.json",
    ]

    summaries = []
    for kf in key_files:
        if os.path.exists(kf):
            try:
                with open(kf, "r", encoding="utf-8") as fh:
                    content = fh.read(800)
                summaries.append(f"--- {kf} ---\n{content}\n")
            except Exception:
                pass

    return f"Project structure (top {max_files} files):\n{tree_str}\n\n{''.join(summaries)}"



async def trim_messages_smart(messages, max_chars: int, summarize_fn=None):
    total = sum(len(str(m.content)) for m in messages)
    if total <= max_chars:
        return messages

    system = [m for m in messages if m.type == "system"]
    rest = [m for m in messages if m.type != "system"]

    # Keep last 8 + tool msgs from last 4 assistant turns
    keep_tail = rest[-8:]
    tool_recent = [m for m in rest if isinstance(m, ToolMessage)][-16:]
    kept_ids = {id(m) for m in keep_tail + tool_recent}
    dropped = [m for m in rest if id(m) not in kept_ids]

    if dropped and summarize_fn:
        summary = await summarize_fn(dropped)
        system = system + [SystemMessage(content=f"Summary of earlier work:\n{summary}")]

    result = system + [m for m in rest if id(m) in kept_ids or m in keep_tail]
    # dedupe preserve order
    seen, out = set(), []
    for m in result:
        if id(m) not in seen:
            seen.add(id(m))
            out.append(m)
    return out


async def needs_plan(user_input:str, model) -> list[str] | None:
    from langchain_core.messages import HumanMessage
    from prompt import PLAN_PROMPT

    resp = await model.ainvoke([HumanMessage(content=PLAN_PROMPT.format(user_input=user_input))])
    text = (resp.content or "").strip()
    if text.upper().startswith("NO_PLAN_NEEDED"):
        return None

    steps = []
    for line in text.splitlines():
        line = line.strip()
        if line and line[0].isdigit():
            steps.append(line.split(".", 1)[-1].strip())
    return steps or None

def _needs_approval(tool: str, args: dict) -> bool:
    if not settings.enable_approval:
        return False
    perm = check(tool, args)
    if perm == "allow":
        return False
    if perm == "deny":
        return True  # handled as auto-deny in tools_node
    return tool in DESTRUCTIVE_TOOLS


def _preview_diff(tool: str, args: dict) -> str:
    path = args.get("path", "")
    if tool == "write_file":
        old = Path(path).read_text(encoding="utf-8") if os.path.exists(path) else ""
        new = args.get("content", "")
    elif tool in ("edit_file", "multi_edit", "apply_diff"):
        old = Path(path).read_text(encoding="utf-8") if os.path.exists(path) else ""
        new = old.replace(args.get("old_string", ""), args.get("new_string", ""), 1)
    else:
        return f"{tool}({args})"
    return "\n".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}",
    ))