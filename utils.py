import subprocess
import os
from pathlib import Path

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

# TODO: Need to check this! and figure out summarizing as well.

def trim_messages(messages, max_chars: int = 120_000):
    """Rough token trimming by character count"""
    total = sum(len(str(m.content)) for m in messages)
    if total <= max_chars:
        return messages

    system = [m for m in messages if getattr(m, "type", None) == "system"]
    rest = [m for m in messages if getattr(m, "type", None) != "system"]

    while rest and sum(len(str(m.content)) for m in system + rest) > max_chars:
        if len(rest) > 4:
            rest.pop(0)
        else:
            break

    return system + rest

def is_complex_request(text: str) -> bool:
    """Heuristic: does this request likely need multi-file planning?"""
    triggers = [
        "app", "project", "application", "website", "full",
        "setup", "scaffold", "create a", "build a", "implement",
        "multiple", "several files", "add feature", "refactor",
    ]
    t = text.lower()
    return any(tr in t for tr in triggers)