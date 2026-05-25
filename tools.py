import pkgutil
from langchain_core.messages import content
from langchain_core.tools import tool
import subprocess
import os
from pathlib import Path
import shutil
import difflib

# Auto-formatting helper

def _auto_format(path:str):
    """Run formatter based on file extension if avilable."""
    ext = Path(path).suffix
    formatters = {
        ".ts": ["npx", "prettier", "--write", path],
        ".tsx": ["npx", "prettier", "--write", path],
        ".js": ["npx", "prettier", "--write", path],
        ".jsx": ["npx", "prettier", "--write", path],
        ".json": ["npx", "prettier", "--write", path],
        ".css": ["npx", "prettier", "--write", path],
        ".py": ["python", "-m", "black", path],
        ".md": ["npx", "prettier", "--write", path],
    }
    if ext in formatters:
        cmd = formatters[ext]
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, capture_output=True, timeout=15)
            except Exception:
                pass

# Sandboxing

ALLOWED_PREFIX = os.path.abspath(os.getcwd())

def _validate_path(path:str) -> str:
    """Ensure that the path always stays within the current working directory"""
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(ALLOWED_PREFIX):
        raise PermissionError(f"Access Denied: {path} is outside of the project root")

    return abs_path

# File tools

@tool
def read_file(path: str) -> str:
    """Read a file from the current working directory. Returns content or error."""
    try:
        path = _validate_path(path)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {path}: {e}"

@tool
def write_file(path: str, content: str) -> str:
    """Write Content to a file. Creates parent directories if needed. Auto-formats after write."""
    try:
        path = _validate_path(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        # Auto-format hook
        _auto_format(path)
        return f"Successfully wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Error writing file: {e}"

@tool
def apply_diff(path: str, old_string: str, new_string: str) -> str:
    """Replace old_string with new_string. Uses exact match then fuzzy fallback."""
    try:
        path = _validate_path(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        if old_string in content:
            new_content = content.replace(old_string, new_string, 1)
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            _auto_format(path)
            return f"Exact diff applied to {path}"

        file_lines = content.splitlines()
        old_lines = old_string.splitlines()
        best_ratio = 0.0
        best_idx = -1

        for i in range(len(file_lines) - len(old_lines) + 1):
            block = "\n".join(file_lines[i:i+len(old_lines)])
            ratio = difflib.SequenceMatcher(None, old_string, block).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = i

        if best_ratio > 0.75:
            block = "\n".join(file_lines[best_idx : best_idx + len(old_lines)])
            new_content = content.replace(block, new_string, 1)
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            _auto_format(path)
            return f"Fuzzy diff applied (similarity {best_ratio:.2f}) to {path}"

        return f"Error: Could not find match in {path}. Best similarity: {best_ratio:.2f}"
    except Exception as e:
        return f"Error applying diff: {e}"

@tool
def list_files(path: str = ".") -> list[str]:
    """List files and directories in the given path"""
    try:
        path = _validate_path(path)
        return "\n".join(sorted(os.listdir(path)))
    except Exception as e:
        return f"Error listing files: {e}"

@tool
def get_project_context() -> str:
    """Get the project context"""
    from utils import get_project_context
    return get_project_context()

@tool
def search_files(path: str, query: str) -> list[str]:
    """Search for query string in files under path"""
    matches = []
    ignore = {".git", "node_modules", "__pycache__", ".venv", "venv"}
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in ignore]
        for fname in files:
            if not fname.endswith((".py", ".tsx", ".ts", ".jsx", ".js", ".json", ".md", ".css", ".html")):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                if query in content:
                    lines = content.splitlines()
                    for i, line in enumerate(lines):
                        if query in line:
                            matches.append(f"{fpath}:{i+1}: {line.strip()}")
            
            except Exception:
                pass
            if len(matches) >= 20:
                break

        if len(matches) >= 20:
            break

    return "\n".join(matches) if matches else f"No matches found for {query} in {path}"

# Git tools

@tool
def git_snapshot(message: str = "agent: auto-save") -> str:
    """Commit all current changes as a snapshot before editing"""
    try:
        # check if it is a git repo
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return "Not a git repository. Skipping snapshot."

        subprocess.run(["git", "add", "-A"], capture_output=True, timeout=10)
        result = subprocess.run(
            ["git", "commit", "-m", message, "--no-verify"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return f"Git snapshot: {message}"
        # Nothing to commit is ok
        if "nothing to commit" in result.stdout.lower() or "nothing to commit" in result.stderr.lower():
            return "No changes to snapshot."
        return f"Git snapshot issue: {result.stderr}"
    except Exception as e:
        return f"Git snapshot error: {e}"

@tool
def git_diff() -> str:
    """Show unstaged changes since last snapshot"""
    try:
        result = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return f"Git diff error: {result.stderr}"

        return result.stdout or "No changes"
    except Exception as e:
        return f"git diff error: {e}"

# Test Runner

@tool
def run_tests(test_path: str = "") -> str:
    """Run the project test suite. Returns output pass/fail status"""
    try:
        # Detect test command
        cmd = None
        if os.path.exists("package.json"):
            with open("package.json", "r", encoding="utf-8") as f:
                import json
                pkg = json.load(f)

            scripts = pkg.get("scripts", {})
            for key in ["test", "test:unit", "jest", "vitest"]:
                if key in scripts:
                    cmd = f"npm run {key}"
                    break

            if not cmd:
                cmd = "npm test"

        elif os.path.exists("pyproject.toml") or os.path.exists("setup.py"):
            cmd = "python -m pytest"
        elif os.path.exists("requirements.txt"):
            cmd = "python -m pytest"
        elif os.path.exists("Cargo.toml"):
            cmd = "cargo test"
        elif os.path.exists("go.mod"):
            cmd = "go test ./..."

        if not cmd:
            return "Could not detect test command. Add tests or configure manually."

        if test_path:
            cmd += f" {test_path}"

        result = subprocess.run(
            cmd.split(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        status = "PASS" if result.returncode == 0 else "FAIL"
        output = result.stdout
        if result.stderr:
            output += "\n--- STDERR ---\n" + result.stderr
        return f"Tests {status} (exit {result.returncode})\n{output[:2000]}"
    except subprocess.TimeoutExpired:
        return "Tests timed out after 120s."
    except Exception as e:
        return f"Test runner error: {e}"