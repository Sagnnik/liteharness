import os
import difflib
from pathlib import Path
from langchain_core.tools import tool
import subprocess
import tempfile
from tools.common import validate_path, auto_format

@tool
def read_file(path:str, offset:int = 1, limit:int | None = None ) -> str:
    """Read a file. Offset/limit are 1-based line numbers"""
    try:
        path = validate_path(path)
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        start = max(0, offset - 1)
        end = start + limit if limit else None
        chunk = lines[start:end]

        numbered = [f"{i+start+1: 4d}| {line}" for i, line in enumerate(chunk)]
        return "\n".join(numbered) if numbered else "(empty file)"
    except Exception as e:
        return f"Error: {e}"

@tool
def write_file(path:str, content:str) -> str:
    """Write content to a file. Creates parent dirs."""
    try:
        path = validate_path(path)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        auto_format(path)
        return f"Wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"Error: {e}"

@tool
def edit_file(path:str, old_string:str, new_string:str, replace_all: bool = False) -> str:
    """Replace old_string with new_string. Uses exact match then fuzzy fallback."""
    try:
        path = validate_path(path)
        p = Path(path)
        content = p.read_text(encoding="utf-8")

        # Exact match logic
        if old_string in content:
            count = -1 if replace_all else 1
            new_content = content.replace(old_string, new_string, count)
            p.write_text(new_content, encoding="utf-8")
            auto_format(path)
            return f"Edited {path}"

        # Fuzzy match fallback 
        # Normalized to handle line-ending variations safely
        lines = content.splitlines()
        old_lines = old_string.splitlines()
        
        if not old_lines or len(old_lines) > len(lines):
            return f"No match found in {path} (old_string layout mismatch)."

        best_ratio, best_idx = 0.0, -1
        for i in range(len(lines) - len(old_lines) + 1):
            block = "\n".join(lines[i:i+len(old_lines)])
            ratio = difflib.SequenceMatcher(None, old_string, block).ratio()
            if ratio > best_ratio:
                best_ratio, best_idx = ratio, i

        if best_ratio > 0.75:
            target_block = "\n".join(lines[best_idx : best_idx + len(old_lines)])
            
            # Re-read or split content explicitly to execute safe replacement
            normalized_content = "\n".join(lines)
            new_normalized = normalized_content.replace(target_block, new_string, 1)
            
            p.write_text(new_normalized, encoding="utf-8")
            auto_format(path)
            return f"Fuzzy edit ({best_ratio:.2f}) on {path}"
            
        return f"No match in {path} (best match confidence: {best_ratio:.2f})"

    except Exception as e:
        return f"Error: {e}"

@tool
def multi_edit(path: str, edits: list[dict]) -> str:
    """Apply multiple edits sequentially to a single file."""
    try:
        path = validate_path(path)
        results = []
        
        for i, ed in enumerate(edits, 1):
            old = ed.get("old_string")
            new = ed.get("new_string")
            rep_all = ed.get("replace_all", False)
            
            if not old or not new:
                return f"Aborted. Edit {i} is missing 'old_string' or 'new_string'."
            
            # calling the underlying function via .fn()
            result_str = edit_file.fn(path, old_string=old, new_string=new, replace_all=rep_all)
            if "No match" in result_str or "Error" in result_str:
                return f"Batch aborted at Edit {i}: {result_str}"
                
            results.append(f"Edit {i}: {result_str}")
            
        return f"Successfully applied all {len(edits)} edits to {path}.\n" + "\n".join(results)
    except Exception as e:
        return f"Error during multi-edit execution: {e}"

@tool
def apply_patch(patch: str) -> str:
    """Apply unified diff patch. Uses git apply --3way if in repo."""
    patch_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
            f.write(patch)
            patch_path = f.name
            
        r = subprocess.run(["git", "apply", "--3way", patch_path], capture_output=True, text=True)
        if r.returncode == 0:
            return "Patch applied via git apply"
        return f"git apply failed: {r.stderr or r.stdout}"
    except Exception as e:
        return f"Error: {e}"
    finally:
        if patch_path and os.path.exists(patch_path):
            os.unlink(patch_path)

@tool
def glob_files(pattern:str) -> str:
    """Find files matching glob pattern under cwd."""
    try:
        matches = []
        for p in Path(".").glob(pattern):
            if p.is_file():
                matches.append(str(p))

        sorted_matches = sorted(matches)[:200]

        return "\n".join(sorted_matches) or "No matches"
    except Exception as e:
        return f"Error: {e}"


@tool
def list_files(path: str = ".") -> str:
    """List contents of a directory. Ignores heavy dependency build paths."""
    try:
        path = validate_path(path)
        # Ignore folders that could break LLM context windows
        ignored_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'build', 'dist'}
        
        entries = []
        with os.scandir(path) as it:
            for entry in it:
                if entry.name in ignored_dirs:
                    continue
                entries.append(entry.name + ("/" if entry.is_dir() else ""))
                
        sorted_entries = sorted(entries)[:300] # Cap output length safely
        output = "\n".join(sorted_entries)
        return output if output else "(empty directory)"
    except Exception as e:
        return f"Error: {e}"


@tool
def add_to_memory(text: str) -> str:
    """Append a note to .ness/NESS.md project memory."""
    from memory import append_memory
    return append_memory(text)
