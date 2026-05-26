import os
from pathlib import Path
import shutil
import subprocess

ALLOWED_PREFIX = Path(os.getcwd()).resolve()

def validate_path(path: str) -> str:
    """
    Safely validates that a path stays strictly inside the project root directory.
    Prevents path traversal and partial-name directory bypasses.
    """
    try:
        abs_path = Path(path).resolve()
        # is_relative_to() ensures the path is structurally nested under ALLOWED_PREFIX
        if not abs_path.is_relative_to(ALLOWED_PREFIX):
            raise PermissionError(f"Access Denied: {path} is outside of the project root")

        return str(abs_path)

    except Exception as e:
        raise ValueError(f"Invalid path: {path} ({e})")

def auto_format(path: str):
    """Formats code files using local ecosystem tooling safely."""
    p = Path(path)
    ext = p.suffix
    formatters = {
        ".py": ["python", "-m", "black", "--", str(p)],
        ".ts": ["npx", "--", "prettier", "--write", str(p)],
        ".tsx": ["npx", "--", "prettier", "--write", str(p)],
        ".js": ["npx", "--", "prettier", "--write", str(p)],
        ".json": ["npx", "--", "prettier", "--write", str(p)],
        ".md": ["npx", "--", "prettier", "--write", str(p)],
    }
    cmd = formatters.get(ext)
    if cmd and shutil.which(cmd[0]):
        try:
            subprocess.run(cmd, capture_output=True, timeout=15)
        except Exception:
            pass