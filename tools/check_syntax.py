from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

import permissions
from permissions import validate_path

Checker = dict[str, Any]

CHECKERS: dict[str, Checker] = {
    "python": {
        "exts": {".py"},
        "file": [
            ["ruff", "check", "--output-format", "concise", "{path}"],
            ["python", "-m", "py_compile", "{path}"],
        ],
        "project": [
            ["ruff", "check", "--output-format", "concise", "."],
            ["python", "-m", "compileall", "-q", "."],
        ],
        "markers": ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile"],
    },
    "javascript": {
        "exts": {".js", ".mjs", ".cjs"},
        "file": [["node", "--check", "{path}"]],
        "project": None,
        "markers": ["package.json"],
    },
    "typescript": {
        "exts": {".ts", ".tsx", ".jsx"},
        "file": None,
        "project": [["tsc", "--noEmit"]],
        "markers": ["tsconfig.json", "package.json"],
    },
    "c": {
        "exts": {".c", ".h"},
        "file": [
            ["clang", "-fsyntax-only", "{path}"],
            ["gcc", "-fsyntax-only", "{path}"],
        ],
        "project": None,
        "markers": ["Makefile", "CMakeLists.txt"],
    },
    "cpp": {
        "exts": {".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".hh"},
        "file": [
            ["clang++", "-fsyntax-only", "-std=c++17", "{path}"],
            ["g++", "-fsyntax-only", "-std=c++17", "{path}"],
        ],
        "project": None,
        "markers": ["Makefile", "CMakeLists.txt"],
    },
    "cuda": {
        "exts": {".cu", ".cuh"},
        "file": [["nvcc", "-c", "-o", "{tmpfile}", "{path}"]],
        "project": None,
        "markers": ["Makefile"],
    },
    "rust": {
        "exts": {".rs"},
        "file": None,
        "project": [
            ["cargo", "check", "--message-format=short"],
            ["cargo", "check"],
        ],
        "markers": ["Cargo.toml"],
    },
    "go": {
        "exts": {".go"},
        "file": [["gofmt", "-e", "{path}"]],
        "project": [["go", "vet", "./..."]],
        "markers": ["go.mod"],
    },
    "elixir": {
        "exts": {".ex", ".exs"},
        "file": None,
        "project": [["mix", "compile", "--warnings-as-errors"]],
        "markers": ["mix.exs"],
    },
    "php": {
        "exts": {".php"},
        "file": [["php", "-l", "{path}"]],
        "project": None,
        "markers": ["composer.json"],
    },
    "ruby": {
        "exts": {".rb"},
        "file": [["ruby", "-c", "{path}"]],
        "project": None,
        "markers": ["Gemfile"],
    },
    "shell": {
        "exts": {".sh", ".bash", ".zsh"},
        "file": [
            ["bash", "-n", "{path}"],
            ["zsh", "-n", "{path}"],
        ],
        "project": None,
        "markers": [],
    },
    "lua": {
        "exts": {".lua"},
        "file": [["luac", "-p", "{path}"]],
        "project": None,
        "markers": [],
    },
    "zig": {
        "exts": {".zig"},
        "file": [["zig", "fmt", "--check", "{path}"]],
        "project": [["zig", "build"]],
        "markers": ["build.zig"],
    },
    "swift": {
        "exts": {".swift"},
        "file": [["swift", "-typecheck", "{path}"]],
        "project": None,
        "markers": ["Package.swift"],
    },
    "java": {
        "exts": {".java"},
        "file": [["javac", "-d", "{tmpdir}", "{path}"]],
        "project": None,
        "markers": ["pom.xml", "build.gradle", "build.gradle.kts"],
    },
}

_ALIASES: dict[str, str] = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "jsx": "typescript",
    "tsx": "typescript",
    "cxx": "cpp",
    "c++": "cpp",
    "hpp": "cpp",
    "bash": "shell",
    "zsh": "shell",
    "sh": "shell",
    "ex": "elixir",
    "exs": "elixir",
}

_ERROR_PATTERNS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+)?:?\s*(?P<msg>.*)$"),
    "c": re.compile(r"^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s*(?P<severity>error|warning|note):\s*(?P<msg>.*)$"),
    "cpp": re.compile(r"^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s*(?P<severity>error|warning|note):\s*(?P<msg>.*)$"),
    "cuda": re.compile(r"^(?P<file>[^\s(]+)\((?P<line>\d+)\):\s*(?P<severity>error|warning)\s*:\s*(?P<msg>.*)$"),
    "rust": re.compile(r"^(?:(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s*)?(?P<severity>error|warning)(?:\[(?P<code>E\d+)\])?:\s*(?P<msg>.*)$"),
    "go": re.compile(r"^(?P<file>[^:]+):(?P<line>\d+):\d+:\s*(?P<msg>.*)$"),
    "php": re.compile(r"^(?P<severity>Parse error|Fatal error|Warning):\s*(?P<msg>.*?)\s+in\s+(?P<file>\S+)\s+on\s+line\s+(?P<line>\d+)$", re.IGNORECASE),
    "ruby": re.compile(r"^(?P<file>[^:]+):(?P<line>\d+):\s*(?P<msg>.*)$"),
    "shell": re.compile(r"^(?P<file>[^:]+): line (?P<line>\d+):\s*(?P<msg>.*)$"),
    "lua": re.compile(r"^(?P<file>[^:]+):(?P<line>\d+):\s*(?P<msg>.*)$"),
    "swift": re.compile(r"^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s*(?P<severity>error|warning|note):\s*(?P<msg>.*)$"),
    "java": re.compile(r"^(?P<file>[^:]+):(?P<line>\d+):\s*(?P<severity>error|warning):\s*(?P<msg>.*)$"),
}

_PYTHON_FILE_LINE = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+)')


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2)


def _normalize_language(hint: str) -> str | None:
    key = hint.strip().lower().lstrip(".")
    if not key:
        return None
    if key in CHECKERS:
        return key
    return _ALIASES.get(key)


def _detect_file_language(path: Path) -> str | None:
    suffix = path.suffix.lower()
    for lang, cfg in CHECKERS.items():
        if suffix in cfg["exts"]:
            return lang
    return None


def _detect_directory_languages(path: Path) -> list[str]:
    languages: set[str] = set()
    for lang, cfg in CHECKERS.items():
        if not cfg.get("project"):
            continue
        markers = cfg.get("markers") or []
        if markers and _find_project_root(path, markers):
            languages.add(lang)
    return sorted(languages)


def _detect_language(path: Path, hint: str) -> tuple[str | None, str | None]:
    if hint:
        lang = _normalize_language(hint)
        if lang:
            return lang, None
        return None, f"No checker available for language '{hint}'"

    if path.is_file():
        lang = _detect_file_language(path)
        if lang:
            return lang, None
        return None, f"No checker available for extension '{path.suffix}'"

    languages = _detect_directory_languages(path)
    if len(languages) == 1:
        return languages[0], None
    if len(languages) > 1:
        return None, f"Multiple checkers match this directory: {', '.join(languages)}. Pass language explicitly."
    return None, "No checker available for this directory. Pass language explicitly."


def _parents_to_root(start: Path) -> Iterator[Path]:
    root = permissions.PROJECT_ROOT.resolve()
    current = start.resolve()
    if not current.is_relative_to(root):
        return

    while True:
        yield current
        if current == root:
            break
        current = current.parent


def _find_project_root(start: Path, markers: list[str]) -> Path | None:
    if not markers:
        return None
    for parent in _parents_to_root(start):
        if any((parent / marker).exists() for marker in markers):
            return parent
    return None


def _resolve_js_bin(name: str, cwd: Path) -> str | None:
    for parent in _parents_to_root(cwd):
        candidate = parent / "node_modules" / ".bin" / name
        if candidate.is_file():
            return str(candidate)
    return shutil.which(name)


def _command_available(template: list[str], cwd: Path) -> bool:
    if not template:
        return False
    if template[0] == "tsc":
        return _resolve_js_bin("tsc", cwd) is not None
    return shutil.which(template[0]) is not None


def _resolve_cmd(template: list[str], path: Path, cwd: Path, temp_dir: str | None) -> list[str]:
    temp_file = str(Path(temp_dir) / "syntax-check.out") if temp_dir else ""
    out: list[str] = []
    for idx, part in enumerate(template):
        if idx == 0 and part == "tsc":
            resolved = _resolve_js_bin("tsc", cwd)
            out.append(resolved or part)
        elif part == "{path}":
            out.append(str(path))
        elif part == "{tmpdir}":
            out.append(temp_dir or "")
        elif part == "{tmpfile}":
            out.append(temp_file)
        else:
            out.append(part)
    return out


def _needs_temp_dir(template: list[str]) -> bool:
    return any(part in {"{tmpdir}", "{tmpfile}"} for part in template)


def _select_template(
    templates: list[list[str]] | None,
    cwd: Path,
    target: Path | None = None,
    language: str = "",
) -> list[str] | None:
    for template in templates or []:
        if language == "shell" and target is not None:
            suffix = target.suffix.lower()
            if suffix == ".zsh" and template[0] != "zsh":
                continue
            if suffix in {".sh", ".bash"} and template[0] != "bash":
                continue
        if _command_available(template, cwd):
            return template
    return None


def _clamp_timeout(value: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = 60
    return max(5, min(parsed, 120))


def _run(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"


def _run_template(template: list[str], path: Path, cwd: Path, timeout: int) -> tuple[list[str], int, str, str]:
    temp_ctx = tempfile.TemporaryDirectory(dir="/tmp") if _needs_temp_dir(template) else None
    try:
        temp_dir = temp_ctx.name if temp_ctx else None
        cmd = _resolve_cmd(template, path, cwd, temp_dir)
        returncode, stdout, stderr = _run(cmd, cwd, timeout)
        return cmd, returncode, stdout, stderr
    finally:
        if temp_ctx:
            temp_ctx.cleanup()


def _repo_relative(path: str, cwd: Path) -> str:
    if not path:
        return ""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    try:
        resolved = candidate.resolve()
        if resolved.is_relative_to(permissions.PROJECT_ROOT.resolve()):
            return str(resolved.relative_to(permissions.PROJECT_ROOT.resolve()))
    except Exception:
        pass
    return path


def _error_from_match(match: re.Match[str], line: str, cwd: Path) -> dict[str, Any]:
    groups = match.groupdict()
    return {
        "file": _repo_relative(groups.get("file", ""), cwd),
        "line": int(groups["line"]) if groups.get("line") and groups["line"].isdigit() else 0,
        "col": int(groups["col"]) if groups.get("col") and groups["col"].isdigit() else 0,
        "severity": groups.get("severity", "error"),
        "message": groups.get("msg", line).strip(),
    }


def _parse_python_tracebacks(raw: str, cwd: Path) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    pending: tuple[str, int] | None = None
    for line in raw.splitlines():
        file_line = _PYTHON_FILE_LINE.search(line)
        if file_line:
            pending = (file_line.group("file"), int(file_line.group("line")))
            continue
        stripped = line.strip()
        if pending and re.match(r"^(SyntaxError|IndentationError|TabError):", stripped):
            filename, line_no = pending
            errors.append(
                {
                    "file": _repo_relative(filename, cwd),
                    "line": line_no,
                    "col": 0,
                    "severity": "error",
                    "message": stripped,
                }
            )
            pending = None
    return errors


def _parse_errors(raw: str, language: str, cwd: Path) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    pattern = _ERROR_PATTERNS.get(language)
    if pattern:
        for line in raw.splitlines():
            match = pattern.match(line.strip())
            if match:
                errors.append(_error_from_match(match, line, cwd))
    if language == "python":
        errors.extend(_parse_python_tracebacks(raw, cwd))
    return errors


@tool
def check_syntax(path: str = ".", language: str = "", timeout: int = 60) -> str:
    """
    Check syntax and lightweight compile/lint errors for supported files or projects.

    Supported languages include Python, JavaScript, TypeScript/JSX/TSX, C/C++,
    CUDA, Rust, Go, Elixir, PHP, Ruby, shell, Lua, Zig, Swift, and Java.

    Files use fast file-level checks when available; directories use project-level
    checks selected by language hints or project markers. Examples: Python uses
    ruff when installed, otherwise py_compile/compileall; JavaScript uses
    node --check; TypeScript uses tsc --noEmit; Rust uses cargo check.

    Commands come from a fixed internal registry, run without a shell, validate
    paths inside the project root, and are bounded by timeout.
    """
    try:
        target = Path(validate_path(path))
        if not target.exists():
            return _json(
                {
                    "ok": False,
                    "language": None,
                    "scope": None,
                    "error": f"{path} does not exist",
                    "command": None,
                    "cwd": None,
                    "errors": [],
                    "raw_output": "",
                }
            )

        lang, error = _detect_language(target, language)
        if error or not lang:
            return _json(
                {
                    "ok": False,
                    "language": lang,
                    "scope": None,
                    "error": error or "No checker available",
                    "command": None,
                    "cwd": None,
                    "errors": [],
                    "raw_output": "",
                }
            )

        cfg = CHECKERS[lang]
        markers = cfg.get("markers", [])
        template: list[str] | None = None
        cwd: Path
        scope: str

        if target.is_file():
            cwd = target.parent
            scope = "file"
            template = _select_template(cfg.get("file"), cwd, target, lang)
            if template is None and cfg.get("project"):
                project_root = _find_project_root(target.parent, markers)
                if project_root:
                    cwd = project_root
                    scope = "project"
                    template = _select_template(cfg.get("project"), cwd, target, lang)
        else:
            cwd = _find_project_root(target, markers) or target
            scope = "project"
            template = _select_template(cfg.get("project"), cwd, target, lang)

        if template is None:
            return _json(
                {
                    "ok": False,
                    "language": lang,
                    "scope": None,
                    "error": f"No syntax checker for {lang} is installed or configured",
                    "command": None,
                    "cwd": None,
                    "errors": [],
                    "raw_output": "",
                }
            )

        cmd, returncode, stdout, stderr = _run_template(template, target, cwd, _clamp_timeout(timeout))
        raw = (stdout + "\n" + stderr).strip()
        ok = returncode == 0
        return _json(
            {
                "ok": ok,
                "language": lang,
                "scope": scope,
                "command": shlex.join(cmd),
                "cwd": str(cwd),
                "errors": [] if ok else _parse_errors(raw, lang, cwd),
                "raw_output": raw[:8000],
            }
        )
    except Exception as exc:
        return _json(
            {
                "ok": False,
                "language": None,
                "scope": None,
                "error": str(exc),
                "command": None,
                "cwd": None,
                "errors": [],
                "raw_output": "",
            }
        )
