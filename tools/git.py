import subprocess
import os
from langchain_core.tools import tool
from pathlib import Path
from tools.common import validate_path
from config import settings

WORKTREE_ROOT = Path(settings.ness_dir) / "worktree"

def _git(args: list[str], timeout:int=15) -> str:
    r= subprocess.run(
        ["git", *args], 
        capture_output=True, 
        text=True, 
        timeout=timeout,
        cwd=WORKTREE_ROOT
    )
    return r.stdout.strip() or "(ok)"

@tool
def git_status() -> str:
    return _git(["status", "--short"])

@tool
def git_diff(path:str = "", cached:bool = False) -> str:
    args = ["diff"]
    if cached:
        args.append("--cached")
    if path:
        args.append(path)
    
    return _git(args)[:12000]

@tool
def git_log(n:int = 20, path:str = "") -> str:
    args = ["log", f"-{n}", "--oneline"]
    if path:
        args.extend(["--", validate_path(path)])
    return _git(args)

@tool
def git_show(rev: str = "HEAD") -> str:
    return _git(["show", rev])[:12000]


@tool
def git_blame(path: str, line: int | None = None) -> str:
    path = validate_path(path)
    args = ["blame", path]
    if line:
        args.extend(["-L", f"{line},{line}"])
    return _git(args)[:8000]


@tool
def git_snapshot(message: str = "agent: auto-save") -> str:
    if subprocess.run(["git", "rev-parse", "--git-dir"], capture_output=True).returncode != 0:
        return "Not a git repo"
    subprocess.run(["git", "add", "-A"], capture_output=True)
    r = subprocess.run(["git", "commit", "-m", message, "--no-verify"], capture_output=True, text=True)
    if r.returncode == 0:
        return f"Snapshot: {message}"
    return "No changes to snapshot"


@tool
def git_commit(message: str, paths: str = "") -> str:
    args = ["commit", "-m", message]
    if paths:
        args.extend(paths.split())
    return _git(args)


@tool
def git_checkout(branch: str, create: bool = False) -> str:
    args = ["checkout"]
    if create:
        args.append("-b")
    args.append(branch)
    return _git(args)


@tool
def git_branch(name: str = "") -> str:
    return _git(["branch", name] if name else ["branch"])


@tool
def git_stash(action: str = "list") -> str:
    return _git(["stash", action])


@tool
def git_worktree_add(branch: str, name: str = "") -> str:
    WORKTREE_ROOT.mkdir(parents=True, exist_ok=True)
    name = name or branch.replace("/", "-")
    wt_path = WORKTREE_ROOT / name
    if wt_path.exists():
        return f"Worktree already exists: {wt_path}"
    return _git(["worktree", "add", str(wt_path), branch])


@tool
def git_worktree_list() -> str:
    return _git(["worktree", "list"])


@tool
def git_worktree_remove(name: str) -> str:
    wt_path = validate_path(str(WORKTREE_ROOT / name))
    return _git(["worktree", "remove", wt_path, "--force"])