from pathlib import Path
from langchain_core.tools import tool
from config import settings
import subprocess
import os
import uuid
import time

NESS = Path(settings.ness_dir)
SHELL_DIR = NESS / "shell"

# Hold a subprocess object
_shell_proc: subprocess.Popen | None = None

def _ensure_shell():
    global _shell_proc
    # check if any process is running and poll to check if it is still running
    if _shell_proc is None or _shell_proc.poll() is not None:
        # launch a background shell
        _shell_proc = subprocess.Popen(
            ["bash"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.getcwd()
        )
    return _shell_proc

@tool
def bash(command: str, timeout:int=30, background:bool = False) -> str:
    """Run bash in persistant shell. background=True writes to .ness/shell/<id>.log"""

    # if background is True, launch a process and send outputs to a log file
    if background:
        SHELL_DIR.mkdir(parents=True, exist_ok=True)
        job_id = str(uuid.uuid4())[:8]
        log = SHELL_DIR / f"{job_id}.log"
        proc = subprocess.Popen(
            command, 
            shell = True, 
            stdout=open(log, "w"), 
            stderr=subprocess.STDOUT, 
            cwd=os.getcwd()
        )
        return f"Background job started: {job_id} (pid: {proc.pid}). Logged to {log}"

    # otherwise, use the persistent shell
    proc = _ensure_shell()
    assert proc.stdin and proc.stdout
    proc.stdin.write(command + "\n")
    # signal string for completion
    proc.stdin.write("echo __DONE__$?\n")
    # always flush the input buffer
    proc.stdin.flush()
    lines, code = [], 0
    
    # create a deadline for the command to complete
    deadline = time.time() + timeout
    while time.time() < deadline:
        # read a line from the output
        line = proc.stdout.readline() 
        if not line:
            break
        if line.startswith("__DONE__"):
            # extract the exit code 0 or 1
            code = int(line.strip().split("__DONE__")[1])
            break
        lines.append(line.rstrip())
    out = "\n".join(lines)[-8000:]

    return f"exit={code}\n{out}"


