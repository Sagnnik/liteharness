from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import tool

from config import settings
from permissions import PROJECT_ROOT, validate_path

SHELL_DIR = Path(settings.ness_dir) / "shells"
JOBS_FILE = "jobs.json"
DEFAULT_OUTPUT_CHARS = 12_000
MAX_OUTPUT_CHARS = 200_000
MAX_TIMEOUT_SECONDS = 600
TERM_GRACE_SECONDS = 2.0
SHELL_STATUSES = {"ok", "failed", "timeout", "running", "killed", "error"}

_job_processes: dict[str, subprocess.Popen[bytes]] = {}


@tool
def shell(
    action: Literal["run", "start", "jobs", "read", "kill"],
    command: str = "",
    timeout: int = 30,
    max_output_chars: int = DEFAULT_OUTPUT_CHARS,
    name: str = "",
    include_finished: bool = True,
    job_id: str = "",
    tail_chars: int = DEFAULT_OUTPUT_CHARS,
    force: bool = False,
) -> str:
    """Execute development operations via the system shell.

    Supported Actions & Required Parameters:
      - 'run': Execute a synchronous foreground command.
        -> Required: `command`
        -> Optional: `timeout`, `max_output_chars`
      - 'start': Run a persistent background process (e.g. dev servers, worker daemons).
        -> Required: `command`
        -> Optional: `name`
      - 'jobs': List all tracked background executions and their runtime states.
        -> Optional: `include_finished`
      - 'read': Retrieve tail logs and runtime status updates for an active or dead job.
        -> Required: `job_id`
        -> Optional: `tail_chars`
      - 'kill': Forcefully terminate or gracefully stop a running background job group.
        -> Required: `job_id`
        -> Optional: `force`
    """
    if action == "run":
        return _shell_run(command, timeout=timeout, max_output_chars=max_output_chars)
    if action == "start":
        return _shell_start(command, name=name)
    if action == "jobs":
        return _shell_jobs(include_finished=include_finished)
    if action == "read":
        return _shell_read(job_id, tail_chars=tail_chars)
    if action == "kill":
        return _shell_kill(job_id, force=force)
    
    return _format_result(
        "error",
        duration_ms=0,
        output=f"Unknown shell action: {action}",
        output_truncated=False,
    )


def _shell_run(command: str, timeout: int = 30, max_output_chars: int = DEFAULT_OUTPUT_CHARS) -> str:
    """Run one isolated foreground command via bash -lc from the project root."""
    started = time.monotonic()
    if not command.strip():
        return _format_result(
            "error",
            duration_ms=0,
            output="Empty shell command.",
            output_truncated=False,
        )

    timeout = _clamp_timeout(timeout)
    max_output_chars = _clamp_output_chars(max_output_chars)
    with tempfile.TemporaryFile() as out:
        try:
            proc = subprocess.Popen(
                ["bash", "-lc", command],
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            return _format_result(
                "error",
                duration_ms=_duration_ms(started),
                output=f"Failed to start bash: {exc}",
                output_truncated=False,
            )

        timed_out = False
        try:
            exit_code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(proc)
            exit_code = proc.poll()

        out.flush()
        output, truncated = _tail_stream(out, max_output_chars)

    if timed_out:
        status = "timeout"
        if output:
            output = f"Command timed out after {timeout}s\n{output}"
        else:
            output = f"Command timed out after {timeout}s"
    elif exit_code == 0:
        status = "ok"
    else:
        status = "failed"

    return _format_result(
        status,
        exit_code=exit_code,
        duration_ms=_duration_ms(started),
        output=output,
        output_truncated=truncated,
    )


def _shell_start(command: str, name: str = "") -> str:
    """Start a background command via bash and log output under .ness/shells."""
    started = time.monotonic()
    if not command.strip():
        return _format_result(
            "error",
            duration_ms=0,
            output="Empty shell command.",
            output_truncated=False,
        )

    shell_dir = _ensure_shell_dir()
    jobs = _refresh_jobs(_load_jobs(shell_dir), shell_dir)
    job_id = _new_job_id(jobs)
    log_path = shell_dir / f"{job_id}.log"
    exit_path = shell_dir / f"{job_id}.exit"
    script_path = shell_dir / f"{job_id}.sh"

    script_path.write_text(_job_script(command, exit_path), encoding="utf-8")
    script_path.chmod(0o600)

    try:
        out = log_path.open("ab")
        try:
            proc = subprocess.Popen(
                ["bash", str(script_path)],
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            out.close()
    except OSError as exc:
        return _format_result(
            "error",
            duration_ms=_duration_ms(started),
            output=f"Failed to start background command: {exc}",
            output_truncated=False,
        )

    pgid = _process_group_id(proc)
    job = {
        "job_id": job_id,
        "name": name.strip(),
        "command": command,
        "pid": proc.pid,
        "pgid": pgid,
        "status": "running",
        "start_time": time.time(),
        "end_time": None,
        "exit_code": None,
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        "exit_path": str(exit_path.relative_to(PROJECT_ROOT)),
        "script_path": str(script_path.relative_to(PROJECT_ROOT)),
    }
    jobs[job_id] = job
    _job_processes[job_id] = proc
    _save_jobs(shell_dir, jobs)

    return _format_result(
        "running",
        duration_ms=_duration_ms(started),
        output=_render_job(job),
        output_truncated=False,
        job_id=job_id,
    )


def _shell_jobs(include_finished: bool = True) -> str:
    """List known background shell jobs and their current status."""
    started = time.monotonic()
    shell_dir = _ensure_shell_dir()
    jobs = _refresh_jobs(_load_jobs(shell_dir), shell_dir)
    _save_jobs(shell_dir, jobs)

    visible = [
        job
        for job in sorted(jobs.values(), key=lambda item: str(item.get("start_time") or ""))
        if include_finished or job.get("status") == "running"
    ]
    if not visible:
        output = "No shell jobs."
    else:
        output = "\n".join(_render_job(job) for job in visible)
    return _format_result(
        "ok",
        duration_ms=_duration_ms(started),
        output=output,
        output_truncated=False,
    )


def _shell_read(job_id: str, tail_chars: int = DEFAULT_OUTPUT_CHARS) -> str:
    """Read a background shell job status and bounded log output."""
    started = time.monotonic()
    shell_dir = _ensure_shell_dir()
    jobs = _refresh_jobs(_load_jobs(shell_dir), shell_dir)
    job = jobs.get(job_id)
    if job is None:
        return _format_result(
            "error",
            duration_ms=_duration_ms(started),
            output=f"Unknown shell job: {job_id}",
            output_truncated=False,
            job_id=job_id,
        )

    _save_jobs(shell_dir, jobs)
    try:
        log_path = _job_path(job, "log_path")
    except ValueError as exc:
        return _format_result(
            "error",
            duration_ms=_duration_ms(started),
            output=str(exc),
            output_truncated=False,
            job_id=job_id,
        )
    output, truncated = _tail_file(log_path, _clamp_output_chars(tail_chars))
    body = _render_job(job)
    if output:
        body = f"{body}\n\n{output}"
    return _format_result(
        str(job.get("status") or "error"),
        exit_code=_optional_int(job.get("exit_code")),
        duration_ms=_duration_ms(started),
        output=body,
        output_truncated=truncated,
        job_id=job_id,
    )


def _shell_kill(job_id: str, force: bool = False) -> str:
    """Terminate a background shell job process group."""
    started = time.monotonic()
    shell_dir = _ensure_shell_dir()
    jobs = _refresh_jobs(_load_jobs(shell_dir), shell_dir)
    job = jobs.get(job_id)
    if job is None:
        return _format_result(
            "error",
            duration_ms=_duration_ms(started),
            output=f"Unknown shell job: {job_id}",
            output_truncated=False,
            job_id=job_id,
        )

    if job.get("status") != "running":
        return _format_result(
            str(job.get("status") or "error"),
            exit_code=_optional_int(job.get("exit_code")),
            duration_ms=_duration_ms(started),
            output=f"Shell job {job_id} is already {job.get('status')}.",
            output_truncated=False,
            job_id=job_id,
        )

    proc = _job_processes.get(job_id)
    pgid = _optional_int(job.get("pgid")) or _optional_int(job.get("pid"))
    if pgid is None:
        job["status"] = "error"
        job["end_time"] = time.time()
        jobs[job_id] = job
        _save_jobs(shell_dir, jobs)
        return _format_result(
            "error",
            duration_ms=_duration_ms(started),
            output=f"Shell job {job_id} has no process group to terminate.",
            output_truncated=False,
            job_id=job_id,
        )

    killed = _kill_process_group(pgid, proc=proc, force=force)
    job["status"] = "killed" if killed else "error"
    job["end_time"] = time.time()
    job["exit_code"] = None
    jobs[job_id] = job
    _save_jobs(shell_dir, jobs)
    _job_processes.pop(job_id, None)
    return _format_result(
        str(job["status"]),
        duration_ms=_duration_ms(started),
        output=_render_job(job),
        output_truncated=False,
        job_id=job_id,
    )


def _ensure_shell_dir() -> Path:
    shell_dir = Path(validate_path(str(SHELL_DIR)))
    shell_dir.mkdir(parents=True, exist_ok=True)
    return shell_dir


def _jobs_path(shell_dir: Path) -> Path:
    return shell_dir / JOBS_FILE


def _load_jobs(shell_dir: Path) -> dict[str, dict[str, Any]]:
    path = _jobs_path(shell_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    jobs = data.get("jobs", data)
    if not isinstance(jobs, dict):
        return {}
    return {
        str(job_id): dict(job)
        for job_id, job in jobs.items()
        if isinstance(job, dict)
    }


def _save_jobs(shell_dir: Path, jobs: dict[str, dict[str, Any]]) -> None:
    path = _jobs_path(shell_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"jobs": jobs}, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _new_job_id(jobs: dict[str, dict[str, Any]]) -> str:
    while True:
        job_id = uuid.uuid4().hex[:8]
        if job_id not in jobs:
            return job_id


def _job_script(command: str, exit_path: Path) -> str:
    exit_file = shlex.quote(str(exit_path))
    payload = shlex.quote(command)
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            f"__liteharness_exit_file={exit_file}",
            "trap 'code=$?; printf \"%s\\n\" \"$code\" > \"$__liteharness_exit_file\"' EXIT",
            f"bash -lc {payload}",
            "",
        ]
    )


def _refresh_jobs(jobs: dict[str, dict[str, Any]], shell_dir: Path) -> dict[str, dict[str, Any]]:
    changed = False
    for job_id, job in list(jobs.items()):
        if job.get("status") != "running":
            continue

        proc = _job_processes.get(job_id)
        exit_code = proc.poll() if proc is not None else None
        if exit_code is None and proc is None:
            try:
                exit_code = _read_exit_code(_job_path(job, "exit_path"))
            except ValueError:
                exit_code = None
        if exit_code is not None:
            job["exit_code"] = exit_code
            job["end_time"] = job.get("end_time") or time.time()
            job["status"] = "ok" if exit_code == 0 else "failed"
            _job_processes.pop(job_id, None)
            changed = True
            continue

        pgid = _optional_int(job.get("pgid")) or _optional_int(job.get("pid"))
        if proc is None and pgid is not None and _process_group_alive(pgid):
            continue

        pid = _optional_int(job.get("pid"))
        if pid is not None and not _pid_alive(pid):
            job["status"] = "error"
            job["end_time"] = job.get("end_time") or time.time()
            _job_processes.pop(job_id, None)
            changed = True

    if changed:
        _save_jobs(shell_dir, jobs)
    return jobs


def _read_exit_code(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _job_path(job: dict[str, Any], key: str) -> Path:
    raw = str(job.get(key) or "")
    if not raw:
        raise ValueError(f"Shell job is missing {key}.")
    return Path(validate_path(raw))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return _process_group_has_live_members(pgid)


def _process_group_has_live_members(pgid: int) -> bool:
    proc_dir = Path("/proc")
    if not proc_dir.exists():
        return True
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
        except OSError:
            continue
        end = stat.rfind(")")
        if end == -1:
            continue
        fields = stat[end + 2 :].split()
        if len(fields) < 3:
            continue
        state = fields[0]
        group = _optional_int(fields[2])
        if group == pgid and state != "Z":
            return True
    return False


def _process_group_id(proc: subprocess.Popen[bytes]) -> int:
    try:
        return os.getpgid(proc.pid)
    except OSError:
        return proc.pid


def _terminate_process_group(proc: subprocess.Popen[bytes]) -> None:
    pgid = _process_group_id(proc)
    _kill_process_group(pgid, proc=proc, force=False)


def _kill_process_group(
    pgid: int,
    *,
    proc: subprocess.Popen[bytes] | None,
    force: bool,
) -> bool:
    if force:
        sent = _send_signal(pgid, signal.SIGKILL)
        return sent and _wait_process_group_exit(pgid, proc, TERM_GRACE_SECONDS)

    sent = _send_signal(pgid, signal.SIGTERM)
    if not sent:
        return False
    if _wait_process_group_exit(pgid, proc, TERM_GRACE_SECONDS):
        return True
    killed = _send_signal(pgid, signal.SIGKILL)
    return killed and _wait_process_group_exit(pgid, proc, TERM_GRACE_SECONDS)


def _wait_process_group_exit(
    pgid: int,
    proc: subprocess.Popen[bytes] | None,
    timeout: float,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None:
            proc.poll()
        if not _process_group_alive(pgid):
            _reap_process(proc)
            return True
        time.sleep(0.05)
    _reap_process(proc)
    return not _process_group_alive(pgid)


def _reap_process(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.returncode is not None:
        return
    try:
        proc.wait(timeout=0)
    except subprocess.TimeoutExpired:
        proc.poll()
    except OSError:
        pass


def _send_signal(pgid: int, sig: signal.Signals) -> bool:
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return True
    except OSError:
        return False


def _tail_file(path: Path, max_chars: int) -> tuple[str, bool]:
    try:
        with path.open("rb") as file:
            return _tail_stream(file, max_chars)
    except OSError:
        return "", False


def _tail_stream(file: Any, max_chars: int) -> tuple[str, bool]:
    file.seek(0, os.SEEK_END)
    size = file.tell()
    if max_chars <= 0:
        return "", size > 0
    max_bytes = min(size, max(max_chars * 4, 4096))
    file.seek(size - max_bytes)
    text, truncated_by_chars = _tail_bytes(file.read(), max_chars)
    return text, size > max_bytes or truncated_by_chars


def _tail_bytes(data: bytes, max_chars: int) -> tuple[str, bool]:
    text = data.decode("utf-8", errors="replace")
    truncated = len(text) > max_chars
    if truncated:
        text = text[-max_chars:]
    return text.rstrip("\n"), truncated


def _clamp_timeout(value: int) -> int:
    try:
        timeout = int(value)
    except (TypeError, ValueError):
        timeout = 30
    return min(max(timeout, 1), MAX_TIMEOUT_SECONDS)


def _clamp_output_chars(value: int) -> int:
    try:
        chars = int(value)
    except (TypeError, ValueError):
        chars = DEFAULT_OUTPUT_CHARS
    return min(max(chars, 0), MAX_OUTPUT_CHARS)


def _duration_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_result(
    status: str,
    *,
    exit_code: int | None = None,
    duration_ms: int,
    output: str,
    output_truncated: bool,
    job_id: str | None = None,
) -> str:
    if status not in SHELL_STATUSES:
        status = "error"
    lines = [
        f"status={status}",
        f"exit_code={'' if exit_code is None else exit_code}",
        f"duration_ms={duration_ms}",
        f"cwd={PROJECT_ROOT}",
        f"output_truncated={'true' if output_truncated else 'false'}",
    ]
    if job_id is not None:
        lines.append(f"job_id={job_id}")
    lines.append("output:")
    if output:
        lines.append(output)
    return "\n".join(lines)


def _render_job(job: dict[str, Any]) -> str:
    name = str(job.get("name") or "")
    label = f" name={name}" if name else ""
    exit_code = job.get("exit_code")
    return (
        f"job_id={job.get('job_id')} status={job.get('status')}"
        f" exit_code={'' if exit_code is None else exit_code}"
        f" pid={job.get('pid')} pgid={job.get('pgid')}"
        f" log={job.get('log_path')}{label}"
        f" command={job.get('command')}"
    )
