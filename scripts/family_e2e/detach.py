"""Starting the runner as a background process that outlives the caller.

On Windows the runner breaks away from the caller's job object, so a caller that kills its job's processes (an agent
harness does after two hours) cannot take a long run with it. Where the job forbids breakaway, the runner is created
through WMI instead, which puts it in no job of the caller; such a process starts with the user's default
environment, not the caller's.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from scripts.family_e2e.manifest import TOOL_CHECKOUT

RUNNER_LOG = "runner.log"
WMI_TIMEOUT_S = 120

# Hidden window: WMI shows a console program's window otherwise.
_WMI_CREATE = (
    "$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; "
    "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{"
    "CommandLine=$env:FAMILY_E2E_COMMAND; CurrentDirectory=$env:FAMILY_E2E_CWD; ProcessStartupInformation=$startup}; "
    "if ($r.ReturnValue -ne 0) { [Console]::Error.WriteLine('Win32_Process.Create returned ' + $r.ReturnValue); exit 1 }; "
    "$r.ProcessId"
)


class DetachError(RuntimeError):
    pass


def runner_command(run_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "scripts.family_e2e",
        "run",
        "--run-dir",
        str(run_dir),
        "--log-file",
        str(run_dir / RUNNER_LOG),
    ]


def spawn_runner(run_dir: Path) -> int:
    """Starts the runner for `run_dir` in the background and returns the started process's pid."""
    args = runner_command(run_dir)
    stdio = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform != "win32":
        return subprocess.Popen(args, cwd=TOOL_CHECKOUT, start_new_session=True, **stdio).pid
    # CREATE_NO_WINDOW, not DETACHED_PROCESS: a venv's python.exe is a launcher, and under DETACHED_PROCESS the
    # interpreter it starts gets a new visible console, which the server and every tool call would then share.
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_BREAKAWAY_FROM_JOB
    try:
        return subprocess.Popen(args, cwd=TOOL_CHECKOUT, creationflags=flags, close_fds=True, **stdio).pid
    except OSError:
        return _spawn_via_wmi(args)


def _spawn_via_wmi(args: list[str]) -> int:
    # Passed through the environment, so neither PowerShell nor WMI has to parse quoting of their own.
    env = os.environ | {"FAMILY_E2E_COMMAND": subprocess.list2cmdline(args), "FAMILY_E2E_CWD": str(TOOL_CHECKOUT)}
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _WMI_CREATE],
        env=env,
        capture_output=True,
        text=True,
        timeout=WMI_TIMEOUT_S,
        check=False,
    )
    if result.returncode != 0:
        raise DetachError(f"starting the runner through WMI failed:\n{result.stderr.strip()}")
    return int(result.stdout.strip().splitlines()[-1])
