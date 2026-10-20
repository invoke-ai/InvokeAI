"""Stopping a child process together with every process it started."""

import subprocess
from typing import Any

import psutil


def kill_process_tree(proc: subprocess.Popen[Any]) -> None:
    """Kill ``proc`` and its descendants, then reap them.

    A command can do its work in a child: the ffmpeg on ``PATH`` (Windows ARM64, where
    imageio-ffmpeg bundles none) can be a package manager's launcher shim -- Chocolatey's and
    Scoop's are -- that runs the real binary as its child, so killing only ``proc`` would leave
    that ffmpeg running.

    The descendants are found through ``proc``. Once it has exited they can no longer be told
    apart from unrelated processes holding a reused parent PID, so they are not looked for; a
    shim waits for its child, so it exits only after that child has.
    """
    if proc.returncode is not None:
        # Reaped already: its PID may belong to an unrelated process by now.
        return
    descendants: list[psutil.Process] = []
    try:
        root = psutil.Process(proc.pid)
    except psutil.Error:
        root = None
    if root is not None:
        try:
            # Suspended, it cannot start a child between the listing below and its kill.
            root.suspend()
        except psutil.Error:
            pass
        try:
            descendants = root.children(recursive=True)
        except psutil.Error:
            pass
    try:
        proc.kill()
    except OSError:
        pass
    for child in descendants:
        try:
            child.kill()
        except psutil.Error:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    psutil.wait_procs(descendants, timeout=5)
