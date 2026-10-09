"""Hide integrated GPUs from HIP on Windows ROCm, before torch initializes it.

HIP on Windows enumerates the Radeon graphics of a Ryzen CPU next to a discrete card. Generation leaves it out of
`generation_devices: auto` (see `TorchDevice._auto_generation_devices`), but its mere visibility costs: measured with
an RX 9060 XT and the gfx1036 iGPU of a Ryzen 7900, with torch 2.12+rocm7.14.1 as with 2.13.0+rocm10.0.0 and without
any Invoke code, the process's first HIP context on the discrete card shows 17.9 GiB of shared GPU memory and its first
kernel launch another 17.9 GiB. None of it is RAM or commit charge, but it is the counter Windows reports paging in,
so Task Manager shows 36 GiB of shared GPU memory and Invoke's paging check reads it as memory pushed out of VRAM. Any
kernel launched on the iGPU itself, even `torch.randn`, crashed the process with an access violation.

`HIP_VISIBLE_DEVICES` avoids all of it, but HIP reads it once, when it initializes, and only HIP can tell which device
is integrated. So a short child process asks torch -- reading device properties creates no context, so the child
maps nothing -- and this process sets the variable before importing torch itself.

The iGPU is hidden whatever `device` and `generation_devices` say, so that `cuda:N` means the same GPU everywhere: the
settings UI lists, validates and saves devices in the numbering torch reports after hiding, and hiding only while both
are `auto` would renumber them on the next start. No setting can usefully name the iGPU anyway: no kernel runs on it.
"""

import json
import logging
import os
import queue
import subprocess
import sys
import threading
from typing import Optional

from invokeai.app.util.torch_cuda_allocator import _installed_torch_is_rocm

# Every variable HIP narrows its device list by. One already set means someone chose the devices; leave them be.
VISIBILITY_ENV_VARS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")

# The child prints its answer and leaves without Python's shutdown. HIP's own teardown can still hang on Windows while
# its DLLs unload, so the probe takes the answer as soon as the line arrives and ends the child itself.
_PROBE = (
    "import json, os, sys, torch; "
    "print(json.dumps([bool(getattr(torch.cuda.get_device_properties(i), 'is_integrated', False)) "
    "for i in range(torch.cuda.device_count())])); "
    "sys.stdout.flush(); os._exit(0)"
)
# Measured 1.6-2.4 s on an RX 9060 XT with a warm torch import; the margin is for a cold import of torch and its ROCm
# libraries after a reboot. A shorter limit would buy little: a HIP that hangs while enumerating devices (seen after a
# driver fault) hangs this process too, as soon as it imports torch.
_PROBE_TIMEOUT_SECONDS = 60
# A child stuck in a driver call cannot be ended at all (seen after a driver fault); it is left behind after this.
_EXIT_WAIT_SECONDS = 5


class ProbeFailed(Exception):
    """The child process could not say which GPUs are integrated."""


def probe_integrated_flags() -> list[bool]:
    """`is_integrated` of every device HIP enumerates, asked in a child process. Raises `ProbeFailed` with the reason."""
    try:
        child = subprocess.Popen(
            [sys.executable, "-c", _PROBE],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise ProbeFailed(f"could not start it ({e})") from e
    answers: "queue.Queue[Optional[list[bool]]]" = queue.Queue()
    threading.Thread(target=_read_answer, args=(child, answers), daemon=True).start()
    try:
        flags = answers.get(timeout=_PROBE_TIMEOUT_SECONDS)
        if flags is None:
            # The output ended without an answer; let the child finish so its exit code can say why.
            _wait_for_exit(child)
    except queue.Empty:
        raise ProbeFailed(f"no answer within {_PROBE_TIMEOUT_SECONDS} s") from None
    finally:
        if child.poll() is None:
            child.kill()
            _wait_for_exit(child)
    if flags is None:
        raise ProbeFailed(f"it exited with code {child.returncode}" if child.returncode else "it gave no answer")
    return flags


def _read_answer(child: subprocess.Popen, answers: "queue.Queue[Optional[list[bool]]]") -> None:
    """Put the first JSON list the child prints, skipping anything else; None once its output ends without one."""
    assert child.stdout is not None
    for line in child.stdout:
        try:
            flags = json.loads(line)
        except ValueError:
            continue
        if isinstance(flags, list):
            answers.put([bool(flag) for flag in flags])
            return
    answers.put(None)


def _wait_for_exit(child: subprocess.Popen) -> None:
    try:
        child.wait(timeout=_EXIT_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def discrete_device_visibility(flags: list[bool]) -> Optional[str]:
    """The `HIP_VISIBLE_DEVICES` value that hides the integrated GPUs, or None when there is nothing to hide.

    Nothing to hide when no GPU is integrated, and nothing *may* be hidden when every GPU is: an APU on its own is
    what the machine generates on.
    """
    discrete = [str(index) for index, integrated in enumerate(flags) if not integrated]
    if not discrete or len(discrete) == len(flags):
        return None
    return ",".join(discrete)


def _torch_is_imported() -> bool:
    return "torch" in sys.modules


def hide_integrated_gpus_on_rocm_windows(logger: logging.Logger) -> None:
    """Set `HIP_VISIBLE_DEVICES` to the discrete GPUs when HIP on Windows also enumerates an integrated one.

    Only when no visibility variable is set: one that is chose the devices already. Must run before torch is imported.
    """
    if sys.platform != "win32" or any(os.environ.get(var) for var in VISIBILITY_ENV_VARS):
        return
    if not _installed_torch_is_rocm():
        return
    if _torch_is_imported():
        logger.warning("ROCm on Windows: torch was imported before integrated GPUs could be hidden from it.")
        return
    try:
        flags = probe_integrated_flags()
    except ProbeFailed as e:
        logger.warning(
            f"ROCm on Windows: could not ask HIP which GPUs are integrated ({e}); leaving all of them visible. An "
            "integrated GPU can then make Windows report tens of GB of shared GPU memory, and generating on it fails. "
            "Set HIP_VISIBLE_DEVICES to the discrete GPUs to choose them yourself."
        )
        return
    visible = discrete_device_visibility(flags)
    if visible is None:
        return
    hidden = [index for index, integrated in enumerate(flags) if integrated]
    os.environ["HIP_VISIBLE_DEVICES"] = visible
    logger.info(
        f"ROCm on Windows: hiding integrated GPU(s) {hidden} from HIP (HIP_VISIBLE_DEVICES={visible}), so `cuda:N` "
        "counts the discrete GPUs only. While HIP can see an integrated GPU, Windows reports tens of GB of shared GPU "
        "memory for Invoke that is not really in use. Set HIP_VISIBLE_DEVICES yourself to choose the devices."
    )
