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
"""

import json
import logging
import os
import subprocess
import sys
from typing import Optional, Union

from invokeai.app.util.torch_cuda_allocator import _installed_torch_is_rocm

# Every variable HIP narrows its device list by. One already set means someone chose the devices; leave them be.
VISIBILITY_ENV_VARS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")

_PROBE = (
    "import json, torch; "
    "print(json.dumps([bool(getattr(torch.cuda.get_device_properties(i), 'is_integrated', False)) "
    "for i in range(torch.cuda.device_count())]))"
)
_PROBE_TIMEOUT_SECONDS = 120


def probe_integrated_flags() -> Optional[list[bool]]:
    """`is_integrated` of every device HIP enumerates, asked in a child process; None if the child cannot answer."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        # torch may print warnings first; the answer is the last line.
        flags = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None
    if not isinstance(flags, list):
        return None
    return [bool(flag) for flag in flags]


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


def hide_integrated_gpus_on_rocm_windows(
    device: str, generation_devices: Union[str, list[str]], logger: logging.Logger
) -> None:
    """Set `HIP_VISIBLE_DEVICES` to the discrete GPUs when HIP on Windows also enumerates an integrated one.

    Only when nothing chose the devices already: no visibility variable is set, and `device` and `generation_devices`
    are both `auto` -- an explicit `cuda:N` counts in HIP's full enumeration, which hiding a device would renumber.
    Must run before torch is imported.
    """
    if sys.platform != "win32" or any(os.environ.get(var) for var in VISIBILITY_ENV_VARS):
        return
    if device != "auto" or generation_devices != "auto":
        return
    if not _installed_torch_is_rocm():
        return
    if _torch_is_imported():
        logger.warning("ROCm on Windows: torch was imported before integrated GPUs could be hidden from it.")
        return
    flags = probe_integrated_flags()
    if flags is None:
        logger.debug("ROCm on Windows: could not ask HIP which GPUs are integrated; leaving all of them visible.")
        return
    visible = discrete_device_visibility(flags)
    if visible is None:
        return
    hidden = [index for index, integrated in enumerate(flags) if integrated]
    os.environ["HIP_VISIBLE_DEVICES"] = visible
    logger.info(
        f"ROCm on Windows: hiding integrated GPU(s) {hidden} from HIP (HIP_VISIBLE_DEVICES={visible}). While HIP can "
        "see one, Windows reports tens of GB of shared GPU memory for Invoke that is not really in use. Set "
        "HIP_VISIBLE_DEVICES yourself to choose the devices."
    )
