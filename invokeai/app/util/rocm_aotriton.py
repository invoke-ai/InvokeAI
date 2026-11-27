"""The `rocm_aotriton_experimental` setting: whether AOTriton's experimental fused SDPA kernels run on AMD GPUs.

PyTorch's ROCm builds ship flash and memory-efficient attention through AOTriton, which marks some GPU architectures
"experimental" (gfx1101, gfx1102, gfx1150, gfx1151 and gfx1200 among them). On those, torch only uses the fused
kernels when `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` is set; otherwise every attention call falls back to the math
kernel, which is slower and materializes the whole score matrix.

Whether the kernels are correct depends on the build as much as on the GPU. On an RX 9060 XT (gfx1200) under torch
2.12+rocm7.14.1, setting the variable made every flash and memory-efficient call fail with `hipErrorInvalidValue`, and
the CLIP text encoder hung on exit. Under torch 2.13.0+rocm10.0.0 every fused case matched the math reference, and
Z-Image at 1024px denoised in 16.7 s instead of 36 s. So `auto` turns the kernels on only for a ROCm 10 build on
architectures they were measured on, and `on`/`off` decide for any build and GPU.

torch reads the variable once per process, at the first fused-kernel check on an experimental architecture. The
server runs attention during startup, so the setting is applied before the app is imported.
"""

import logging
import os
import re
from collections.abc import Iterable
from typing import Optional, Union

from invokeai.app.services.config.config_default import ROCM_AOTRITON

AOTRITON_EXPERIMENTAL_ENV = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"

# Architectures on which the experimental kernels were measured correct under `MIN_AUTO_ROCM_MAJOR`, with the
# measurement: gfx1200 (RX 9060 XT), Windows, torch 2.13.0+rocm10.0.0 -- CLIP, Z-Image, FLUX and VAE attention shapes
# all within the math kernel's own error, Z-Image 1024px 16.7 s instead of 36 s. Add an architecture only with a
# measurement of its own.
AUTO_MEASURED_ARCHS = frozenset({"gfx1200"})
MIN_AUTO_ROCM_MAJOR = 10

# The values `c10::utils::check_env` reads as true.
_TRUTHY = {"1", "on", "yes", "true", "y"}


def rocm_major_version(rocm_version: Optional[str], torch_version: str) -> Optional[int]:
    """The ROCm major version torch was built against, or None when neither source names one.

    `torch.version.rocm` (`10.0.0` on AMD's wheels) first, then the local label of the torch version (`2.13.0+rocm7.2`)
    for a build that leaves it unset. Not `torch.version.hip`: AMD's ROCm 10 wheels report HIP 7.15.
    """
    match = re.match(r"(\d+)\.", rocm_version or "") or re.search(r"\+rocm(\d+)", torch_version)
    return int(match.group(1)) if match else None


def resolve_aotriton_experimental(
    setting: ROCM_AOTRITON,
    exported: Optional[str],
    rocm_major: Optional[int],
    archs: Iterable[str],
) -> tuple[Optional[str], str]:
    """Decide what `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` should be.

    Returns the value to export (None: leave the environment as it is) and the reason, for the log.

    Args:
        setting: The `rocm_aotriton_experimental` setting.
        exported: The variable's value already in the environment, if any. It always wins.
        rocm_major: The ROCm major version of the torch build (see `rocm_major_version`).
        archs: The architectures of the GPUs generation runs on, e.g. `{"gfx1200"}`.
    """
    if exported is not None:
        return None, f"{AOTRITON_EXPERIMENTAL_ENV}={exported} is set in the environment"
    if setting == "on":
        return "1", "rocm_aotriton_experimental is 'on'"
    if setting == "off":
        return "0", "rocm_aotriton_experimental is 'off'"
    archs = sorted(set(archs))
    if rocm_major is None or rocm_major < MIN_AUTO_ROCM_MAJOR:
        build = "a ROCm build without a version label" if rocm_major is None else f"ROCm {rocm_major}"
        return None, f"auto: not measured on {build}"
    if not archs:
        return None, "auto: no ROCm GPU to generate on"
    unmeasured = [arch for arch in archs if arch not in AUTO_MEASURED_ARCHS]
    if unmeasured:
        return None, f"auto: not measured on {', '.join(unmeasured)}"
    return "1", f"auto: measured on {', '.join(archs)} with ROCm {rocm_major}"


def _arch_name(gcn_arch_name: str) -> str:
    """`gfx90a:sramecc+:xnack-` -> `gfx90a`."""
    return gcn_arch_name.split(":", 1)[0]


def apply_rocm_aotriton_setting(
    setting: ROCM_AOTRITON,
    generation_devices: Union[str, list[str]],
    logger: logging.Logger,
) -> None:
    """Export `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` as the setting decides, and log the effective value.

    Must run after the allocator is configured (this imports torch) and before anything runs attention. A no-op on
    builds other than ROCm, apart from a warning for an explicit `on`.
    """
    import torch

    if torch.version.hip is None:
        if setting == "on":
            logger.warning("rocm_aotriton_experimental is 'on', but this PyTorch is not a ROCm build; ignoring it.")
        return

    from invokeai.backend.util.devices import TorchDevice

    archs: set[str] = set()
    try:
        devices = TorchDevice.get_generation_devices(generation_devices) or [TorchDevice.choose_torch_device()]
        for device in devices:
            if device.type == "cuda":
                archs.add(_arch_name(torch.cuda.get_device_properties(device).gcnArchName))
    except Exception as e:
        logger.warning(f"Could not read the GPU architecture for rocm_aotriton_experimental: {e}")
        archs.clear()

    exported = os.environ.get(AOTRITON_EXPERIMENTAL_ENV)
    value, reason = resolve_aotriton_experimental(
        setting, exported, rocm_major_version(getattr(torch.version, "rocm", None), torch.__version__), archs
    )
    if exported is not None and setting != "auto" and (exported.strip().lower() in _TRUTHY) != (setting == "on"):
        logger.warning(
            f"rocm_aotriton_experimental is '{setting}', but {AOTRITON_EXPERIMENTAL_ENV}={exported} is set in the "
            "environment and takes precedence."
        )
    if value is not None:
        os.environ[AOTRITON_EXPERIMENTAL_ENV] = value

    effective = os.environ.get(AOTRITON_EXPERIMENTAL_ENV)
    enabled = effective is not None and effective.strip().lower() in _TRUTHY
    gpus = ", ".join(sorted(archs)) or "unknown GPU"
    if enabled:
        logger.info(f"ROCm ({gpus}): AOTriton's experimental fused attention kernels are on ({reason}).")
    else:
        logger.info(
            f"ROCm ({gpus}): AOTriton's experimental fused attention kernels are off ({reason}). On a GPU that "
            "AOTriton marks experimental, attention then runs on the slower math kernel; set "
            "rocm_aotriton_experimental: on to try the fused kernels."
        )
