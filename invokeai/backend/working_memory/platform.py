import os
import sys
from collections.abc import Mapping
from dataclasses import astuple, dataclass, replace
from typing import Final, Literal, Optional, cast, get_args

import torch

from invokeai.app.util.torch_cuda_allocator import ALLOCATOR_CONF_ENV_VARS, expandable_segments_enabled
from invokeai.backend.working_memory.counters import Backend, backend_for

ConvBackend = Literal["cudnn", "miopen", "mps", "xpu", "cpu"]
Allocator = Literal["native", "cudaMallocAsync", "expandable"]
"""torch's caching allocator: the native one, the native one in expandable-segments mode, or the driver's
``cudaMallocAsync`` pool (``hipMallocAsync`` on a HIP build, which torch reports under the same name)."""
OperatingSystem = Literal["windows", "linux", "other"]


@dataclass(frozen=True)
class Platform:
    """The traits of the machine an operation runs on, as coefficients are keyed. Each field holds the true value:
    what was measured, and what was not, is `priced_as`'s business.

    `conv_backend` and `allocator` are the keys an `EstimatorSpec` names. `os` refines them only where an entry says
    so: an entry whose platform names an `os` applies there and takes precedence over the entry that does not, so the
    operating system enters the pricing of exactly the platforms measured on one OS or measured to differ by it, and
    of no other. Both are MIOpen: its native allocator demands up to 1.6x more on Windows than on Linux, and expandable
    segments, Invoke's Windows default, are measured only on Windows.
    """

    conv_backend: ConvBackend
    allocator: Allocator
    os: OperatingSystem


MEASURED: Final = frozenset(
    {
        ("cudnn", "native", None),
        ("cudnn", "cudaMallocAsync", None),
        ("miopen", "native", None),
        ("miopen", "expandable", "windows"),
    }
)
"""The (convolution backend, allocator, OS) combinations coefficients are measured on, None for every OS. Shipped data
carries entries for these only."""


def is_measured(platform: Platform) -> bool:
    return {(platform.conv_backend, platform.allocator, None), astuple(platform)} & MEASURED != set()


def priced_as(platform: Platform) -> Platform:
    """The measured platform an operation on `platform` is priced as: `platform` itself where it is measured. The OS
    stays: it only selects among the entries of the measured pair.

    The one place that borrowing is declared; the resolver applies it and names it in an estimate's `source`.

    - A convolution backend nothing is measured on (MPS, XPU, CPU) takes cuDNN under the native allocator. On MPS that
      pairs with a score-matrix term the SDPA probe always charges there, so the total is not thin.
    - An allocator not measured with its backend on this OS takes that backend's native allocator: cuDNN in
      expandable-segments mode, MIOpen under ``hipMallocAsync``, and MIOpen with expandable segments off Windows (a
      user override there; Linux defaults to native). On cuDNN, native is the larger of the two measured allocators
      (``cudaMallocAsync`` needs 25-35% less); on MIOpen, native is the larger too, on Windows by up to 1.6x.
    """
    if is_measured(platform):
        return platform
    if platform.conv_backend in ("mps", "xpu", "cpu"):
        return replace(platform, conv_backend="cudnn", allocator="native")
    return replace(platform, allocator="native")


_CONV_BACKENDS: Final[Mapping[Optional[Backend], ConvBackend]] = {
    "cuda": "cudnn",
    "hip": "miopen",
    "mps": "mps",
    "xpu": "xpu",
    None: "cpu",
}
"""The convolution backend of each GPU backend. A HIP build reports its GPUs as ``cuda`` devices, so the torch build --
not the device string -- separates MIOpen from cuDNN, and a `cpu_only` model on a ROCm build is not on MIOpen. Any
device type other than CUDA, MPS and XPU runs torch's CPU kernels, or none at all, and is priced as the CPU."""


def _allocator(backend: Optional[Backend], allocator_backend: Optional[str], environ: Mapping[str, str]) -> Allocator:
    """The caching allocator of a `backend` device under torch's allocator backend and these allocator variables:
    `native` for anything but a CUDA/HIP GPU. The driver's pool ignores expandable segments, so torch's backend
    decides first; expandable segments are a mode of the native allocator."""
    if backend not in ("cuda", "hip"):
        return "native"
    if allocator_backend == "cudaMallocAsync":
        return "cudaMallocAsync"
    return "expandable" if expandable_segments_enabled(environ) else "native"


def _allocator_for(device: torch.device) -> Allocator:
    """The caching allocator an operation on `device` allocates through.

    The backend is fixed before torch initializes CUDA, so torch's answer holds for the process. The variables are read
    from the environment the model cache reads them from. A ``cuda`` device on a build without one (a test pinning the
    device) has no backend to ask.
    """
    backend = backend_for(device)
    allocator_backend = (
        torch.cuda.get_allocator_backend() if backend in ("cuda", "hip") and torch.cuda.is_available() else None
    )
    return _allocator(backend, allocator_backend, os.environ)


def _host_os() -> OperatingSystem:
    """The operating system this process runs on, as platforms name it."""
    if sys.platform == "win32":
        return "windows"
    return "linux" if sys.platform.startswith("linux") else "other"


def platform_for(device: torch.device) -> Platform:
    """The platform an operation on `device` runs on."""
    return Platform(_CONV_BACKENDS[backend_for(device)], _allocator_for(device), _host_os())


def platform_for_environment(backend: Backend, hard: Mapping[str, str]) -> Platform:
    """The platform `platform_for` gives a `backend` device in the environment whose hard facts are `hard`
    (`fingerprint.environment_for`), as a calibration store records them: torch's allocator backend, the allocator
    variables and the OS. Raises ValueError when they lack a fact the platform needs.
    """
    os_name = hard.get("os")
    if os_name not in get_args(OperatingSystem):
        raise ValueError(f"The environment records no operating system the platform knows: os={os_name!r}.")
    allocator_backend = hard.get("allocator_backend")
    if backend in ("cuda", "hip") and allocator_backend is None:
        raise ValueError(f"The environment of a {backend} device records no allocator_backend.")
    environ = {var: hard.get(f"env.{var}", "") for var in ALLOCATOR_CONF_ENV_VARS}
    allocator = _allocator(backend, allocator_backend, environ)
    return Platform(_CONV_BACKENDS[backend], allocator, cast(OperatingSystem, os_name))
