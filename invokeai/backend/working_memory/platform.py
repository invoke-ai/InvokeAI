import sys
from dataclasses import astuple, dataclass, replace
from typing import Final, Literal

import torch

from invokeai.app.util.torch_cuda_allocator import expandable_segments_enabled

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


def _allocator_for(device: torch.device) -> Allocator:
    """The caching allocator an operation on `device` allocates through: `native` for anything but a CUDA/HIP GPU.

    The backend is fixed before torch initializes CUDA, so torch's answer holds for the process; it is asked first,
    because the driver's pool ignores expandable segments. Expandable segments are a mode of the native allocator, read
    from the environment the model cache reads them from. A ``cuda`` device on a build without one (a test pinning the
    device) has no backend to ask.
    """
    if device.type != "cuda":
        return "native"
    if torch.cuda.is_available() and torch.cuda.get_allocator_backend() == "cudaMallocAsync":
        return "cudaMallocAsync"
    return "expandable" if expandable_segments_enabled() else "native"


def _host_os() -> OperatingSystem:
    if sys.platform == "win32":
        return "windows"
    return "linux" if sys.platform.startswith("linux") else "other"


def platform_for(device: torch.device) -> Platform:
    """The platform an operation on `device` runs on.

    A HIP build reports ``device.type == "cuda"``, so the torch build -- not the device string -- is what separates
    MIOpen from cuDNN, and a `cpu_only` model on a ROCm build is not on MIOpen. Any device type other than CUDA, MPS and
    XPU runs torch's CPU kernels, or none at all, and is priced as the CPU.
    """
    allocator, os = _allocator_for(device), _host_os()
    if device.type == "cuda":
        return Platform("miopen" if torch.version.hip is not None else "cudnn", allocator, os)
    if device.type == "mps":
        return Platform("mps", allocator, os)
    if device.type == "xpu":
        return Platform("xpu", allocator, os)
    return Platform("cpu", allocator, os)
