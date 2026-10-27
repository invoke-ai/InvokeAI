"""What a measurement is keyed by: the class of device it ran on, and the software environment that decides how much
memory an operation takes there.

A device class names a kind of card, not one card: identical cards share it, whatever their index or PCI slot, so
measurements taken on one apply to the other. An environment splits into hard facts, any change of which can change an
operation's memory (the operating system, torch and runtime versions, the convolution library, the allocator and its
configuration, the attention path, InvokeAI's memory settings), and soft facts recorded for diagnosis only (full
versions, the driver, the OS build). Both are produced deterministically here; which change makes a measurement stale
is the caller's decision.
"""

import ctypes
import ctypes.util
import hashlib
import json
import os
import platform
import sys
from collections.abc import Mapping
from typing import Final, Literal, Optional

import psutil
import torch
from pydantic import BaseModel, ConfigDict, computed_field

from invokeai.backend.util import nvml
from invokeai.backend.util.attention import rocm_sdpa_chunks_math
from invokeai.backend.working_memory.counters import Backend, backend_for
from invokeai.version import __version__

_VENDORS: Final[Mapping[Backend, str]] = {"cuda": "nvidia", "hip": "amd", "mps": "apple", "xpu": "intel"}

HARD_ENVIRONMENT_VARIABLES: Final = (
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_HIP_ALLOC_CONF",
    "MIOPEN_FIND_MODE",
    "PYTORCH_TUNABLEOP_ENABLED",
    # The head width above which the ROCm SDPA guard sends attention to the chunked math kernel (`util/attention.py`).
    "INVOKE_ROCM_FUSED_SDPA_MAX_HEAD_DIM",
    # Whether AOTriton's fused attention runs on GPUs PyTorch marks experimental; without it attention takes the math
    # kernel. What `rocm_aotriton_experimental` resolved to (`app/util/rocm_aotriton.py`), or a value set beforehand.
    "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL",
    # Native or decomposed causal 3D convolutions on ROCm (`backend/wan/rocm_causal_conv3d.py`).
    "INVOKEAI_ROCM_CONV3D",
)
"""Environment variables that change how much memory an operation takes: the allocator's configuration, MIOpen's
algorithm search, TunableOp's GEMM selection, and the ROCm attention and convolution kernel choices. Recorded as
``env.<name>``, empty when unset."""


def _short_hash(fields: Mapping[str, object]) -> str:
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class DeviceClass(BaseModel):
    """A kind of compute device. Identical cards share one, whatever their index or PCI location."""

    model_config = ConfigDict(frozen=True)

    vendor: str
    name: str
    total_mib: int
    backend: Backend
    arch: Optional[str]
    """``gcnArchName`` on HIP (e.g. ``gfx1200``), ``sm_<major><minor>`` on CUDA, the reported architecture on XPU."""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def id(self) -> str:
        """A short stable hash of the other fields. Derived, never given."""
        return _short_hash(self.model_dump(exclude={"id"}))


def _mps_name() -> str:
    """The Mac's chip, e.g. ``Apple M2 Max``; MPS names no device."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        size = ctypes.c_size_t(0)
        if libc.sysctlbyname(b"machdep.cpu.brand_string", None, ctypes.byref(size), None, 0) == 0 and size.value:
            buffer = ctypes.create_string_buffer(size.value)
            if libc.sysctlbyname(b"machdep.cpu.brand_string", buffer, ctypes.byref(size), None, 0) == 0:
                return buffer.value.decode("utf-8", errors="replace")
    except Exception:
        pass
    return "Apple GPU"


def device_class_for(device: torch.device) -> DeviceClass:
    """The class of `device`. Raises ValueError for a device that is not a GPU."""
    backend = backend_for(device)
    if backend is None:
        raise ValueError(f"{device} is not a GPU; only GPU devices have a device class.")
    if backend in ("cuda", "hip"):
        props = torch.cuda.get_device_properties(device)
        name, total = torch.cuda.get_device_name(device), int(props.total_memory)
        arch = str(props.gcnArchName) if backend == "hip" else f"sm_{props.major}{props.minor}"
    elif backend == "xpu":
        props = torch.xpu.get_device_properties(device)
        name, total = str(props.name), int(props.total_memory)
        architecture = getattr(props, "architecture", None)
        arch = None if architecture is None else str(architecture)
    else:
        # Unified memory: the device's share is the machine's RAM.
        name, total, arch = _mps_name(), int(psutil.virtual_memory().total), None
    return DeviceClass(vendor=_VENDORS[backend], name=name, total_mib=total // 2**20, backend=backend, arch=arch)


class Environment(BaseModel):
    """The software a measurement ran under: hard facts that change an operation's memory, soft facts that do not."""

    model_config = ConfigDict(frozen=True)

    hard: dict[str, str]
    soft: dict[str, Optional[str]]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def id(self) -> str:
        """A short stable hash of the hard facts. Soft facts do not enter it."""
        return _short_hash(self.hard)


def _major_minor(version: str) -> str:
    return ".".join(version.split("+", 1)[0].split(".")[:2])


def _conv_library_major() -> Optional[str]:
    """The major version of cuDNN, or of MIOpen on a HIP build, as loaded at runtime."""
    try:
        if torch.backends.cudnn.version() is None:
            return None
        # (major, minor, patch) on either build; the packed integer `version()` returns is encoded per library.
        return str(torch._C._cudnn.getRuntimeVersion()[0])
    except Exception:
        return None


def os_family(platform_name: str) -> Literal["windows", "linux", "other"]:
    """The operating system family of a ``sys.platform`` value, as a hard fact: the driver model differs (WDDM pages
    an allocation that does not fit into shared memory instead of failing it), so the same card and torch build can
    take different memory on Windows and on Linux."""
    if platform_name == "win32":
        return "windows"
    return "linux" if platform_name.startswith("linux") else "other"


def environment_for(device: torch.device, settings: Mapping[str, str | int | float | bool | None]) -> Environment:
    """The environment an operation on `device` runs under. Every key of `settings` is a hard fact, as
    ``setting.<key>``: the caller passes the InvokeAI settings that change memory."""
    backend = backend_for(device)
    hard: dict[str, str] = {"os": os_family(sys.platform), "torch": _major_minor(torch.__version__)}
    if backend in ("cuda", "hip"):
        runtime = torch.version.hip if backend == "hip" else torch.version.cuda
        hard[backend] = _major_minor(runtime or "")
        hard["miopen" if backend == "hip" else "cudnn"] = _conv_library_major() or ""
        hard["allocator_backend"] = torch.cuda.get_allocator_backend()
        hard["rocm_sdpa_guard"] = str(rocm_sdpa_chunks_math(device.type)).lower()
    elif backend == "xpu":
        hard["xpu"] = str(getattr(torch.version, "xpu", None) or "")
    for name in HARD_ENVIRONMENT_VARIABLES:
        hard[f"env.{name}"] = os.environ.get(name, "")
    for key, value in settings.items():
        hard[f"setting.{key}"] = str(value)

    soft: dict[str, Optional[str]] = {
        "torch_build": torch.__version__,
        "driver": nvml.driver_version() if backend == "cuda" else None,
        "os_build": platform.platform(),
        "invokeai": __version__,
    }
    return Environment(hard=hard, soft=soft)
