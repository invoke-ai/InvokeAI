"""Minimal NVML queries through ctypes: this process's memory on an NVIDIA device, and the driver version.

NVML ships with the NVIDIA driver (``libnvidia-ml.so.1`` on Linux, ``nvml.dll`` in System32 on Windows), so this needs
no package. Under WDDM, Windows reports no per-process memory through NVML; the working-memory probe reads it through
D3DKMT there (`wddm.process_local_usage`), and uses NVML on Windows for the driver version only.

Everything here is best-effort: any failure yields ``None`` ("unknown") rather than raising.
"""

import ctypes
import functools
import os
import sys
import threading
from typing import Optional

import torch

from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.util.pci import pci_address

_NVML_SUCCESS = 0
_NVML_ERROR_INSUFFICIENT_SIZE = 7
# `usedGpuMemory` when the driver cannot attribute memory to the process (always so under WDDM).
_NVML_VALUE_NOT_AVAILABLE = 0xFFFFFFFFFFFFFFFF
# NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE.
_DRIVER_VERSION_BUFFER_SIZE = 80
# Room for processes that start between the count and the fill call.
_SPARE_PROCESS_ENTRIES = 8

_LIBRARY_NAMES = ("nvml.dll",) if sys.platform == "win32" else ("libnvidia-ml.so.1", "libnvidia-ml.so")


class _ProcessInfo(ctypes.Structure):
    """``nvmlProcessInfo_t``, the entry ``nvmlDeviceGetComputeRunningProcesses_v3`` fills."""

    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]


_lock = threading.Lock()
# Per torch device index: the NVML device handle, or None once the lookup has failed.
_handles: dict[int, Optional[ctypes.c_void_p]] = {}


def _configure_prototypes(lib: ctypes.CDLL) -> None:
    """Declare every entry point used here, so ctypes passes handles at full pointer width."""
    prototypes = {
        "nvmlInit_v2": [],
        "nvmlDeviceGetHandleByPciBusId_v2": [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)],
        "nvmlDeviceGetComputeRunningProcesses_v3": [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.c_void_p,
        ],
        "nvmlSystemGetDriverVersion": [ctypes.c_char_p, ctypes.c_uint],
    }
    for name, argtypes in prototypes.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int


@functools.lru_cache(maxsize=1)
def _load() -> Optional[ctypes.CDLL]:
    """Open and initialise NVML once per process; None where there is no NVIDIA driver."""
    for name in _LIBRARY_NAMES:
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        try:
            _configure_prototypes(lib)
            if lib.nvmlInit_v2() != _NVML_SUCCESS:
                return None
        except Exception as exc:
            InvokeAILogger.get_logger(__name__).debug(f"NVML unusable: {exc}")
            return None
        return lib
    return None


def _handle_for(lib: ctypes.CDLL, device: torch.device) -> Optional[ctypes.c_void_p]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    with _lock:
        if index not in _handles:
            handle = ctypes.c_void_p()
            found = lib.nvmlDeviceGetHandleByPciBusId_v2(pci_address(index).encode(), ctypes.byref(handle))
            _handles[index] = handle if found == _NVML_SUCCESS else None
        return _handles[index]


def process_used_memory(device: torch.device) -> Optional[int]:
    """Bytes NVML attributes to this process on `device`: everything the process holds there, the CUDA context
    included. ``None`` off CUDA builds, without NVML, and wherever the driver does not attribute memory to processes."""
    if device.type != "cuda" or torch.version.hip is not None:
        return None
    try:
        lib = _load()
        if lib is None:
            return None
        handle = _handle_for(lib, device)
        if handle is None:
            return None
        count = ctypes.c_uint(0)
        status = lib.nvmlDeviceGetComputeRunningProcesses_v3(handle, ctypes.byref(count), None)
        if status != _NVML_ERROR_INSUFFICIENT_SIZE:
            # Success with an empty buffer means no process at all, so not this one either.
            return None
        count = ctypes.c_uint(count.value + _SPARE_PROCESS_ENTRIES)
        infos = (_ProcessInfo * count.value)()
        if lib.nvmlDeviceGetComputeRunningProcesses_v3(handle, ctypes.byref(count), infos) != _NVML_SUCCESS:
            return None
        # NVML lists host PIDs: in a container without the host's PID namespace no entry is this process (unknown).
        pid = os.getpid()
        for info in infos[: count.value]:
            if info.pid == pid:
                return None if info.usedGpuMemory == _NVML_VALUE_NOT_AVAILABLE else int(info.usedGpuMemory)
        return None
    except Exception:
        InvokeAILogger.get_logger(__name__).debug(f"NVML process memory lookup for {device} failed", exc_info=True)
        return None


def driver_version() -> Optional[str]:
    """The NVIDIA driver version, e.g. ``"581.57"``, or None without NVML."""
    try:
        lib = _load()
        if lib is None:
            return None
        buffer = ctypes.create_string_buffer(_DRIVER_VERSION_BUFFER_SIZE)
        if lib.nvmlSystemGetDriverVersion(buffer, _DRIVER_VERSION_BUFFER_SIZE) != _NVML_SUCCESS:
            return None
        return buffer.value.decode("ascii", errors="replace") or None
    except Exception:
        return None


def reset_cache() -> None:
    """Forget the loader and the device handles (tests only). NVML itself stays initialised."""
    with _lock:
        _handles.clear()
    _load.cache_clear()
