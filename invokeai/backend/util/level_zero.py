"""Minimal Level Zero queries for facts torch does not surface on XPU.

Two things are read here, both through the Level Zero loader that is already present wherever
``torch+xpu`` is installed (it arrives with the ``intel-sycl-rt`` runtime dependency) -- so no
new package and no compiled extension:

* **Is a device integrated?** torch's ``XPUDeviceProp`` exposes ``architecture`` and
  ``gpu_eu_count`` but nothing equivalent to ``ZE_DEVICE_PROPERTY_FLAG_INTEGRATED``.
* **Driver-global free VRAM.** ``torch.xpu.mem_get_info()`` is the preferred source, but it
  requires the SYCL ``ext_intel_free_memory`` aspect, which is missing on a number of driver
  and kernel combinations (passthrough VMs, WSL2). Sysman's ``zesMemoryGetState`` reports the
  same driver-global figure and is often available when the aspect is not.

Note that Sysman is not a guaranteed substitute: torch's query bottoms out in the same layer
(a failing ``mem_get_info`` frequently reports "SysMan support is unavailable"), so where
Sysman itself is broken this returns ``None`` too. It is strictly better than the caller's
process-local estimate and never worse.

Everything here is best-effort: any failure yields ``None`` ("unknown") rather than raising,
and callers must keep their existing behaviour when the answer is unknown.
"""

import ctypes
import ctypes.util
import os
import threading
from typing import Optional

import torch

from invokeai.backend.util.logging import InvokeAILogger

ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x3
ZE_DEVICE_PROPERTY_FLAG_INTEGRATED = 1 << 0
ZE_MAX_DEVICE_NAME = 256
ZE_MAX_DEVICE_UUID_SIZE = 16
ZES_STRUCTURE_TYPE_MEM_STATE = 0x1E

_LOADER_NAMES = ("ze_loader", "libze_loader.so.1", "ze_loader.dll")

_lock = threading.Lock()
_integrated_by_index: Optional[dict[int, bool]] = None
_probe_attempted = False

_sysman_lock = threading.Lock()
_sysman_lib: Optional[ctypes.CDLL] = None
_sysman_modules: Optional[dict[int, list[ctypes.c_void_p]]] = None
_sysman_attempted = False


class _ZeDeviceUUID(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint8 * ZE_MAX_DEVICE_UUID_SIZE)]


class _ZesMemState(ctypes.Structure):
    """``zes_mem_state_t``. Five fields, unchanged since Level Zero 1.0."""

    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("health", ctypes.c_uint32),
        ("free", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
    ]


class _ZeDeviceProperties(ctypes.Structure):
    """``ze_device_properties_t`` (Level Zero 1.x). Field order is API-stable."""

    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("type", ctypes.c_uint32),
        ("vendorId", ctypes.c_uint32),
        ("deviceId", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("subdeviceId", ctypes.c_uint32),
        ("coreClockRate", ctypes.c_uint32),
        ("maxMemAllocSize", ctypes.c_uint64),
        ("maxHardwareContexts", ctypes.c_uint32),
        ("maxCommandQueuePriority", ctypes.c_uint32),
        ("numThreadsPerEU", ctypes.c_uint32),
        ("physicalEUSimdWidth", ctypes.c_uint32),
        ("numEUsPerSubslice", ctypes.c_uint32),
        ("numSubslicesPerSlice", ctypes.c_uint32),
        ("numSlices", ctypes.c_uint32),
        ("timerResolution", ctypes.c_uint64),
        ("timestampValidBits", ctypes.c_uint32),
        ("kernelTimestampValidBits", ctypes.c_uint32),
        ("uuid", _ZeDeviceUUID),
        ("name", ctypes.c_char * ZE_MAX_DEVICE_NAME),
    ]


def _load_loader() -> Optional[ctypes.CDLL]:
    for name in _LOADER_NAMES:
        path = ctypes.util.find_library(name) or name
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


def _enumerate(lib: ctypes.CDLL, fn_name: str, parent: Optional[ctypes.c_void_p] = None) -> list[ctypes.c_void_p]:
    """Level Zero's two-call enumeration: ask for the count, then fill an array."""
    fn = getattr(lib, fn_name)
    count = ctypes.c_uint32(0)
    head = [parent] if parent is not None else []
    if fn(*head, ctypes.byref(count), None) != 0:
        raise RuntimeError(f"{fn_name} failed")
    if count.value == 0:
        return []
    arr = (ctypes.c_void_p * count.value)()
    if fn(*head, ctypes.byref(count), arr) != 0:
        raise RuntimeError(f"{fn_name} failed")
    return list(arr[: count.value])


def _probe_integrated_flags() -> Optional[dict[int, bool]]:
    lib = _load_loader()
    if lib is None:
        return None
    try:
        if lib.zeInit(0) != 0:
            return None
        devices: list[ctypes.c_void_p] = []
        for driver in _enumerate(lib, "zeDriverGet"):
            devices += _enumerate(lib, "zeDeviceGet", driver)

        # Level Zero's enumeration order is only meaningfully comparable to torch's xpu:N
        # ordering when both see the same set. If the counts disagree (ZE_AFFINITY_MASK,
        # a flat/composite tile hierarchy, a non-Intel L0 driver in the list), decline to
        # answer rather than risk mislabelling a device.
        if len(devices) != torch.xpu.device_count():
            return None

        flags: dict[int, bool] = {}
        for index, device in enumerate(devices):
            props = _ZeDeviceProperties()
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
            if lib.zeDeviceGetProperties(device, ctypes.byref(props)) != 0:
                return None
            flags[index] = bool(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED)
        return flags
    except Exception as exc:
        InvokeAILogger.get_logger(__name__).debug(f"Level Zero integrated-GPU probe failed: {exc}")
        return None


def xpu_device_is_integrated(device: torch.device) -> Optional[bool]:
    """Return True/False if ``device`` is an integrated GPU, or None if it cannot be determined.

    Integrated GPUs share memory with the CPU, so they should not be treated as having
    dedicated VRAM, and ``generation_devices: auto`` should not pair one with a discrete card.
    """
    global _integrated_by_index, _probe_attempted

    if device.type != "xpu":
        return False
    index = device.index
    if index is None:
        return None

    with _lock:
        if not _probe_attempted:
            _probe_attempted = True
            _integrated_by_index = _probe_integrated_flags()
        flags = _integrated_by_index

    if flags is None:
        return None
    return flags.get(index)


def _init_sysman() -> tuple[Optional[ctypes.CDLL], Optional[dict[int, list[ctypes.c_void_p]]]]:
    """Initialise Sysman and cache each device's memory-module handles.

    Handles stay valid for the life of the process, so only the per-call state query
    (``zesMemoryGetState``) has to be repeated.
    """
    lib = _load_loader()
    if lib is None:
        return None, None
    try:
        # Older runtimes gate Sysman behind this; newer ones use zesInit and ignore it. Setting it
        # only helps if Level Zero has not initialised yet, hence setdefault rather than assignment.
        os.environ.setdefault("ZES_ENABLE_SYSMAN", "1")
        if lib.zesInit(0) != 0:
            return None, None
        devices: list[ctypes.c_void_p] = []
        for driver in _enumerate(lib, "zesDriverGet"):
            devices += _enumerate(lib, "zesDeviceGet", driver)
        # Same ordering caveat as the integrated-flag probe: decline rather than mislabel.
        if len(devices) != torch.xpu.device_count():
            return None, None
        modules: dict[int, list[ctypes.c_void_p]] = {}
        for index, device in enumerate(devices):
            handles = _enumerate(lib, "zesDeviceEnumMemoryModules", device)
            if not handles:
                return None, None
            modules[index] = handles
        return lib, modules
    except Exception as exc:
        InvokeAILogger.get_logger(__name__).debug(f"Level Zero Sysman init failed: {exc}")
        return None, None


def xpu_memory_info(device: torch.device) -> Optional[tuple[int, int]]:
    """Return driver-global ``(free, total)`` VRAM in bytes, or None if Sysman cannot answer.

    Unlike ``total_memory - memory_reserved()``, this accounts for VRAM held by other processes
    on the same device, which is what the model cache's budgeting arithmetic assumes.
    """
    global _sysman_lib, _sysman_modules, _sysman_attempted

    if device.type != "xpu":
        return None
    index = device.index
    if index is None:
        # Resolve an index-less device the same way TorchDevice.normalize does. Callers currently
        # always pass a concrete device (tensor.device and the cache's execution device are both
        # indexed), but silently returning None here would skip the driver-global query and drop
        # straight to the blind estimate -- a quiet accuracy regression rather than a visible one.
        try:
            index = torch.xpu.current_device()
        except Exception:
            return None

    with _sysman_lock:
        if not _sysman_attempted:
            _sysman_attempted = True
            _sysman_lib, _sysman_modules = _init_sysman()
        lib, modules = _sysman_lib, _sysman_modules

    if lib is None or modules is None:
        return None
    handles = modules.get(index)
    if not handles:
        return None

    free = 0
    total = 0
    for handle in handles:
        state = _ZesMemState()
        state.stype = ZES_STRUCTURE_TYPE_MEM_STATE
        try:
            if lib.zesMemoryGetState(handle, ctypes.byref(state)) != 0:
                return None
        except Exception:
            return None
        free += int(state.free)
        total += int(state.size)

    if total <= 0:
        return None
    return (free, total)


def reset_cache() -> None:
    """Clear the memoised probe results (tests only)."""
    global _integrated_by_index, _probe_attempted, _sysman_lib, _sysman_modules, _sysman_attempted
    with _lock:
        _integrated_by_index = None
        _probe_attempted = False
    with _sysman_lock:
        _sysman_lib = None
        _sysman_modules = None
        _sysman_attempted = False
