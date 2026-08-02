"""Minimal Level Zero queries for properties torch does not surface on XPU.

torch's ``XPUDeviceProp`` exposes ``architecture`` and ``gpu_eu_count`` but nothing that
distinguishes an integrated GPU from a discrete one. Level Zero does, via
``ZE_DEVICE_PROPERTY_FLAG_INTEGRATED``, and its loader is already present wherever
``torch+xpu`` is installed (it arrives with the ``intel-sycl-rt`` runtime dependency), so
this needs no new package and no compiled extension.

Everything here is best-effort: any failure yields ``None`` ("unknown") rather than raising,
and callers must keep their existing behaviour when the answer is unknown.
"""

import ctypes
import ctypes.util
import threading
from typing import Optional

import torch

from invokeai.backend.util.logging import InvokeAILogger

ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x3
ZE_DEVICE_PROPERTY_FLAG_INTEGRATED = 1 << 0
ZE_MAX_DEVICE_NAME = 256
ZE_MAX_DEVICE_UUID_SIZE = 16

_LOADER_NAMES = ("ze_loader", "libze_loader.so.1", "ze_loader.dll")

_lock = threading.Lock()
_integrated_by_index: Optional[dict[int, bool]] = None
_probe_attempted = False


class _ZeDeviceUUID(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint8 * ZE_MAX_DEVICE_UUID_SIZE)]


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


def reset_cache() -> None:
    """Clear the memoised probe result (tests only)."""
    global _integrated_by_index, _probe_attempted
    with _lock:
        _integrated_by_index = None
        _probe_attempted = False
