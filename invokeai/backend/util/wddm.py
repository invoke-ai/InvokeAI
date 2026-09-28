"""The Windows video-memory budget of a ROCm device, read from the WDDM kernel thunks in gdi32.

On a ROCm build under Windows, ``torch.cuda.mem_get_info`` reports the device total minus this process's own usage. It
does not see other processes, and it keeps reporting free memory after Windows has started to page this process's
allocations out -- Windows does not fail an allocation that exceeds what it will keep resident, it places it in shared
system memory, silently and for as long as the allocation lives. What it will keep resident is the per-process budget
``D3DKMTQueryVideoMemoryInfo`` reports (the figure ``IDXGIAdapter3::QueryVideoMemoryInfo`` returns, without COM).

Measured on an RX 9060 XT (16 GB, torch 2.12+rocm7.14): alone, the budget is 15.09 of 15.92 GiB and paging starts
once the committed local usage passes budget + 0.38 GiB; next to another GPU process, Windows lowers the budget within
about a second. What this process can still allocate before paging is the budget minus its live allocations -- the
device total minus torch's free figure. Not minus ``CurrentUsage``: that also counts memory HIP keeps after a free and
hands back to the next allocation (1-2 GiB measured, still held after ``empty_cache()``), so it would hide what an
offload just freed.

Whether Windows has actually paged this process out is not visible through D3DKMT (``CurrentUsage`` counts paged bytes
as local); it is visible in the ``GPU Process Memory`` performance counters Task Manager shows, read here through PDH.

Everything here is best-effort: any failure yields ``None`` ("unknown") rather than raising, and callers must keep their
existing behaviour when the answer is unknown.
"""

import ctypes
import functools
import os
import sys
import threading
from dataclasses import dataclass
from typing import Optional

import torch

from invokeai.backend.util.logging import InvokeAILogger

_KMTQAITYPE_ADAPTERADDRESS = 6
_MEMORY_SEGMENT_GROUP_LOCAL = 0
_PDH_FMT_LARGE = 0x00000400
_PDH_MORE_DATA = 0x800007D2
# The English counter path; PdhAddEnglishCounterW resolves it on a localized Windows too.
_PDH_SHARED_USAGE_COUNTER = r"\GPU Process Memory(*)\Shared Usage"
# PDH_CSTATUS_VALID_DATA and PDH_CSTATUS_NEW_DATA; any other status leaves `largeValue` undefined.
_PDH_VALID_DATA = (0x00000000, 0x00000001)


# Win32 types spelled as fixed-width ctypes, so this module imports on every platform.
_UINT = ctypes.c_uint32
_ULONG = ctypes.c_uint32
_LONG = ctypes.c_int32
_BOOL = ctypes.c_int32


class _Luid(ctypes.Structure):
    _fields_ = [("LowPart", _UINT), ("HighPart", _LONG)]


class _AdapterInfo(ctypes.Structure):
    """``D3DKMT_ADAPTERINFO``."""

    _fields_ = [
        ("hAdapter", _UINT),
        ("AdapterLuid", _Luid),
        ("NumOfSources", _ULONG),
        ("bPrecisePresentRegionsPreferred", _BOOL),
    ]


class _EnumAdapters2(ctypes.Structure):
    """``D3DKMT_ENUMADAPTERS2``."""

    _fields_ = [("NumAdapters", _ULONG), ("pAdapters", ctypes.POINTER(_AdapterInfo))]


class _QueryAdapterInfo(ctypes.Structure):
    """``D3DKMT_QUERYADAPTERINFO``."""

    _fields_ = [
        ("hAdapter", _UINT),
        ("Type", ctypes.c_int),
        ("pPrivateDriverData", ctypes.c_void_p),
        ("PrivateDriverDataSize", _UINT),
    ]


class _AdapterAddress(ctypes.Structure):
    """``D3DKMT_ADAPTERADDRESS``: the adapter's PCI location."""

    _fields_ = [("BusNumber", _UINT), ("DeviceNumber", _UINT), ("FunctionNumber", _UINT)]


class _CloseAdapter(ctypes.Structure):
    """``D3DKMT_CLOSEADAPTER``."""

    _fields_ = [("hAdapter", _UINT)]


class _QueryVideoMemoryInfo(ctypes.Structure):
    """``D3DKMT_QUERYVIDEOMEMORYINFO``. A null ``hProcess`` means the calling process."""

    _fields_ = [
        ("hProcess", ctypes.c_void_p),
        ("hAdapter", _UINT),
        ("MemorySegmentGroup", ctypes.c_int),
        ("Budget", ctypes.c_uint64),
        ("CurrentUsage", ctypes.c_uint64),
        ("CurrentReservation", ctypes.c_uint64),
        ("AvailableForReservation", ctypes.c_uint64),
        ("PhysicalAdapterIndex", _UINT),
    ]


class _PdhCounterValue(ctypes.Structure):
    """``PDH_FMT_COUNTERVALUE`` read as ``PDH_FMT_LARGE``."""

    _fields_ = [("CStatus", ctypes.c_uint32), ("largeValue", ctypes.c_int64)]


class _PdhCounterValueItem(ctypes.Structure):
    """``PDH_FMT_COUNTERVALUE_ITEM_W``."""

    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _PdhCounterValue)]


_GDI32_FUNCTIONS = (
    "D3DKMTEnumAdapters2",
    "D3DKMTQueryAdapterInfo",
    "D3DKMTQueryVideoMemoryInfo",
    "D3DKMTCloseAdapter",
)


@dataclass(frozen=True)
class _Adapter:
    handle: int
    luid: int
    total_bytes: int


_lock = threading.Lock()
# Per torch device index: the matching WDDM adapter, or None once resolution has failed.
_adapters: dict[int, Optional[_Adapter]] = {}

_pdh_lock = threading.Lock()
# One PDH query per counter path, opened on first use: path -> (query, counter) or None when that counter is absent.
# Per counter so that a counter this build does not have disables only what reads it.
_pdh_queries: dict[str, Optional[tuple[ctypes.c_void_p, ctypes.c_void_p]]] = {}


def _supported(device: torch.device) -> bool:
    return sys.platform == "win32" and torch.version.hip is not None and device.type == "cuda"


@functools.lru_cache(maxsize=1)
def _load_gdi32() -> Optional[ctypes.CDLL]:
    """Open gdi32 once and declare every thunk as taking a struct pointer and returning an NTSTATUS."""
    try:
        lib = ctypes.WinDLL("gdi32")
        for name in _GDI32_FUNCTIONS:
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = ctypes.c_long
    except (AttributeError, OSError) as exc:
        InvokeAILogger.get_logger(__name__).debug(f"WDDM thunks unavailable: {exc}")
        return None
    return lib


@functools.lru_cache(maxsize=1)
def _load_pdh() -> Optional[ctypes.CDLL]:
    """Open pdh once and declare the few entry points used; every one returns a PDH_STATUS."""
    try:
        lib = ctypes.WinDLL("pdh")
        prototypes = {
            "PdhOpenQueryW": [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
            "PdhAddEnglishCounterW": [
                ctypes.c_void_p,
                ctypes.c_wchar_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
            ],
            "PdhCollectQueryData": [ctypes.c_void_p],
            "PdhGetFormattedCounterArrayW": [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_void_p,
            ],
            "PdhCloseQuery": [ctypes.c_void_p],
        }
        for name, argtypes in prototypes.items():
            fn = getattr(lib, name)
            fn.argtypes = argtypes
            fn.restype = ctypes.c_uint32
    except (AttributeError, OSError) as exc:
        InvokeAILogger.get_logger(__name__).debug(f"PDH unavailable: {exc}")
        return None
    return lib


def _close(lib: ctypes.CDLL, handle: int) -> None:
    try:
        lib.D3DKMTCloseAdapter(ctypes.byref(_CloseAdapter(handle)))
    except Exception:
        pass


def _resolve_adapter(lib: ctypes.CDLL, index: int) -> Optional[_Adapter]:
    """Open the WDDM adapter at the torch device's PCI location, closing every other handle the enumeration opened.

    ROCm's device properties carry no LUID, so the PCI bus and device numbers are the link. Anything but exactly one
    match is treated as unknown rather than guessed.
    """
    props = torch.cuda.get_device_properties(index)
    location = (int(props.pci_bus_id), int(props.pci_device_id))

    enum = _EnumAdapters2(0, None)
    if lib.D3DKMTEnumAdapters2(ctypes.byref(enum)) != 0:
        return None
    infos = (_AdapterInfo * enum.NumAdapters)()
    enum.pAdapters = infos
    if lib.D3DKMTEnumAdapters2(ctypes.byref(enum)) != 0:
        return None

    matches: list[_AdapterInfo] = []
    pending = list(infos[: enum.NumAdapters])
    current: list[_AdapterInfo] = []
    try:
        while pending:
            # Held in `current` while it is processed: popped from `pending` and not yet in `matches`, a handle whose
            # query raises would otherwise be the one handle the cleanup below misses.
            current = [pending.pop()]
            info = current[0]
            address = _AdapterAddress()
            query = _QueryAdapterInfo(
                info.hAdapter,
                _KMTQAITYPE_ADAPTERADDRESS,
                ctypes.cast(ctypes.byref(address), ctypes.c_void_p),
                ctypes.sizeof(address),
            )
            if (
                lib.D3DKMTQueryAdapterInfo(ctypes.byref(query)) == 0
                and (address.BusNumber, address.DeviceNumber) == location
            ):
                matches.append(info)
            else:
                _close(lib, info.hAdapter)
            current = []
    except Exception:
        # Every handle the enumeration opened belongs to this process until it is closed: the ones matched so far, the
        # one being processed, and the ones this loop never reached.
        for info in matches + current + pending:
            _close(lib, info.hAdapter)
        raise

    if len(matches) != 1:
        for info in matches:
            _close(lib, info.hAdapter)
        return None
    luid = (matches[0].AdapterLuid.HighPart << 32) | matches[0].AdapterLuid.LowPart
    return _Adapter(handle=matches[0].hAdapter, luid=luid, total_bytes=int(props.total_memory))


def _adapter_for(device: torch.device) -> Optional[tuple[ctypes.CDLL, _Adapter]]:
    lib = _load_gdi32()
    if lib is None:
        return None
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
    except Exception:
        return None
    with _lock:
        if index not in _adapters:
            try:
                _adapters[index] = _resolve_adapter(lib, index)
            except Exception as exc:
                InvokeAILogger.get_logger(__name__).debug(f"WDDM adapter lookup for {device} failed: {exc}")
                _adapters[index] = None
        adapter = _adapters[index]
    return None if adapter is None else (lib, adapter)


def local_video_memory(device: torch.device) -> Optional[tuple[int, int]]:
    """``(budget, current_usage)`` for this process on the device's adapter, both driver-side figures in bytes.

    ``None`` off Windows ROCm, and whenever the driver cannot answer or answers something implausible. Called per
    generation and per decode, so every failure answers "unknown" here rather than reaching those callers.
    """
    if not _supported(device):
        return None
    try:
        resolved = _adapter_for(device)
        if resolved is None:
            return None
        lib, adapter = resolved
        info = _QueryVideoMemoryInfo(None, adapter.handle, _MEMORY_SEGMENT_GROUP_LOCAL)
        if lib.D3DKMTQueryVideoMemoryInfo(ctypes.byref(info)) != 0:
            return None
        if not 0 < info.Budget <= adapter.total_bytes:
            return None
        # Clamped, not rejected: `CurrentUsage` counts bytes Windows has already paged out, so it can exceed the
        # adapter's own size exactly when the process over-commits -- and dropping the pair there would also take the
        # budget away from the model cache's free-VRAM cap, which is the caller that needs it most in that state.
        return int(info.Budget), max(0, min(int(info.CurrentUsage), adapter.total_bytes))
    except Exception:
        InvokeAILogger.get_logger(__name__).debug(f"WDDM memory lookup for {device} failed", exc_info=True)
        return None


def video_memory_budget(device: torch.device) -> Optional[int]:
    """This process's WDDM budget on the device's adapter, in bytes: what Windows keeps resident before paging."""
    local = local_video_memory(device)
    return None if local is None else local[0]


def _open_counter_query(pdh: ctypes.CDLL, path: str) -> Optional[tuple[ctypes.c_void_p, ctypes.c_void_p]]:
    query = ctypes.c_void_p()
    if pdh.PdhOpenQueryW(None, None, ctypes.byref(query)) != 0:
        return None
    counter = ctypes.c_void_p()
    if pdh.PdhAddEnglishCounterW(query, path, None, ctypes.byref(counter)) != 0:
        pdh.PdhCloseQuery(query)
        return None
    return query, counter


def _counter_values(path: str) -> dict[str, int]:
    """That counter's value per instance, collected once. ``{}`` when the counter or the loader is unavailable."""
    pdh = _load_pdh()
    if pdh is None:
        return {}
    with _pdh_lock:
        try:
            if path not in _pdh_queries:
                _pdh_queries[path] = _open_counter_query(pdh, path)
            opened = _pdh_queries[path]
            if opened is None:
                return {}
            return _usage_by_instance(pdh, *opened)
        except Exception:
            return {}


def _usage_by_instance(pdh: ctypes.CDLL, query: ctypes.c_void_p, counter: ctypes.c_void_p) -> dict[str, int]:
    """Collect the query once; that counter's value per instance, keyed by its lower-case instance name."""
    if pdh.PdhCollectQueryData(query) != 0:
        return {}
    size, count = ctypes.c_uint32(0), ctypes.c_uint32(0)
    status = pdh.PdhGetFormattedCounterArrayW(counter, _PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), None)
    if status != _PDH_MORE_DATA:
        return {}
    buffer = (ctypes.c_byte * size.value)()
    status = pdh.PdhGetFormattedCounterArrayW(counter, _PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), buffer)
    if status != 0:
        return {}
    items = ctypes.cast(buffer, ctypes.POINTER(_PdhCounterValueItem))
    return {
        items[i].szName.lower(): int(items[i].FmtValue.largeValue)
        for i in range(count.value)
        if items[i].szName and items[i].FmtValue.CStatus in _PDH_VALID_DATA
    }


def paged_bytes(device: torch.device) -> Optional[int]:
    """Bytes of this process's memory on the device's adapter that Windows currently keeps in shared system memory.

    Allocations land there when they did not fit the budget or found no contiguous VRAM, and they stay there; a few
    hundred MB also appear briefly while weights are copied in. ``None`` off Windows ROCm and whenever the counters
    cannot answer, including before this process has used the adapter at all.

    The first call opens this counter's PDH query (~0.3 s); each later call collects it once (~0.3 ms).
    """
    if not _supported(device):
        return None
    resolved = _adapter_for(device)
    if resolved is None:
        return None
    instance = f"pid_{os.getpid()}_{_adapter_instance(resolved[1].luid)}"
    return _counter_values(_PDH_SHARED_USAGE_COUNTER).get(instance)


def _adapter_instance(luid: int) -> str:
    return f"luid_0x{(luid >> 32) & 0xFFFFFFFF:08x}_0x{luid & 0xFFFFFFFF:08x}_phys_0"


def reset_cache() -> None:
    """Close the cached adapter handles and the PDH query, and forget the loaders (tests only)."""
    with _lock:
        opened = [adapter for adapter in _adapters.values() if adapter is not None]
        _adapters.clear()
    lib = _load_gdi32() if opened else None
    if lib is not None:
        for adapter in opened:
            _close(lib, adapter.handle)
    _load_gdi32.cache_clear()

    with _pdh_lock:
        opened_queries = [opened for opened in _pdh_queries.values() if opened is not None]
        _pdh_queries.clear()
    pdh = _load_pdh() if opened_queries else None
    if pdh is not None:
        for query, _counter in opened_queries:
            pdh.PdhCloseQuery(query)
    _load_pdh.cache_clear()
