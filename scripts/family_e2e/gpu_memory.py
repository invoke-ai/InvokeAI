"""Driver-level video memory of another process, sampled without a GPU context of our own.

The allocator's own figure (``torch.cuda.max_memory_allocated``) misses its pools and the cuDNN/MIOpen workspaces, so
the real peak runs 25-75 % above it. What the driver charges the process is what decides whether a card of a given size
copes, and Windows reports it per process and adapter through the WDDM kernel thunks: ``D3DKMTQueryVideoMemoryInfo``
on another process's handle answers in about a microsecond, so the dedicated (local segment) usage can be sampled every
few milliseconds. The bytes Windows has paged out to system memory are only visible in the ``GPU Process Memory``
performance counters (``Shared Usage``), read through PDH at a slower rate.

The client never imports torch: a CUDA/HIP context of its own would sit on the same adapter and distort what the
server can allocate. Windows (WDDM) only for now; it covers NVIDIA and AMD/ROCm alike.
"""

import bisect
import ctypes
import functools
import logging
import operator
import re
import sys
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

_STATUS_SUCCESS = 0x00000000
# The target has no allocation context on the adapter: it never opened it, or has closed it again.
_STATUS_INVALID_PARAMETER = 0xC000000D
_STATUS_ACCESS_DENIED = 0xC0000022
# D3DKMTQueryVideoMemoryInfo answers STATUS_ACCESS_DENIED for PROCESS_QUERY_LIMITED_INFORMATION; it needs the full
# query right (measured on Windows 10 22H2 with an NVIDIA and an AMD driver).
_PROCESS_QUERY_INFORMATION = 0x0400
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5
_ERROR_INVALID_PARAMETER = 87

_KMTQAITYPE_ADAPTERREGISTRYINFO = 8
_KMTQAITYPE_ADAPTERTYPE = 15
# D3DKMT_KMD_DRIVER_VERSION: the kernel-mode driver's version, the one Device Manager shows.
_KMTQAITYPE_KMD_DRIVER_VERSION = 49
# D3DKMT_ADAPTERTYPE bits.
_ADAPTER_RENDER_SUPPORTED = 1 << 0
_ADAPTER_SOFTWARE_DEVICE = 1 << 2
_ADAPTER_COMPUTE_ONLY = 1 << 11
_MEMORY_SEGMENT_GROUP_LOCAL = 0
_MAX_PATH = 260

_PDH_FMT_LARGE = 0x00000400
_PDH_MORE_DATA = 0x800007D2
_PDH_NO_DATA = 0x800007D5
# PDH_CSTATUS_VALID_DATA and PDH_CSTATUS_NEW_DATA; any other status leaves `largeValue` undefined.
_PDH_VALID_DATA = (0x00000000, 0x00000001)
# English paths; PdhAddEnglishCounterW resolves them on a localized Windows too.
_PDH_DEDICATED_USAGE = r"\GPU Process Memory(*)\Dedicated Usage"
_PDH_SHARED_USAGE = r"\GPU Process Memory(*)\Shared Usage"
_PDH_INSTANCE = re.compile(r"pid_(\d+)_luid_0x([0-9a-f]{8})_0x([0-9a-f]{8})_phys_(\d+)", re.IGNORECASE)

_logger = logging.getLogger(__name__)

# Win32 types spelled as fixed-width ctypes, so this module imports on every platform.
_UINT = ctypes.c_uint32
_LONG = ctypes.c_int32
_NTSTATUS = ctypes.c_uint32


class UnsupportedPlatformError(RuntimeError):
    """GPU memory sampling is not implemented for this operating system."""


@dataclass(frozen=True)
class Adapter:
    luid: int
    """``(HighPart << 32) | LowPart``, as D3DKMT and the PDH instance names spell it."""
    description: str
    driver_version: str | None = None
    """As Device Manager shows it, e.g. ``32.0.16.1047``; ``None`` when the driver does not report one."""


@dataclass(frozen=True)
class ProcessUsage:
    dedicated_bytes: int
    shared_bytes: int


@dataclass(frozen=True)
class MemoryTrace:
    """Samples of one process's usage on one adapter, stamped with the client's ``time.perf_counter()``.

    The trace is a sample-and-hold step function: a value stands from its timestamp until the next sample.
    """

    method: str
    interval_ms: float
    """Mean spacing of the dedicated samples; the configured interval while they span no measurable time."""
    dedicated: tuple[tuple[float, int], ...]
    """Local-segment ``CurrentUsage`` in bytes."""
    shared: tuple[tuple[float, int], ...]
    """PDH ``Shared Usage`` in bytes: what Windows keeps of this process's allocations in system memory."""

    def peak(self, t0: float | None = None, t1: float | None = None) -> int | None:
        """The highest dedicated usage in ``[t0, t1]``: the value in effect at ``t0`` and every sample up to ``t1``.

        Counting the value in effect at ``t0`` keeps a window shorter than the sampling interval from coming back
        empty. ``None`` bounds are open; ``None`` is returned when no sample falls on or before ``t1``.
        """
        if t0 is not None and t1 is not None and t1 < t0:
            raise ValueError(f"window ends before it starts: [{t0}, {t1}]")
        samples = self.dedicated
        lo = 0
        if t0 is not None:
            lo = bisect.bisect_left(samples, t0, key=_sample_time)
            if lo > 0 and (lo == len(samples) or samples[lo][0] > t0):
                lo -= 1
        hi = len(samples) if t1 is None else bisect.bisect_right(samples, t1, key=_sample_time)
        return max((value for _, value in samples[lo:hi]), default=None)

    def at(self, t: float) -> int | None:
        """The dedicated usage in effect at ``t``: the last sample at or before it."""
        index = bisect.bisect_right(self.dedicated, t, key=_sample_time)
        return self.dedicated[index - 1][1] if index else None


_sample_time = operator.itemgetter(0)


class _Luid(ctypes.Structure):
    _fields_ = [("LowPart", _UINT), ("HighPart", _LONG)]


class _AdapterInfo(ctypes.Structure):
    """``D3DKMT_ADAPTERINFO``."""

    _fields_ = [("hAdapter", _UINT), ("AdapterLuid", _Luid), ("NumOfSources", _UINT), ("bPrecise", _LONG)]


class _EnumAdapters2(ctypes.Structure):
    """``D3DKMT_ENUMADAPTERS2``."""

    _fields_ = [("NumAdapters", _UINT), ("pAdapters", ctypes.POINTER(_AdapterInfo))]


class _QueryAdapterInfo(ctypes.Structure):
    """``D3DKMT_QUERYADAPTERINFO``."""

    _fields_ = [("hAdapter", _UINT), ("Type", ctypes.c_int), ("pData", ctypes.c_void_p), ("DataSize", _UINT)]


class _AdapterRegistryInfo(ctypes.Structure):
    """``D3DKMT_ADAPTERREGISTRYINFO``."""

    _fields_ = [
        ("AdapterString", ctypes.c_wchar * _MAX_PATH),
        ("BiosString", ctypes.c_wchar * _MAX_PATH),
        ("DacType", ctypes.c_wchar * _MAX_PATH),
        ("ChipType", ctypes.c_wchar * _MAX_PATH),
    ]


class _OpenAdapterFromLuid(ctypes.Structure):
    """``D3DKMT_OPENADAPTERFROMLUID``."""

    _fields_ = [("AdapterLuid", _Luid), ("hAdapter", _UINT)]


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


def _require_windows() -> None:
    if sys.platform != "win32":
        raise UnsupportedPlatformError(
            f"GPU memory sampling is implemented for Windows (WDDM) only, not {sys.platform}"
        )


@functools.cache
def _win32() -> SimpleNamespace:
    """gdi32, kernel32 and pdh with the prototypes used here; loaded on first use, since only Windows has them."""
    gdi32 = ctypes.WinDLL("gdi32")
    for name in (
        "D3DKMTEnumAdapters2",
        "D3DKMTQueryAdapterInfo",
        "D3DKMTOpenAdapterFromLuid",
        "D3DKMTQueryVideoMemoryInfo",
        "D3DKMTCloseAdapter",
    ):
        fn = getattr(gdi32, name)
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = _NTSTATUS

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int32
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int32

    pdh = ctypes.WinDLL("pdh")
    prototypes = {
        "PdhOpenQueryW": [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
        "PdhAddEnglishCounterW": [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
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
        fn = getattr(pdh, name)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_uint32
    return SimpleNamespace(gdi32=gdi32, kernel32=kernel32, pdh=pdh)


def _check(status: int, call: str) -> None:
    if status != _STATUS_SUCCESS:
        raise OSError(f"{call} failed with status {status:#010x}")


def _luid_value(luid: _Luid) -> int:
    return ((luid.HighPart & 0xFFFFFFFF) << 32) | luid.LowPart


def _luid_struct(value: int) -> _Luid:
    return _Luid(value & 0xFFFFFFFF, value >> 32)


def _query_adapter_info(handle: int, info_type: int, data: ctypes.Structure | ctypes.c_uint32 | ctypes.c_uint64) -> int:
    query = _QueryAdapterInfo(handle, info_type, ctypes.cast(ctypes.byref(data), ctypes.c_void_p), ctypes.sizeof(data))
    return _win32().gdi32.D3DKMTQueryAdapterInfo(ctypes.byref(query))


def _describe(info: _AdapterInfo) -> Adapter | None:
    """The adapter, or ``None`` for one that cannot hold a compute process's memory or cannot be named."""
    adapter_type = _UINT()
    if _query_adapter_info(info.hAdapter, _KMTQAITYPE_ADAPTERTYPE, adapter_type) != _STATUS_SUCCESS:
        return None
    # The Microsoft Basic Render Driver is a software device; display-only adapters (indirect displays) run nothing.
    if adapter_type.value & _ADAPTER_SOFTWARE_DEVICE:
        return None
    if not adapter_type.value & (_ADAPTER_RENDER_SUPPORTED | _ADAPTER_COMPUTE_ONLY):
        return None
    registry = _AdapterRegistryInfo()
    if _query_adapter_info(info.hAdapter, _KMTQAITYPE_ADAPTERREGISTRYINFO, registry) != _STATUS_SUCCESS:
        return None
    version = ctypes.c_uint64()
    status = _query_adapter_info(info.hAdapter, _KMTQAITYPE_KMD_DRIVER_VERSION, version)
    return Adapter(
        luid=_luid_value(info.AdapterLuid),
        description=registry.AdapterString,
        driver_version=_format_driver_version(version.value) if status == _STATUS_SUCCESS else None,
    )


def _format_driver_version(packed: int) -> str | None:
    """A driver version packed as four 16-bit fields, most significant first, in dotted form; 0 is unknown."""
    if not packed:
        return None
    return ".".join(str(packed >> shift & 0xFFFF) for shift in (48, 32, 16, 0))


def list_adapters() -> list[Adapter]:
    """The hardware GPU adapters, in the order the kernel enumerates them."""
    _require_windows()
    gdi32 = _win32().gdi32
    # A null array asks for the size: the call answers the most adapters it could return.
    enum = _EnumAdapters2(0, None)
    _check(gdi32.D3DKMTEnumAdapters2(ctypes.byref(enum)), "D3DKMTEnumAdapters2")
    infos = (_AdapterInfo * enum.NumAdapters)()
    enum.pAdapters = infos
    _check(gdi32.D3DKMTEnumAdapters2(ctypes.byref(enum)), "D3DKMTEnumAdapters2")
    opened = infos[: enum.NumAdapters]
    try:
        return [adapter for adapter in map(_describe, opened) if adapter is not None]
    finally:
        for info in opened:
            gdi32.D3DKMTCloseAdapter(ctypes.byref(_CloseAdapter(info.hAdapter)))


def _select_adapter(adapters: list[Adapter], description_substring: str) -> Adapter:
    needle = description_substring.casefold()
    matches = [adapter for adapter in adapters if needle in adapter.description.casefold()]
    if len(matches) == 1:
        return matches[0]
    available = "; ".join(f"{adapter.description} (LUID {adapter.luid:#x})" for adapter in adapters) or "none"
    found = "No adapter matches" if not matches else f"{len(matches)} adapters match"
    raise LookupError(f"{found} {description_substring!r}. Adapters: {available}")


def find_adapter(description_substring: str) -> Adapter:
    """The one adapter whose description contains the substring, ignoring case. ``LookupError`` for none or several."""
    return _select_adapter(list_adapters(), description_substring)


def _parse_instance(name: str) -> tuple[int, int] | None:
    """``(pid, luid)`` of a ``GPU Process Memory`` instance on physical adapter 0, else ``None``."""
    match = _PDH_INSTANCE.fullmatch(name)
    if match is None or match[4] != "0":
        return None
    return int(match[1]), (int(match[2], 16) << 32) | int(match[3], 16)


class _GpuProcessMemoryCounters:
    """One PDH query over every process's ``GPU Process Memory`` instances.

    Only the first query in a process is slow to open (about 0.35 s while PDH loads the provider); a collection costs
    well under a millisecond.
    """

    def __init__(self) -> None:
        pdh = _win32().pdh
        self._query = ctypes.c_void_p()
        _check(pdh.PdhOpenQueryW(None, None, ctypes.byref(self._query)), "PdhOpenQueryW")
        try:
            self._dedicated = self._add(_PDH_DEDICATED_USAGE)
            self._shared = self._add(_PDH_SHARED_USAGE)
        except BaseException:
            self.close()
            raise

    def _add(self, path: str) -> ctypes.c_void_p:
        counter = ctypes.c_void_p()
        status = _win32().pdh.PdhAddEnglishCounterW(self._query, path, None, ctypes.byref(counter))
        _check(status, f"PdhAddEnglishCounterW({path})")
        return counter

    def collect(self) -> dict[tuple[int, int], ProcessUsage]:
        """Usage keyed by ``(pid, luid)``, for every process that has opened an adapter."""
        status = _win32().pdh.PdhCollectQueryData(self._query)
        if status == _PDH_NO_DATA:
            return {}
        _check(status, "PdhCollectQueryData")
        dedicated = self._values(self._dedicated)
        shared = self._values(self._shared)
        return {
            key: ProcessUsage(dedicated.get(key, 0), shared.get(key, 0)) for key in dedicated.keys() | shared.keys()
        }

    def _values(self, counter: ctypes.c_void_p) -> dict[tuple[int, int], int]:
        pdh = _win32().pdh
        size, count = ctypes.c_uint32(0), ctypes.c_uint32(0)
        status = pdh.PdhGetFormattedCounterArrayW(
            counter, _PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), None
        )
        if status == _STATUS_SUCCESS:
            return {}
        if status != _PDH_MORE_DATA:
            _check(status, "PdhGetFormattedCounterArrayW")
        buffer = (ctypes.c_byte * size.value)()
        status = pdh.PdhGetFormattedCounterArrayW(
            counter, _PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), buffer
        )
        _check(status, "PdhGetFormattedCounterArrayW")
        items = ctypes.cast(buffer, ctypes.POINTER(_PdhCounterValueItem))
        values: dict[tuple[int, int], int] = {}
        for item in items[: count.value]:
            key = _parse_instance(item.szName or "")
            if key is not None and item.FmtValue.CStatus in _PDH_VALID_DATA:
                values[key] = int(item.FmtValue.largeValue)
        return values

    def close(self) -> None:
        if self._query:
            _win32().pdh.PdhCloseQuery(self._query)
            self._query = ctypes.c_void_p()


def process_usage_on_adapter(adapter: Adapter) -> dict[int, ProcessUsage]:
    """Every process that has opened the adapter, by pid, with its dedicated and shared usage."""
    _require_windows()
    counters = _GpuProcessMemoryCounters()
    try:
        usage = counters.collect()
    finally:
        counters.close()
    return {pid: value for (pid, luid), value in usage.items() if luid == adapter.luid}


class ProcessMemorySampler:
    """Samples one process's usage on one adapter on a background thread, from ``start()`` until ``stop()``.

    Dedicated usage is read every ``interval_s``, shared usage every ``shared_interval_s``. A process that has not
    opened the adapter yet reads as zero. Sampling ends early when the process exits.

    ``pid`` must be the process that allocates. On Windows the ``python.exe`` in a venv's ``Scripts`` folder and the
    console-script ``.exe`` files there are launchers: they run the interpreter as a child process and read as zero.
    """

    def __init__(self, pid: int, adapter: Adapter, interval_s: float = 0.003, shared_interval_s: float = 0.1) -> None:
        _require_windows()
        if pid <= 0:
            raise ValueError(f"invalid pid {pid}")
        if interval_s <= 0 or shared_interval_s <= 0:
            raise ValueError(f"sampling intervals must be positive, got {interval_s} and {shared_interval_s}")
        self._pid = pid
        self._adapter = adapter
        self._interval_s = interval_s
        self._shared_interval_s = shared_interval_s
        self._dedicated: list[tuple[float, int]] = []
        self._shared: list[tuple[float, int]] = []
        self._process: int | None = None
        self._adapter_handle: int | None = None
        self._counters: _GpuProcessMemoryCounters | None = None
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._trace: MemoryTrace | None = None
        self._warned: set[str] = set()

    def start(self) -> None:
        """Open the process and the adapter, take one sample and start the sampling thread."""
        if self._thread is not None or self._trace is not None:
            raise RuntimeError("a sampler starts once")
        try:
            self._open()
            self._sample_dedicated(raise_on_failure=True)
            self._sample_shared(raise_on_failure=True)
        except BaseException:
            self._close()
            raise
        self._thread = threading.Thread(target=self._run, name=f"gpu-memory-sampler-{self._pid}", daemon=True)
        self._thread.start()

    def stop(self) -> MemoryTrace:
        """Stop sampling, release the handles and return the trace; later calls return the same trace."""
        if self._trace is None:
            self._stopping.set()
            if self._thread is not None:
                self._thread.join()
            self._close()
            self._trace = MemoryTrace(
                method="d3dkmt",
                interval_ms=self._mean_interval_ms(),
                dedicated=tuple(self._dedicated),
                shared=tuple(self._shared),
            )
        return self._trace

    def _open(self) -> None:
        win32 = _win32()
        process = win32.kernel32.OpenProcess(_PROCESS_QUERY_INFORMATION, 0, self._pid)
        if not process:
            error = ctypes.get_last_error()
            if error == _ERROR_INVALID_PARAMETER:
                raise ProcessLookupError(f"no process with pid {self._pid}")
            if error == _ERROR_ACCESS_DENIED:
                raise PermissionError(f"access denied opening process {self._pid} for memory queries")
            raise ctypes.WinError(error)
        self._process = process
        if self._exited():
            raise ProcessLookupError(f"process {self._pid} has exited")

        opened = _OpenAdapterFromLuid(_luid_struct(self._adapter.luid), 0)
        status = win32.gdi32.D3DKMTOpenAdapterFromLuid(ctypes.byref(opened))
        _check(status, f"D3DKMTOpenAdapterFromLuid for {self._adapter.description} (LUID {self._adapter.luid:#x})")
        self._adapter_handle = opened.hAdapter
        # The target's STATUS_INVALID_PARAMETER only means "no allocation on this adapter" once the same call has
        # succeeded for this process, which proves the adapter handle and the struct layout.
        status, _ = self._query_usage(None)
        _check(status, "D3DKMTQueryVideoMemoryInfo for the sampling process")
        self._counters = _GpuProcessMemoryCounters()

    def _close(self) -> None:
        win32 = _win32()
        if self._counters is not None:
            self._counters.close()
            self._counters = None
        if self._adapter_handle is not None:
            win32.gdi32.D3DKMTCloseAdapter(ctypes.byref(_CloseAdapter(self._adapter_handle)))
            self._adapter_handle = None
        if self._process is not None:
            win32.kernel32.CloseHandle(self._process)
            self._process = None

    def _query_usage(self, process: int | None) -> tuple[int, int]:
        assert self._adapter_handle is not None
        info = _QueryVideoMemoryInfo(process, self._adapter_handle, _MEMORY_SEGMENT_GROUP_LOCAL)
        status = _win32().gdi32.D3DKMTQueryVideoMemoryInfo(ctypes.byref(info))
        return status, int(info.CurrentUsage)

    def _exited(self) -> bool:
        code = ctypes.c_uint32()
        if not _win32().kernel32.GetExitCodeProcess(self._process, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return code.value != _STILL_ACTIVE

    def _sample_dedicated(self, raise_on_failure: bool = False) -> bool:
        """Append one dedicated sample; ``False`` once the process has exited."""
        t = time.perf_counter()
        status, usage = self._query_usage(self._process)
        if status == _STATUS_SUCCESS:
            self._dedicated.append((t, usage))
            return True
        if self._exited():
            if raise_on_failure:
                raise ProcessLookupError(f"process {self._pid} has exited")
            return False
        if status == _STATUS_INVALID_PARAMETER:
            self._dedicated.append((t, 0))
            return True
        if raise_on_failure:
            if status == _STATUS_ACCESS_DENIED:
                raise PermissionError(f"access denied querying video memory of process {self._pid}")
            _check(status, f"D3DKMTQueryVideoMemoryInfo for process {self._pid}")
        self._warn_once("dedicated", f"D3DKMTQueryVideoMemoryInfo failed with status {status:#010x}; sample skipped")
        return True

    def _sample_shared(self, raise_on_failure: bool = False) -> None:
        assert self._counters is not None
        t = time.perf_counter()
        try:
            usage = self._counters.collect().get((self._pid, self._adapter.luid))
        except OSError as error:
            if raise_on_failure:
                raise
            self._warn_once("shared", f"{error}; shared sample skipped")
            return
        self._shared.append((t, 0 if usage is None else usage.shared_bytes))

    def _warn_once(self, series: str, message: str) -> None:
        if series not in self._warned:
            self._warned.add(series)
            _logger.warning(f"GPU memory sampler for process {self._pid}: {message} (further failures not logged)")

    def _run(self) -> None:
        clock = time.perf_counter
        next_dedicated = clock() + self._interval_s
        next_shared = clock() + self._shared_interval_s
        while not self._stopping.is_set():
            delay = next_dedicated - clock()
            if delay > 0:
                time.sleep(delay)
            if not self._sample_dedicated():
                return
            now = clock()
            if now >= next_shared:
                self._sample_shared()
                next_shared = now + self._shared_interval_s
            # A fixed rate: a late wake-up shortens the next sleep instead of delaying every later sample, and after a
            # stall (GIL, suspend) sampling resumes from now rather than bursting to catch up.
            next_dedicated = max(next_dedicated + self._interval_s, now)

    def _mean_interval_ms(self) -> float:
        span = self._dedicated[-1][0] - self._dedicated[0][0] if len(self._dedicated) > 1 else 0.0
        if span <= 0:
            return self._interval_s * 1000
        return span / (len(self._dedicated) - 1) * 1000
