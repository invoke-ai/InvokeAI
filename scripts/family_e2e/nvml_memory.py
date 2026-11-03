"""Driver-level video memory of another process on Linux with an NVIDIA GPU, read through NVML.

The Linux counterpart of `gpu_memory`, with the same interface and types. NVML ships with the NVIDIA driver
(``libnvidia-ml.so.1``) and lists every compute process on a device with ``usedGpuMemory``: what the driver has
allocated for that process, its CUDA context included. One ``nvmlDeviceGetComputeRunningProcesses_v3`` call returns
the whole list, so the sampler reads it at a fixed rate and picks out its process; the one-off listing behind the
foreign-load checks adds the graphics processes. The Linux driver does not page video memory out to system memory,
so there is no shared series.

Like `gpu_memory`, this never imports torch: a CUDA context of the client's own would sit on the same device and
distort what the server can allocate. AMD (ROCm) GPUs on Linux are not supported yet.
"""

import ctypes
import functools
import logging
import sys
import threading
import time

import psutil

from scripts.family_e2e.gpu_memory import Adapter, MemoryTrace, ProcessUsage, UnsupportedPlatformError, _select_adapter

_LIBRARY_NAMES = ("libnvidia-ml.so.1", "libnvidia-ml.so")
_NVML_SUCCESS = 0
_NVML_ERROR_NOT_SUPPORTED = 3
_NVML_ERROR_INSUFFICIENT_SIZE = 7
_NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18
# `usedGpuMemory` where the driver does not account memory per process: vGPU guests, MIG, some containers.
_NVML_VALUE_NOT_AVAILABLE = 0xFFFFFFFFFFFFFFFF
# NVML_DEVICE_NAME_V2_BUFFER_SIZE, NVML_DEVICE_UUID_V2_BUFFER_SIZE and NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE.
_NAME_BUFFER_SIZE = 96
_UUID_BUFFER_SIZE = 96
_DRIVER_VERSION_BUFFER_SIZE = 80
# Room for processes that start between NVML's count and the fill.
_SPARE_PROCESS_ENTRIES = 8
_LIST_ATTEMPTS = 4
_AMD_NOTE = "AMD (ROCm) GPUs on Linux are not supported yet"

_logger = logging.getLogger(__name__)


class _ProcessInfo(ctypes.Structure):
    """``nvmlProcessInfo_t``, the entry ``nvmlDeviceGetComputeRunningProcesses_v3`` fills."""

    _fields_ = [
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    ]


_PROCESS_LIST = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(_ProcessInfo)]
_PROTOTYPES = {
    "nvmlInit_v2": [],
    "nvmlErrorString": [ctypes.c_int],
    "nvmlSystemGetDriverVersion": [ctypes.c_char_p, ctypes.c_uint],
    "nvmlDeviceGetCount_v2": [ctypes.POINTER(ctypes.c_uint)],
    "nvmlDeviceGetHandleByIndex_v2": [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
    "nvmlDeviceGetHandleByUUID": [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)],
    "nvmlDeviceGetName": [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint],
    "nvmlDeviceGetUUID": [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint],
    "nvmlDeviceGetComputeRunningProcesses_v3": _PROCESS_LIST,
    "nvmlDeviceGetGraphicsRunningProcesses_v3": _PROCESS_LIST,
}


def _open_library() -> ctypes.CDLL:
    for name in _LIBRARY_NAMES:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise UnsupportedPlatformError(
        f"GPU memory sampling on {sys.platform} reads NVML, which comes with the NVIDIA driver, but neither "
        f"{' nor '.join(_LIBRARY_NAMES)} loads. {_AMD_NOTE}."
    )


def _declare(lib: ctypes.CDLL) -> None:
    """Declares the prototypes used here, so ctypes passes device handles at full pointer width."""
    for function, argtypes in _PROTOTYPES.items():
        try:
            fn = getattr(lib, function)
        except AttributeError:
            raise UnsupportedPlatformError(f"NVML has no {function}: the NVIDIA driver is too old") from None
        fn.argtypes = argtypes
        fn.restype = ctypes.c_char_p if function == "nvmlErrorString" else ctypes.c_int


@functools.cache
def _nvml() -> ctypes.CDLL:
    """NVML, initialised on first use and never shut down: every later call and sampler thread shares the one init."""
    lib = _open_library()
    _declare(lib)
    status = lib.nvmlInit_v2()
    if status == _NVML_ERROR_LIB_RM_VERSION_MISMATCH:
        raise UnsupportedPlatformError(
            "NVML does not match the loaded NVIDIA kernel module (driver/library version mismatch), as after a driver "
            "upgrade until the machine restarts; reboot, then try again"
        )
    if status != _NVML_SUCCESS:
        raise UnsupportedPlatformError(f"nvmlInit_v2 failed: {_error_string(lib, status)} (NVML error {status})")
    return lib


def _error_string(lib: ctypes.CDLL, status: int) -> str:
    text = lib.nvmlErrorString(status)
    return text.decode("utf-8", errors="replace") if text else "unknown error"


def _check(lib: ctypes.CDLL, status: int, call: str) -> None:
    if status != _NVML_SUCCESS:
        raise OSError(f"{call} failed: {_error_string(lib, status)} (NVML error {status})")


def _text(buffer: ctypes.Array[ctypes.c_char]) -> str:
    return buffer.value.decode("utf-8", errors="replace")


def list_adapters() -> list[Adapter]:
    """The NVIDIA GPUs NVML sees, in its enumeration order.

    A GPU NVML cannot query (no permission in a cgroup, lost after an Xid error) is left out with a warning, so the
    others stay usable.
    """
    lib = _nvml()
    driver = ctypes.create_string_buffer(_DRIVER_VERSION_BUFFER_SIZE)
    _check(lib, lib.nvmlSystemGetDriverVersion(driver, len(driver)), "nvmlSystemGetDriverVersion")
    count = ctypes.c_uint()
    _check(lib, lib.nvmlDeviceGetCount_v2(ctypes.byref(count)), "nvmlDeviceGetCount_v2")
    adapters = []
    for index in range(count.value):
        try:
            adapters.append(_describe(lib, index, _text(driver) or None))
        except OSError as error:
            _logger.warning(f"NVIDIA GPU {index} left out: {error}")
    return adapters


def _describe(lib: ctypes.CDLL, index: int, driver_version: str | None) -> Adapter:
    handle = ctypes.c_void_p()
    _check(
        lib, lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)), f"nvmlDeviceGetHandleByIndex_v2({index})"
    )
    name = ctypes.create_string_buffer(_NAME_BUFFER_SIZE)
    _check(lib, lib.nvmlDeviceGetName(handle, name, len(name)), f"nvmlDeviceGetName({index})")
    uuid = ctypes.create_string_buffer(_UUID_BUFFER_SIZE)
    _check(lib, lib.nvmlDeviceGetUUID(handle, uuid, len(uuid)), f"nvmlDeviceGetUUID({index})")
    return Adapter(luid=None, uuid=_text(uuid), description=_text(name), driver_version=driver_version)


def find_adapter(name_or_uuid: str) -> Adapter:
    """The NVIDIA GPU with this UUID, else the one whose name contains the text, ignoring case.

    ``LookupError`` for none or several; the message lists every GPU with its UUID. A UUID tells identical cards
    apart, e.g. two runs side by side, one per GPU; the runner pins each server to its UUID.
    """
    adapters = list_adapters()
    for adapter in adapters:
        if adapter.uuid is not None and adapter.uuid.casefold() == name_or_uuid.casefold():
            return adapter
    return _select_adapter(
        adapters,
        name_or_uuid,
        no_match_hint=f"NVML lists NVIDIA GPUs only; {_AMD_NOTE}",
        several_hint="set gpu to the UUID of one of them; the runner pins the server to that GPU",
    )


def _not_attributed(adapter: Adapter) -> UnsupportedPlatformError:
    return UnsupportedPlatformError(
        f"NVML does not attribute memory to processes on {adapter.description} (usedGpuMemory is not available, as "
        "in vGPU guests, MIG instances and some containers); sampling needs a GPU the driver accounts per process"
    )


class _ProcessTable:
    """The compute (or graphics) processes on one GPU, read into entries kept between reads: one NVML call while they
    fit."""

    def __init__(self, adapter: Adapter, graphics: bool = False) -> None:
        if adapter.uuid is None:
            raise ValueError(f"{adapter.description} is not an NVML device")
        self._lib = _nvml()
        self._handle = ctypes.c_void_p()
        status = self._lib.nvmlDeviceGetHandleByUUID(adapter.uuid.encode("ascii"), ctypes.byref(self._handle))
        _check(self._lib, status, f"nvmlDeviceGetHandleByUUID for {adapter.description}")
        self._graphics = graphics
        self._function = (
            "nvmlDeviceGetGraphicsRunningProcesses_v3" if graphics else "nvmlDeviceGetComputeRunningProcesses_v3"
        )
        self._list = getattr(self._lib, self._function)
        self._entries = (_ProcessInfo * _SPARE_PROCESS_ENTRIES)()

    def read(self) -> dict[int, int]:
        """``usedGpuMemory`` by pid, ``_NVML_VALUE_NOT_AVAILABLE`` where the driver does not attribute memory.

        A graphics listing the device does not support (compute-only cards and modes) has no processes.
        """
        for _ in range(_LIST_ATTEMPTS):
            count = ctypes.c_uint(len(self._entries))
            status = self._list(self._handle, ctypes.byref(count), self._entries)
            if status == _NVML_SUCCESS:
                return {entry.pid: entry.usedGpuMemory for entry in self._entries[: count.value]}
            if status == _NVML_ERROR_NOT_SUPPORTED and self._graphics:
                return {}
            if status != _NVML_ERROR_INSUFFICIENT_SIZE:
                _check(self._lib, status, self._function)
            # NVML has set the count it needs; the spare entries leave room for processes that start meanwhile.
            self._entries = (_ProcessInfo * (count.value + _SPARE_PROCESS_ENTRIES))()
        raise OSError(f"{self._function}: the process count kept growing over {_LIST_ATTEMPTS} attempts")


def process_usage_on_adapter(adapter: Adapter) -> dict[int, ProcessUsage]:
    """Every compute and graphics process on the GPU, by pid, with its memory as dedicated usage; NVML has no shared
    usage.

    A process with both kinds of context is listed in both lists, with the same memory, so the larger value stands.
    A graphics process whose memory NVML does not attribute is left out: the compute list already tells whether the
    driver accounts memory per process.
    """
    compute = _ProcessTable(adapter).read()
    if _NVML_VALUE_NOT_AVAILABLE in compute.values():
        raise _not_attributed(adapter)
    used = dict(compute)
    for pid, value in _ProcessTable(adapter, graphics=True).read().items():
        if value != _NVML_VALUE_NOT_AVAILABLE:
            used[pid] = max(used.get(pid, 0), value)
    return {pid: ProcessUsage(dedicated_bytes=value, shared_bytes=0) for pid, value in used.items()}


class ProcessMemorySampler:
    """Samples one process's memory on one NVIDIA GPU on a background thread, from ``start()`` until ``stop()``.

    ``usedGpuMemory`` is read every ``interval_s``. A process NVML does not list (it has not opened the GPU yet, or
    holds nothing there) reads as zero. Sampling ends early when the process exits.

    ``pid`` must be the process that allocates: NVML charges memory to the process that holds it, not to its parent.
    """

    def __init__(self, pid: int, adapter: Adapter, interval_s: float = 0.005) -> None:
        if pid <= 0:
            raise ValueError(f"invalid pid {pid}")
        if interval_s <= 0:
            raise ValueError(f"the sampling interval must be positive, got {interval_s}")
        self._pid = pid
        self._adapter = adapter
        self._interval_s = interval_s
        self._samples: list[tuple[float, int]] = []
        self._process: psutil.Process | None = None
        self._table: _ProcessTable | None = None
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._trace: MemoryTrace | None = None
        self._warned: set[str] = set()

    def start(self) -> None:
        """Resolve the process and the GPU, take one sample and start the sampling thread."""
        if self._thread is not None or self._trace is not None:
            raise RuntimeError("a sampler starts once")
        try:
            # Holds the create time, so a later process that reuses the pid does not pass for this one.
            self._process = psutil.Process(self._pid)
        except psutil.NoSuchProcess:
            raise ProcessLookupError(f"no process with pid {self._pid}") from None
        self._table = _ProcessTable(self._adapter)
        self._sample(raise_on_failure=True)
        self._thread = threading.Thread(target=self._run, name=f"gpu-memory-sampler-{self._pid}", daemon=True)
        self._thread.start()

    def stop(self) -> MemoryTrace:
        """Stop sampling and return the trace; later calls return the same trace."""
        if self._trace is None:
            self._stopping.set()
            if self._thread is not None:
                self._thread.join()
            self._trace = MemoryTrace(
                method="nvml", interval_ms=self._mean_interval_ms(), dedicated=tuple(self._samples), shared=()
            )
        return self._trace

    def _exited(self) -> bool:
        assert self._process is not None
        try:
            # psutil counts a zombie as running, but its memory went with its exit.
            return not self._process.is_running() or self._process.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return True

    def _sample(self, raise_on_failure: bool = False) -> bool:
        """Append one sample; ``False`` once the process has exited."""
        assert self._table is not None
        t = time.perf_counter()
        try:
            used = self._table.read().get(self._pid)
        except OSError as error:
            if raise_on_failure:
                raise
            self._warn_once("error", f"{error}; sample skipped")
            return True
        if used is None:
            if self._exited():
                if raise_on_failure:
                    raise ProcessLookupError(f"process {self._pid} has exited")
                return False
            used = 0
        elif used == _NVML_VALUE_NOT_AVAILABLE:
            if raise_on_failure:
                raise _not_attributed(self._adapter)
            self._warn_once("unavailable", "NVML reports its memory as not available; sample skipped")
            return True
        self._samples.append((t, used))
        return True

    def _warn_once(self, kind: str, message: str) -> None:
        if kind not in self._warned:
            self._warned.add(kind)
            _logger.warning(f"GPU memory sampler for process {self._pid}: {message} (further failures not logged)")

    def _run(self) -> None:
        clock = time.perf_counter
        next_sample = clock() + self._interval_s
        while not self._stopping.is_set():
            delay = next_sample - clock()
            if delay > 0:
                time.sleep(delay)
            if not self._sample():
                return
            # A fixed rate, as in gpu_memory: a late wake-up shortens the next sleep, and after a stall sampling
            # resumes from now rather than bursting to catch up.
            next_sample = max(next_sample + self._interval_s, clock())

    def _mean_interval_ms(self) -> float:
        span = self._samples[-1][0] - self._samples[0][0] if len(self._samples) > 1 else 0.0
        if span <= 0:
            return self._interval_s * 1000
        return span / (len(self._samples) - 1) * 1000
