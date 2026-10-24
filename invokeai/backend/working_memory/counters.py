"""This process's memory on a compute device, as the working-memory probe reads it at the boundaries of a window.

Two kinds of counter, and neither alone is an operation's demand:

- torch's caching allocator: the bytes it reserved from the driver and the bytes live tensors occupy, with exact peaks
  since the last reset, on CUDA, ROCm and XPU. MPS keeps current values only; there `PeakSampler` keeps the peaks.
- the driver: what this process holds on the device, including what torch's allocator does not see -- cuBLAS and
  cuDNN workspaces, kernel modules, and on ROCm the pool HIP keeps after a free.

Measured on an RTX 4090 (torch 2.13+cu130, Windows): the driver delta equals torch's reserved-peak delta to three
decimals over VAE decode and encode, matmul, SDPA with and without a mask, and convolutions; first use outside the
allocator adds to the driver alone (+16 MiB at the first matmul, +2 MiB at the first convolution). Under
``backend:cudaMallocAsync`` the reserved peak can sit below the allocated peak (128 vs 136 MiB). On an RTX PRO 6000
Blackwell Server Edition (torch 2.13+cu130, Linux, driver 580.178) NVML's per-process figure matches ``nvidia-smi``,
and its delta over a VAE decode equals the reserved delta (1044 MiB at 1024px); in the process's first window, first use
outside the allocator adds to the driver alone (736 vs 626 MiB reserved at 768px). On an RX 9060 XT (torch 2.13+rocm10,
Windows, expandable segments) the driver keeps HIP's retained pool after ``empty_cache()``, so its delta reads below
reserved (4.3 vs 6.4 GiB for a 1024px decode), except for the first MIOpen decode, which grows it beyond (2.0 vs
1.72 GiB). So the probe takes the largest delta of the counters it has.

The driver is read at boundaries only: torch's peaks are exact, so what a boundary read misses is memory outside torch's
allocator that came and went between two boundaries.

Every reader is best-effort: a failure reads as None. A peak reset or a synchronization that fails raises instead: the
window would go on with peaks from before its boundary or with work still in flight, so the probe, which never lets an
exception reach a generation, marks the window invalid.
"""

import dataclasses
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Optional

import torch

from invokeai.backend.util import nvml, wddm
from invokeai.backend.util.level_zero import xpu_memory_info
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.util.pci import pci_address

Backend = Literal["cuda", "hip", "mps", "xpu"]

Method = Literal["windows-d3dkmt", "nvml", "fdinfo", "mps-driver", "xpu-sysman", "torch-only"]
"""Where a reading's driver figures come from.

- ``windows-d3dkmt``: this process's ``CurrentUsage`` in the adapter's local segment (`wddm.process_local_usage`), on
  CUDA and ROCm under Windows, with the paged bytes from PDH.
- ``nvml``: the ``usedGpuMemory`` NVML attributes to this process, on CUDA under Linux.
- ``fdinfo``: ``drm-memory-vram`` summed over this process's DRM clients on the device, on ROCm under Linux.
- ``mps-driver``: ``torch.mps.driver_allocated_memory()``.
- ``xpu-sysman``: the device's used memory from Level Zero Sysman. Device-wide, so it guards against another process
  loading the device during a window and is never taken as demand.
- ``torch-only``: no driver figure; the allocator's counters alone.
"""

VERIFIED_METHODS: Final[frozenset[tuple[Method, Backend]]] = frozenset(
    {("windows-d3dkmt", "cuda"), ("windows-d3dkmt", "hip"), ("nvml", "cuda")}
)
"""The (method, backend) pairs whose readings were checked on real hardware: the RTX 4090, the RTX PRO 6000 and the RX
9060 XT above. The other readers follow their vendors' documentation, and a window measured with them says so
(``verified=False``)."""

PEAK_SAMPLER_INTERVAL_S: float = 0.003
"""How often `PeakSampler` samples. A few milliseconds is far below an operation's allocation lifetimes and costs one
getter call per sample."""

_FDINFO_DIR = Path("/proc/self/fdinfo")
_FDINFO_UNITS: Final[Mapping[str, int]] = {"": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30}

# torch.cuda.memory_stats / torch.xpu.memory_stats keys, per CounterReading field.
_STAT_KEYS: Final[Mapping[str, str]] = {
    "reserved": "reserved_bytes.all.current",
    "allocated": "allocated_bytes.all.current",
    "peak_reserved": "reserved_bytes.all.peak",
    "peak_allocated": "allocated_bytes.all.peak",
    "num_ooms": "num_ooms",
    "num_alloc_retries": "num_alloc_retries",
}


@dataclass(frozen=True)
class CounterReading:
    """One reading of a device's counters for this process, in bytes; None where the platform has no such counter or
    its reader failed."""

    reserved: Optional[int]
    allocated: Optional[int]
    peak_reserved: Optional[int]
    """Since the last `reset_peaks`, like the other peaks."""
    peak_allocated: Optional[int]
    num_ooms: Optional[int]
    """Allocations that failed, cumulative for the process."""
    num_alloc_retries: Optional[int]
    """Allocations that only succeeded after the allocator released its cache and retried, cumulative. A retry
    squeezes the peak below what the operation would have taken with room to spare."""
    driver: Optional[int]
    """This process's memory on the device as the driver counts it."""
    peak_driver: Optional[int]
    """Only where a sampler keeps it (MPS); elsewhere the driver is read at boundaries."""
    paged: Optional[int]
    """This process's bytes Windows keeps in shared system memory for the device's adapter."""
    device_used: Optional[int]
    """Used memory of the whole device, every process together (XPU)."""
    method: Method


def backend_for(device: torch.device) -> Optional[Backend]:
    """The GPU backend `device` runs on; None for a device that is not a GPU. A HIP build reports ``cuda`` devices."""
    if device.type == "cuda":
        return "hip" if torch.version.hip is not None else "cuda"
    if device.type in ("mps", "xpu"):
        return device.type  # type: ignore[return-value]
    return None


def _allocator_stats(device: torch.device) -> Mapping[str, int]:
    try:
        if device.type == "cuda":
            return torch.cuda.memory_stats(device)
        if device.type == "xpu":
            return torch.xpu.memory_stats(device)
        if device.type == "mps":
            return {_STAT_KEYS["allocated"]: torch.mps.current_allocated_memory()}
    except Exception:
        InvokeAILogger.get_logger(__name__).debug(f"Allocator statistics for {device} unavailable", exc_info=True)
    return {}


def _fdinfo_size(value: str) -> Optional[int]:
    number, _, unit = value.strip().partition(" ")
    scale = _FDINFO_UNITS.get(unit.strip())
    return int(number) * scale if number.isdigit() and scale is not None else None


def fdinfo_vram_bytes(pdev: str) -> Optional[int]:
    """VRAM this process holds on the DRM device at PCI address `pdev`, from ``/proc/self/fdinfo``.

    Each open DRM file reports its client's ``drm-memory-vram``. Duplicated descriptors share a client and report the
    same memory, so each ``drm-client-id`` counts once. None when no descriptor of this process is a client of that
    device.
    """
    try:
        entries = list(_FDINFO_DIR.iterdir())
    except OSError:
        return None
    total: Optional[int] = None
    clients: set[str] = set()
    for entry in entries:
        try:
            text = entry.read_text(encoding="utf-8", errors="replace")
        except OSError:
            # Closed since the listing, or not readable.
            continue
        fields = {key.strip(): value.strip() for key, _, value in (line.partition(":") for line in text.splitlines())}
        if fields.get("drm-pdev", "").lower() != pdev.lower():
            continue
        client = fields.get("drm-client-id")
        if client is not None:
            if client in clients:
                continue
            clients.add(client)
        vram = _fdinfo_size(fields.get("drm-memory-vram", ""))
        if vram is not None:
            total = vram if total is None else total + vram
    return total


def driver_method(backend: Optional[Backend], platform_name: Optional[str] = None) -> Method:
    """The reader a `backend` device's driver figure comes from on the OS `platform_name` (a ``sys.platform`` value,
    this process's by default), ``torch-only`` where there is none. A reading that fails falls back to ``torch-only``
    too."""
    platform_name = sys.platform if platform_name is None else platform_name
    if backend in ("cuda", "hip") and platform_name == "win32":
        return "windows-d3dkmt"
    if backend == "cuda" and platform_name.startswith("linux"):
        return "nvml"
    if backend == "hip" and platform_name.startswith("linux"):
        return "fdinfo"
    if backend == "mps":
        return "mps-driver"
    if backend == "xpu":
        return "xpu-sysman"
    return "torch-only"


def method_verified(backend: Backend, platform_name: Optional[str] = None) -> bool:
    """Whether the windows of a `backend` device on this OS read a driver figure verified on real hardware."""
    return (driver_method(backend, platform_name), backend) in VERIFIED_METHODS


def _driver_reading(device: torch.device) -> tuple[Optional[int], Optional[int], Optional[int], Method]:
    """``(driver, paged, device_used, method)`` for `device` on this platform."""
    backend = backend_for(device)
    driver: Optional[int] = None
    paged: Optional[int] = None
    device_used: Optional[int] = None
    method = driver_method(backend)
    try:
        if method == "windows-d3dkmt":
            driver, paged = wddm.process_local_usage(device), wddm.process_paged_bytes(device)
        elif method == "nvml":
            driver = nvml.process_used_memory(device)
        elif method == "fdinfo":
            index = device.index if device.index is not None else torch.cuda.current_device()
            driver = fdinfo_vram_bytes(pci_address(index))
        elif method == "mps-driver":
            driver = int(torch.mps.driver_allocated_memory())
        elif method == "xpu-sysman":
            info = xpu_memory_info(device)
            device_used = None if info is None else info[1] - info[0]
    except Exception:
        InvokeAILogger.get_logger(__name__).debug(f"Driver memory reading for {device} failed", exc_info=True)
        return None, None, None, "torch-only"
    if driver is None and device_used is None:
        method = "torch-only"
    return driver, paged, device_used, method


def read_counters(device: torch.device) -> CounterReading:
    """This process's counters on `device` now. Peaks are torch's, since the last `reset_peaks`; MPS has none."""
    stats = _allocator_stats(device)
    values = {field: (None if key not in stats else int(stats[key])) for field, key in _STAT_KEYS.items()}
    driver, paged, device_used, method = _driver_reading(device)
    return CounterReading(
        **values, driver=driver, peak_driver=None, paged=paged, device_used=device_used, method=method
    )


def reset_peaks(device: torch.device) -> None:
    """Start torch's peak counters over at the current values. Cumulative counters (OOMs, retries) are kept. Raises
    what torch raises."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "xpu":
        torch.xpu.reset_peak_memory_stats(device)


def synchronize(device: torch.device) -> None:
    """Wait for the work queued on `device`, so frees it implies have happened. Raises what torch raises."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _larger(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None:
        return b
    return a if b is None else max(a, b)


class PeakSampler:
    """The largest values `read` returned since the last `reset`, sampled every `PEAK_SAMPLER_INTERVAL_S` on a daemon
    thread between `start` and `stop`.

    For a device whose counters have no peaks: without it, a window would see only the values at its boundaries and
    miss everything that came and went in between. `read` returns one value per counter, None where it has none; a
    `read` that raises is a sample of Nones.
    """

    def __init__(self, read: Callable[[], tuple[Optional[int], ...]]) -> None:
        self._read = read
        # Every path reads under the lock, so a sample read before a `reset` cannot be folded after it and carry the
        # earlier interval's peak into the new one. A `reset` waits for one `read` at most: two getters on MPS.
        self._lock = threading.Lock()
        self._peaks: tuple[Optional[int], ...] = ()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self) -> tuple[Optional[int], ...]:
        try:
            return self._read()
        except Exception:
            return ()

    def _fold_sample(self) -> tuple[Optional[int], ...]:
        with self._lock:
            sample = self._sample()
            width = max(len(self._peaks), len(sample))
            peaks = self._peaks + (None,) * (width - len(self._peaks))
            sample = sample + (None,) * (width - len(sample))
            self._peaks = tuple(_larger(peak, value) for peak, value in zip(peaks, sample, strict=True))
            return self._peaks

    def _run(self) -> None:
        while not self._stop.wait(PEAK_SAMPLER_INTERVAL_S):
            self._fold_sample()

    def start(self) -> None:
        self.reset()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="working-memory-peak-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def reset(self) -> None:
        """Start the peaks over at the values now."""
        with self._lock:
            self._peaks = self._sample()

    def peaks(self) -> tuple[Optional[int], ...]:
        """The peaks since the last reset, the values now included."""
        return self._fold_sample()


def _mps_values() -> tuple[Optional[int], ...]:
    return int(torch.mps.current_allocated_memory()), int(torch.mps.driver_allocated_memory())


class DeviceCounters:
    """One device's counters for the length of a window: readings, peak resets and synchronization.

    The peaks are torch's where it keeps them. On MPS, which keeps none, a `PeakSampler` keeps the allocated and the
    driver peaks from construction until `close`.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._sampler = PeakSampler(_mps_values) if device.type == "mps" else None
        if self._sampler is not None:
            self._sampler.start()

    def read(self) -> CounterReading:
        reading = read_counters(self.device)
        if self._sampler is None:
            return reading
        peak_allocated, peak_driver = (self._sampler.peaks() + (None, None))[:2]
        return dataclasses.replace(
            reading,
            peak_allocated=_larger(peak_allocated, reading.allocated),
            peak_driver=_larger(peak_driver, reading.driver),
        )

    def reset_peaks(self) -> None:
        reset_peaks(self.device)
        if self._sampler is not None:
            self._sampler.reset()

    def synchronize(self) -> None:
        synchronize(self.device)

    def close(self) -> None:
        if self._sampler is not None:
            self._sampler.stop()
