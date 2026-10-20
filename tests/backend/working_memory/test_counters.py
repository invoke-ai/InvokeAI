"""The device counters: which driver reader each platform uses, the DRM fdinfo reader, and the MPS peak sampler.

The OS and torch readers behind them are stand-ins: what is under test is that each platform reads through its own
reader, labels the reading with it, and turns any failure into "unknown" -- and that a peak reset or a synchronization
that fails raises, since it has no "unknown" to fall back on.
"""

import sys
import threading
from pathlib import Path
from typing import Callable

import pytest
import torch

from invokeai.backend.util import nvml, wddm
from invokeai.backend.working_memory import counters

GIB = 2**30
MIB = 2**20
PDEV = "0000:03:00.0"
CUDA0 = torch.device("cuda", 0)

_STATS = {
    "reserved_bytes.all.current": 3 * GIB,
    "allocated_bytes.all.current": 2 * GIB,
    "reserved_bytes.all.peak": 5 * GIB,
    "allocated_bytes.all.peak": 4 * GIB,
    "num_ooms": 0,
    "num_alloc_retries": 1,
}


def _fdinfo(pdev: str, client: int, vram: str) -> str:
    return (
        "pos:\t0\nflags:\t02100002\nmnt_id:\t26\nino:\t1063\ndrm-driver:\tamdgpu\n"
        f"drm-client-id:\t{client}\ndrm-pdev:\t{pdev}\npasid:\t32790\ndrm-memory-vram:\t{vram}\n"
        "drm-memory-gtt:\t2048 KiB\ndrm-memory-cpu:\t0 KiB\namd-evicted-vram:\t0 KiB\n"
    )


def test_fdinfo_sums_each_client_of_the_device_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "3").write_text("pos:\t0\nflags:\t0100000\nmnt_id:\t26\nino:\t1063\n")  # not a DRM file
    (tmp_path / "7").write_text(_fdinfo(PDEV, 11, "1048576 KiB"))
    (tmp_path / "8").write_text(_fdinfo(PDEV, 11, "1048576 KiB"))  # a duplicated descriptor of the same client
    (tmp_path / "9").write_text(_fdinfo(PDEV, 12, "512 MiB"))
    (tmp_path / "10").write_text(_fdinfo("0000:0c:00.0", 13, "4194304 KiB"))  # the integrated GPU
    monkeypatch.setattr(counters, "_FDINFO_DIR", tmp_path)

    assert counters.fdinfo_vram_bytes(PDEV) == 1 * GIB + 512 * MIB
    assert counters.fdinfo_vram_bytes("0000:0C:00.0") == 4 * GIB


def test_fdinfo_without_a_client_of_the_device_is_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "7").write_text(_fdinfo("0000:0c:00.0", 13, "4194304 KiB"))
    monkeypatch.setattr(counters, "_FDINFO_DIR", tmp_path)
    assert counters.fdinfo_vram_bytes(PDEV) is None

    monkeypatch.setattr(counters, "_FDINFO_DIR", tmp_path / "missing")
    assert counters.fdinfo_vram_bytes(PDEV) is None


def _raise(*_args: object) -> None:
    raise RuntimeError("the driver did not answer")


def _windows(monkeypatch: pytest.MonkeyPatch, reader: Callable) -> None:
    monkeypatch.setattr(wddm, "process_local_usage", reader)
    monkeypatch.setattr(wddm, "process_paged_bytes", lambda device: 64 * MIB)


def _nvml(monkeypatch: pytest.MonkeyPatch, reader: Callable) -> None:
    monkeypatch.setattr(nvml, "process_used_memory", reader)


def _fdinfo_reader(monkeypatch: pytest.MonkeyPatch, reader: Callable) -> None:
    monkeypatch.setattr(counters, "pci_address", lambda index: PDEV)
    monkeypatch.setattr(counters, "fdinfo_vram_bytes", reader)


_CUDA_PLATFORMS = [
    pytest.param("win32", None, _windows, "windows-d3dkmt", id="windows-cuda"),
    pytest.param("win32", "7.14.0", _windows, "windows-d3dkmt", id="windows-rocm"),
    pytest.param("linux", None, _nvml, "nvml", id="linux-cuda"),
    pytest.param("linux", "7.2.0", _fdinfo_reader, "fdinfo", id="linux-rocm"),
]


@pytest.mark.parametrize(("platform", "hip", "install", "method"), _CUDA_PLATFORMS)
def test_each_gpu_platform_reads_its_driver_through_its_own_reader(
    monkeypatch: pytest.MonkeyPatch, platform: str, hip: str | None, install: Callable, method: str
):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda device: dict(_STATS))
    install(monkeypatch, lambda device: 6 * GIB)

    reading = counters.read_counters(CUDA0)

    assert (reading.driver, reading.method) == (6 * GIB, method)
    assert (reading.reserved, reading.allocated, reading.peak_reserved, reading.peak_allocated) == (
        3 * GIB,
        2 * GIB,
        5 * GIB,
        4 * GIB,
    )
    assert (reading.num_ooms, reading.num_alloc_retries) == (0, 1)
    assert reading.paged == (64 * MIB if platform == "win32" else None)


@pytest.mark.parametrize(("platform", "hip", "install", "method"), _CUDA_PLATFORMS)
def test_a_failing_reader_reads_as_unknown_and_never_raises(
    monkeypatch: pytest.MonkeyPatch, platform: str, hip: str | None, install: Callable, method: str
):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "memory_stats", _raise)
    install(monkeypatch, _raise)

    reading = counters.read_counters(CUDA0)

    assert (reading.driver, reading.reserved, reading.peak_allocated, reading.method) == (
        None,
        None,
        None,
        "torch-only",
    )


@pytest.mark.parametrize(
    ("device", "module", "function"),
    [
        pytest.param(CUDA0, torch.cuda, "reset_peak_memory_stats", id="cuda-reset"),
        pytest.param(CUDA0, torch.cuda, "synchronize", id="cuda-synchronize"),
        pytest.param(torch.device("xpu", 0), torch.xpu, "reset_peak_memory_stats", id="xpu-reset"),
        pytest.param(torch.device("xpu", 0), torch.xpu, "synchronize", id="xpu-synchronize"),
        pytest.param(torch.device("mps"), torch.mps, "synchronize", id="mps-synchronize"),
    ],
)
def test_a_peak_reset_or_synchronization_that_fails_raises(
    monkeypatch: pytest.MonkeyPatch, device: torch.device, module: object, function: str
):
    """A window that went on would measure from an earlier operation's peaks, or with that work still in flight."""
    monkeypatch.setattr(module, function, _raise)
    # What the MPS peak sampler reads while it runs.
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 1 * GIB)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 2 * GIB)

    device_counters = counters.DeviceCounters(device)
    try:
        with pytest.raises(RuntimeError, match="the driver did not answer"):
            if function == "synchronize":
                device_counters.synchronize()
            else:
                device_counters.reset_peaks()
    finally:
        device_counters.close()


def test_xpu_reads_the_device_wide_usage_as_a_guard_not_as_this_processs_driver(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.xpu, "memory_stats", lambda device: dict(_STATS))
    monkeypatch.setattr(counters, "xpu_memory_info", lambda device: (10 * GIB, 16 * GIB))

    reading = counters.read_counters(torch.device("xpu", 0))

    assert (reading.driver, reading.device_used, reading.method) == (None, 6 * GIB, "xpu-sysman")
    assert reading.reserved == 3 * GIB

    monkeypatch.setattr(counters, "xpu_memory_info", lambda device: None)
    assert counters.read_counters(torch.device("xpu", 0)).method == "torch-only"


class _FakeMps:
    """The two MPS getters, over a value the test moves."""

    def __init__(self) -> None:
        self.allocated, self.driver = 1 * GIB, 2 * GIB
        self.spike_reads = 0
        self.spike_seen = threading.Event()

    def current_allocated_memory(self) -> int:
        if self.allocated == 5 * GIB and threading.current_thread() is not threading.main_thread():
            # Two sampler reads of the spike: the first has been folded into the peaks by the time of the second.
            self.spike_reads += 1
            if self.spike_reads >= 2:
                self.spike_seen.set()
        return self.allocated

    def driver_allocated_memory(self) -> int:
        return self.driver


def test_on_mps_a_sampler_keeps_the_peak_that_came_and_went_between_two_reads(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeMps()
    monkeypatch.setattr(torch.mps, "current_allocated_memory", fake.current_allocated_memory)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", fake.driver_allocated_memory)
    monkeypatch.setattr(torch.mps, "synchronize", lambda: None)
    monkeypatch.setattr(counters, "PEAK_SAMPLER_INTERVAL_S", 0.0005)

    device_counters = counters.DeviceCounters(torch.device("mps"))
    try:
        fake.allocated, fake.driver = 5 * GIB, 7 * GIB
        assert fake.spike_seen.wait(timeout=30)
        fake.allocated, fake.driver = 1 * GIB, 2 * GIB

        reading = device_counters.read()
        assert (reading.allocated, reading.driver, reading.method) == (1 * GIB, 2 * GIB, "mps-driver")
        assert (reading.peak_allocated, reading.peak_driver) == (5 * GIB, 7 * GIB)
        assert reading.peak_reserved is None, "MPS reserves nothing torch reports"

        device_counters.reset_peaks()
        assert (device_counters.read().peak_allocated, device_counters.read().peak_driver) == (1 * GIB, 2 * GIB)
    finally:
        device_counters.close()
    assert not device_counters._sampler.running  # type: ignore[union-attr]


def test_on_mps_failing_getters_read_as_unknown_and_the_sampler_still_stops(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.mps, "current_allocated_memory", _raise)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", _raise)

    device_counters = counters.DeviceCounters(torch.device("mps"))
    try:
        reading = device_counters.read()
    finally:
        device_counters.close()

    assert (reading.allocated, reading.peak_allocated, reading.driver, reading.method) == (
        None,
        None,
        None,
        "torch-only",
    )
    assert not device_counters._sampler.running  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("backend", "platform", "method", "verified"),
    [
        ("cuda", "win32", "windows-d3dkmt", True),
        ("hip", "win32", "windows-d3dkmt", True),
        ("cuda", "linux", "nvml", False),
        ("hip", "linux", "fdinfo", False),
        ("mps", "darwin", "mps-driver", False),
        ("xpu", "linux", "xpu-sysman", False),
        ("cuda", "darwin", "torch-only", False),
    ],
)
def test_the_driver_reader_and_its_verification_follow_backend_and_os(
    monkeypatch: pytest.MonkeyPatch, backend: str, platform: str, method: str, verified: bool
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    assert (counters.driver_method(backend), counters.method_verified(backend)) == (method, verified)  # type: ignore[arg-type]
