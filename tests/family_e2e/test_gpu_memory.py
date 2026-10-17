import subprocess
import sys
import time

import pytest

from scripts.family_e2e import gpu_memory
from scripts.family_e2e.gpu_memory import (
    Adapter,
    MemoryTrace,
    ProcessMemorySampler,
    UnsupportedPlatformError,
    find_adapter,
    list_adapters,
    process_usage_on_adapter,
)

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="the sampler reads WDDM, which only Windows has")


def _trace(*samples: tuple[float, int]) -> MemoryTrace:
    return MemoryTrace(method="d3dkmt", interval_ms=3.0, dedicated=samples, shared=())


STEPS = _trace((1.0, 10), (2.0, 30), (3.0, 20))


def test_empty_trace_has_no_level_or_peak() -> None:
    trace = _trace()
    assert trace.at(1.0) is None
    assert trace.peak() is None
    assert trace.peak(0.0, 1.0) is None


@pytest.mark.parametrize(("t", "expected"), [(0.5, None), (1.0, 10), (1.99, 10), (2.0, 30), (10.0, 20)])
def test_at_holds_the_last_sample_at_or_before_t(t: float, expected: int | None) -> None:
    assert STEPS.at(t) == expected


@pytest.mark.parametrize(
    ("t0", "t1", "expected"),
    [
        (None, None, 30),
        (None, 1.5, 10),
        (2.0, None, 30),
        (1.5, 2.0, 30),  # the end is inclusive
        (2.2, 2.4, 30),  # no sample inside: the value in effect at t0
        (2.5, 3.0, 30),  # the value in effect at t0 counts ...
        (3.0, 4.0, 20),  # ... unless a sample at exactly t0 supersedes it
        (4.0, 5.0, 20),  # after the last sample its value holds
        (0.0, 0.5, None),  # before the first sample
    ],
)
def test_peak_is_the_highest_level_in_the_window(t0: float | None, t1: float | None, expected: int | None) -> None:
    assert STEPS.peak(t0, t1) == expected


def test_samples_sharing_a_timestamp_all_count() -> None:
    # Samples taken within one tick of a coarse clock share a timestamp; all of them belong to that instant.
    trace = _trace((1.0, 10), (1.0, 50), (1.0, 20), (2.0, 5))
    assert trace.at(1.0) == 20
    assert trace.peak(1.0, 1.0) == 50
    assert trace.peak(1.5, 1.6) == 20
    assert trace.peak(2.0, 3.0) == 5


def test_peak_rejects_a_window_that_ends_before_it_starts() -> None:
    with pytest.raises(ValueError):
        STEPS.peak(2.0, 1.0)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("pid_50992_luid_0x00000000_0x00022A61_phys_0", (50992, 0x22A61)),
        ("pid_7_luid_0x00000001_0x0000beef_phys_0", (7, (1 << 32) | 0xBEEF)),
        ("pid_7_luid_0x00000000_0x00022A61_phys_1", None),  # second physical GPU of a linked adapter
        ("pid_7_luid_0x22A61_phys_0", None),
        ("_Total", None),
    ],
)
def test_pdh_instance_names_parse_to_pid_and_luid(name: str, expected: tuple[int, int] | None) -> None:
    assert gpu_memory._parse_instance(name) == expected


@pytest.mark.parametrize("luid", [0x22A61, (1 << 32) | 0xBEEF, (0x80000000 << 32) | 1])
def test_luid_survives_the_d3dkmt_struct_and_matches_the_pdh_spelling(luid: int) -> None:
    # HighPart is a signed LONG; a set top bit must not turn the LUID negative.
    assert gpu_memory._luid_value(gpu_memory._luid_struct(luid)) == luid
    pdh_name = f"pid_1_luid_0x{luid >> 32:08X}_0x{luid & 0xFFFFFFFF:08X}_phys_0"
    assert gpu_memory._parse_instance(pdh_name) == (1, luid)


@pytest.mark.parametrize(
    ("packed", "expected"),
    [
        # As D3DKMT reports the RTX 4090 and RX 9060 XT drivers, next to what Device Manager shows for them.
        (0x0020000000100417, "32.0.16.1047"),
        (0x00200000794103EC, "32.0.31041.1004"),
        (0xFFFF000100020003, "65535.1.2.3"),
        (0, None),  # an unset version is unknown, not "0.0.0.0"
    ],
)
def test_driver_version_reads_as_device_manager_shows_it(packed: int, expected: str | None) -> None:
    assert gpu_memory._format_driver_version(packed) == expected


ADAPTERS = [
    Adapter(luid=0x29ADC, description="AMD Radeon(TM) Graphics"),
    Adapter(luid=0x25CBD, description="AMD Radeon RX 9060 XT"),
    Adapter(luid=0x22A61, description="NVIDIA GeForce RTX 4090"),
]


def test_adapter_is_selected_by_case_insensitive_substring() -> None:
    assert gpu_memory._select_adapter(ADAPTERS, "rx 9060") == ADAPTERS[1]


@pytest.mark.parametrize(
    ("adapters", "needle", "message"),
    [
        (ADAPTERS, "Radeon", "2 adapters match 'Radeon'"),
        (ADAPTERS, "Arc", "No adapter matches 'Arc'"),
        ([], "4090", "Adapters: none"),
    ],
)
def test_adapter_selection_needs_exactly_one_match(adapters: list[Adapter], needle: str, message: str) -> None:
    with pytest.raises(LookupError) as error:
        gpu_memory._select_adapter(adapters, needle)
    assert message in str(error.value)
    # The message lists what is there, so the caller can pick a better substring.
    for adapter in adapters:
        assert adapter.description in str(error.value)


def test_other_platforms_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(UnsupportedPlatformError):
        list_adapters()
    with pytest.raises(UnsupportedPlatformError):
        find_adapter("4090")
    with pytest.raises(UnsupportedPlatformError):
        process_usage_on_adapter(ADAPTERS[2])
    with pytest.raises(UnsupportedPlatformError):
        ProcessMemorySampler(1234, ADAPTERS[2])


@windows_only
def test_stop_without_start_returns_one_empty_trace() -> None:
    sampler = ProcessMemorySampler(1234, ADAPTERS[2], interval_s=0.005)
    trace = sampler.stop()
    assert trace == MemoryTrace(method="d3dkmt", interval_ms=5.0, dedicated=(), shared=())
    assert sampler.stop() is trace


@windows_only
def test_start_refuses_a_process_that_has_exited() -> None:
    process = subprocess.Popen(["cmd.exe", "/c", "exit 0"])
    process.wait()
    # Popen still holds its handle, so the pid cannot have been reused by another process.
    sampler = ProcessMemorySampler(process.pid, ADAPTERS[2])
    with pytest.raises(ProcessLookupError, match="has exited"):
        sampler.start()
    assert sampler.stop().dedicated == ()


_CUDA_CHILD = """
import os, sys, torch
if not torch.cuda.is_available():
    print("unavailable", flush=True)
    sys.exit()
torch.zeros(1, device="cuda")
torch.cuda.synchronize()
print(os.getpid(), flush=True)
sys.stdin.readline()
x = torch.empty(2**28, dtype=torch.float32, device="cuda")
x.fill_(1)
torch.cuda.synchronize()
print("allocated", flush=True)
sys.stdin.readline()
del x
torch.cuda.empty_cache()
torch.cuda.synchronize()
print("freed", flush=True)
sys.stdin.readline()
"""


@pytest.mark.slow
@windows_only
def test_sampler_follows_a_cuda_allocation_in_another_process() -> None:
    """Needs exactly one NVIDIA adapter and torch with CUDA. The child prints its own pid: under a venv launcher,
    ``Popen.pid`` is the launcher, not the interpreter that allocates."""
    try:
        adapter = find_adapter("NVIDIA")
    except LookupError as error:
        pytest.skip(str(error))
    child = subprocess.Popen(
        [sys.executable, "-c", _CUDA_CHILD], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    assert child.stdin is not None and child.stdout is not None
    sampler = None
    try:
        first = child.stdout.readline().strip()
        if not first.isdigit():
            pytest.skip(f"no CUDA in the child: {first!r}")
        sampler = ProcessMemorySampler(int(first), adapter)
        sampler.start()

        def step(expected: str) -> float:
            """Trigger the child's next step; the instant just before it, read against the trace afterwards."""
            t = time.perf_counter()
            child.stdin.write("\n")
            child.stdin.flush()
            assert child.stdout.readline().strip() == expected
            # Let samples land after the step before anything else changes.
            time.sleep(0.05)
            return t

        # Each instant is taken immediately before the next change, so the level read there must not show it yet.
        t_before_allocation = step("allocated")
        t_before_free = step("freed")
        t_end = time.perf_counter()
        trace = sampler.stop()
    finally:
        if sampler is not None:
            sampler.stop()
        child.communicate("\n", timeout=60)

    gib = 2**30
    context = trace.at(t_before_allocation)
    allocated = trace.at(t_before_free)
    assert context is not None and allocated is not None
    assert context > 0
    assert abs(allocated - context - gib) < gib / 16
    peak = trace.peak(t_before_allocation, t_before_free)
    assert peak is not None and 0 <= peak - allocated < gib / 16
    # The CUDA caching allocator hands the block back to the driver on empty_cache().
    freed = trace.at(t_end)
    assert freed is not None and freed < allocated - gib * 15 / 16
