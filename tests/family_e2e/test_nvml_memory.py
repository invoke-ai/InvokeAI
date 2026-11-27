"""The Linux NVML sampler against a scripted stand-in for libnvidia-ml.

The stand-in holds the module to its declared prototypes: every entry point converts its arguments through the
argtypes the module declares, as a ctypes function does, and process entries are written as raw bytes at NVML's ABI
offsets, so the module's entry struct must match the driver's layout. The real library, its calling convention and
the restypes are exercised only by the hardware test at the end, which needs Linux, an NVIDIA GPU and torch with CUDA.
"""

import ctypes
import logging
import os
import statistics
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from scripts.family_e2e import nvml_memory, runner
from scripts.family_e2e.gpu_memory import Adapter, MemoryTrace, ProcessUsage, UnsupportedPlatformError
from scripts.family_e2e.nvml_memory import ProcessMemorySampler, find_adapter, list_adapters, process_usage_on_adapter

GIB = 2**30
MIB = 2**20
NOT_AVAILABLE = 0xFFFFFFFFFFFFFFFF
DRIVER = "580.95.05"
RTX_6000 = ("NVIDIA RTX 6000 Ada Generation", "GPU-5f1c2e3a-1111-4000-8000-000000000001")
L4 = ("NVIDIA L4", "GPU-9b8a7c6d-2222-4000-8000-000000000002")
RTX_6000_ADAPTER = Adapter(luid=None, uuid=RTX_6000[1], description=RTX_6000[0], driver_version=DRIVER)
# NVML's codes for the failures scripted here.
NOT_SUPPORTED = 3
NO_PERMISSION = 4
NOT_FOUND = 6
INSUFFICIENT_SIZE = 7
GPU_IS_LOST = 15
LIB_RM_VERSION_MISMATCH = 18
# nvmlProcessInfo_t as the driver writes it: pid, padding, usedGpuMemory, gpuInstanceId, computeInstanceId.
PROCESS_ENTRY = struct.Struct("=I4xQII")

linux_only = pytest.mark.skipif(sys.platform != "linux", reason="the NVML sampler is the Linux backend")


class _Declared:
    """An entry point of the fake. Like a ctypes function it takes the argtypes and restype the module declares, and
    converts each argument through its argtype, so a call ctypes would refuse fails here too."""

    def __init__(self, implementation: Callable[..., Any]) -> None:
        self._implementation = implementation
        self.argtypes: list[Any] | None = None
        self.restype: Any = None

    def __call__(self, *args: Any) -> Any:
        name = self._implementation.__name__
        assert self.argtypes is not None and self.restype is not None, f"{name} is called without a declared prototype"
        assert len(args) == len(self.argtypes), f"{name} takes {len(self.argtypes)} arguments, got {len(args)}"
        for argtype, arg in zip(self.argtypes, args, strict=True):
            argtype.from_param(arg)
        return self._implementation(*args)


class _FakeNvml:
    """Devices as (name, uuid). `processes` and `graphics` map a device's uuid to its compute and graphics processes
    as [(pid, used bytes)]; `failing` holds device indices whose handle lookup NVML refuses.

    While a sampler runs, tests change what NVML answers through `change` and `wait_until_listed`.
    """

    def __init__(self, *devices: tuple[str, str], init_status: int = 0) -> None:
        self.devices = list(devices)
        self.processes: dict[str, list[tuple[int, int]]] = {uuid: [] for _, uuid in devices}
        self.graphics: dict[str, list[tuple[int, int]]] = {uuid: [] for _, uuid in devices}
        self.failing: set[int] = set()
        self.init_status = init_status
        self.inits = 0
        self.list_status = 0
        """Returned by every process listing while not 0."""
        self.graphics_status = 0
        """Returned by every graphics process listing while not 0."""
        self.before_list: Callable[[], None] | None = None
        """Runs at the start of every process listing."""
        self._listed = threading.Condition()
        self._version = 0
        self._served = 0
        for name in dir(self):
            if name.startswith("nvml"):
                setattr(self, name, _Declared(getattr(self, name)))

    def nvmlInit_v2(self) -> int:
        self.inits += 1
        return self.init_status

    def nvmlErrorString(self, status: int) -> bytes:
        texts = {NO_PERMISSION: b"Insufficient Permissions", NOT_FOUND: b"Not Found", GPU_IS_LOST: b"GPU is lost"}
        return texts.get(status, b"Unknown Error")

    def nvmlSystemGetDriverVersion(self, buffer, length: int) -> int:
        buffer.value = DRIVER.encode()
        return 0

    def nvmlDeviceGetCount_v2(self, count_ref) -> int:
        count_ref._obj.value = len(self.devices)
        return 0

    def nvmlDeviceGetHandleByIndex_v2(self, index: int, handle_ref) -> int:
        if index in self.failing:
            return NO_PERMISSION
        handle_ref._obj.value = index + 1  # a device handle is a non-null pointer
        return 0

    def nvmlDeviceGetHandleByUUID(self, uuid: bytes, handle_ref) -> int:
        uuids = [device_uuid.encode() for _, device_uuid in self.devices]
        if uuid not in uuids:
            return NOT_FOUND
        handle_ref._obj.value = uuids.index(uuid) + 1
        return 0

    def nvmlDeviceGetName(self, handle: ctypes.c_void_p, buffer, length: int) -> int:
        buffer.value = self.devices[handle.value - 1][0].encode()
        return 0

    def nvmlDeviceGetUUID(self, handle: ctypes.c_void_p, buffer, length: int) -> int:
        buffer.value = self.devices[handle.value - 1][1].encode()
        return 0

    def nvmlDeviceGetComputeRunningProcesses_v3(self, handle: ctypes.c_void_p, count_ref, entries) -> int:
        return self._list(self.processes, handle, count_ref, entries)

    def nvmlDeviceGetGraphicsRunningProcesses_v3(self, handle: ctypes.c_void_p, count_ref, entries) -> int:
        return self.graphics_status or self._list(self.graphics, handle, count_ref, entries)

    def _list(self, table: dict[str, list[tuple[int, int]]], handle: ctypes.c_void_p, count_ref, entries) -> int:
        if self.before_list is not None:
            self.before_list()
        with self._listed:
            version, status = self._version, self.list_status
            processes = list(table[self.devices[handle.value - 1][1]])
        try:
            if status:
                return status
            count = count_ref._obj
            if entries is None or count.value < len(processes):
                count.value = len(processes)
                return INSUFFICIENT_SIZE if processes else 0
            for index, (pid, used) in enumerate(processes):
                PROCESS_ENTRY.pack_into(entries, index * PROCESS_ENTRY.size, pid, used, 0, 0)
            count.value = len(processes)
            return 0
        finally:
            with self._listed:
                self._served = max(self._served, version)
                self._listed.notify_all()

    def change(self, uuid: str, processes: list[tuple[int, int]] | None = None, list_status: int | None = None) -> None:
        """What later listings of the device answer: its compute processes, or a failure status (0 for none)."""
        with self._listed:
            if processes is not None:
                self.processes[uuid] = processes
            if list_status is not None:
                self.list_status = list_status
            self._version += 1

    def wait_until_listed(self) -> None:
        """Blocks until a listing has answered with the state of the last `change`. A sampler stores what it read
        there before its `stop()` returns."""
        with self._listed:
            target = self._version
            assert self._listed.wait_for(lambda: self._served >= target, timeout=30), "the sampler stopped listing"


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[_FakeNvml], _FakeNvml]]:
    """Replaces libnvidia-ml with a fake; the module's NVML init is reset around each test."""
    nvml_memory._nvml.cache_clear()

    def installer(fake: _FakeNvml) -> _FakeNvml:
        monkeypatch.setattr(nvml_memory, "_open_library", lambda: fake)
        return fake

    yield installer
    nvml_memory._nvml.cache_clear()


def _exited_child() -> subprocess.Popen[bytes]:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child


def test_adapters_are_named_by_nvml_with_their_uuid_and_the_driver(install) -> None:
    fake = install(_FakeNvml(RTX_6000, L4))

    assert list_adapters() == [
        RTX_6000_ADAPTER,
        Adapter(luid=None, uuid=L4[1], description=L4[0], driver_version=DRIVER),
    ]
    assert list_adapters()[0].key == RTX_6000[1]
    assert fake.inits == 1, "NVML is initialised once per process"


def test_a_gpu_nvml_cannot_query_is_left_out_with_a_warning(install, caplog: pytest.LogCaptureFixture) -> None:
    fake = install(_FakeNvml(L4, RTX_6000))
    fake.failing = {0}

    with caplog.at_level(logging.WARNING, logger=nvml_memory.__name__):
        assert list_adapters() == [RTX_6000_ADAPTER]
        assert find_adapter("6000") == RTX_6000_ADAPTER
    assert any(
        "NVIDIA GPU 0 left out" in r.getMessage() and "Insufficient Permissions" in r.getMessage()
        for r in caplog.records
    )


def test_an_adapter_is_found_by_name_and_a_miss_lists_the_gpus_and_notes_amd(install) -> None:
    install(_FakeNvml(RTX_6000, L4))

    assert find_adapter("rtx 6000") == RTX_6000_ADAPTER
    with pytest.raises(LookupError) as miss:
        find_adapter("Radeon")
    assert "No adapter matches 'Radeon'" in str(miss.value)
    assert f"{RTX_6000[0]} ({RTX_6000[1]})" in str(miss.value)
    assert "AMD (ROCm) GPUs on Linux are not supported yet" in str(miss.value)
    with pytest.raises(LookupError) as several:
        find_adapter("NVIDIA")
    assert "2 adapters match 'NVIDIA'" in str(several.value)
    assert "AMD" not in str(several.value)


def test_identical_gpus_are_told_apart_by_their_uuid(install) -> None:
    name = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    first, second = (
        (name, "GPU-aaaaaaaa-1111-4000-8000-000000000001"),
        (name, "GPU-bbbbbbbb-2222-4000-8000-000000000002"),
    )
    install(_FakeNvml(first, second))

    assert find_adapter(second[1]).uuid == second[1]
    assert find_adapter(first[1].upper()).uuid == first[1]
    with pytest.raises(LookupError) as ambiguous:
        find_adapter("RTX PRO 6000")
    message = str(ambiguous.value)
    assert "2 adapters match 'RTX PRO 6000'" in message
    assert f"{name} ({first[1]})" in message and f"{name} ({second[1]})" in message
    assert "set gpu to the UUID of one of them; the runner pins the server to that GPU" in message


def test_open_gpu_on_linux_reads_memory_and_samples_through_nvml(
    install, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = install(_FakeNvml(RTX_6000))
    me = os.getpid()
    fake.processes[RTX_6000[1]] = [(me, GIB)]
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(runner, "WSL_KERNEL_RELEASE", tmp_path / "absent")

    gpu = runner.open_gpu("6000")

    assert gpu.adapter == RTX_6000_ADAPTER
    assert gpu.process_usage() == {me: ProcessUsage(dedicated_bytes=GIB, shared_bytes=0)}
    sampler = gpu.sampler(me)
    sampler.start()
    trace = sampler.stop()
    assert trace.method == "nvml" and trace.dedicated and trace.dedicated[0][1] == GIB
    with pytest.raises(runner.RunnerError, match="AMD"):
        runner.open_gpu("Radeon")


def test_open_gpu_refuses_linux_under_wsl(install, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install(_FakeNvml(RTX_6000))
    release = tmp_path / "osrelease"
    release.write_text("5.15.167.4-microsoft-standard-WSL2\n", encoding="utf-8")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(runner, "WSL_KERNEL_RELEASE", release)

    with pytest.raises(UnsupportedPlatformError, match="WSL.*run the tool on Windows"):
        runner.open_gpu("6000")


def test_without_the_nvml_library_sampling_is_unsupported(install, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nvml_memory, "_LIBRARY_NAMES", ("libnvidia-ml-absent.so.1",))
    with pytest.raises(UnsupportedPlatformError, match=r"reads NVML.*AMD \(ROCm\) GPUs on Linux are not supported"):
        list_adapters()


def test_a_driver_and_library_version_mismatch_asks_for_a_reboot(install) -> None:
    install(_FakeNvml(RTX_6000, init_status=LIB_RM_VERSION_MISMATCH))
    with pytest.raises(UnsupportedPlatformError, match="driver upgrade.*reboot"):
        list_adapters()


def test_process_usage_merges_compute_and_graphics_processes_of_that_gpu_only(install) -> None:
    fake = install(_FakeNvml(RTX_6000, L4))
    fake.processes[RTX_6000[1]] = [(10, GIB), (11, 2 * GIB)]
    # 11 has both kinds of context and is listed in both; 14's memory is not attributed and cannot count.
    fake.graphics[RTX_6000[1]] = [(11, 2 * GIB), (13, 512 * MIB), (14, NOT_AVAILABLE)]
    fake.processes[L4[1]] = [(12, 3 * GIB)]

    assert process_usage_on_adapter(RTX_6000_ADAPTER) == {
        10: ProcessUsage(dedicated_bytes=GIB, shared_bytes=0),
        11: ProcessUsage(dedicated_bytes=2 * GIB, shared_bytes=0),
        13: ProcessUsage(dedicated_bytes=512 * MIB, shared_bytes=0),
    }


def test_a_gpu_without_graphics_process_listing_reports_its_compute_processes(install) -> None:
    fake = install(_FakeNvml(RTX_6000))
    fake.processes[RTX_6000[1]] = [(10, GIB)]
    fake.graphics_status = NOT_SUPPORTED

    assert process_usage_on_adapter(RTX_6000_ADAPTER) == {10: ProcessUsage(dedicated_bytes=GIB, shared_bytes=0)}


def test_a_process_list_that_grows_while_it_is_read_is_read_again(install) -> None:
    fake = install(_FakeNvml(RTX_6000))
    processes = fake.processes[RTX_6000[1]]
    calls = 0

    def thirty_more_on_the_first_two_calls() -> None:
        # More than the spare entries each time, so each answer outgrows the room the previous one made.
        nonlocal calls
        calls += 1
        if calls <= 2:
            start = 100 + len(processes)
            processes.extend((pid, GIB + pid) for pid in range(start, start + 30))

    fake.before_list = thirty_more_on_the_first_two_calls

    usage = process_usage_on_adapter(RTX_6000_ADAPTER)

    assert len(usage) == 60
    assert {pid: value.dedicated_bytes for pid, value in usage.items()} == dict(processes)


def test_a_process_list_that_never_settles_fails_instead_of_looping(install) -> None:
    fake = install(_FakeNvml(RTX_6000))
    processes = fake.processes[RTX_6000[1]]
    new_pids = iter(range(100, 100_000))
    fake.before_list = lambda: processes.extend((next(new_pids), GIB) for _ in range(100))

    with pytest.raises(OSError, match="kept growing"):
        process_usage_on_adapter(RTX_6000_ADAPTER)


def test_memory_the_driver_does_not_attribute_is_refused(install) -> None:
    fake = install(_FakeNvml(RTX_6000))
    fake.processes[RTX_6000[1]] = [(os.getpid(), NOT_AVAILABLE)]

    with pytest.raises(UnsupportedPlatformError, match="does not attribute memory to processes"):
        process_usage_on_adapter(RTX_6000_ADAPTER)
    sampler = ProcessMemorySampler(os.getpid(), RTX_6000_ADAPTER)
    with pytest.raises(UnsupportedPlatformError, match="does not attribute memory to processes"):
        sampler.start()
    assert sampler.stop().dedicated == ()


def test_sampler_reads_its_process_and_zero_while_nvml_does_not_list_it(install) -> None:
    fake = install(_FakeNvml(RTX_6000))
    me, other = os.getpid(), 4_000_001
    fake.processes[RTX_6000[1]] = [(other, 5 * GIB)]
    sampler = ProcessMemorySampler(me, RTX_6000_ADAPTER)
    sampler.start()
    try:
        fake.change(RTX_6000[1], processes=[(other, 5 * GIB), (me, 3 * GIB)])
        fake.wait_until_listed()
    finally:
        trace = sampler.stop()

    values = [value for _, value in trace.dedicated]
    assert values[0] == 0 and values[-1] == 3 * GIB
    assert set(values) == {0, 3 * GIB}, "only the sampled process counts"
    assert (trace.method, trace.shared) == ("nvml", ())
    assert sampler.stop() is trace


def test_sampler_stops_when_its_process_exits(install) -> None:
    fake = install(_FakeNvml(RTX_6000))
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    assert child.stdin is not None
    fake.processes[RTX_6000[1]] = [(child.pid, GIB)]
    sampler = ProcessMemorySampler(child.pid, RTX_6000_ADAPTER)
    try:
        sampler.start()
    finally:
        child.stdin.close()
        child.wait(timeout=60)
    try:
        fake.change(RTX_6000[1], processes=[])
        fake.wait_until_listed()
    finally:
        trace = sampler.stop()

    # A process that is gone ends the trace; it does not read as zero from then on.
    assert trace.dedicated and {value for _, value in trace.dedicated} == {GIB}


def test_start_refuses_a_process_that_has_exited(install) -> None:
    install(_FakeNvml(RTX_6000))
    sampler = ProcessMemorySampler(_exited_child().pid, RTX_6000_ADAPTER)
    with pytest.raises(ProcessLookupError):
        sampler.start()
    assert sampler.stop().dedicated == ()


def test_nvml_failures_fail_start_and_only_skip_samples_once_running(install, caplog: pytest.LogCaptureFixture) -> None:
    fake = install(_FakeNvml(RTX_6000))
    me = os.getpid()
    fake.list_status = GPU_IS_LOST
    with pytest.raises(OSError, match="GPU is lost"):
        ProcessMemorySampler(me, RTX_6000_ADAPTER).start()

    fake.list_status = 0
    sampler = ProcessMemorySampler(me, RTX_6000_ADAPTER)
    with caplog.at_level(logging.WARNING, logger=nvml_memory.__name__):
        sampler.start()
        try:
            # Twice, so at least two listings fail.
            for _ in range(2):
                fake.change(RTX_6000[1], list_status=GPU_IS_LOST)
                fake.wait_until_listed()
            fake.change(RTX_6000[1], processes=[(me, 2 * GIB)], list_status=0)
            fake.wait_until_listed()
        finally:
            trace = sampler.stop()

    assert trace.dedicated[0][1] == 0 and trace.dedicated[-1][1] == 2 * GIB
    assert sum("GPU is lost" in record.getMessage() for record in caplog.records) == 1


def test_stop_without_start_returns_one_empty_trace() -> None:
    sampler = ProcessMemorySampler(1234, RTX_6000_ADAPTER)
    trace = sampler.stop()
    assert trace == MemoryTrace(method="nvml", interval_ms=5.0, dedicated=(), shared=())
    assert sampler.stop() is trace


def test_the_process_entry_matches_the_nvml_layout() -> None:
    """The driver writes entries by offset: a shifted field or stride reads a neighbour's bytes, it does not fail."""
    assert nvml_memory._ProcessInfo.usedGpuMemory.offset == 8
    assert ctypes.sizeof(nvml_memory._ProcessInfo) == PROCESS_ENTRY.size == 24


_CUDA_CHILD = """
import os, sys, torch
if not torch.cuda.is_available():
    print("unavailable", flush=True)
    sys.exit()
torch.zeros(1, device="cuda")
torch.cuda.synchronize()
# CUDA device 0 after CUDA_VISIBLE_DEVICES, spelled as NVML spells its UUID.
print(os.getpid(), "GPU-" + str(torch.cuda.get_device_properties(0).uuid), flush=True)
sys.stdin.readline()
x = torch.empty(int(sys.argv[1]), dtype=torch.uint8, device="cuda")
x.fill_(1)
torch.cuda.synchronize()
print("allocated", flush=True)
sys.stdin.readline()
"""


def _end_child(child: subprocess.Popen[str]) -> None:
    """Lets the child finish, or kills it. Raises nothing, so a failure of the test itself stays the one reported."""
    try:
        child.communicate("\n", timeout=60)
    except Exception:
        child.kill()
        child.wait()


def _tail(path: Path, lines: int = 20) -> str:
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


@pytest.mark.slow
@linux_only
def test_two_cuda_processes_are_each_charged_their_own_memory(
    record_property: Callable[[str, object], None], tmp_path: Path
) -> None:
    """Needs NVML and torch with CUDA. Two processes at once, with different allocations: an entry stride that does
    not match the driver's would hand one process the other's memory, or garbage.

    The GPU is the children's CUDA device 0, by its UUID, so the test honours CUDA_VISIBLE_DEVICES on a machine with
    several GPUs. This process stays without a CUDA context, like the tool."""
    if runner._under_wsl():
        pytest.skip("the tool refuses WSL: NVML there reports no per-process memory")
    try:
        list_adapters()
    except UnsupportedPlatformError as error:
        pytest.skip(str(error))
    # NVML's own error text comes back through nvmlErrorString's declared restype.
    absent = Adapter(luid=None, uuid="GPU-00000000-0000-0000-0000-000000000000", description="no such GPU")
    with pytest.raises(OSError, match=r"nvmlDeviceGetHandleByUUID .* failed: (?!unknown error).+ \(NVML error \d+\)"):
        nvml_memory._ProcessTable(absent)
    sizes = (GIB, 2 * GIB)
    # stderr into files, not pipes: torch's warnings could fill a pipe nobody reads and block the child.
    logs = [tmp_path / f"child-{index}.log" for index in range(len(sizes))]
    children = []
    for size, log in zip(sizes, logs, strict=True):
        with log.open("wb") as stderr:
            children.append(
                subprocess.Popen(
                    [sys.executable, "-c", _CUDA_CHILD, str(size)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                    text=True,
                )
            )
    sampler = None
    try:
        pids, uuids = [], set()
        for child, log in zip(children, logs, strict=True):
            assert child.stdout is not None
            first = child.stdout.readline().split()
            if first == ["unavailable"]:
                pytest.skip("torch in this environment has no CUDA")
            if len(first) != 2 or not first[0].isdigit():
                pytest.fail(f"the CUDA child failed (stdout {first!r}); its stderr ends with:\n{_tail(log)}")
            pids.append(int(first[0]))
            uuids.add(first[1])
        assert len(uuids) == 1, f"the children run on different GPUs: {uuids}"
        # Not caught: a UUID NVML does not list is a failure, not a reason to skip.
        adapter = find_adapter(uuids.pop())
        sampler = ProcessMemorySampler(pids[1], adapter)
        sampler.start()
        contexts = process_usage_on_adapter(adapter)

        def allocate(child: subprocess.Popen[str]) -> float:
            """Let the child allocate; the instant just before it, read against the trace afterwards."""
            assert child.stdin is not None and child.stdout is not None
            t = time.perf_counter()
            child.stdin.write("\n")
            child.stdin.flush()
            assert child.stdout.readline().strip() == "allocated"
            # Let samples land after the allocation before anything else changes.
            time.sleep(0.05)
            return t

        t_before_first = allocate(children[0])
        t_before_second = allocate(children[1])
        t_end = time.perf_counter()
        allocated = process_usage_on_adapter(adapter)

        table = nvml_memory._ProcessTable(adapter)
        latencies = []
        for _ in range(2000):
            t0 = time.perf_counter()
            table.read()
            latencies.append(time.perf_counter() - t0)
        trace = sampler.stop()
    finally:
        if sampler is not None:
            sampler.stop()
        for child in children:
            _end_child(child)

    tolerance = GIB / 16
    for pid, size in zip(pids, sizes, strict=True):
        assert contexts[pid].dedicated_bytes > 0, "a CUDA context holds memory"
        assert abs(allocated[pid].dedicated_bytes - contexts[pid].dedicated_bytes - size) < tolerance
    second_context, before_second, after_second = (trace.at(t) for t in (t_before_first, t_before_second, t_end))
    assert second_context is not None and before_second is not None and after_second is not None
    assert abs(before_second - second_context) < tolerance, "the first child's allocation is not the second's"
    assert abs(after_second - second_context - sizes[1]) < tolerance

    cuts = statistics.quantiles(latencies, n=100)
    p50_us, p99_us = cuts[49] * 1e6, cuts[98] * 1e6
    record_property("nvml_process_list_p50_us", round(p50_us, 1))
    record_property("nvml_process_list_p99_us", round(p99_us, 1))
    record_property("sampler_interval_ms", round(trace.interval_ms, 3))
    print(
        f"NVML process listing over {len(latencies)} calls: p50 {p50_us:.0f} us, p99 {p99_us:.0f} us; "
        f"sampler interval {trace.interval_ms:.2f} ms"
    )
