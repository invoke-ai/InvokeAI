"""The NVML readers, against a scripted stand-in for the NVML library.

The stand-in fills the same ctypes structures the module passes, so the two-call process enumeration, the PCI lookup
and the struct layout are exercised; only the driver behind them is fake.
"""

import ctypes
import functools
import os
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from invokeai.backend.util import nvml

GIB = 2**30
DEVICE = torch.device("cuda", 0)
OUR_BUS_ID = b"0000:03:00.0"


class _FakeNvml:
    """A device at `OUR_BUS_ID` running `processes` as (pid, used bytes)."""

    def __init__(self, processes: list[tuple[int, int]], driver: str = "581.57"):
        self.processes = processes
        self.driver = driver
        self.bus_ids: list[bytes] = []

    def nvmlDeviceGetHandleByPciBusId_v2(self, bus_id: bytes, handle_ref) -> int:
        self.bus_ids.append(bus_id)
        if bus_id != OUR_BUS_ID:
            return 6  # NVML_ERROR_NOT_FOUND
        handle_ref._obj.value = 0x5EED
        return 0

    def nvmlDeviceGetComputeRunningProcesses_v3(self, handle, count_ref, infos) -> int:
        count = count_ref._obj
        if infos is None or count.value < len(self.processes):
            count.value = len(self.processes)
            return 7 if self.processes else 0  # NVML_ERROR_INSUFFICIENT_SIZE
        for info, (pid, used) in zip(infos, self.processes, strict=False):
            info.pid, info.usedGpuMemory = pid, used
        count.value = len(self.processes)
        return 0

    def nvmlSystemGetDriverVersion(self, buffer, length: int) -> int:
        buffer.value = self.driver.encode()
        return 0


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch):
    """A CUDA build whose device 0 sits at PCI bus 3. Returns an installer for a fake NVML (or None for no NVML)."""
    monkeypatch.setattr(torch.version, "hip", None)
    props = SimpleNamespace(pci_domain_id=0, pci_bus_id=3, pci_device_id=0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: props)
    # The host's PID namespace, wherever the tests run (a CI container may have one of its own).
    monkeypatch.setattr(nvml, "_in_initial_pid_namespace", lambda: True)
    nvml.reset_cache()

    def installer(fake: Optional[_FakeNvml]) -> Optional[_FakeNvml]:
        monkeypatch.setattr(nvml, "_load", functools.lru_cache(maxsize=1)(lambda: fake))
        return fake

    yield installer
    nvml.reset_cache()


def test_reads_this_processs_memory_on_the_device_at_its_pci_address(install):
    me = os.getpid()
    fake = install(_FakeNvml([(me + 1, 9 * GIB), (me, 3 * GIB), (me + 2, GIB)]))

    assert nvml.process_used_memory(DEVICE) == 3 * GIB
    assert nvml.process_used_memory(DEVICE) == 3 * GIB
    assert fake.bus_ids == [OUR_BUS_ID], "the handle is looked up once"


@pytest.mark.parametrize(
    "processes",
    [[(1, GIB)], [], [(os.getpid(), 0xFFFFFFFFFFFFFFFF)]],
    ids=["not-listed", "no-processes", "not-attributed"],
)
def test_unknown_when_nvml_does_not_attribute_memory_to_this_process(install, processes):
    install(_FakeNvml(processes))
    assert nvml.process_used_memory(DEVICE) is None


def test_unknown_without_nvml_off_cuda_and_when_a_call_raises(install, monkeypatch: pytest.MonkeyPatch):
    install(None)
    assert nvml.process_used_memory(DEVICE) is None
    assert nvml.driver_version() is None

    fake = install(_FakeNvml([(os.getpid(), GIB)]))
    monkeypatch.setattr(torch.version, "hip", "7.2.0")
    assert nvml.process_used_memory(DEVICE) is None
    assert fake.bus_ids == [], "a HIP build never asks NVML"

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(fake, "nvmlDeviceGetComputeRunningProcesses_v3", lambda *args: 1 / 0)
    assert nvml.process_used_memory(DEVICE) is None


def test_reads_the_driver_version(install):
    install(_FakeNvml([], driver="581.57"))
    assert nvml.driver_version() == "581.57"


def test_the_process_entry_matches_the_nvml_layout():
    """The driver writes entries by offset: a shifted field would read a neighbour's bytes, not fail."""
    assert nvml._ProcessInfo.usedGpuMemory.offset == 8
    assert ctypes.sizeof(nvml._ProcessInfo) == 24


def test_unknown_in_a_pid_namespace_of_its_own_where_an_unrelated_host_process_has_its_id(
    install, monkeypatch: pytest.MonkeyPatch
):
    """NVML lists the host's process ids: in a container without the host's PID namespace, this process's own id can
    be another process's on the host."""
    install(_FakeNvml([(os.getpid(), 5 * GIB)]))
    monkeypatch.setattr(nvml, "_in_initial_pid_namespace", lambda: False, raising=False)

    assert nvml.process_used_memory(DEVICE) is None


def test_the_pid_namespace_is_read_from_procfs(tmp_path, monkeypatch: pytest.MonkeyPatch):
    namespace = tmp_path / "pid"
    namespace.write_bytes(b"")
    monkeypatch.setattr(nvml, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(nvml, "_PID_NAMESPACE_LINK", str(namespace))

    assert not nvml._in_initial_pid_namespace()
    monkeypatch.setattr(nvml, "_INITIAL_PID_NAMESPACE_INODE", os.stat(namespace).st_ino)
    assert nvml._in_initial_pid_namespace()
    namespace.unlink()
    assert not nvml._in_initial_pid_namespace(), "unreadable is not the host's namespace"
    monkeypatch.setattr(nvml, "sys", SimpleNamespace(platform="win32"))
    assert nvml._in_initial_pid_namespace(), "no PID namespaces off Linux"
