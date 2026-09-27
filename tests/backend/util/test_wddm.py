"""The Windows video-memory readers, against scripted stand-ins for the gdi32 kernel thunks and for PDH.

The stand-ins are driven through the same ctypes structures the module passes, so struct layout, the two-call adapter
enumeration, the PCI match, handle cleanup and the counter-instance naming are exercised; only the OS behind them is fake.
"""

import ctypes
import functools
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from invokeai.backend.util import wddm
from invokeai.backend.util.devices import TorchDevice

GIB = 1024**3
DEVICE = torch.device("cuda", 0)
OUR_BUS, OUR_DEVICE = 3, 0


def _luid_low(handle: int) -> int:
    return 0x22A00 + handle


class _FakePdh:
    """A PDH with the module's two counters: per-process shared usage, and adapter-wide dedicated usage."""

    def __init__(self, instances: dict[str, int], adapter_instances: dict[str, int] | None = None):
        self.instances = instances
        self.adapter_instances = adapter_instances or {}
        self.counter_paths: list[str] = []
        self.closed = 0
        self._alive: list[object] = []
        self._by_handle: dict[int, dict[str, int]] = {}

    def PdhOpenQueryW(self, source, user_data, query_ref):
        query_ref._obj.value = 1
        return 0

    def PdhAddEnglishCounterW(self, query, path, user_data, counter_ref):
        self.counter_paths.append(path)
        handle = 2 + len(self.counter_paths)
        counter_ref._obj.value = handle
        self._by_handle[handle] = self.adapter_instances if "Adapter" in path else self.instances
        return 0

    def PdhCollectQueryData(self, query):
        return 0

    def PdhGetFormattedCounterArrayW(self, counter, fmt, size_ref, count_ref, buffer):
        instances = self._by_handle[counter.value if hasattr(counter, "value") else counter]
        count = len(instances)
        if buffer is None:
            size_ref._obj.value = ctypes.sizeof(wddm._PdhCounterValueItem) * count
            count_ref._obj.value = count
            return wddm._PDH_MORE_DATA
        items = (wddm._PdhCounterValueItem * count).from_buffer(buffer)
        for item, (name, value) in zip(items, instances.items(), strict=True):
            item.szName = name
            item.FmtValue.largeValue = value
        self._alive.append(items)  # the name buffers live as long as the array object
        count_ref._obj.value = count
        return 0

    def PdhCloseQuery(self, query):
        self.closed += 1
        return 0


class _FakeGdi32:
    """Adapters as (handle, bus, device); every thunk returns NTSTATUS 0 unless told otherwise."""

    def __init__(self, adapters, budget=15 * GIB, query_status=0):
        self.adapters = adapters
        self.budget = budget
        self.query_status = query_status
        self.enum_calls = 0
        self.closed: list[int] = []

    def D3DKMTEnumAdapters2(self, ref):
        self.enum_calls += 1
        enum = ref._obj
        if not enum.pAdapters:
            enum.NumAdapters = len(self.adapters)
            return 0
        for i, (handle, _bus, _device) in enumerate(self.adapters):
            enum.pAdapters[i].hAdapter = handle
            enum.pAdapters[i].AdapterLuid.LowPart = _luid_low(handle)
            enum.pAdapters[i].AdapterLuid.HighPart = 0
        return 0

    def D3DKMTQueryAdapterInfo(self, ref):
        query = ref._obj
        address = wddm._AdapterAddress.from_address(query.pPrivateDriverData)
        _handle, bus, device = next(a for a in self.adapters if a[0] == query.hAdapter)
        address.BusNumber, address.DeviceNumber = bus, device
        return 0

    def D3DKMTQueryVideoMemoryInfo(self, ref):
        info = ref._obj
        info.Budget = self.budget
        return self.query_status

    def D3DKMTCloseAdapter(self, ref):
        self.closed.append(ref._obj.hAdapter)
        return 0


@pytest.fixture
def windows_rocm(monkeypatch: pytest.MonkeyPatch):
    """A Windows ROCm build whose device 0 sits at PCI bus 3 with 16 GiB. Returns an installer for a fake gdi32."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(torch.version, "hip", "7.14.0")
    props = SimpleNamespace(pci_bus_id=OUR_BUS, pci_device_id=OUR_DEVICE, total_memory=16 * GIB)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: props)
    wddm.reset_cache()

    def install(fake: _FakeGdi32) -> _FakeGdi32:
        monkeypatch.setattr(wddm, "_load_gdi32", functools.lru_cache(maxsize=1)(lambda: fake))
        return fake

    yield install
    wddm.reset_cache()


def test_reads_the_budget_of_the_adapter_at_the_devices_pci_location(windows_rocm):
    gdi32 = windows_rocm(_FakeGdi32([(10, 7, 0), (20, OUR_BUS, OUR_DEVICE), (30, 0xFFFFFFFF, 0xFFFF)], budget=15 * GIB))

    assert wddm.video_memory_budget(DEVICE) == 15 * GIB
    assert sorted(gdi32.closed) == [10, 30], "the other adapters' handles are closed, ours stays open"


def test_resolves_the_adapter_once(windows_rocm):
    gdi32 = windows_rocm(_FakeGdi32([(20, OUR_BUS, OUR_DEVICE)]))

    wddm.video_memory_budget(DEVICE)
    gdi32.budget = 12 * GIB  # another GPU process started
    assert wddm.video_memory_budget(DEVICE) == 12 * GIB
    assert gdi32.enum_calls == 2, "one count call and one fill call, for the first query only"


@pytest.mark.parametrize(
    ("adapters", "kwargs"),
    [
        ([(10, 7, 0)], {}),
        ([(20, OUR_BUS, OUR_DEVICE), (21, OUR_BUS, OUR_DEVICE)], {}),
        ([(20, OUR_BUS, OUR_DEVICE)], {"query_status": -1073741823}),
        ([(20, OUR_BUS, OUR_DEVICE)], {"budget": 0}),
        ([(20, OUR_BUS, OUR_DEVICE)], {"budget": 17 * GIB}),
    ],
    ids=[
        "no-adapter-at-the-location",
        "two-adapters-at-the-location",
        "query-fails",
        "zero-budget",
        "budget-over-total",
    ],
)
def test_unknown_rather_than_guessed(windows_rocm, adapters, kwargs):
    gdi32 = windows_rocm(_FakeGdi32(adapters, **kwargs))

    assert wddm.video_memory_budget(DEVICE) is None
    if len(adapters) == 2:
        assert sorted(gdi32.closed) == [20, 21], "an ambiguous match must not leak either handle"


@pytest.mark.parametrize(
    ("platform", "hip", "device"),
    [("linux", "7.2.0", DEVICE), ("win32", None, DEVICE), ("win32", "7.14.0", torch.device("cpu"))],
    ids=["linux-rocm", "windows-cuda", "cpu-device"],
)
def test_elsewhere_gdi32_is_never_touched(windows_rocm, monkeypatch, platform, hip, device):
    def loader():
        raise AssertionError("gdi32 must not be loaded outside Windows ROCm")

    monkeypatch.setattr(wddm, "_load_gdi32", functools.lru_cache(maxsize=1)(loader))
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(torch.version, "hip", hip)

    assert wddm.video_memory_budget(device) is None


def test_reset_closes_the_cached_handle(windows_rocm):
    gdi32 = windows_rocm(_FakeGdi32([(20, OUR_BUS, OUR_DEVICE)]))
    wddm.video_memory_budget(DEVICE)

    wddm.reset_cache()

    assert gdi32.closed == [20]


def _instance(pid: int, handle: int) -> str:
    return f"pid_{pid}_luid_0x00000000_0x{_luid_low(handle):08X}_phys_0"


def test_paged_bytes_reads_this_process_on_the_devices_adapter(windows_rocm, monkeypatch):
    windows_rocm(_FakeGdi32([(10, 7, 0), (20, OUR_BUS, OUR_DEVICE)]))
    me = os.getpid()
    pdh = _FakePdh(
        {
            _instance(me, 10): 5 * GIB,  # this process, other adapter
            _instance(me + 1, 20): 7 * GIB,  # another process, our adapter
            _instance(me, 20): 3 * GIB,  # this process, our adapter (PDH may report upper-case hex)
        }
    )
    monkeypatch.setattr(wddm, "_load_pdh", functools.lru_cache(maxsize=1)(lambda: pdh))

    assert wddm.paged_bytes(DEVICE) == 3 * GIB
    assert wddm.paged_bytes(DEVICE) == 3 * GIB
    assert pdh.counter_paths == [
        r"\GPU Process Memory(*)\Shared Usage",
        r"\GPU Adapter Memory(*)\Dedicated Usage",
    ], "one query with both counters, opened on first use"

    wddm.reset_cache()
    assert pdh.closed == 1


def _adapter_instance(handle: int) -> str:
    return f"luid_0x00000000_0x{_luid_low(handle):08x}_phys_0"


@pytest.mark.parametrize(
    ("adapter_gib", "ours_gib", "expected_gib"),
    [(8.4, 0.2, 8.2), (12.2, 12.1, 0.0), (0.2, 6.0, 0.0)],
    ids=["another-process-holds-the-card", "all-of-it-is-ours", "adapter-reads-below-our-own"],
)
def test_other_process_local_bytes_subtracts_our_own_usage(
    windows_rocm, monkeypatch, adapter_gib, ours_gib, expected_gib
):
    """What bounds a decode is what OTHER processes hold: the budget does not track them (measured 15.09 GiB next to
    an 8 GiB holder), and the per-process PDH counter swapped its instances while two processes shared the card, so
    this comes from the adapter-wide figure minus torch's own-usage view. A reading below our own clamps to zero."""
    windows_rocm(_FakeGdi32([(20, OUR_BUS, OUR_DEVICE)]))
    total = 16 * GIB
    pdh = _FakePdh({}, adapter_instances={_adapter_instance(20): int(adapter_gib * GIB)})
    monkeypatch.setattr(wddm, "_load_pdh", functools.lru_cache(maxsize=1)(lambda: pdh))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (total - int(ours_gib * GIB), total))

    assert wddm.other_process_local_bytes(DEVICE) == pytest.approx(int(expected_gib * GIB), abs=GIB // 10)


def test_other_process_local_bytes_is_unknown_without_an_adapter_reading(windows_rocm, monkeypatch):
    windows_rocm(_FakeGdi32([(20, OUR_BUS, OUR_DEVICE)]))
    pdh = _FakePdh({}, adapter_instances={_adapter_instance(21): 8 * GIB})  # a different adapter
    monkeypatch.setattr(wddm, "_load_pdh", functools.lru_cache(maxsize=1)(lambda: pdh))

    assert wddm.other_process_local_bytes(DEVICE) is None


def test_paged_bytes_is_unknown_before_this_process_used_the_adapter(windows_rocm, monkeypatch):
    windows_rocm(_FakeGdi32([(20, OUR_BUS, OUR_DEVICE)]))
    pdh = _FakePdh({_instance(os.getpid() + 1, 20): GIB})
    monkeypatch.setattr(wddm, "_load_pdh", functools.lru_cache(maxsize=1)(lambda: pdh))

    assert wddm.paged_bytes(DEVICE) is None


def test_paged_bytes_elsewhere_never_touches_pdh(monkeypatch):
    def loader():
        raise AssertionError("pdh must not be loaded outside Windows ROCm")

    monkeypatch.setattr(wddm, "_load_pdh", functools.lru_cache(maxsize=1)(loader))
    monkeypatch.setattr(sys, "platform", "linux")

    assert wddm.paged_bytes(DEVICE) is None


def test_structures_match_the_d3dkmt_layout():
    """The kernel reads these by offset: a shifted field would read a neighbour's bytes, not fail."""
    assert ctypes.sizeof(wddm._AdapterInfo) == 20
    assert wddm._QueryVideoMemoryInfo.Budget.offset == 16
    assert ctypes.sizeof(wddm._QueryVideoMemoryInfo) == 56
    # PDH_FMT_COUNTERVALUE_ITEM_W on 64-bit: a name pointer, then the value whose 8-byte union follows a DWORD status.
    assert wddm._PdhCounterValueItem.FmtValue.offset == 8
    assert wddm._PdhCounterValue.largeValue.offset == 8
    assert ctypes.sizeof(wddm._PdhCounterValueItem) == 24


needs_windows_rocm = pytest.mark.skipif(
    sys.platform != "win32" or torch.version.hip is None or not torch.cuda.is_available(),
    reason="needs a ROCm GPU under Windows",
)


@pytest.mark.slow
@needs_windows_rocm
def test_on_hardware_freed_memory_counts_as_headroom_again():
    """HIP keeps memory after a free and hands it to the next allocation; WDDM still counts it as this process's usage.
    The capped free figure must recover anyway, or an offload the model cache just made would look like no progress."""
    wddm.reset_cache()
    device = torch.device("cuda", 0)
    torch.empty(1, device=device)
    torch.cuda.empty_cache()
    assert wddm.video_memory_budget(device) is not None
    before, _ = TorchDevice.cuda_mem_get_info(device)

    block = torch.empty(2 * GIB, dtype=torch.uint8, device=device).fill_(1)
    torch.cuda.synchronize()
    held, _ = TorchDevice.cuda_mem_get_info(device)
    del block
    torch.cuda.empty_cache()
    after, _ = TorchDevice.cuda_mem_get_info(device)

    assert before - held >= 1.9 * GIB
    assert after >= before - 64 * 2**20


@pytest.mark.slow
@needs_windows_rocm
def test_on_hardware_the_paged_amount_of_this_process_is_found():
    """The warning depends on finding this process's instance under the name the code builds; if that is wrong,
    `paged_bytes` quietly answers None forever."""
    wddm.reset_cache()
    device = torch.device("cuda", 0)
    block = torch.empty(64 * 2**20, dtype=torch.uint8, device=device).fill_(1)
    torch.cuda.synchronize()

    paged = wddm.paged_bytes(device)

    assert paged is not None and paged >= 0
    del block
