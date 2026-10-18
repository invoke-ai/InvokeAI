"""Fresh windows (`measurement.measure`): one operation measured alone, as the calibration scripts measure their points.

They run on the probe tests' scripted device, whose release returns the allocator's cached blocks when
`TorchDevice.empty_cache` runs. What a window's readings mean -- the demand, the method, the reasons -- is shared with
the probe's windows and tested there; these cover what a fresh window does around the operation.
"""

from typing import Iterator

import pytest
import torch

from invokeai.backend.util import devices
from invokeai.backend.working_memory import counters, measurement
from invokeai.backend.working_memory.measurement import measure
from tests.backend.working_memory.test_probe import FakeCounters, FakeDevice

GIB = 2**30
MIB = 2**20


@pytest.fixture
def device(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeDevice]:
    fake = FakeDevice()
    opened: list[FakeCounters] = []

    def device_counters(torch_device: torch.device) -> FakeCounters:
        opened.append(FakeCounters(fake))
        return opened[-1]

    monkeypatch.setattr(devices, "EMPTY_CACHE_OBSERVERS", [fake])
    monkeypatch.setattr(counters, "DeviceCounters", device_counters)
    monkeypatch.setattr(measurement, "SPILL_SETTLE_S", 0.0)
    yield fake
    assert opened and all(c.closed for c in opened), "every window closes its counters"


def test_a_fresh_window_takes_what_the_operation_took_over_a_released_baseline(device: FakeDevice):
    device.alloc(1 * GIB)  # the weights and inputs, in place
    device.alloc(3 * GIB)
    device.free(3 * GIB)  # an earlier forward's blocks, still cached, and its peak

    def forward() -> None:
        device.alloc(2 * GIB)
        device.free(2 * GIB)

    window = measure(torch.device("cpu"), forward)

    # Served from the cached blocks, the forward would grow neither the driver nor reserved; under the earlier peak,
    # neither peak would show it.
    assert (window.driver, window.reserved, window.allocated, window.demand) == (2 * GIB, 2 * GIB, 2 * GIB, 2 * GIB)
    assert (window.reasons, window.counter_method) == ((), "windows-d3dkmt")


def test_an_operation_out_of_memory_gives_an_invalid_window_and_other_errors_propagate(device: FakeDevice):
    def out_of_memory() -> None:
        device.num_ooms += 1  # the allocator counts the allocation it failed
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 GiB")

    def wrong_shapes() -> None:
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied")

    window = measure(torch.device("cpu"), out_of_memory)
    with pytest.raises(RuntimeError, match="shapes"):
        measure(torch.device("cpu"), wrong_shapes)

    assert window.reasons == ("oom",)


@pytest.mark.parametrize(
    ("paged", "reasons"),
    [([0, 600 * MIB, 9 * MIB], ()), ([0, 600 * MIB, 600 * MIB], ("spill",))],
    ids=["transient", "persistent"],
)
def test_only_spill_that_persists_after_the_operation_voids_a_fresh_window(
    device: FakeDevice, paged: list[int], reasons: tuple[str, ...]
):
    """The readings at the baseline, at the close and after the settle time."""
    device.paged = list(paged)

    window = measure(torch.device("cpu"), lambda: None)

    assert (window.paged, window.reasons) == (600 * MIB, reasons)
