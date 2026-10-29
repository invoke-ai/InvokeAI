"""An operation's working memory, from this process's counter readings at the two ends of a window.

A window spans one operation on a compute device with its weights and inputs in place. It opens with a synchronize, a
release of the allocator's cached blocks, the baseline reading and a reset of the peaks: with nothing cached, whatever
the operation takes shows as growth. It closes with a synchronize and a reading. Per counter (the driver's figure,
torch's reserved and allocated bytes; see `counters`) the window takes its peak's growth over the baseline, and the
operation's *demand* is the largest growth of the counters that could be read: no counter alone covers every platform
(`counters` has the measurements).

A window is invalid when its readings show that

- an allocation failed (``oom``), or only succeeded after the allocator released its cache and retried
  (``alloc_retry``). A retry squeezes the peak: an untiled 3072px FLUX VAE decode on an RTX 4090 read 22.5 GiB at the
  card's limit, ~1280 bytes per pixel instead of ~2150;
- Windows still keeps more than `SPILL_TOLERANCE_BYTES` of the process in shared memory `SPILL_SETTLE_S` after the close
  (``spill``). Right after a decode the counter spikes transiently, 589 MiB, and is back at 9 MiB at the next read;
- a device-wide counter (XPU) grew by more than this process's own reserved bytes: another process's load
  (``foreign_device_load``);
- no counter could be read (``counters_unavailable``).

Two kinds of window take these definitions. The in-server probe's (`probe`) follow an operation through co-locks, nested
windows and releases, in segments, and have reasons of their own. A *fresh window* (`measure`) is one operation alone:
the calibration scripts measure their points that way, each in a fresh process with the server's runtime
(`invokeai.app.run_app.apply_server_runtime`).
"""

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Final, Literal, Optional

import torch

from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.working_memory import counters
from invokeai.backend.working_memory.counters import CounterReading, Method

MIB = 2**20

SPILL_TOLERANCE_BYTES: Final = 256 * MIB
"""Shared-memory bytes above the window's baseline that count as spill once they persist."""
SPILL_SETTLE_S: float = 0.5
"""How long a close waits before re-reading a raised shared-memory counter. Module-level so tests can shorten it."""
FOREIGN_DEVICE_TOLERANCE_BYTES: Final = 256 * MIB
"""Growth of a device-wide counter (XPU) beyond this process's own reserved growth that marks another process's load."""

CounterName = Literal["driver", "reserved", "allocated"]
COUNTERS: Final[tuple[CounterName, ...]] = ("driver", "reserved", "allocated")

ReadingReason = Literal["oom", "alloc_retry", "spill", "foreign_device_load", "counters_unavailable"]
"""Why a window's readings void it. The probe has reasons of its own on top (`probe.Reason`)."""


def counter_value(reading: CounterReading, name: CounterName) -> Optional[int]:
    """`name`'s value in `reading`, as a baseline takes it."""
    return reading.driver if name == "driver" else reading.reserved if name == "reserved" else reading.allocated


def counter_peak(reading: CounterReading, name: CounterName) -> Optional[int]:
    """`name`'s peak since the last reset in `reading`. The driver keeps none outside a sampler (MPS): its value then."""
    if name == "driver":
        return reading.peak_driver if reading.peak_driver is not None else reading.driver
    return reading.peak_reserved if name == "reserved" else reading.peak_allocated


def demand(deltas: Iterable[Optional[int]]) -> Optional[int]:
    """The demand of an operation whose counters grew by `deltas`: the largest that could be read, None if none could."""
    readable = [delta for delta in deltas if delta is not None]
    return max(readable) if readable else None


def window_method(first: CounterReading, driver_readable: bool) -> Method:
    """Where a window's driver figure came from: the method of its baseline `first`, unless a driver figure read there
    dropped out later (`driver_readable` False), which leaves the window to torch's counters."""
    return "torch-only" if first.driver is not None and not driver_readable else first.method


def _rose(before: Optional[int], after: Optional[int]) -> bool:
    return before is not None and after is not None and after > before


def reading_reasons(first: CounterReading, final: CounterReading) -> list[ReadingReason]:
    """Why the readings at a window's open (`first`) and close (`final`) void it, spill and unreadable counters aside."""
    reasons: list[ReadingReason] = []
    if _rose(first.num_ooms, final.num_ooms):
        reasons.append("oom")
    if _rose(first.num_alloc_retries, final.num_alloc_retries):
        reasons.append("alloc_retry")
    if (
        first.device_used is not None
        and final.device_used is not None
        and first.reserved is not None
        and final.reserved is not None
        and (final.device_used - first.device_used) - (final.reserved - first.reserved) > FOREIGN_DEVICE_TOLERANCE_BYTES
    ):
        reasons.append("foreign_device_load")
    return reasons


def spilled(first: CounterReading, final: CounterReading, read: Callable[[], CounterReading]) -> bool:
    """Whether the process stays paged into shared memory after a window: `final` holds more than
    `SPILL_TOLERANCE_BYTES` above `first`, and so does a `read` `SPILL_SETTLE_S` later."""
    baseline = first.paged
    if baseline is None or final.paged is None or final.paged - baseline <= SPILL_TOLERANCE_BYTES:
        return False
    time.sleep(SPILL_SETTLE_S)
    settled = read().paged
    return settled is None or settled - baseline > SPILL_TOLERANCE_BYTES


@dataclass(frozen=True)
class FreshWindow:
    """One operation measured as a window of its own: per counter its peak's growth over the baseline, None where it
    could not be read."""

    driver: Optional[int]
    reserved: Optional[int]
    allocated: Optional[int]
    paged: Optional[int]
    """How many more of the process's bytes Windows kept in shared memory at the close than at the baseline. A record:
    `spilled` judges spill after the settle time."""
    demand: Optional[int]
    counter_method: Method
    """Where the driver figure came from (`window_method`)."""
    reasons: tuple[ReadingReason, ...]
    """Why the window is invalid; none for a valid one."""


def fresh_window(first: CounterReading, final: CounterReading, *, oom: bool, spill: bool) -> FreshWindow:
    """The fresh window of an operation between `first`, read after the release before it, and `final`, read after it.

    `oom` says the operation ran out of memory, `spill` that the process stayed paged (`spilled`).
    """
    deltas: dict[CounterName, Optional[int]] = {}
    for name in COUNTERS:
        base, top = counter_value(first, name), counter_peak(final, name)
        deltas[name] = None if base is None or top is None else top - base
    reasons: list[ReadingReason] = ["oom"] if oom else []
    reasons += [reason for reason in reading_reasons(first, final) if reason not in reasons]
    if spill:
        reasons.append("spill")
    operation_demand = demand(deltas.values())
    if operation_demand is None:
        reasons.append("counters_unavailable")
    return FreshWindow(
        driver=deltas["driver"],
        reserved=deltas["reserved"],
        allocated=deltas["allocated"],
        paged=None if first.paged is None or final.paged is None else final.paged - first.paged,
        demand=operation_demand,
        counter_method=window_method(first, driver_readable=deltas["driver"] is not None),
        reasons=tuple(reasons),
    )


def measure(device: torch.device, operation: Callable[[], object]) -> FreshWindow:
    """Run `operation` on `device` as a fresh window: synchronize, release the allocator's cached blocks, read the
    baseline and reset the peaks; run it; synchronize and read.

    An operation that runs out of memory gives an invalid window (``oom``); its other errors propagate, and so does a
    synchronization or a peak reset that fails (`counters`): no reading after it could be trusted, so there is no
    window. Its output is released after the closing reading.
    """
    device_counters = counters.DeviceCounters(device)
    try:
        device_counters.synchronize()
        TorchDevice.empty_cache(force=True)
        first = device_counters.read()
        device_counters.reset_peaks()
        output, oom = None, False
        try:
            output = operation()
        except RuntimeError as e:  # torch.OutOfMemoryError among them
            if "out of memory" not in str(e).lower():
                raise
            oom = True
        device_counters.synchronize()
        final = device_counters.read()
        spill = spilled(first, final, device_counters.read)
        del output
        return fresh_window(first, final, oom=oom, spill=spill)
    finally:
        device_counters.close()
