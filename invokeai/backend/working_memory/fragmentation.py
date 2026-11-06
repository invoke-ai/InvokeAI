"""When a measured window's demand is the allocator's fragmentation rather than the operation's.

A long-running server's caching allocator grows its reserve around the same tensors: reserved bytes climb while the
allocated bytes, what the operation really holds, stay put. A window taken then measures the allocator's history, and a
line fitted on it over-reserves every later operation. One rule, `fragmented`, read by
`scripts/working_memory/build_shipped.py`, which flags such points for review, and by the user fit
(`app/services/working_memory_calibration/`), which leaves them out. It judges measurements: the fit-eligible windows
of one operation at one size from one attempt (a calibration attempt, or one run without one), in the order they ran.
Windows of one attempt at one size ran back to back on one allocator state; another attempt's ran on another, before
or after offloads and other work, so each attempt is judged on its own and a recalibration can bring a size back.

- `grew_around_flat_allocation`: across a measurement's windows, reserved grew by `FRAGMENTATION_GROWTH` from the
  first window to the largest while allocated stayed within `ALLOCATED_FLAT`.
- `ratio_outlier`: a measurement's largest window has a reserved-to-allocated ratio `RATIO_OUTLIER` above the median of
  the same line's smaller measurements that are not flagged, from any attempt: the ratio is the operation's on the
  card, which no attempt owns. This catches a size whose every window ran fragmented, so that it shows no growth of its
  own; a single window, too. Measured on an RX 9060 XT (fp32 SDXL decode): 1216px measured 1.47 against 1.10 at 576px
  and 832px, its one fit-eligible window 1.33 times their median, while the fp16 sizes of the same run stayed within
  1.06 of theirs.
"""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Final, Optional

FRAGMENTATION_GROWTH: Final = 1.10
"""Reserved growth from a measurement's first window to its largest that, with flat allocated bytes, flags
fragmentation."""

ALLOCATED_FLAT: Final = 0.02
"""How far allocated bytes may spread across a measurement's windows and still count as flat."""

RATIO_OUTLIER: Final = 1.2
"""How far a size's reserved-to-allocated ratio may exceed the median of the same line's smaller sizes before it
counts as fragmentation."""


@dataclass(frozen=True)
class Measurement:
    """The fit-eligible windows of one operation at one size from one attempt, in the order they ran."""

    line: Hashable
    """What its sizes are compared within: one line (`fit.Line.suffix`) of one estimator version on one card and
    platform or environment."""
    size: int
    """The operation's size on that line (`fit.Line.size`)."""
    demand: Sequence[int]
    reserved: Sequence[Optional[int]]
    """Each window's reserved delta."""
    allocated: Sequence[Optional[int]]
    reference: bool = True
    """Whether larger sizes may be compared with it; a measurement still being taken is judged, but sets no standard."""


def fragmented(measurements: Sequence[Measurement]) -> list[bool]:
    """Per measurement, whether its windows measured fragmentation: `grew_around_flat_allocation`, or a
    `ratio_outlier` of its largest window against the smaller measurements of its line that are not flagged and may
    be a `reference`."""
    flags = [
        grew_around_flat_allocation(measurement.demand, measurement.reserved, measurement.allocated)
        for measurement in measurements
    ]
    lines: dict[Hashable, list[int]] = {}
    for index, measurement in enumerate(measurements):
        if not flags[index]:
            lines.setdefault(measurement.line, []).append(index)
    for members in lines.values():
        kept: list[tuple[int, float]] = []
        for index in sorted(members, key=lambda index: measurements[index].size):
            measurement = measurements[index]
            largest = list(measurement.demand).index(max(measurement.demand))
            ratio = reserved_ratio(measurement.reserved[largest], measurement.allocated[largest])
            if ratio_outlier(ratio, [kept_ratio for size, kept_ratio in kept if size < measurement.size]):
                flags[index] = True
            elif ratio is not None and measurement.reference:
                kept.append((measurement.size, ratio))
    return flags


def grew_around_flat_allocation(
    demand: Sequence[int], reserved: Sequence[Optional[int]], allocated: Sequence[Optional[int]]
) -> bool:
    """Whether the windows of one measurement, in the order they ran, show reserved growing by `FRAGMENTATION_GROWTH`
    from the first window to the one of the largest demand while allocated stayed flat. False when that largest
    window is the first, or a counter is missing."""
    largest = list(demand).index(max(demand))
    first, peak = reserved[0], reserved[largest]
    if largest == 0 or first is None or peak is None or None in allocated:
        return False
    flat = [value for value in allocated if value is not None]
    return peak >= FRAGMENTATION_GROWTH * first and max(flat) <= (1 + ALLOCATED_FLAT) * min(flat)


def reserved_ratio(reserved: Optional[int], allocated: Optional[int]) -> Optional[float]:
    """Reserved over allocated bytes of one window; None where either is missing or allocated is not positive."""
    if reserved is None or allocated is None or allocated <= 0:
        return None
    return reserved / allocated


def ratio_outlier(ratio: Optional[float], smaller: Sequence[float]) -> bool:
    """Whether a size's `ratio` is `RATIO_OUTLIER` times the median ratio of the same line's `smaller` sizes; False
    without a ratio or a smaller size to compare with."""
    if ratio is None or not smaller:
        return False
    return ratio >= RATIO_OUTLIER * median(smaller)
