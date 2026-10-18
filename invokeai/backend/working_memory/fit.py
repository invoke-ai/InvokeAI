"""The shipped fit: the values under `data/` reproduced from their measured points.

The AutoencoderKL estimators price their convolutions as a line in the output's pixel count, one per operation and
dtype: `k_<operation>_<dtype>` bytes per pixel plus `b_<operation>_<dtype>` bytes, the base a process pays whatever the
size. Each carried pair is the upper envelope of the points it is resolved for (see `upper_envelope`), plus
`SHIPPED_MARGIN`. A point joins the fit of the entry its estimator resolves to on its platform -- the lender's, where
the estimator borrows -- when it is a single-sample, untiled operation of that pair (`in_fit` unless measured out of
regime) with no weight growth. Tiled points bound their estimates but fit nothing: the tile term reuses the untiled
pair. Nor does a point whose VAE an in-place cast grew (`upcast_weight_bytes`): its demand includes the growth, which
the estimate prices as a term of its own. A point that bounds its estimate from below only
(`MeasuredPoint.holds_upper_bound`, a `probe_demand` window) is covered by the line and can raise it, but takes no part
in choosing it: the fresh-process points cap a line.

`tests/backend/working_memory/test_shipped_coefficients.py` holds every carried value that is not a declared stand-in
to `fitted_values` of the shipped points, so a new point that moves a fit fails there with the value to ship.

A user's own calibration fits the same lines from its windows (`user_pair`, read by `user_fit.py`), with
`USER_MARGIN`, the same margin.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Final, NamedTuple, Optional

from invokeai.backend.working_memory.coefficients import CoefficientEntry, MeasuredPoint, WorkingMemoryData, find_entry

SHIPPED_MARGIN: Final = Fraction(1, 20)
"""What the shipped slope and intercept add over the envelope of their points: 5%, for the runs and cards the points
do not cover."""

USER_MARGIN: Final = SHIPPED_MARGIN
"""What a user calibration's slope and intercept add over the envelope of its points: the shipped margin. A user line
measures the card it prices, and the windows that would inflate it (fragmented, partly resident) never join it."""

MIN_AREA_RATIO: Final = 2
"""How far apart the sizes a line is fitted on must lie, as a ratio of pixel counts: closer points pin the intercept
no better than their noise."""

_MIB: Final = 2**20

EntryKey = tuple[str, int, tuple[tuple[str, str], ...]]
"""An entry's estimator id, version and platform keys: what a point is resolved by."""


def entry_key(entry: CoefficientEntry | MeasuredPoint) -> EntryKey:
    return (entry.estimator, entry.version, tuple(sorted(entry.platform.items())))


class FitPoint(NamedTuple):
    pixels: int
    bytes: int
    caps: bool
    """Whether the point takes part in choosing the line (`MeasuredPoint.holds_upper_bound`); every point is covered."""


def upper_envelope(
    points: Sequence[tuple[int, int]], capping: Optional[Sequence[tuple[int, int]]] = None
) -> tuple[Fraction, Fraction]:
    """The line ``y = k * x + b`` on or above every ``(x, y)`` point, with ``b >= 0``, that over-reserves least.

    The candidates are the edges of the upper convex hull of the points and the origin: each such line lies on or
    above every point, and the origin keeps its intercept non-negative. Of those, the one whose largest ratio to the
    highest reading of a size is smallest wins, the steeper one on a tie, since it is the more conservative past the
    largest point. Only the highest reading counts: a lower one at a measured size is covered whatever the line, and
    rating against it would tilt the line though no point lies above it. With `capping`, a subset of the points, only
    its readings rate the edges: the others are covered but cap nothing. Without any, all points rate them. Raises
    unless the points span two sizes at least `MIN_AREA_RATIO` apart.
    """
    if not points or any(x <= 0 or y <= 0 for x, y in points):
        raise ValueError(f"an envelope needs points with positive sizes and bytes, got {list(points)}")
    sizes = sorted({x for x, _ in points})
    if len(sizes) < 2 or sizes[-1] < MIN_AREA_RATIO * sizes[0]:
        raise ValueError(
            f"an envelope needs two sizes at least {MIN_AREA_RATIO}x apart in pixels, got {sizes}; measure more sizes"
        )

    highest = {x: max(y for px, y in points if px == x) for x in sizes}
    hull: list[tuple[int, int]] = [(0, 0)]
    for point in highest.items():
        # Drop the last vertex while it is not above the chord from the one before it to this point.
        while len(hull) >= 2:
            (x0, y0), (x1, y1) = hull[-2], hull[-1]
            if (x1 - x0) * (point[1] - y0) - (y1 - y0) * (point[0] - x0) < 0:
                break
            hull.pop()
        hull.append(point)

    rated = {x: max(y for px, y in capping if px == x) for x in sorted({x for x, _ in capping})} if capping else highest
    best: tuple[Fraction, Fraction, Fraction] | None = None
    for (x0, y0), (x1, y1) in zip(hull, hull[1:], strict=False):
        k = Fraction(y1 - y0, x1 - x0)
        b = y0 - k * x0
        if k < 0:
            continue
        worst = max((k * x + b) / y for x, y in rated.items())
        if best is None or (worst, -k) < (best[0], -best[1]):
            best = (worst, k, b)
    assert best is not None, "the edge from the origin always rises"
    return best[1], best[2]


def shipped_pair(
    points: Sequence[tuple[int, int]], capping: Optional[Sequence[tuple[int, int]]] = None
) -> tuple[int, int]:
    """The slope and intercept to ship for these points: the envelope plus `SHIPPED_MARGIN`, the slope rounded up to
    a whole byte per pixel and the intercept to a whole MiB."""
    k, b = upper_envelope(points, capping)
    margin = 1 + SHIPPED_MARGIN
    return math.ceil(k * margin), math.ceil(b * margin / _MIB) * _MIB


def user_pair(points: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """The slope and intercept a user calibration's points give a line: the envelope plus `USER_MARGIN`, the slope
    rounded up to a whole byte per pixel and the intercept to a whole MiB.

    Every user point is a `probe_demand` window, so no point caps the line and all of them rate its edges (the
    fallback of `upper_envelope`). Raises like `upper_envelope` unless the points span two sizes `MIN_AREA_RATIO`
    apart."""
    k, b = upper_envelope(points)
    margin = 1 + USER_MARGIN
    return math.ceil(k * margin), math.ceil(b * margin / _MIB) * _MIB


def line_of(features: Mapping[str, object]) -> Optional[tuple[str, int]]:
    """The convolution line an operation of these features is priced on, ``<operation>_<dtype>``, and its size in
    output pixels; None for features that name no such line."""
    operation, dtype = features.get("operation"), features.get("dtype")
    height, width = features.get("height"), features.get("width")
    if not isinstance(operation, str) or not isinstance(dtype, str):
        return None
    if not isinstance(height, int) or not isinstance(width, int):
        return None
    return f"{operation}_{dtype}", height * width


def fits_a_line(features: Mapping[str, object]) -> bool:
    """Whether a point of these features can join its line's fit: an untiled single-sample operation of an
    AutoencoderKL whose VAE no in-place cast grew. Tiled points reuse the untiled line, the cast's growth is a term of
    its own, and the lines are the AutoencoderKL convolutions': the SD estimator also prices the tiny autoencoder
    (`vae_class` AutoencoderTiny) on them, a different network whose windows say nothing about those lines.
    Estimators that name no `vae_class` price an AutoencoderKL only."""
    return (
        features.get("tile") is None
        and features.get("batch_size", 1) == 1
        and not features.get("upcast_weight_bytes", 0)
        and features.get("vae_class", "AutoencoderKL") == "AutoencoderKL"
        and line_of(features) is not None
    )


def _pairs(entry: CoefficientEntry) -> set[str]:
    """The `<operation>_<dtype>` suffixes an entry carries both halves of a convolution line for."""
    names = set(entry.values or {})
    return {name[2:] for name in names if name.startswith("k_") and f"b_{name[2:]}" in names}


def fit_points(data: Iterable[WorkingMemoryData]) -> Mapping[tuple[EntryKey, str], list[FitPoint]]:
    """Per carried entry and line, the points that line is fitted on.

    A point belongs to the entry its own estimator's entry resolves to on its platform, one `values_from` hop, and to
    the line its operation and dtype name. Lines no point measures are absent.
    """
    files = list(data)
    entries = [entry for file in files for entry in file.coefficients]
    groups: dict[tuple[EntryKey, str], list[FitPoint]] = {}
    for point in (point for file in files for point in file.points):
        found = line_of(point.features)
        if not point.in_fit or found is None or not fits_a_line(point.features):
            continue
        owner = carrier_of(entries, point)
        suffix, pixels = found
        if suffix in _pairs(owner):
            line = groups.setdefault((entry_key(owner), suffix), [])
            line.append(FitPoint(pixels, point.measured_bytes, point.holds_upper_bound))
    return groups


def carrier_of(entries: Sequence[CoefficientEntry], point: MeasuredPoint) -> CoefficientEntry:
    """The entry that carries the values `point` is evaluated with: its estimator's entry on its platform, or the one
    that entry borrows from. A point names the OS only where it measured an OS-specific entry's platform."""
    where = f"{point.estimator} point on {point.platform}"
    entry = find_entry(entries, point.estimator, point.version, point.platform, where)
    if entry.values_from is None:
        return entry
    return find_entry(entries, entry.values_from.estimator, entry.values_from.version, point.platform, where)


def fitted_values(data: Iterable[WorkingMemoryData]) -> Mapping[EntryKey, Mapping[str, int]]:
    """Per entry that carries values, what `shipped_pair` gives each line it does not declare a stand-in.

    Raises, naming the entry and line, where the points cannot fit that line.
    """
    files = list(data)
    groups = fit_points(files)
    fitted: dict[EntryKey, dict[str, int]] = {}
    for entry in (entry for file in files for entry in file.coefficients if entry.values is not None):
        key = entry_key(entry)
        for suffix in sorted(_pairs(entry)):
            if {f"k_{suffix}", f"b_{suffix}"} <= set(entry.stand_ins):
                continue
            line = groups.get((key, suffix), [])
            try:
                k, b = shipped_pair([(p.pixels, p.bytes) for p in line], [(p.pixels, p.bytes) for p in line if p.caps])
            except ValueError as e:
                raise ValueError(f"{entry.estimator} on {entry.platform}, {suffix}: {e}") from e
            fitted.setdefault(key, {}).update({f"k_{suffix}": k, f"b_{suffix}": b})
    return fitted
