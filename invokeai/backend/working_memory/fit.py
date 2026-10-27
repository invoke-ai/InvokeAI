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
in choosing it: the fresh-process points cap a line. A line is fitted only once its fresh-process points alone span
two sizes `MIN_AREA_RATIO` apart (`fitted_pair`); until then it stays a stand-in, which the other points bound from
below.

`tests/backend/working_memory/test_shipped_coefficients.py` holds every carried value on such a line that is not a
declared stand-in to `fitted_values` of the shipped points, so a new point that moves a fit fails there with the value
to ship. A value on no pixel line (`line_values`) is hand-set: the MOSS-Image denoise's `k_token`, bytes per attended
token with no base. Its points still decide and bound it (`bounds.py`); this module fits pixel lines only.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Final, NamedTuple, Optional

from invokeai.backend.working_memory.coefficients import CoefficientEntry, MeasuredPoint, WorkingMemoryData, find_entry

SHIPPED_MARGIN: Final = Fraction(1, 20)
"""What the shipped slope and intercept add over the envelope of their points: 5%, for the runs and cards the points
do not cover."""

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


def spans(sizes: Iterable[int]) -> bool:
    """Whether the pixel counts include two at least `MIN_AREA_RATIO` apart: enough to fit a line on."""
    distinct = sorted(set(sizes))
    return len(distinct) >= 2 and distinct[-1] >= MIN_AREA_RATIO * distinct[0]


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
    if not spans(sizes):
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


def _pairs(entry: CoefficientEntry) -> set[str]:
    """The `<operation>_<dtype>` suffixes an entry carries both halves of a convolution line for."""
    names = set(entry.values or {})
    return {name[2:] for name in names if name.startswith("k_") and f"b_{name[2:]}" in names}


def line_values(entry: CoefficientEntry) -> set[str]:
    """The values of `entry` that `fitted_values` reproduces: both halves of each pixel line it carries, stand-ins
    included. Any other value is hand-set (see the module docstring).

    Where per-estimator lines take over (`EstimatorSpec.line`, PR #9919), a line in attended tokens such as the
    MOSS-Image denoise's is fitted too, and this distinction goes."""
    return {f"{part}_{suffix}" for suffix in _pairs(entry) for part in ("k", "b")}


def fit_points(data: Iterable[WorkingMemoryData]) -> Mapping[tuple[EntryKey, str], list[FitPoint]]:
    """Per carried entry and line, the points that line is fitted on.

    A point belongs to the entry its own estimator's entry resolves to on its platform, one `values_from` hop, and to
    the line its operation and dtype name. Lines no point measures are absent.
    """
    files = list(data)
    entries = [entry for file in files for entry in file.coefficients]
    groups: dict[tuple[EntryKey, str], list[FitPoint]] = {}
    for point in (point for file in files for point in file.points):
        features = point.features
        if (
            not point.in_fit
            or features.get("tile") is not None
            or features.get("batch_size", 1) != 1
            or features.get("upcast_weight_bytes", 0)
        ):
            continue
        owner = carrier_of(entries, point)
        suffix = f"{features.get('operation')}_{features.get('dtype')}"
        if suffix in _pairs(owner):
            pixels = int(features["height"]) * int(features["width"])  # type: ignore[arg-type]
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
    """Per entry that carries values, what `fitted_pair` gives each line it does not declare a stand-in.

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
            try:
                k, b = fitted_pair(groups.get((key, suffix), []))
            except ValueError as e:
                raise ValueError(f"{entry.estimator} on {entry.platform}, {suffix}: {e}") from e
            fitted.setdefault(key, {}).update({f"k_{suffix}": k, f"b_{suffix}": b})
    return fitted


def capping_spans(line: Sequence[FitPoint]) -> bool:
    """Whether a line's fresh-process points alone span the sizes a fit needs, as only then is it fitted. Warm
    windows, or one fresh size with warm windows at others, could pin it below what a cold first run needs."""
    return spans(point.pixels for point in line if point.caps)


def fitted_pair(line: Sequence[FitPoint]) -> tuple[int, int]:
    """The pair to ship for a line (`shipped_pair`): the coefficient-wise larger of its fit on all its points, rated on the
    fresh-process ones, and its fit on the fresh-process points alone. A probe point above the hull can drop a fresh
    vertex and so lower one half of the line; the larger of the two keeps both at the fresh-process fit at least.

    Raises unless the fresh-process points span (`capping_spans`)."""
    capping = [(p.pixels, p.bytes) for p in line if p.caps]
    if not capping_spans(line):
        raise ValueError(
            f"its fresh-process points measure {sorted({x for x, _ in capping})} pixels, not two sizes at least "
            f"{MIN_AREA_RATIO}x apart; keep it a stand-in until they do"
        )
    k_all, b_all = shipped_pair([(p.pixels, p.bytes) for p in line], capping)
    k_fresh, b_fresh = shipped_pair(capping)
    return max(k_all, k_fresh), max(b_all, b_fresh)
