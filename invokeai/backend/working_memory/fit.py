"""The shipped fit: the values under `data/` reproduced from their measured points.

The AutoencoderKL estimators price their convolutions as a line in the output's pixel count, one per operation and
dtype: `k_<operation>_<dtype>` bytes per pixel plus `b_<operation>_<dtype>` bytes, the base a process pays whatever the
size. Each carried pair is the upper envelope of the fresh-process points it is resolved for (see `upper_envelope`),
plus `SHIPPED_MARGIN`, raised where the line's other points need it (`raised_pair`). A point joins the fit of the entry
its estimator resolves to on its platform -- the lender's, where the estimator borrows -- when it is a single-sample,
untiled operation of that pair (`in_fit` unless measured out of regime) with no weight growth. Tiled points bound their
estimates but fit nothing: the tile term reuses the untiled pair. Nor does a point whose VAE an in-place cast grew
(`upcast_weight_bytes`): its demand includes the growth, which the estimate prices as a term of its own. A point that
bounds its estimate from below only (`MeasuredPoint.holds_upper_bound`, a `probe_demand` window) takes no part in the
envelope: the fresh-process points cap a line. It is covered by the line all the same. Where the envelope covers it by
the margin, the pair stays as it is; where not, `raised_pair` raises it by the least that keeps the upper bounds of
every point priced on the line (`bounds.line_bounds`), found exactly among all lines, and refuses, naming the points
that conflict, only where no line keeps them (`LineConflict`). A line is fitted only once its fresh-process points alone
span two sizes `MIN_AREA_RATIO` apart (`fitted_pair`); until then it stays a stand-in, which the other points bound from
below and raise the same way.

`tests/backend/working_memory/test_shipped_coefficients.py` holds every carried value on such a line that is not a
declared stand-in to `fitted_values` of the shipped points, so a new point that moves a fit fails there with the value
to ship. A value on no pixel line is hand-set, which `hand_set_values` allows only for an estimator keyed by no platform
trait: the MOSS-Image denoise's `k_token`, bytes per attended token with no base. Its points still decide and bound it
there; this module fits pixel lines only.

A user's own calibration fits the same lines from its windows (`user_pair`, read by `user_fit.py`), with
`USER_MARGIN`, the same margin. No fresh-process point caps a user line and no floor holds it, so its envelope always
covers its points and is never raised.
"""

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from fractions import Fraction
from itertools import combinations
from typing import TYPE_CHECKING, Final, NamedTuple, Optional

from invokeai.backend.working_memory.coefficients import CoefficientEntry, MeasuredPoint, WorkingMemoryData, find_entry

if TYPE_CHECKING:
    from invokeai.backend.working_memory.spec import EstimatorSpec

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

LineKey = tuple[EntryKey, str]
"""A line: the entry that carries its values, and its ``<operation>_<dtype>`` suffix."""


def entry_key(entry: CoefficientEntry | MeasuredPoint) -> EntryKey:
    return (entry.estimator, entry.version, tuple(sorted(entry.platform.items())))


class FitPoint(NamedTuple):
    pixels: int
    bytes: int
    caps: bool
    """Whether the point takes part in choosing the line (`MeasuredPoint.holds_upper_bound`); every point is covered."""


class LineBound(NamedTuple):
    """An upper bound a point sets on a line it is priced on (`bounds.line_bounds`): its estimate's terms on the line,
    ``k * pixels + b * bases``, at most `ceiling` bytes."""

    pixels: Fraction
    bases: Fraction
    ceiling: Fraction
    point: str
    """Names the point where no line keeps the bound: its id (`bounds.point_id`), which two points can share, with
    the bytes it measured and the most its estimate may reserve."""


class LineConflict(ValueError):
    """No line at or above a line's floor covers its points within the upper bounds of the points priced on it: the
    data conflicts with the bounds, rather than measuring the line too little."""


def spans(sizes: Iterable[int]) -> bool:
    """Whether the pixel counts include two at least `MIN_AREA_RATIO` apart: enough to fit a line on."""
    distinct = sorted(set(sizes))
    return len(distinct) >= 2 and distinct[-1] >= MIN_AREA_RATIO * distinct[0]


def upper_envelope(points: Sequence[tuple[int, int]]) -> tuple[Fraction, Fraction]:
    """The line ``y = k * x + b`` on or above every ``(x, y)`` point, with ``b >= 0``, that over-reserves least.

    The candidates are the edges of the upper convex hull of the points and the origin: each such line lies on or
    above every point, and the origin keeps its intercept non-negative. Of those, the one whose largest ratio to the
    highest reading of a size is smallest wins, the steeper one on a tie, since it is the more conservative past the
    largest point. Only the highest reading counts: a lower one at a measured size is covered whatever the line, and
    rating against it would tilt the line though no point lies above it. Raises unless the points span two sizes at
    least `MIN_AREA_RATIO` apart.
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

    best: tuple[Fraction, Fraction, Fraction] | None = None
    for (x0, y0), (x1, y1) in zip(hull, hull[1:], strict=False):
        k = Fraction(y1 - y0, x1 - x0)
        b = y0 - k * x0
        if k < 0:
            continue
        worst = max((k * x + b) / y for x, y in highest.items())
        if best is None or (worst, -k) < (best[0], -best[1]):
            best = (worst, k, b)
    assert best is not None, "the edge from the origin always rises"
    return best[1], best[2]


def shipped_pair(points: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """The slope and intercept to ship for these points: the envelope plus `SHIPPED_MARGIN`, the slope rounded up to
    a whole byte per pixel and the intercept to a whole MiB."""
    k, b = upper_envelope(points)
    margin = 1 + SHIPPED_MARGIN
    return math.ceil(k * margin), math.ceil(b * margin / _MIB) * _MIB


def user_pair(points: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """The slope and intercept a user calibration's points give a line: the envelope plus `USER_MARGIN`, the slope
    rounded up to a whole byte per pixel and the intercept to a whole MiB.

    Every user point is a `probe_demand` window, so all of them rate the envelope's edges, and none caps the line or
    floors it: the envelope covers them all, with no raise (`raised_pair`) to search for. Raises like
    `upper_envelope` unless the points span two sizes `MIN_AREA_RATIO` apart."""
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


def hand_set_values(entry: CoefficientEntry, spec: "EstimatorSpec") -> set[str]:
    """The values `entry` may carry that no fit reproduces: those on no pixel line, for a `spec` keyed by no platform
    trait. Any other spec prices convolutions, whose every value is half of a line, so a value off one there is a typo
    or a term the fit never checks, and none is exempt."""
    if spec.platform_keys:
        return set()
    lines = {f"{part}_{suffix}" for suffix in _pairs(entry) for part in ("k", "b")}
    return set(entry.values or {}) - lines


def fit_points(data: Iterable[WorkingMemoryData]) -> Mapping[LineKey, list[FitPoint]]:
    """Per carried entry and line, the points that line is fitted on.

    A point belongs to the entry its own estimator's entry resolves to on its platform, one `values_from` hop, and to
    the line its operation and dtype name. Lines no point measures are absent.
    """
    files = list(data)
    entries = [entry for file in files for entry in file.coefficients]
    groups: dict[LineKey, list[FitPoint]] = {}
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


def fitted_values(
    data: Iterable[WorkingMemoryData], bounds: Mapping[LineKey, Sequence[LineBound]]
) -> Mapping[EntryKey, Mapping[str, float]]:
    """Per entry that carries values, what `fitted_pair` gives each line it does not declare a stand-in, within the
    upper bounds `bounds` holds for it (`bounds.line_bounds` of the same data).

    Raises, naming the entry and line, where the points cannot fit that line: `LineConflict` where they conflict with
    the upper bounds, `ValueError` where they do not measure it well enough.
    """
    files = list(data)
    groups = fit_points(files)
    fitted: dict[EntryKey, dict[str, float]] = {}
    for entry in (entry for file in files for entry in file.coefficients if entry.values is not None):
        key = entry_key(entry)
        for suffix in sorted(_pairs(entry)):
            if {f"k_{suffix}", f"b_{suffix}"} <= set(entry.stand_ins):
                continue
            try:
                k, b = fitted_pair(groups.get((key, suffix), []), bounds.get((key, suffix), ()))
            except LineConflict as e:
                raise LineConflict(f"{entry.estimator} on {entry.platform}, {suffix}: {e}") from e
            except ValueError as e:
                raise ValueError(f"{entry.estimator} on {entry.platform}, {suffix}: {e}") from e
            fitted.setdefault(key, {}).update({f"k_{suffix}": k, f"b_{suffix}": b})
    return fitted


def capping_spans(line: Sequence[FitPoint]) -> bool:
    """Whether a line's fresh-process points alone span the sizes a fit needs, as only then is it fitted. Warm
    windows, or one fresh size with warm windows at others, could pin it below what a cold first run needs."""
    return spans(point.pixels for point in line if point.caps)


def fitted_pair(line: Sequence[FitPoint], bounds: Sequence[LineBound]) -> tuple[float, float]:
    """The pair to ship for a line: the fit on its fresh-process points alone (`shipped_pair`), raised by
    `raised_pair` to cover its other points within `bounds`. A `probe_demand` point thus never lowers either half of
    the fresh-process fit, and one that fit already covers leaves it as it is.

    Raises unless the fresh-process points span (`capping_spans`), and where no raise keeps every bound."""
    capping = [(p.pixels, p.bytes) for p in line if p.caps]
    if not capping_spans(line):
        raise ValueError(
            f"its fresh-process points measure {sorted({x for x, _ in capping})} pixels, not two sizes at least "
            f"{MIN_AREA_RATIO}x apart; keep it a stand-in until they do"
        )
    return raised_pair(shipped_pair(capping), line, bounds)


class _HalfPlane(NamedTuple):
    """``k * pixels + b * bases >= limit``, or ``<=`` where `upper`, on a candidate line ``(k, b)``."""

    pixels: Fraction
    bases: Fraction
    limit: Fraction
    upper: bool
    why: str

    def holds(self, k: Fraction, b: Fraction) -> bool:
        value = k * self.pixels + b * self.bases
        return value <= self.limit if self.upper else value >= self.limit


def _corners(constraints: Sequence[_HalfPlane]) -> Iterator[tuple[Fraction, Fraction]]:
    """Every line where the boundaries of two constraints meet and all of them hold: the corners of the region of
    lines that satisfy them, exactly. With both floors among them the region has a corner unless it is empty."""
    for p, q in combinations(constraints, 2):
        det = p.pixels * q.bases - p.bases * q.pixels
        if det:
            k = (p.limit * q.bases - p.bases * q.limit) / det
            b = (p.pixels * q.limit - p.limit * q.pixels) / det
            if all(c.holds(k, b) for c in constraints):
                yield k, b


def _conflict(floors: Sequence[_HalfPlane], others: Sequence[_HalfPlane]) -> list[_HalfPlane]:
    """Of `others`, a set that no line at or above the floors satisfies, though one satisfies each of its proper
    subsets: each constraint is dropped while the rest still conflict. In two dimensions at most three remain."""
    kept = list(others)
    for constraint in others:
        rest = [c for c in kept if c is not constraint]
        if next(_corners([*floors, *rest]), None) is None:
            kept = rest
    return kept


def raised_pair(
    floor: tuple[float, float], line: Sequence[FitPoint], bounds: Sequence[LineBound]
) -> tuple[float, float]:
    """The lowest line at or above `floor`, per coefficient, that covers every point of `line` by `SHIPPED_MARGIN`
    and keeps every one of `bounds`.

    `floor` itself where it covers every point by the margin: a point it covers changes nothing, and `bounds` are not
    consulted, since the floor's own bounds are the data test's to hold. Otherwise every line is a candidate, not only
    the edges of a hull: each constraint is a half-plane in ``(k, b)``, so the lines that satisfy them all form a convex
    region, and the lowest of them is one of its corners, which `_corners` enumerates exactly. The lowest line is the
    one with the smallest reservation averaged over the sizes the line's fresh-process points span (``k`` times the
    middle of that range, plus ``b``), or all its points' sizes where none caps it; the steeper on a tie, as
    `upper_envelope` breaks them. Probe sizes do not move that measure, so a line raised for some probes stays where it
    is when another probe it covers arrives. The pair is rounded up as `shipped_pair` rounds it, and `bounds` are kept
    with room for that rounding: one byte per pixel and one MiB of base.

    Raises `LineConflict`, naming the points that cannot be satisfied together, where no line at or above `floor`
    covers the points within `bounds`. Raises `ValueError` where a raise is needed and `line` has fresh-process points
    but `bounds` states none: those points always bound the line (`bounds.line_bounds`), so empty `bounds` there means
    the caller left them out, and a raise without them could break every one."""
    k_floor, b_floor = Fraction(floor[0]), Fraction(floor[1])
    margin = 1 + SHIPPED_MARGIN
    highest: dict[int, int] = {}
    for point in line:
        highest[point.pixels] = max(highest.get(point.pixels, 0), point.bytes)
    # Every line at or above the floor covers what the floor covers: only the other points constrain the raise.
    uncovered = [
        _HalfPlane(Fraction(x), Fraction(1), margin * y, False, f"{y / _MIB:.0f} MiB at {x} px")
        for x, y in sorted(highest.items())
        if k_floor * x + b_floor < margin * y
    ]
    if not uncovered:
        return floor
    if not bounds and any(point.caps for point in line):
        raise ValueError(
            "the line's fresh-process points bound its raise, but no upper bounds are given for it; pass "
            "`bounds.line_bounds` of the same data"
        )
    floors = [
        _HalfPlane(Fraction(1), Fraction(0), k_floor, False, "the slope floor"),
        _HalfPlane(Fraction(0), Fraction(1), b_floor, False, "the base floor"),
    ]
    caps = [
        _HalfPlane(bound.pixels, bound.bases, bound.ceiling - bound.pixels - bound.bases * _MIB, True, bound.point)
        for bound in bounds
    ]
    sizes = [point.pixels for point in line if point.caps] or list(highest)
    middle = Fraction(min(sizes) + max(sizes), 2)
    best = min(_corners([*floors, *uncovered, *caps]), key=lambda kb: (kb[0] * middle + kb[1], -kb[0]), default=None)
    if best is None:
        conflict = _conflict(floors, [*uncovered, *caps])
        covers = "; ".join(c.why for c in conflict if not c.upper)
        within = "; ".join(c.why for c in conflict if c.upper)
        at = f"k={floor[0]:.0f}, b={floor[1] / _MIB:.0f} MiB"
        if not covers:
            # Covering points never conflicts with the floors alone: the floor itself breaks a bound.
            raise LineConflict(f"the line's floor {at} already exceeds the upper bound of {within}")
        raise LineConflict(
            f"no line at or above {at} covers {covers} by {SHIPPED_MARGIN:.0%} within the upper bound of {within}"
        )
    k, b = best
    return math.ceil(k), math.ceil(b / _MIB) * _MIB
