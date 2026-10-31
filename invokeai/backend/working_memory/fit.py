"""The shipped fit: the values under `data/` reproduced from their measured points.

An estimator's fitted terms price a line per operation and dtype: `k_<suffix>` bytes per unit of the line's size plus
`b_<suffix>` bytes, the base a process pays whatever the size. Its `EstimatorSpec.line` says which line an operation's
features price on, their size on it, and whether a measurement of them can join its fit (`Line`). The AutoencoderKL
estimators price their convolutions on a line in the output's pixels (`convolution_line`); the Z-Image and FLUX.2 denoise
price their forwards on ones in attended tokens. Each carried pair is the upper envelope of the fresh-process points it
is resolved for (see `upper_envelope`), plus `SHIPPED_MARGIN`, raised where the line's other points need it
(`raised_pair`). A point joins the fit of the entry its estimator resolves to on its platform -- the lender's, where the
estimator borrows -- when it is `in_fit` (not measured out of regime) and its line admits it.

A line is fitted on its points' demand as measured; no term is subtracted from it. So a line admits only measurements
whose demand is the line's alone. Where an estimate adds a term of its own on top of the line, a measurement of it fits
nothing: a tiled VAE operation (the tile term reuses the untiled pair), a VAE an in-place cast grew
(`upcast_weight_bytes`), and every denoise forward with an analytic term. Such a measurement still bounds the line from
above where it holds an upper bound (`bounds.line_bounds`). The VAE mid-block score matrix is not such a term: an
estimate reserves the larger of it and the convolutions, which peak apart, and on the builds that materialize it the
convolutions stay the larger over the measured sizes (`vae_working_memory._peak_of_two_phases`); the CUDA points that
force a math kernel are out of fit. A point that bounds its estimate from below only (`MeasuredPoint.holds_upper_bound`,
a `probe_demand` window) takes no part in the envelope: the fresh-process points cap a line. It is covered by the line
all the same. Where the envelope covers it by the margin, the pair stays as it is; where not, `raised_pair` raises it by
the least that keeps the upper bounds of every point priced on the line (`bounds.line_bounds`), found exactly among all
lines, and refuses, naming the points that conflict, only where no line keeps them (`LineConflict`). A line is fitted
only once its fresh-process points alone span two sizes `MIN_AREA_RATIO` apart (`fitted_pair`); until then it stays a
stand-in, which the other points bound from below and raise the same way.

`tests/backend/working_memory/test_shipped_coefficients.py` holds every carried value that is not a declared stand-in
to `fitted_values` of the shipped points, so a new point that moves a fit fails there with the value to ship. A
`legacy` entry's hand-set values are neither fitted nor held to a fit. A value on no line its entry carries is hand-set
too, which `hand_set_values` allows only for an estimator keyed by no platform trait: the MOSS-Image denoise's
`k_token`, bytes per attended token, whose entry ships no base for its line. Its points still decide and bound it
there.

A user's own calibration fits the same lines from its windows (`user_pair`, read by `user_fit.py`), with
`USER_MARGIN`, the same margin, for the estimators that opt in (`EstimatorSpec.user_calibrated`). No fresh-process
point caps a user line and no floor holds it, so its envelope always covers its points and is never raised.
"""

import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from fractions import Fraction
from itertools import combinations
from typing import TYPE_CHECKING, Final, NamedTuple, Optional

from invokeai.backend.working_memory.coefficients import CoefficientEntry, MeasuredPoint, WorkingMemoryData, find_entry
from invokeai.backend.working_memory.estimate import FeatureValue, WorkingMemoryTerm

if TYPE_CHECKING:
    from invokeai.backend.working_memory.spec import EstimatorSpec

SHIPPED_MARGIN: Final = Fraction(1, 20)
"""What the shipped slope and intercept add over the envelope of their points: 5%, for the runs and cards the points
do not cover."""

USER_MARGIN: Final = SHIPPED_MARGIN
"""What a user calibration's slope and intercept add over the envelope of its points: the shipped margin. A user line
measures the card it prices, and the windows that would inflate it (fragmented, partly resident) never join it."""

MIN_AREA_RATIO: Final = 2
"""How far apart the sizes a line is fitted on must lie, as a ratio of its sizes (pixel or token counts): closer points
pin the intercept no better than their noise."""

_MIB: Final = 2**20

EntryKey = tuple[str, int, tuple[tuple[str, str], ...]]
"""An entry's estimator id, version and platform keys: what a point is resolved by."""

LineKey = tuple[EntryKey, str]
"""A line: the entry that carries its values, and its suffix (`Line.suffix`)."""


class Line(NamedTuple):
    """Where an operation's features place it on the line its estimator's fitted terms price it on."""

    suffix: str
    """``<operation>_<dtype>``, naming the line's coefficients ``k_<suffix>`` and ``b_<suffix>``."""
    size: int
    """The operation's size on the line, in the unit its slope is per: output pixels, attended tokens."""
    fits: bool
    """Whether a measurement of these features can join the line's fit: its demand is the line's alone."""
    label: str
    """How a point of these features reads in its id (`bounds.point_id`), e.g. ``decode-bf16-1024x1024-tile512``, with
    ``-scores`` where a VAE operation's build materializes the mid-block score matrix."""


LineFunction = Callable[[Mapping[str, FeatureValue]], Optional[Line]]
"""What an `EstimatorSpec` declares about its fitted line: the `Line` an operation's features place it on, or None for
features that name no line."""


def off_line_reasons(
    terms: Callable[[Mapping[str, FeatureValue], Mapping[str, float]], Iterable[WorkingMemoryTerm]],
    features: Mapping[str, FeatureValue],
    coefficients: Iterable[str],
) -> list[str]:
    """What puts an operation off the line its fitted terms price: an attention mask, which narrows the kernels SDPA can
    pick (``has_attention_mask``), and each analytic term of `terms` that is not zero, as ``<name>=<bytes>``. Empty for
    an operation whose demand is the line's alone. The terms are priced with every coefficient at zero, so only what
    the operation adds on top of the line remains; a term added to an estimator takes the operations it prices out of
    the fit by itself. The estimators label their points with the reasons in this order."""
    unpriced = dict.fromkeys(coefficients, 0.0)
    reasons = ["attention_mask"] if features.get("has_attention_mask") else []
    return reasons + [
        f"{term.name}={int(term.bytes)}"
        for term in terms(features, unpriced)
        if term.coefficient is None and term.bytes
    ]


def convolution_line(features: Mapping[str, FeatureValue]) -> Optional[Line]:
    """The AutoencoderKL convolution line ``<operation>_<dtype>``, in output pixels: every VAE estimator's.

    An untiled single-sample operation of an AutoencoderKL whose VAE no in-place cast grew fits it. Tiled operations
    reuse the untiled line, the cast's growth is a term of its own, and the lines are the AutoencoderKL convolutions':
    the SD estimator also prices the tiny autoencoder (`vae_class` AutoencoderTiny) on them, a different network whose
    windows say nothing about those lines. Estimators that name no `vae_class` price an AutoencoderKL only.
    """
    operation, dtype = features.get("operation"), features.get("dtype")
    height, width = features.get("height"), features.get("width")
    if not isinstance(operation, str) or not isinstance(dtype, str):
        return None
    if not isinstance(height, int) or not isinstance(width, int):
        return None
    tile, batch = features.get("tile"), features.get("batch_size", 1)
    fits = (
        tile is None
        and batch == 1
        and not features.get("upcast_weight_bytes", 0)
        and features.get("vae_class", "AutoencoderKL") == "AutoencoderKL"
    )
    tiled = f"-tile{tile}" if tile is not None else ""
    batched = f"-b{batch}" if batch != 1 else ""
    scored = "-scores" if features.get("score_matrix_bytes") else ""
    label = f"{operation}-{dtype}-{height}x{width}{tiled}{batched}{scored}"
    return Line(f"{operation}_{dtype}", height * width, fits, label)


def entry_key(entry: CoefficientEntry | MeasuredPoint) -> EntryKey:
    return (entry.estimator, entry.version, tuple(sorted(entry.platform.items())))


class FitPoint(NamedTuple):
    size: int
    bytes: int
    caps: bool
    """Whether the point takes part in choosing the line (`MeasuredPoint.holds_upper_bound`); every point is covered."""


class LineBound(NamedTuple):
    """An upper bound a point sets on a line it is priced on (`bounds.line_bounds`): its estimate's terms on the line,
    ``k * size + b * bases``, at most `ceiling` bytes. `size` is in the line's unit and counts every unit the estimate
    prices at the slope: a batch's samples, a tile with its overlap."""

    size: Fraction
    bases: Fraction
    ceiling: Fraction
    point: str
    """Names the point where no line keeps the bound: its id (`bounds.point_id`), which two points can share, with
    the bytes it measured and the most its estimate may reserve."""


class LineConflict(ValueError):
    """No line at or above a line's floor covers its points within the upper bounds of the points priced on it: the
    data conflicts with the bounds, rather than measuring the line too little."""


def spans(sizes: Iterable[int]) -> bool:
    """Whether the sizes include two at least `MIN_AREA_RATIO` apart: enough to fit a line on."""
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
            f"an envelope needs two sizes at least {MIN_AREA_RATIO}x apart, got {sizes}; measure more sizes"
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
    a whole byte per unit of size and the intercept to a whole MiB."""
    k, b = upper_envelope(points)
    margin = 1 + SHIPPED_MARGIN
    return math.ceil(k * margin), math.ceil(b * margin / _MIB) * _MIB


def user_pair(points: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """The slope and intercept a user calibration's points give a line: the envelope plus `USER_MARGIN`, the slope
    rounded up to a whole byte per unit of size and the intercept to a whole MiB.

    Every user point is a `probe_demand` window, so all of them rate the envelope's edges, and none caps the line or
    floors it: the envelope covers them all, with no raise (`raised_pair`) to search for. Raises like
    `upper_envelope` unless the points span two sizes `MIN_AREA_RATIO` apart."""
    k, b = upper_envelope(points)
    margin = 1 + USER_MARGIN
    return math.ceil(k * margin), math.ceil(b * margin / _MIB) * _MIB


def carried_lines(entry: CoefficientEntry) -> set[str]:
    """The suffixes an entry carries both halves of a line for."""
    names = set(entry.values or {})
    return {name[2:] for name in names if name.startswith("k_") and f"b_{name[2:]}" in names}


def hand_set_values(entry: CoefficientEntry, spec: "EstimatorSpec") -> set[str]:
    """The values `entry` may carry that no fit reproduces: those on no line it carries (`carried_lines`), for a `spec`
    keyed by no platform trait. The MOSS-Image denoise declares its line in attended tokens, but its entry carries the
    slope alone; shipping a fitted base (`b_token`) with it makes that a carried line, and its exemption goes. Any
    other spec prices lines whose every value is half of one, so a value off one there is a typo or a term the fit
    never checks, and none is exempt."""
    if spec.platform_keys:
        return set()
    lines = {f"{part}_{suffix}" for suffix in carried_lines(entry) for part in ("k", "b")}
    return set(entry.values or {}) - lines


def fit_points(
    data: Iterable[WorkingMemoryData], specs: Mapping[str, "EstimatorSpec"]
) -> Mapping[LineKey, list[FitPoint]]:
    """Per carried entry and line, the points that line is fitted on.

    A point belongs to the entry its own estimator's entry resolves to on its platform, one `values_from` hop, and to
    the line its estimator's `EstimatorSpec.line` places its features on, where that admits them. `specs` holds the
    spec of every estimator a point names. Lines no point measures are absent.
    """
    files = list(data)
    entries = [entry for file in files for entry in file.coefficients]
    groups: dict[LineKey, list[FitPoint]] = {}
    for point in (point for file in files for point in file.points):
        if not point.in_fit:
            continue
        line = specs[point.estimator].line(point.features)
        if line is None or not line.fits:
            continue
        owner = carrier_of(entries, point)
        if line.suffix in carried_lines(owner):
            fitted_on = groups.setdefault((entry_key(owner), line.suffix), [])
            fitted_on.append(FitPoint(line.size, point.measured_bytes, point.holds_upper_bound))
    return groups


def carrier_of(entries: Sequence[CoefficientEntry], point: MeasuredPoint) -> CoefficientEntry:
    """The entry that carries the values `point` is evaluated with: its estimator's entry on its platform, or the one
    that entry borrows from. A point names the OS only where it measured an OS-specific entry's platform."""
    where = f"{point.estimator} point on {point.platform}"
    entry = find_entry(entries, point.estimator, point.version, point.platform, where)
    if entry.values_from is None:
        return entry
    return find_entry(entries, entry.values_from.estimator, entry.values_from.version, point.platform, where)


def entry_fit(
    entry: CoefficientEntry,
    groups: Mapping[LineKey, Sequence[FitPoint]],
    bounds: Mapping[LineKey, Sequence[LineBound]],
) -> tuple[dict[str, float], dict[str, str], dict[str, str]]:
    """What `fitted_pair` gives each line `entry` carries and does not declare a stand-in, from `groups`
    (`fit_points`) within `bounds` (`bounds.line_bounds` of the same data); per such line its points cannot fit, why;
    and per such line whose points conflict with the bounds (`LineConflict`), why."""
    key = entry_key(entry)
    values: dict[str, float] = {}
    problems: dict[str, str] = {}
    conflicts: dict[str, str] = {}
    for suffix in sorted(carried_lines(entry)):
        if {f"k_{suffix}", f"b_{suffix}"} <= set(entry.stand_ins):
            continue
        line = groups.get((key, suffix), [])
        if not line:
            problems[suffix] = "no in-fit point measures it"
            continue
        try:
            k, b = fitted_pair(line, bounds.get((key, suffix), ()))
        except LineConflict as e:
            conflicts[suffix] = str(e)
            continue
        except ValueError as e:
            problems[suffix] = str(e)
            continue
        values |= {f"k_{suffix}": k, f"b_{suffix}": b}
    return values, problems, conflicts


def fitted_values(
    data: Iterable[WorkingMemoryData],
    specs: Mapping[str, "EstimatorSpec"],
    bounds: Mapping[LineKey, Sequence[LineBound]],
) -> Mapping[EntryKey, Mapping[str, float]]:
    """Per entry that carries values and is not `legacy`, what `fitted_pair` gives each line it does not declare a
    stand-in (`entry_fit`), within the upper bounds `bounds` holds for it (`bounds.line_bounds` of the same data).

    Raises, naming the entry and line, where the points cannot fit that line: `ValueError` where they do not measure
    it well enough, `LineConflict` where they conflict with the upper bounds.
    """
    files = list(data)
    groups = fit_points(files, specs)
    fitted: dict[EntryKey, dict[str, float]] = {}
    for entry in (entry for file in files for entry in file.coefficients):
        if entry.values is None or entry.legacy:
            continue
        values, problems, conflicts = entry_fit(entry, groups, bounds)
        if problems:
            suffix, problem = next(iter(problems.items()))
            raise ValueError(f"{entry.estimator} on {entry.platform}, {suffix}: {problem}")
        if conflicts:
            suffix, conflict = next(iter(conflicts.items()))
            raise LineConflict(f"{entry.estimator} on {entry.platform}, {suffix}: {conflict}")
        if values:
            fitted[entry_key(entry)] = values
    return fitted


def capping_spans(line: Sequence[FitPoint]) -> bool:
    """Whether a line's fresh-process points alone span the sizes a fit needs, as only then is it fitted. Warm
    windows, or one fresh size with warm windows at others, could pin it below what a cold first run needs."""
    return spans(point.size for point in line if point.caps)


def fitted_pair(line: Sequence[FitPoint], bounds: Sequence[LineBound]) -> tuple[float, float]:
    """The pair to ship for a line: the fit on its fresh-process points alone (`shipped_pair`), raised by
    `raised_pair` to cover its other points within `bounds`. A `probe_demand` point thus never lowers either half of
    the fresh-process fit, and one that fit already covers leaves it as it is.

    Raises unless the fresh-process points span (`capping_spans`), and where no raise keeps every bound."""
    capping = [(p.size, p.bytes) for p in line if p.caps]
    if not capping_spans(line):
        raise ValueError(
            f"its fresh-process points measure sizes {sorted({x for x, _ in capping})}, not two at least "
            f"{MIN_AREA_RATIO}x apart; keep it a stand-in until they do"
        )
    return raised_pair(shipped_pair(capping), line, bounds)


class _HalfPlane(NamedTuple):
    """``k * size + b * bases >= limit``, or ``<=`` where `upper`, on a candidate line ``(k, b)``."""

    size: Fraction
    bases: Fraction
    limit: Fraction
    upper: bool
    why: str

    def holds(self, k: Fraction, b: Fraction) -> bool:
        value = k * self.size + b * self.bases
        return value <= self.limit if self.upper else value >= self.limit


def _corners(constraints: Sequence[_HalfPlane]) -> Iterator[tuple[Fraction, Fraction]]:
    """Every line where the boundaries of two constraints meet and all of them hold: the corners of the region of
    lines that satisfy them, exactly. With both floors among them the region has a corner unless it is empty."""
    for p, q in combinations(constraints, 2):
        det = p.size * q.bases - p.bases * q.size
        if det:
            k = (p.limit * q.bases - p.bases * q.limit) / det
            b = (p.size * q.limit - p.limit * q.size) / det
            if all(c.holds(k, b) for c in constraints):
                yield k, b


def _lowest_rounded(constraints: Sequence[_HalfPlane], middle: Fraction) -> Optional[tuple[int, int]]:
    """The rounded line that satisfies every one of `constraints` with the smallest ``k * middle + b``, the steeper on
    a tie; None where no rounded line does. A rounded line is what ships: a whole byte of slope per unit of size and a
    whole MiB of base, ``(k, b)`` with ``k`` an integer and ``b`` a multiple of a MiB.

    Exact and complete. The rounded lines that can satisfy the constraints lie in the real region they bound, so the
    search runs over each whole MiB of base from the region's lowest base up. At a base, every constraint is a bound on
    the slope, and the lowest whole slope within them is that base's best line, since the objective rises with the
    slope. The search stops above the region's highest base, which caps that bound it in the base imply; without
    such a cap, past every coverage limit, above which no constraint moves with the base; and once a base alone,
    at the region's lowest slope, costs more than the best line found."""
    corners = list(_corners(constraints))
    if not corners:
        return None
    k_least = min(k for k, _ in corners)
    b_least = min(b for _, b in corners)
    if any(c.upper and c.bases > 0 for c in constraints):
        b_most = max(b for _, b in corners)
    else:
        # Unbounded in the base: above every limit, a larger base satisfies nothing more.
        b_most = max([b for _, b in corners] + [c.limit / c.bases for c in constraints if c.bases > 0]) + _MIB
    best: Optional[tuple[Fraction, int, int]] = None
    mib = math.ceil(b_least / _MIB)
    while mib * _MIB <= b_most and (best is None or k_least * middle + mib * _MIB <= best[0]):
        b = mib * _MIB
        mib += 1
        k_low, k_high = k_least, None
        feasible = True
        for c in constraints:
            rest = c.limit - c.bases * b
            if not c.size:
                feasible = feasible and (rest >= 0 if c.upper else rest <= 0)
            elif c.upper:
                k_high = rest / c.size if k_high is None else min(k_high, rest / c.size)
            else:
                k_low = max(k_low, rest / c.size)
        k = math.ceil(k_low)
        if not feasible or (k_high is not None and k > k_high):
            continue
        cost = k * middle + b
        if best is None or (cost, -k) < (best[0], -best[1]):
            best = (cost, k, b)
    return None if best is None else (best[1], best[2])


def _conflict(floors: Sequence[_HalfPlane], others: Sequence[_HalfPlane], middle: Fraction) -> list[_HalfPlane]:
    """Of `others`, a set that no rounded line at or above the floors satisfies (`_lowest_rounded`), though one
    satisfies each of its proper subsets: each constraint is dropped while the rest still conflict."""
    kept = list(others)
    for constraint in others:
        rest = [c for c in kept if c is not constraint]
        if _lowest_rounded([*floors, *rest], middle) is None:
            kept = rest
    return kept


def raised_pair(
    floor: tuple[float, float], line: Sequence[FitPoint], bounds: Sequence[LineBound]
) -> tuple[float, float]:
    """The lowest rounded line at or above `floor`, per coefficient, that covers every point of `line` by
    `SHIPPED_MARGIN` and keeps every one of `bounds`.

    `floor` itself where it covers every point by the margin: a point it covers changes nothing, and `bounds` are not
    consulted, since the floor's own bounds are the data test's to hold. Otherwise every line is a candidate, not only
    the edges of a hull, and the candidates are the lines that ship: rounded as `shipped_pair` rounds, a whole byte of
    slope per unit of size and a whole MiB of base. Each constraint is a half-plane in ``(k, b)``, so the lines that
    satisfy them all form a convex region; `_lowest_rounded` finds the lowest rounded line in it exactly, checking
    each candidate against the constraints as they are, with no allowance. The lowest line is the one with the smallest
    reservation averaged over the sizes the line's fresh-process points span (``k`` times the middle of that range,
    plus ``b``), or all its points' sizes where none caps it; the steeper on a tie, as `upper_envelope` breaks them.
    Probe sizes do not move that measure, so a line raised for some probes stays where it is when another probe it
    covers arrives.

    Raises `LineConflict`, naming the points that cannot be satisfied together, where no rounded line at or above
    `floor` covers the points within `bounds`, also where only a line between whole bytes or MiB would. Raises
    `ValueError` where a raise is needed and `line` has fresh-process points but `bounds` states none: those points
    always bound the line (`bounds.line_bounds`), so empty `bounds` there means the caller left them out, and a raise
    without them could break every one."""
    k_floor, b_floor = Fraction(floor[0]), Fraction(floor[1])
    margin = 1 + SHIPPED_MARGIN
    highest: dict[int, int] = {}
    for point in line:
        highest[point.size] = max(highest.get(point.size, 0), point.bytes)
    # Every line at or above the floor covers what the floor covers: only the other points constrain the raise.
    uncovered = [
        _HalfPlane(Fraction(x), Fraction(1), margin * y, False, f"{y / _MIB:.0f} MiB at size {x}")
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
    caps = [_HalfPlane(bound.size, bound.bases, bound.ceiling, True, bound.point) for bound in bounds]
    sizes = [point.size for point in line if point.caps] or list(highest)
    middle = Fraction(min(sizes) + max(sizes), 2)
    best = _lowest_rounded([*floors, *uncovered, *caps], middle)
    if best is None:
        conflict = _conflict(floors, [*uncovered, *caps], middle)
        covers = "; ".join(c.why for c in conflict if not c.upper)
        within = "; ".join(c.why for c in conflict if c.upper)
        at = f"k={floor[0]:.0f}, b={floor[1] / _MIB:.0f} MiB"
        if not covers:
            # Covering points never conflicts with the floors alone: the floor itself breaks a bound.
            raise LineConflict(f"the line's floor {at} already exceeds the upper bound of {within}")
        raise LineConflict(
            f"no line at or above {at} covers {covers} by {SHIPPED_MARGIN:.0%} within the upper bound of {within}"
        )
    return best
