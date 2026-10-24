"""The shipped fit: the values under `data/` reproduced from their measured points.

An estimator's fitted terms price a line per operation and dtype: `k_<suffix>` bytes per unit of the line's size plus
`b_<suffix>` bytes, the base a process pays whatever the size. Its `EstimatorSpec.line` says which line an operation's
features price on, their size on it, and whether a measurement of them can join its fit (`Line`). The AutoencoderKL
estimators price their convolutions on a line in the output's pixels (`convolution_line`); the Z-Image and FLUX.2 denoise
price their forwards on ones in attended tokens. Each carried pair is the upper envelope of the points it is
resolved for (see `upper_envelope`), plus `SHIPPED_MARGIN`. A point joins the fit of the entry its estimator resolves
to on its platform -- the lender's, where the estimator borrows -- when it is `in_fit` (not measured out of regime) and
its line admits it.

A line is fitted on its points' demand as measured; no term is subtracted from it. So a line admits only measurements
whose demand is the line's alone. Where an estimate adds a term of its own on top of the line, a measurement of it
fits nothing: a tiled VAE operation (the tile term reuses the untiled pair), a VAE an in-place cast grew
(`upcast_weight_bytes`), and every denoise forward with an analytic term. The VAE mid-block score matrix is not such a
term: an estimate reserves the larger of it and the convolutions, which peak apart, and on the builds that materialize
it the convolutions stay the larger over the measured sizes (`vae_working_memory._peak_of_two_phases`); the CUDA points
that force a math kernel are out of fit. A point that bounds its estimate from below only
(`MeasuredPoint.holds_upper_bound`, a `probe_demand` window) is covered by the line and can raise it, but takes no part
in choosing it: the fresh-process points cap a line. A line is fitted only once its fresh-process points alone span
two sizes `MIN_AREA_RATIO` apart (`fitted_pair`); until then it stays a stand-in, which the other points bound from
below.

`tests/backend/working_memory/test_shipped_coefficients.py` holds every carried value that is not a declared stand-in
to `fitted_values` of the shipped points, so a new point that moves a fit fails there with the value to ship. A
`legacy` entry's hand-set values are neither fitted nor held to a fit.

A user's own calibration fits the same lines from its windows (`user_pair`, read by `user_fit.py`), with
`USER_MARGIN`, the same margin, for the estimators that opt in (`EstimatorSpec.user_calibrated`).
"""

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
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


class Line(NamedTuple):
    """Where an operation's features place it on the line its estimator's fitted terms price it on."""

    suffix: str
    """``<operation>_<dtype>``, naming the line's coefficients ``k_<suffix>`` and ``b_<suffix>``."""
    size: int
    """The operation's size on the line, in the unit its slope is per: output pixels, attended tokens."""
    fits: bool
    """Whether a measurement of these features can join the line's fit: its demand is the line's alone."""
    label: str
    """How a point of these features reads in its id (`bounds.point_id`), e.g. ``decode-bf16-1024x1024-tile512``."""


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
    return Line(f"{operation}_{dtype}", height * width, fits, f"{operation}-{dtype}-{height}x{width}{tiled}{batched}")


def entry_key(entry: CoefficientEntry | MeasuredPoint) -> EntryKey:
    return (entry.estimator, entry.version, tuple(sorted(entry.platform.items())))


class FitPoint(NamedTuple):
    size: int
    bytes: int
    caps: bool
    """Whether the point takes part in choosing the line (`MeasuredPoint.holds_upper_bound`); every point is covered."""


def spans(sizes: Iterable[int]) -> bool:
    """Whether the sizes include two at least `MIN_AREA_RATIO` apart: enough to fit a line on."""
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
    a whole byte per unit of size and the intercept to a whole MiB."""
    k, b = upper_envelope(points, capping)
    margin = 1 + SHIPPED_MARGIN
    return math.ceil(k * margin), math.ceil(b * margin / _MIB) * _MIB


def user_pair(points: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """The slope and intercept a user calibration's points give a line: the envelope plus `USER_MARGIN`, the slope
    rounded up to a whole byte per unit of size and the intercept to a whole MiB.

    Every user point is a `probe_demand` window, so no point caps the line and all of them rate its edges (the
    fallback of `upper_envelope`). Raises like `upper_envelope` unless the points span two sizes `MIN_AREA_RATIO`
    apart."""
    k, b = upper_envelope(points)
    margin = 1 + USER_MARGIN
    return math.ceil(k * margin), math.ceil(b * margin / _MIB) * _MIB


def carried_lines(entry: CoefficientEntry) -> set[str]:
    """The suffixes an entry carries both halves of a line for."""
    names = set(entry.values or {})
    return {name[2:] for name in names if name.startswith("k_") and f"b_{name[2:]}" in names}


def fit_points(
    data: Iterable[WorkingMemoryData], specs: Mapping[str, "EstimatorSpec"]
) -> Mapping[tuple[EntryKey, str], list[FitPoint]]:
    """Per carried entry and line, the points that line is fitted on.

    A point belongs to the entry its own estimator's entry resolves to on its platform, one `values_from` hop, and to
    the line its estimator's `EstimatorSpec.line` places its features on, where that admits them. `specs` holds the
    spec of every estimator a point names. Lines no point measures are absent.
    """
    files = list(data)
    entries = [entry for file in files for entry in file.coefficients]
    groups: dict[tuple[EntryKey, str], list[FitPoint]] = {}
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
    entry: CoefficientEntry, groups: Mapping[tuple[EntryKey, str], Sequence[FitPoint]]
) -> tuple[dict[str, int], dict[str, str]]:
    """What `fitted_pair` gives each line `entry` carries and does not declare a stand-in, from `groups`
    (`fit_points`); and, per such line its points cannot fit, why."""
    key = entry_key(entry)
    values: dict[str, int] = {}
    problems: dict[str, str] = {}
    for suffix in sorted(carried_lines(entry)):
        if {f"k_{suffix}", f"b_{suffix}"} <= set(entry.stand_ins):
            continue
        line = groups.get((key, suffix), [])
        if not line:
            problems[suffix] = "no in-fit point measures it"
            continue
        try:
            k, b = fitted_pair(line)
        except ValueError as e:
            problems[suffix] = str(e)
            continue
        values |= {f"k_{suffix}": k, f"b_{suffix}": b}
    return values, problems


def fitted_values(
    data: Iterable[WorkingMemoryData], specs: Mapping[str, "EstimatorSpec"]
) -> Mapping[EntryKey, Mapping[str, int]]:
    """Per entry that carries values and is not `legacy`, what `fitted_pair` gives each line it does not declare a
    stand-in (`entry_fit`).

    Raises, naming the entry and line, where the points cannot fit that line.
    """
    files = list(data)
    groups = fit_points(files, specs)
    fitted: dict[EntryKey, dict[str, int]] = {}
    for entry in (entry for file in files for entry in file.coefficients):
        if entry.values is None or entry.legacy:
            continue
        values, problems = entry_fit(entry, groups)
        if problems:
            suffix, problem = next(iter(problems.items()))
            raise ValueError(f"{entry.estimator} on {entry.platform}, {suffix}: {problem}")
        if values:
            fitted[entry_key(entry)] = values
    return fitted


def capping_spans(line: Sequence[FitPoint]) -> bool:
    """Whether a line's fresh-process points alone span the sizes a fit needs, as only then is it fitted. Warm
    windows, or one fresh size with warm windows at others, could pin it below what a cold first run needs."""
    return spans(point.size for point in line if point.caps)


def fitted_pair(line: Sequence[FitPoint]) -> tuple[int, int]:
    """The pair to ship for a line (`shipped_pair`): the coefficient-wise larger of its fit on all its points, rated on the
    fresh-process ones, and its fit on the fresh-process points alone. A probe point above the hull can drop a fresh
    vertex and so lower one half of the line; the larger of the two keeps both at the fresh-process fit at least.

    Raises unless the fresh-process points span (`capping_spans`)."""
    capping = [(p.size, p.bytes) for p in line if p.caps]
    if not capping_spans(line):
        raise ValueError(
            f"its fresh-process points measure sizes {sorted({x for x, _ in capping})}, not two at least "
            f"{MIN_AREA_RATIO}x apart; keep it a stand-in until they do"
        )
    k_all, b_all = shipped_pair([(p.size, p.bytes) for p in line], capping)
    k_fresh, b_fresh = shipped_pair(capping)
    return max(k_all, k_fresh), max(b_all, b_fresh)
