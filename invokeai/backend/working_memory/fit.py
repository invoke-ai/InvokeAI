"""The shipped fit: the values under `data/` reproduced from their measured points.

An estimator's fitted terms price a line per operation and dtype: `k_<suffix>` bytes per unit of the line's size plus
`b_<suffix>` bytes, the base a process pays whatever the size. Its `EstimatorSpec.line` says which line an operation's
features price on, their size on it, and whether a measurement of them can join its fit (`Line`). The AutoencoderKL
estimators price their convolutions on a line in the output's pixels (`convolution_line`); the Z-Image and FLUX.2
denoise price their forwards on ones in attended tokens. Each carried pair is the upper envelope of the points it is
resolved for (see `upper_envelope`), plus `SHIPPED_MARGIN`. A point joins the fit of the entry that carries its line on
its platform -- its estimator's own entry where that carries the line, the lender's where it borrows the line -- when it
is `in_fit` (not measured out of regime) and its line admits it.

A line is fitted on its points' demand above what the estimate's other fitted lines price them at. Most lines sit on
none, so their points' demand is fitted as measured. A line that sits on another (`Line.sits_on`) is a tiled VAE
operation's: the estimate prices its tile on the untiled convolution line at the tile area, and its tiled line
(`tiled_<operation>_<dtype>`, in the image's pixels) what tiling adds on top. The tiled line covers each point above
the untiled demand the point's own estimator measured at that area, on the same platform, operation and dtype: the
highest of its in-fit, fresh-process points of the untiled line there (`FitPoint.reference`, the untiled references a
tiling run measures at each tile's side). Only where no such point exists does the untiled line's fitted price there
stand in. The reason is the tiles nobody measured: the shipped untiled line lies above the measured demand by a margin
that differs per tile size, and covered above the line, a tiled line would take that margin in at the measured tiles
and miss it at the others. Covered above the measured demand, it prices what tiling adds to the untiled demand at the
tile, which carries to any tile whose growth is no larger than the measured tiles'. Each point is rated on its whole
estimate, the untiled line's fitted price at its tile included, so the 1.3x bound of `bounds.py` holds for what is
reserved. Both need the untiled line's fit first, which is the dependency order: the lines that sit on none, then the
lines that sit on one of them, never a chain (`base_values`, `fitted_values`). The data test and
`scripts/working_memory/build_shipped.py` fit in this order too, both through `fitted_values`.

A line may have a second phase, which peaks apart from it (`Line.assembly`): a tiled decode assembles the image from
its decoded tiles after the last one, priced on ``assembled_decode_<dtype>`` in the same size, the image's pixels. The
estimate reserves the larger, ``max(untiled + T, A)`` for the tiled line ``T`` on the untiled line at the tile and the
assembly line ``A``, so neither line alone is an envelope of the points. Both are fitted together, from the tiled
line's points with the same offsets, by the deterministic search of `two_phase_envelope`: each candidate ``A`` is the
zero line or a line through two of the points' readings, ``T`` is the `upper_envelope` of the points above it, and the
candidate whose estimates, both lines as they ship with `SHIPPED_MARGIN`, over-reserve the capping points least wins,
the zero ``A`` on a tie (`shipped_two_phase`). Where no point needs the second phase, ``A`` is the zero line and the
estimate the line alone.

A measurement whose demand includes an analytic term of its estimate fits no line: a VAE an in-place cast grew
(`upcast_weight_bytes`), a batch, and every denoise forward with an analytic term. The VAE mid-block score matrix is
not such a term: an estimate reserves the larger of it and the convolutions, which peak apart, and on the builds that
materialize it the convolutions stay the larger over the measured sizes (`vae_working_memory._peak_of_two_phases`); the
CUDA points that force a math kernel are out of fit. A point that bounds its estimate from below only (a
`probe_demand` window, `MeasuredPoint.holds_upper_bound`, or a tiled VAE operation on a tile the app never sets,
`Line.caps`) is covered by the line and can raise it, but takes no part in choosing it: the other fresh-process points
cap a line. A line is fitted only once its capping points alone span two sizes `MIN_AREA_RATIO` apart
(`capping_spans`); until then it stays a stand-in, which the other points bound from below. A probe window on a line
never lowers a half of it either: each coefficient ships at the larger of the fit on all points
and the fit on the fresh-process points alone, both rated on the capping ones (`fitted_pair`), for a line with an
assembly phase both lines of each two-phase fit (`fitted_two_phase`). A fresh point below `SMALLEST_APP_SIDE` shapes
both fits; it only rates neither. A line that sits on another reads its points above their offsets in both fits
(`point_reading`), on the line beneath as it ships. So a probe window on the line beneath, which raises that line,
raises the offsets and ratings the line above is read on, and the line above can come out lower in a coefficient:
part of what the tile costs moves to the line beneath, and each point stays covered and within its bound.

`tests/backend/working_memory/test_shipped_coefficients.py` holds every carried value that is not a declared stand-in
to `fitted_values` of the shipped points, so a new point that moves a fit fails there with the value to ship. A
`legacy` entry's hand-set values are neither fitted nor held to a fit. A value on no line its entry carries
(`line_values`) is hand-set too: the MOSS-Image denoise's `k_token`, bytes per attended token, whose entry ships no
base for its line. Its points still decide and bound it (`bounds.py`), and the data test allows such a value only for
an estimator keyed by no platform trait.

A user's own calibration fits the same lines from its windows (`user_pair`, read by `user_fit.py`), with
`USER_MARGIN`, the same margin, for the estimators that opt in (`EstimatorSpec.user_calibrated`): the lines whose
demand is their own (`Line.own_demand`). A line that sits on another, and its assembly line, are the shipped fit's
alone.
"""

import itertools
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
    """Whether a measurement of these features can join the line's fit: its demand is the line's alone, or the line's
    above the line it sits on."""
    label: str
    """How a point of these features reads in its id (`bounds.point_id`), e.g. ``decode-bf16-1024x1024-tile512``."""
    sits_on: Optional[tuple[str, int]] = None
    """The line the estimate prices this operation on beneath this one, as its suffix and the operation's size on it,
    or None. A tiled VAE operation sits on the untiled line at its tile area. The line named sits on none."""
    assembly: Optional[str] = None
    """The suffix of the line that prices this operation's second phase, in the same size, or None. A tiled decode
    assembles the image from its decoded tiles once the last tile ran (``assembled_decode_<dtype>``). The estimate
    reserves the larger phase: this line on the one it sits on, or the assembly line. The two are fitted together on
    this line's points (`two_phase_envelope`)."""
    caps: bool = True
    """Whether a fresh-process measurement of these features caps the line: rates its fit and holds its estimate to
    the upper bound of `bounds.py`. False for a VAE operation below `SMALLEST_APP_SIDE`, on a smaller tile or a smaller
    image, whose measurement bounds its estimate from below only."""

    @property
    def own_demand(self) -> bool:
        """Whether a measurement of these features measures this line's demand alone: it fits, and sits on no other
        line. The lines a user calibration fits."""
        return self.fits and self.sits_on is None


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


SMALLEST_APP_SIDE: Final = 512
"""The side, in output pixels, below which a VAE operation's measurement bounds its estimate from below only: it is
covered, but it neither rates nor caps a line (`Line.caps`), since a line held to it would over-reserve the sizes the
app runs, whose demand grows less per pixel.

- A tile: the upscale tab's smallest, `UPSCALE_TILE_SIZE_MIN` in
  `invokeai/frontend/webv2/src/features/upscale/core/settings.ts`, and the size the nodes' default resolves to for
  FLUX.1 and FLUX.2 (1024 px for SD). A smaller one is set by hand in the workflow editor only.
- An untiled image of fewer pixels than a square of this side: the model cache reserves at least the 3 GiB default
  `device_working_mem_gb`, and every shipped estimate stays below that up to 384 px a side (at 256 px, the smallest
  measured, below 1.4 GiB), so over-reserving such an image costs nothing by default. CogView4's decode on MIOpen
  passes it from about 448 px."""


def convolution_line(features: Mapping[str, FeatureValue]) -> Optional[Line]:
    """The AutoencoderKL convolution line of an operation: every VAE estimator's.

    An untiled operation prices on ``<operation>_<dtype>`` in output pixels. A tiled one prices on
    ``tiled_<operation>_<dtype>`` in the image's pixels, all samples' together, and sits on the untiled line at the
    area of its tile clamped to the image per axis: an estimator records a covering tile as no tile, since the VAE
    runs the single pass then. A tiled decode has a second phase, after its last tile: the decoded tiles, the rows
    blended from them and the image concatenated from those, all live at once, and the node's post-processing. It
    prices on the assembly line ``assembled_decode_<dtype>`` in the same size (`Line.assembly`). A tiled encode
    assembles only the moments, at an eighth of each side, which its tiled line covers. A tile below
    `SMALLEST_APP_SIDE`, or an untiled image of fewer pixels than its square, caps nothing (`Line.caps`).

    A single-sample operation of an AutoencoderKL whose VAE no in-place cast grew fits its line. The cast's growth is a
    term of its own, a batch prices its base once per sample, and the lines are the AutoencoderKL convolutions': the SD
    estimator also prices the tiny autoencoder (`vae_class` AutoencoderTiny) on them, a different network whose windows
    say nothing about those lines. Estimators that name no `vae_class` price an AutoencoderKL only.
    """
    operation, dtype = features.get("operation"), features.get("dtype")
    height, width = features.get("height"), features.get("width")
    if not isinstance(operation, str) or not isinstance(dtype, str):
        return None
    if not isinstance(height, int) or not isinstance(width, int):
        return None
    tile, batch = features.get("tile"), features.get("batch_size", 1)
    fits = (
        batch == 1
        and not features.get("upcast_weight_bytes", 0)
        and features.get("vae_class", "AutoencoderKL") == "AutoencoderKL"
    )
    batched = f"-b{batch}" if batch != 1 else ""
    label = f"{operation}-{dtype}-{height}x{width}"
    if not isinstance(tile, int):
        caps = height * width >= SMALLEST_APP_SIDE * SMALLEST_APP_SIDE
        return Line(f"{operation}_{dtype}", height * width, fits, f"{label}{batched}", caps=caps)
    samples = batch if isinstance(batch, int) else 1
    tile_area = min(tile, height) * min(tile, width)
    return Line(
        f"tiled_{operation}_{dtype}",
        height * width * samples,
        fits,
        f"{label}-tile{tile}{batched}",
        sits_on=(f"{operation}_{dtype}", tile_area),
        assembly=f"assembled_decode_{dtype}" if operation == "decode" else None,
        caps=tile >= SMALLEST_APP_SIDE,
    )


def entry_key(entry: CoefficientEntry | MeasuredPoint) -> EntryKey:
    return (entry.estimator, entry.version, tuple(sorted(entry.platform.items())))


class FitPoint(NamedTuple):
    size: int
    bytes: int
    caps: bool
    """Whether the point takes part in choosing the line (`MeasuredPoint.holds_upper_bound` and `Line.caps`); every
    point is covered."""
    base: Optional[tuple[EntryKey, str, int]] = None
    """For a point of a line that sits on another (`Line.sits_on`): the entry that carries that line for the point,
    its suffix, and the point's size on it."""
    assembly: Optional[str] = None
    """For a point of a line with an assembly phase (`Line.assembly`): the assembly line's suffix."""
    reference: Optional[int] = None
    """For a point of a line that sits on another: the highest untiled demand its own estimator measured on its
    platform at its size on the line beneath, of its in-fit points that hold the upper bound; None where none did."""
    fresh: bool = False
    """Whether the point is a fresh-process measurement (`MeasuredPoint.holds_upper_bound`), not a warm `probe_demand`
    window. A fresh point below `SMALLEST_APP_SIDE` caps nothing but still shapes the fit a line is held to at least
    (`fitted_pair`)."""


def spans(sizes: Iterable[int]) -> bool:
    """Whether the sizes include two at least `MIN_AREA_RATIO` apart: enough to fit a line on."""
    distinct = sorted(set(sizes))
    return len(distinct) >= 2 and distinct[-1] >= MIN_AREA_RATIO * distinct[0]


Reading = tuple[int, int] | tuple[int, int, Fraction] | tuple[int, int, Fraction, Fraction]
"""A point as `upper_envelope` reads it: its size and measured bytes; for a line that sits on another, its offset, the
demand above which the line covers it (0 without); and the bytes its estimate prices it at beneath the line, which its
rating counts (the offset without)."""


def _with_offset(points: Sequence[Reading]) -> list[tuple[int, int, Fraction, Fraction]]:
    readings = []
    for x, y, *rest in points:
        offset = Fraction(rest[0]) if rest else Fraction(0)
        readings.append((x, y, offset, Fraction(rest[1]) if len(rest) > 1 else offset))
    return readings


def upper_envelope(points: Sequence[Reading], capping: Optional[Sequence[Reading]] = None) -> tuple[Fraction, Fraction]:
    """The line ``y = k * x + b`` with ``k, b >= 0`` on or above every point's demand above its offset, ``y - offset``,
    that over-reserves least.

    The candidates are the rising edges of the upper convex hull of those demands and the origin, and the flat line
    through its highest vertex: each such line lies on or above every point, and the origin keeps its intercept
    non-negative. A falling edge would price larger images below smaller ones. Of those, the one whose largest ratio of
    a point's estimate, what it is priced at beneath the line plus ``k * x + b``, to its reading is smallest wins, the
    steeper one on a tie, since it is the more conservative past the largest point. Only the highest reading of a size
    and offset counts: a lower one of the same measurement is covered whatever the line, and rating against it would
    tilt the line though no point lies above it. Points that share a size but not an offset are different operations, a
    tiled one at two tiles, and each rates the edges. With `capping`, a subset of the points, only its readings rate
    the edges: the others are covered but cap nothing. Without any, all points rate them. Where no point's demand lies
    above its offset, the line is zero: the estimate beneath it covers every point. Raises unless the points span two
    sizes at least `MIN_AREA_RATIO` apart.
    """
    readings = _with_offset(points)
    _check_sizes(points, readings)
    sizes = sorted({x for x, *_ in readings})

    highest = {x: max(y - o for px, y, o, _ in readings if px == x) for x in sizes}
    hull: list[tuple[int, Fraction]] = [(0, Fraction(0))]
    for point in highest.items():
        # Drop the last vertex while it is not above the chord from the one before it to this point.
        while len(hull) >= 2:
            (x0, y0), (x1, y1) = hull[-2], hull[-1]
            if (x1 - x0) * (point[1] - y0) - (y1 - y0) * (point[0] - x0) < 0:
                break
            hull.pop()
        hull.append(point)

    rated: dict[tuple[int, Fraction, Fraction], int] = {}
    for x, y, o, r in _with_offset(capping) if capping else readings:
        rated[(x, o, r)] = max(y, rated.get((x, o, r), y))
    edges = [((y1 - y0) / (x1 - x0), x0, y0) for (x0, y0), (x1, y1) in zip(hull, hull[1:], strict=False)]
    candidates = [(k, y0 - k * x0) for k, x0, y0 in edges if k >= 0]
    # Past the hull's highest vertex its edges fall; the flattest line `k >= 0` allows through that vertex stands in
    # for them. It is the last rising edge's equal or worse wherever the hull only rises, so it changes nothing there,
    # and where no edge rises (every demand below its offset) it is the zero line.
    candidates.append((Fraction(0), max(y for _, y in hull)))
    best: tuple[Fraction, Fraction, Fraction] | None = None
    for k, b in candidates:
        worst = max((r + k * x + b) / y for (x, _, r), y in rated.items())
        if best is None or (worst, -k) < (best[0], -best[1]):
            best = (worst, k, b)
    assert best is not None, "the flat line is always a candidate"
    return best[1], best[2]


LinePair = tuple[Fraction, Fraction]
"""A line's slope and intercept."""


def two_phase_envelope(
    points: Sequence[Reading], capping: Optional[Sequence[Reading]] = None
) -> tuple[LinePair, LinePair]:
    """A line that sits on another and its assembly line (`Line.assembly`), chosen together: ``T`` above each point's
    offset, ``A`` on its own, such that ``max(offset + T(x), A(x))`` lies on or above every point and, as the lines
    ship, over-reserves least. A deterministic search:

    1. The candidates for ``A``: the zero line, and every line ``k >= 0`` through the readings ``(x_i, y_i)`` and
       ``(x_j, y_j)`` of two points with ``x_i < x_j``. ``A``'s intercept may be negative: it only counts where ``A``
       is the larger phase.
    2. Each point above a candidate's ``A`` is left to ``T``: the `upper_envelope` of those points, rated on those of
       them that cap (on all of them where none does), or the zero line if no point is left. A candidate whose points
       left span no two sizes `MIN_AREA_RATIO` apart cannot fit ``T`` and is skipped.
    3. Each candidate is rated as it ships: both lines plus `SHIPPED_MARGIN` and rounded (`_with_margin`), ``T'`` and
       ``A'``, on the capping points (without `capping`, on all), by the ratio of a point's estimate
       ``max(priced + T'(x), A'(x))`` to its reading, ``priced`` what it is priced at beneath ``T``, the highest
       reading of a size and offset only, as `upper_envelope` rates. The smallest largest ratio wins, then the
       smallest sum of ratios. A tie goes to the zero ``A``, so that no assembly line ships that no point needs;
       between two other candidates, to the steeper ``A``, the more conservative past the largest point, then to the
       higher intercept. ``A`` decides ``T``, so that orders every candidate.

    Rated before the margin, the search would favour the assembly line where it should not: the margin scales the
    whole phase a line prices, for ``A`` a point's whole demand and for ``T`` only what tiling adds above the tile, so a
    point ``A`` prices would ship 5% above the ratio it was rated at. The margin plays no part in which points ``A``
    covers: that is step 2, before it.

    With the zero ``A`` every point is left to ``T``: the line alone, as `upper_envelope` fits it. Returns the chosen
    lines before the margin, which `shipped_two_phase` adds. Raises unless the points span two sizes at least
    `MIN_AREA_RATIO` apart.
    """
    readings = _with_offset(points)
    _check_sizes(points, readings)
    rated: dict[tuple[int, Fraction, Fraction], int] = {}
    for x, y, o, r in _with_offset(capping) if capping else readings:
        rated[(x, o, r)] = max(y, rated.get((x, o, r), y))
    capping_readings = _with_offset(capping) if capping else []

    measured = sorted({(x, y) for x, y, _, _ in readings})
    assemblies = {(Fraction(0), Fraction(0))}
    for (x0, y0), (x1, y1) in itertools.combinations(measured, 2):
        if x0 < x1 and y1 >= y0:
            k = Fraction(y1 - y0, x1 - x0)
            assemblies.add((k, y0 - k * x0))

    best: Optional[tuple[tuple[Fraction, Fraction, tuple[Fraction, ...]], LinePair, LinePair]] = None
    for k_a, b_a in assemblies:
        left = [reading for reading in readings if reading[1] > k_a * reading[0] + b_a]
        if left:
            left_capping = [reading for reading in capping_readings if reading[1] > k_a * reading[0] + b_a]
            try:
                line = upper_envelope(left, left_capping)
            except ValueError:
                continue
        else:
            line = (Fraction(0), Fraction(0))
        (k_t, b_t), (k_s, b_s) = _with_margin(*line, SHIPPED_MARGIN), _with_margin(k_a, b_a, SHIPPED_MARGIN)
        ratios = [max(r + k_t * x + b_t, Fraction(k_s * x + b_s)) / y for (x, _, r), y in rated.items()]
        tie_break = (Fraction(0),) if k_a == b_a == 0 else (Fraction(1), -k_a, -b_a)
        rating = (max(ratios), sum(ratios, Fraction(0)), tie_break)
        if best is None or rating < best[0]:
            best = (rating, line, (k_a, b_a))
    assert best is not None, "the zero assembly line leaves every point to the line, which fits them"
    return best[1], best[2]


def _check_sizes(points: Sequence[Reading], readings: Sequence[tuple[int, int, Fraction, Fraction]]) -> None:
    if not readings or any(x <= 0 or y <= 0 for x, y, *_ in readings):
        raise ValueError(f"an envelope needs points with positive sizes and bytes, got {list(points)}")
    sizes = sorted({x for x, *_ in readings})
    if not spans(sizes):
        raise ValueError(
            f"an envelope needs two sizes at least {MIN_AREA_RATIO}x apart, got {sizes}; measure more sizes"
        )


def _with_margin(k: Fraction, b: Fraction, margin: Fraction) -> tuple[int, int]:
    """A line scaled by ``1 + margin``, slope and intercept alike, so that it grows wherever it is positive, a
    negative intercept included; the slope rounded up to a whole byte per unit of size and the intercept to a whole
    MiB, a negative one towards zero."""
    scale = 1 + margin
    return math.ceil(k * scale), math.ceil(b * scale / _MIB) * _MIB


def shipped_pair(points: Sequence[Reading], capping: Optional[Sequence[Reading]] = None) -> tuple[int, int]:
    """The slope and intercept to ship for these points: the envelope plus `SHIPPED_MARGIN`, the slope rounded up to
    a whole byte per unit of size and the intercept to a whole MiB."""
    return _with_margin(*upper_envelope(points, capping), SHIPPED_MARGIN)


def shipped_two_phase(
    points: Sequence[Reading], capping: Optional[Sequence[Reading]] = None
) -> tuple[tuple[int, int], tuple[int, int]]:
    """The slope and intercept to ship for a line with an assembly phase and for its assembly line: the pair
    `two_phase_envelope` chooses, each plus `SHIPPED_MARGIN` and rounded up as `shipped_pair` rounds."""
    line, assembly = two_phase_envelope(points, capping)
    return _with_margin(*line, SHIPPED_MARGIN), _with_margin(*assembly, SHIPPED_MARGIN)


def user_pair(points: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """The slope and intercept a user calibration's points give a line: the envelope plus `USER_MARGIN`, the slope
    rounded up to a whole byte per unit of size and the intercept to a whole MiB.

    Every user point is a `probe_demand` window, so no point caps the line and all of them rate its edges (the
    fallback of `upper_envelope`). Raises like `upper_envelope` unless the points span two sizes `MIN_AREA_RATIO`
    apart."""
    return _with_margin(*upper_envelope(points), USER_MARGIN)


def carried_lines(entry: CoefficientEntry) -> set[str]:
    """The suffixes an entry carries both halves of a line for."""
    names = set(entry.values or {})
    return {name[2:] for name in names if name.startswith("k_") and f"b_{name[2:]}" in names}


def line_values(entry: CoefficientEntry) -> set[str]:
    """The values of `entry` that `fitted_values` reproduces: both halves of each line it carries (`carried_lines`),
    stand-ins included. Any other value is hand-set (see the module docstring), which the data test allows only for an
    estimator keyed by no platform trait.

    The MOSS-Image denoise declares its line in attended tokens, but its entry carries the slope alone; shipping a
    fitted base (`b_token`) with it makes that a carried line, and this exemption goes."""
    return {f"{part}_{suffix}" for suffix in carried_lines(entry) for part in ("k", "b")}


def fit_points(
    data: Iterable[WorkingMemoryData], specs: Mapping[str, "EstimatorSpec"]
) -> Mapping[tuple[EntryKey, str], list[FitPoint]]:
    """Per carried entry and line, the points that line is fitted on.

    A point belongs to the line its estimator's `EstimatorSpec.line` places its features on, where that admits them,
    of the entry that carries that line on its platform (`carrier_of`). A point of a line that sits on another names
    the entry that carries that one (`FitPoint.base`) and the untiled demand its own estimator measured there
    (`FitPoint.reference`). A point of a line with an assembly phase measures the assembly line too, which the same
    entry carries, and is listed under both (`FitPoint.assembly`). `specs` holds the spec of every estimator a point
    names. Lines no point measures are absent. Raises where a line sits on one no entry
    carries, or on one that sits on another, and where its entry does not carry its assembly line.
    """
    files = list(data)
    entries = [entry for file in files for entry in file.coefficients]
    placed = [
        (point, line)
        for point in (point for file in files for point in file.points)
        if point.in_fit and (line := specs[point.estimator].line(point.features)) is not None and line.fits
    ]
    references: dict[tuple[str, int, tuple[tuple[str, str], ...], str, int], int] = {}
    for point, line in placed:
        if line.sits_on is None and point.holds_upper_bound:
            key = (point.estimator, point.version, tuple(sorted(point.platform.items())), line.suffix, line.size)
            references[key] = max(point.measured_bytes, references.get(key, 0))
    groups: dict[tuple[EntryKey, str], list[FitPoint]] = {}
    for point, line in placed:
        owner = carrier_of(entries, point, line.suffix)
        if line.suffix not in carried_lines(owner):
            continue
        base, reference = None, None
        if line.sits_on is not None:
            suffix, size = line.sits_on
            carrier = carrier_of(entries, point, suffix)
            if suffix not in carried_lines(carrier):
                raise ValueError(f"{point.estimator} on {point.platform}: {line.suffix} sits on {suffix}, not carried")
            base = (entry_key(carrier), suffix, size)
            platform = tuple(sorted(point.platform.items()))
            reference = references.get((point.estimator, point.version, platform, suffix, size))
        fresh = point.holds_upper_bound
        caps = fresh and line.caps
        fit_point = FitPoint(line.size, point.measured_bytes, caps, base, line.assembly, reference, fresh)
        groups.setdefault((entry_key(owner), line.suffix), []).append(fit_point)
        if line.assembly is not None:
            if line.assembly not in carried_lines(owner):
                raise ValueError(
                    f"{point.estimator} on {point.platform}: {line.suffix} pairs with {line.assembly}, not carried"
                )
            groups.setdefault((entry_key(owner), line.assembly), []).append(fit_point)
    sitting = {key for key, points in groups.items() if any(point.base for point in points)}
    for (owner_key, suffix), points in groups.items():
        chained = sorted({point.base[1] for point in points if point.base and point.base[:2] in sitting})
        if chained:
            raise ValueError(f"{owner_key[0]} on {dict(owner_key[2])}: {suffix} sits on {chained}, which sit on others")
    return groups


def carrier_of(entries: Sequence[CoefficientEntry], point: MeasuredPoint, suffix: str) -> CoefficientEntry:
    """The entry that carries line `suffix` for `point`: its estimator's entry on its platform where that carries the
    line itself, the one that entry borrows from otherwise. A point names the OS only where it measured an OS-specific
    entry's platform."""
    where = f"{point.estimator} point on {point.platform}"
    entry = find_entry(entries, point.estimator, point.version, point.platform, where)
    if entry.values_from is None or suffix in carried_lines(entry):
        return entry
    return find_entry(entries, entry.values_from.estimator, entry.values_from.version, point.platform, where)


def point_reading(point: FitPoint, bases: Mapping[EntryKey, Mapping[str, float]]) -> Reading:
    """A point as the envelopes read it: covered above the untiled demand its estimator measured at its tile, where it
    did, and above the line beneath otherwise; rated on what that line prices it at, plus the line's own. Public for
    `scripts/working_memory/build_shipped.py`, which raises a stand-in on its points as the fit reads them."""
    if point.base is None:
        return (point.size, point.bytes)
    key, suffix, size = point.base
    values = bases[key]
    priced = Fraction(values[f"k_{suffix}"]) * size + Fraction(values[f"b_{suffix}"])
    return (point.size, point.bytes, priced if point.reference is None else Fraction(point.reference), priced)


def entry_fit(
    entry: CoefficientEntry,
    groups: Mapping[tuple[EntryKey, str], Sequence[FitPoint]],
    bases: Optional[Mapping[EntryKey, Mapping[str, float]]],
) -> tuple[dict[str, int], dict[str, str]]:
    """What `fitted_pair` gives each line `entry` carries and does not declare a stand-in, from `groups`
    (`fit_points`); and, per such line its points cannot fit, why.

    A line that sits on another is rated on what the line beneath prices its points at, with `bases`
    (`base_values`). Given None, as `base_values` fits the lines that sit on none first, such a line is a problem: it
    cannot be fitted yet. A line with an assembly phase is fitted together with its assembly line
    (`fitted_two_phase`), which gets its values there."""
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
        if any(point.assembly == suffix for point in line):
            continue
        if bases is None and any(point.base for point in line):
            problems[suffix] = "it sits on another line, whose fit it needs first (fit.base_values)"
            continue
        readings = [point_reading(point, bases or {}) for point in line]
        capping = [reading for reading, point in zip(readings, line, strict=True) if point.caps]
        fresh = [reading for reading, point in zip(readings, line, strict=True) if point.fresh]
        assembly = line[0].assembly
        try:
            if assembly is None:
                pair = fitted_pair(readings, fresh, capping)
                values |= dict(zip((f"k_{suffix}", f"b_{suffix}"), pair, strict=True))
            else:
                pair, assembled = fitted_two_phase(readings, fresh, capping)
                values |= dict(zip((f"k_{suffix}", f"b_{suffix}"), pair, strict=True))
                values |= dict(zip((f"k_{assembly}", f"b_{assembly}"), assembled, strict=True))
        except ValueError as e:
            problems[suffix] = str(e)
    return values, problems


def base_values(
    entries: Iterable[CoefficientEntry], groups: Mapping[tuple[EntryKey, str], Sequence[FitPoint]]
) -> dict[EntryKey, dict[str, float]]:
    """Per entry that carries values, what a line that sits on one of its lines prices that line with: its carried
    values, with the fit of each line that sits on none in their place where the entry is fitted (not `legacy`, not a
    stand-in). The first step of the fit's order: the lines that sit on one are the problems of this step, and are
    fitted on its result (`fitted_values`)."""
    bases: dict[EntryKey, dict[str, float]] = {}
    for entry in entries:
        if entry.values is None:
            continue
        fitted = {} if entry.legacy else entry_fit(entry, groups, None)[0]
        bases[entry_key(entry)] = dict(entry.values) | fitted
    return bases


def fitted_values(
    data: Iterable[WorkingMemoryData], specs: Mapping[str, "EstimatorSpec"]
) -> Mapping[EntryKey, Mapping[str, int]]:
    """Per entry that carries values and is not `legacy`, what `fitted_pair` gives each line it does not declare a
    stand-in, and `fitted_two_phase` each line with an assembly phase together with its assembly line (`entry_fit`):
    the lines that sit on none first, then those that sit on one, rated with that fit (`base_values`).

    Raises, naming the entry and line, where the points cannot fit that line.
    """
    files = list(data)
    groups = fit_points(files, specs)
    entries = [entry for file in files for entry in file.coefficients]
    bases = base_values(entries, groups)
    fitted: dict[EntryKey, dict[str, int]] = {}
    for entry in entries:
        if entry.values is None or entry.legacy:
            continue
        values, problems = entry_fit(entry, groups, bases)
        if problems:
            suffix, problem = next(iter(problems.items()))
            raise ValueError(f"{entry.estimator} on {entry.platform}, {suffix}: {problem}")
        if values:
            fitted[entry_key(entry)] = values
    return fitted


def capping_spans(line: Sequence[FitPoint]) -> bool:
    """Whether a line's capping points (`FitPoint.caps`) alone span the sizes a fit needs, as only then is it fitted.
    Warm windows, or one fresh size with warm windows or points below `SMALLEST_APP_SIDE` at others, could pin it below
    what a cold first run at a size the app sets needs."""
    return spans(point.size for point in line if point.caps)


def _check_capping(capping: Sequence[Reading]) -> None:
    sizes = sorted({x for x, *_ in capping})
    if not spans(sizes):
        raise ValueError(
            f"its capping points measure sizes {sizes}, not two at least {MIN_AREA_RATIO}x apart; keep it a "
            "stand-in until they do"
        )


def _larger(first: tuple[int, int], second: tuple[int, int]) -> tuple[int, int]:
    return max(first[0], second[0]), max(first[1], second[1])


def fitted_pair(readings: Sequence[Reading], fresh: Sequence[Reading], capping: Sequence[Reading]) -> tuple[int, int]:
    """The pair to ship for a line's readings (`shipped_pair`), `fresh` those of its fresh-process points and
    `capping` those of its capping points: the coefficient-wise larger of its fit on all of them and its fit on the
    fresh ones alone, both rated on the capping ones. A probe window above the hull can drop a fresh vertex and so
    lower one half of the line; the larger of the two keeps both at the fresh fit at least. A fresh point that caps
    nothing (below `SMALLEST_APP_SIDE`) is a cold measurement: it shapes both fits, and rates neither.

    Raises unless the capping readings span (`capping_spans`)."""
    _check_capping(capping)
    return _larger(shipped_pair(readings, capping), shipped_pair(fresh, capping))


def fitted_two_phase(
    readings: Sequence[Reading], fresh: Sequence[Reading], capping: Sequence[Reading]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """The pairs to ship for a line with an assembly phase and for its assembly line (`shipped_two_phase`): per line
    and coefficient, the larger of the two-phase fit on all the readings and the two-phase fit on the fresh ones
    alone, both rated on the capping ones. Each estimate ``max(priced + T(x), A(x))`` grows with every coefficient, so
    the result lies on or above both fits' estimates at every size: a probe window on this line can raise either line
    but lower neither. Both fits read the readings on the line beneath as it ships, which a probe window there can
    raise (`point_reading`).

    Raises unless the capping readings span (`capping_spans`)."""
    _check_capping(capping)
    line, assembly = shipped_two_phase(readings, capping)
    fresh_line, fresh_assembly = shipped_two_phase(fresh, capping)
    return _larger(line, fresh_line), _larger(assembly, fresh_assembly)
