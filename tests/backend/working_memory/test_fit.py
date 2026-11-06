"""The shipped fit: an upper envelope of the points, with a non-negative base, plus the shipped margin, raised where
other points lie above it by the least the upper bounds allow."""

from dataclasses import replace
from fractions import Fraction

import pytest

from invokeai.backend.working_memory.bounds import bound_violation, evaluate_point, line_bounds, point_id
from invokeai.backend.working_memory.coefficients import CoefficientEntry, MeasuredPoint, ValuesFrom, WorkingMemoryData
from invokeai.backend.working_memory.estimate import WorkingMemoryEstimate, WorkingMemoryTerm
from invokeai.backend.working_memory.fit import (
    SHIPPED_MARGIN,
    SMALLEST_APP_SIDE,
    FitPoint,
    Line,
    LineBound,
    LineConflict,
    base_values,
    convolution_line,
    entry_fit,
    entry_key,
    fit_points,
    fitted_values,
    hand_set_values,
    point_reading,
    raised_pair,
    shipped_pair,
    shipped_two_phase,
    two_phase_envelope,
    upper_envelope,
)
from invokeai.backend.working_memory.spec import EstimatorSpec, import_estimator_modules
from invokeai.backend.z_image import denoise_working_memory
from invokeai.backend.z_image.denoise_working_memory import Z_IMAGE_DENOISE_SPEC

MIB = 2**20
GIB = 2**30


def _no_terms(features, values):
    return ()


# Throwaway estimators on the convolution line every VAE estimator declares.
SPECS = {
    estimator: EstimatorSpec(
        id=estimator,
        version=1,
        coefficients=frozenset(),
        platform_keys=("conv_backend",),
        terms=_no_terms,
        line=convolution_line,
    )
    for estimator in ("lender", "borrower")
}

# A process base plus a per-pixel cost, read with the scatter real points have: the shape the VAE points take.
SCATTERED = [(262144, 940 * MIB), (589824, 1540 * MIB), (1048576, 2380 * MIB), (1638400, 3120 * MIB)]
PROPORTIONAL = [(262144, 1100 * MIB), (589824, 2457 * MIB), (1048576, 4330 * MIB), (4194304, 17194 * MIB)]


@pytest.mark.parametrize("points", [SCATTERED, PROPORTIONAL], ids=["with-a-base", "proportional"])
def test_the_envelope_covers_every_point_with_a_non_negative_base(points) -> None:
    k, b = upper_envelope(points)
    assert b >= 0
    assert all(k * x + b >= y for x, y in points)


def test_the_envelope_is_the_hull_edge_that_over_reserves_its_worst_point_least() -> None:
    """Through the origin and the 512px point the line covers everything too, but over-reserves 1280px by 1.48x; the
    edge from 512px to 1024px over-reserves its worst point, 1280px, by 1.11x."""
    k, b = upper_envelope(SCATTERED)
    assert (k, b) == (Fraction(600 * MIB, 327680), 940 * MIB - Fraction(600 * MIB, 327680) * 262144)
    assert max((k * x + b) / y for x, y in SCATTERED) < Fraction(112, 100)


def test_the_largest_reading_of_a_size_is_the_one_covered() -> None:
    k, b = upper_envelope([(262144, 1000 * MIB), (262144, 1100 * MIB), (1048576, 4400 * MIB)])
    assert k * 262144 + b >= 1100 * MIB


def test_a_tie_between_two_edges_goes_to_the_steeper_one() -> None:
    """Both outer edges over-reserve their worst point by 1.25x; the steeper is the more conservative past the largest
    point."""
    assert upper_envelope([(1, 10), (2, 15), (4, 20)]) == (5, 5)


def test_a_lower_reading_at_a_measured_size_does_not_move_the_line() -> None:
    """Rated against every reading, the 12 at size 4 would hand the line to the flatter edge (it over-reserves 12 by
    1.67x instead of 2.08x) though no point lies above the line the highest readings give."""
    assert upper_envelope([(1, 10), (2, 15), (4, 20), (4, 12)]) == upper_envelope([(1, 10), (2, 15), (4, 20)])


def test_a_falling_edge_is_never_the_line_and_the_flat_line_through_the_peak_stands_in() -> None:
    """A line that falls with the size would price larger images below smaller ones. Past the hull's peak the flattest
    line `k >= 0` allows through it over-reserves the 5 at size 4 by 2x; the edge from the origin would by 8x."""
    assert upper_envelope([(1, 10), (4, 5)]) == (0, 10)


def test_a_line_that_sits_on_another_covers_the_demand_above_it_and_is_rated_on_the_whole_estimate() -> None:
    """Two tiles of one tiled operation: the 1000-byte tile measures 100 and 200 bytes above its untiled price, the
    4000-byte one 100 below and level. The hull of the highest demands above the offsets runs through 100 at size 1
    and 200 at size 4; of its edges and the flat line, the edge between them over-reserves its worst point least,
    rated on offset plus line against the reading: 1.06x at the larger tile, where the edge from the origin would
    reserve 1.17x of the smaller tile's 1200."""
    points = [
        (1, 1100, Fraction(1000)),
        (4, 1200, Fraction(1000)),
        (2, 3900, Fraction(4000)),
        (4, 4000, Fraction(4000)),
    ]

    k, b = upper_envelope(points)

    assert (k, b) == (Fraction(100, 3), Fraction(200, 3))
    assert all(o + k * x + b >= y for x, y, o in points)


def test_a_line_that_sits_on_another_is_zero_where_nothing_lies_above_its_offset() -> None:
    assert upper_envelope([(1, 900, Fraction(1000)), (4, 950, Fraction(1000))]) == (0, 0)


def _floored(x: int, y: int, floor: int) -> tuple[int, int, Fraction, Fraction, Fraction]:
    """A point of a line that sits on none beneath, with what its estimate reserves whatever the line: its floor, the
    shipped margin included, which covers the point where it is at least the reading plus that margin."""
    return (x, y, Fraction(0), Fraction(0), Fraction(floor))


def test_a_point_its_floor_covers_lifts_no_line() -> None:
    """The 100 at size 8 would pull the line to it; its floor reserves 105 whatever the line, the 100 and the shipped
    margin, so the line stays the one the other points give. A floor of 104 covers the 100 without the margin a line
    would ship, and leaves the point to the line."""
    points = [(1, 10), (2, 15), (4, 20)]

    assert upper_envelope([*points, _floored(8, 100, 104)]) == upper_envelope([*points, (8, 100)])
    assert upper_envelope([*points, (8, 100)]) != upper_envelope(points)
    assert upper_envelope([*points, _floored(8, 100, 105)]) == upper_envelope(points) == (5, 5)


def test_a_point_its_floor_covers_rates_a_line_that_prices_it_above_the_floor() -> None:
    """Without the point at 16 the edges (5, 5) and (2.5, 10) both over-reserve their worst point 1.25x, and the
    steeper wins. Its floor covers the 50 at 16 with 53, but (5, 5) prices 85 there, 1.7x what it measured, where
    (2.5, 10) prices it under the floor: what the estimate reserves rates."""
    points = [(1, 10), (2, 15), (4, 20)]

    assert upper_envelope([*points, _floored(16, 50, 53)]) == (Fraction(5, 2), 10)


def test_a_point_its_floor_covers_rates_no_line_that_prices_it_below_the_floor() -> None:
    """The floor reserves 90 for the 50 at 16 whichever edge wins: (5, 5) prices it at 90 and (2.5, 10) at 50, neither
    above it. Rated on what the edges price, 1.8x against 1.0x, the point would hand the line to the flatter edge,
    though it reserves the same under both; rated on nothing, the tie between them goes to the steeper as without it."""
    points = [(1, 10), (2, 15), (4, 20)]

    assert upper_envelope([*points, _floored(16, 50, 90)]) == upper_envelope(points) == (5, 5)


def test_a_floor_no_line_reaches_does_not_decide_between_the_lines() -> None:
    """The floor reserves 2.5x the 10 at size 1 whatever the line. Counted as each edge's worst ratio, it would tie
    every edge and hand the line to the steepest, (10, 0), which over-reserves the 40 at 8 by 2x; left out, the edge
    (5, 10) wins, the 1.25x it holds the others to."""
    points = [(2, 20), (4, 30), (8, 40)]

    assert upper_envelope([*points, _floored(1, 10, 25)]) == upper_envelope(points) == (5, 10)


def test_a_line_is_held_against_the_floor_as_it_ships() -> None:
    """The floor, the margin in it, reserves 87 for the 50 at 16. The edge (5, 5) prices that point at 85 before the
    margin, under the floor, and at 89.25 as it ships, above it: it is the shipped line that would reserve more than
    the floor, so the point rates it, 1.7x, and the flatter edge (2.5, 10) wins, as at a floor of 85."""
    points = [(1, 10), (2, 15), (4, 20)]

    assert upper_envelope([*points, _floored(16, 50, 87)]) == (Fraction(5, 2), 10)


def test_a_point_its_floor_covers_rates_no_assembly_line_that_prices_it_below_the_floor() -> None:
    """The assembly line through the readings at 6 and 8 Mi pixels prices the tile's decodes best (`FLAT_THEN_STEEP`).
    At 16 Mi pixels it prices the 300 MiB measured there at 2.6x, under the 800 MiB its floor reserves anyway; the
    tiled line alone would price it at 1.8x, but over-reserves the flat points 1.89x. The floor decides that point
    whichever line ships, so the assembly line still wins."""
    x, y, offset = 16 * MIB, 300 * MIB, Fraction(100 * MIB)
    covered = (x, y, offset, offset, Fraction(800 * MIB))

    assert two_phase_envelope([*FLAT_THEN_STEEP, covered]) == two_phase_envelope(FLAT_THEN_STEEP)


def test_a_floor_no_candidate_reaches_does_not_decide_between_the_candidates() -> None:
    """The floor reserves 3x the 60 MiB at 1 Mi pixels whichever lines ship. Counted as every candidate's worst ratio,
    it would tie them all and leave the choice to the sum of ratios, which takes an assembly line, 7.5 bytes per pixel
    from 80 MiB; left out, the tiled line alone, 5 bytes per pixel, wins on the worst of the other points, as it does
    without the covered point."""
    decodes = _decodes((4, 110, 100), (8, 140, 100), (12, 140, 100))
    offset = Fraction(100 * MIB)
    covered = (1 * MIB, 60 * MIB, offset, offset, Fraction(180 * MIB))

    assert two_phase_envelope([*decodes, covered]) == two_phase_envelope(decodes) == ((5, 0), (0, 0))


def test_no_assembly_line_passes_through_a_point_its_floor_covers() -> None:
    """The floor covers the 150 MiB at 2 Mi pixels, with the margin. An assembly line through it and the 350 MiB at 8,
    33.3 bytes per pixel from 83 MiB, would rate better than the tiled line alone; but that point needs no line, so no
    candidate is drawn through it, and the tiled line alone, 31.25 bytes per pixel, wins."""
    decodes = _decodes((3, 140, 100), (4, 180, 100), (8, 350, 100))
    offset = Fraction(100 * MIB)
    covered = (2 * MIB, 150 * MIB, offset, offset, Fraction(240 * MIB))

    assert two_phase_envelope([*decodes, covered]) == ((Fraction(125, 4), 0), (0, 0))


def test_the_shipped_margin_is_five_percent() -> None:
    assert SHIPPED_MARGIN == Fraction(5, 100)


def test_the_shipped_pair_adds_the_margin_and_rounds_up() -> None:
    k, b = upper_envelope(SCATTERED)
    shipped_k, shipped_b = shipped_pair(SCATTERED)
    assert shipped_k >= k * (1 + SHIPPED_MARGIN) > shipped_k - 1
    assert shipped_b % MIB == 0
    assert shipped_b >= b * (1 + SHIPPED_MARGIN) > shipped_b - MIB


@pytest.mark.parametrize(
    "points",
    [
        pytest.param([], id="none"),
        pytest.param([(1048576, 4330 * MIB)], id="one-size"),
        pytest.param([(1048576, 4330 * MIB), (1048576, 4300 * MIB)], id="one-size-twice"),
        pytest.param([(1048576, 4330 * MIB), (1638400, 6748 * MIB)], id="sizes-too-close"),
    ],
)
def test_a_line_needs_two_sizes_at_least_twice_apart(points) -> None:
    with pytest.raises(ValueError):
        upper_envelope(points)


def _point(estimator: str, operation: str, px: int, mib: int, **extra) -> MeasuredPoint:
    features = {"operation": operation, "dtype": "bf16", "height": px, "width": px, "tile": None} | extra
    return MeasuredPoint(
        estimator=estimator,
        version=1,
        platform={"conv_backend": "cudnn"},
        features=features,
        measured_bytes=mib * MIB,
        method="reserved_peak",
        device="-",
        provenance="-",
    )


def _entry(estimator: str, values=None, lender: str | None = None) -> CoefficientEntry:
    return CoefficientEntry(
        estimator=estimator,
        version=1,
        platform={"conv_backend": "cudnn"},
        values=values,
        values_from=ValuesFrom(estimator=lender, version=1) if lender else None,
        provenance="-",
    )


def test_a_line_is_fitted_on_the_single_sample_untiled_points_of_every_member_that_borrows_it() -> None:
    values = {"k_decode_bf16": 0.0, "b_decode_bf16": 0.0}
    data = WorkingMemoryData(
        schema_version=1,
        coefficients=[_entry("lender", values), _entry("borrower", lender="lender")],
        points=[
            _point("lender", "decode", 512, 1100),
            _point("borrower", "decode", 1024, 4400),
            # None of these is the line's: another operation, a tile, a batch, a point out of regime, and a point whose
            # demand includes an in-place upcast the estimate prices as a term of its own.
            _point("borrower", "encode", 1024, 9000),
            _point("borrower", "decode", 2048, 1000, tile=512),
            _point("borrower", "decode", 1024, 9000, batch_size=2),
            _point("borrower", "decode", 2048, 90000, upcast_weight_bytes=160 * MIB),
            MeasuredPoint.model_validate(_point("lender", "decode", 2048, 90000).model_dump() | {"in_fit": False}),
        ],
    )
    fitted = fitted_values([data], SPECS, {})[("lender", 1, (("conv_backend", "cudnn"),))]
    assert (fitted["k_decode_bf16"], fitted["b_decode_bf16"]) == shipped_pair(
        [(512 * 512, 1100 * MIB), (1024 * 1024, 4400 * MIB)]
    )


TILED_POINTS = [
    _point("borrower", "encode", 2048, 1250, tile=512),
    _point("borrower", "encode", 4096, 1500, tile=512),
    _point("borrower", "encode", 2048, 4500, tile=1024),
]
"""Tiled encodes: a line that sits on the untiled one, with no assembly phase."""

UNTILED = [(512 * 512, 1100 * MIB), (1024 * 1024, 4400 * MIB)]


def _untiled(operation: str) -> list[MeasuredPoint]:
    """The untiled points of `operation` the tiled lines below sit on (`UNTILED`), one of each estimator."""
    return [_point("lender", operation, 512, 1100), _point("borrower", operation, 1024, 4400)]


def test_a_tiled_line_covers_what_its_points_measured_above_their_estimators_untiled_demand_at_the_tile() -> None:
    """The borrower carries its tiled line and borrows the untiled one. Each tiled point is covered above the untiled
    demand its own estimator measured at its tile: 4400 MiB at the 1024px tile. At the 512px tile the borrower measured
    nothing untiled (the 512px point is the lender's), so the untiled line's fit stands in, fitted first. Every point
    is rated on what is reserved for it, that fit at its tile plus the tiled line. Covered above the untiled line at
    1024px instead, the 1024px tile's 4500 MiB would add nothing, and the line would not carry what tiling adds there
    to another tile."""
    data = WorkingMemoryData(
        schema_version=1,
        coefficients=[
            _entry("lender", {"k_encode_bf16": 0.0, "b_encode_bf16": 0.0}),
            _entry("borrower", {"k_tiled_encode_bf16": 0.0, "b_tiled_encode_bf16": 0.0}, lender="lender"),
        ],
        points=[*_untiled("encode"), *TILED_POINTS],
    )

    fitted = fitted_values([data], SPECS, {})

    k, b = shipped_pair(UNTILED)
    assert fitted[("lender", 1, (("conv_backend", "cudnn"),))] == {"k_encode_bf16": k, "b_encode_bf16": b}

    def at(tile: int) -> Fraction:
        return Fraction(k * tile * tile + b)

    readings = [
        (2048 * 2048, 1250 * MIB, at(512), at(512)),
        (4096 * 4096, 1500 * MIB, at(512), at(512)),
        (2048 * 2048, 4500 * MIB, Fraction(4400 * MIB), at(1024)),
    ]
    tiled = shipped_pair(readings)
    assert fitted[("borrower", 1, (("conv_backend", "cudnn"),))] == {
        "k_tiled_encode_bf16": tiled[0],
        "b_tiled_encode_bf16": tiled[1],
    }
    assert tiled != shipped_pair([(x, y, rate) for x, y, _, rate in readings]), "over the untiled line instead"
    assert tiled != shipped_pair([reading[:2] for reading in readings]), "on the raw demand"


COUNTED_PER_PIXEL = 20
"""What the floor of `_floored_terms` counts per image pixel, as a tiled VAE operation counts its buffers."""


def _floored_terms(features, values):
    """A tiled operation's two phases as the VAE estimators price them: the tile on the untiled line plus the tiled
    line, or the floor, the counted buffers and the untiled line's base, whichever is larger. An untiled one is its
    line."""
    line = convolution_line(features)
    assert line is not None
    if line.sits_on is None:
        return (
            WorkingMemoryTerm.fitted("convolutions", f"k_{line.suffix}", line.size, values),
            WorkingMemoryTerm.fitted("convolution_base", f"b_{line.suffix}", 1, values),
        )
    untiled, tile_area = line.sits_on
    tile_phase = (
        WorkingMemoryTerm.fitted("convolutions", f"k_{untiled}", tile_area, values),
        WorkingMemoryTerm.fitted("convolution_base", f"b_{untiled}", 1, values),
        WorkingMemoryTerm.fitted("tiled", f"k_{line.suffix}", line.size, values),
        WorkingMemoryTerm.fitted("tiled_base", f"b_{line.suffix}", 1, values),
    )
    floor_phase = (
        WorkingMemoryTerm("buffers", COUNTED_PER_PIXEL * line.size, True),
        WorkingMemoryTerm.fitted("floor_base", f"b_{untiled}", 1, values),
    )
    floor_wins = sum(term.bytes for term in floor_phase) > sum(term.bytes for term in tile_phase)
    return (
        *(replace(term, included=not floor_wins) for term in tile_phase),
        *(replace(term, included=floor_wins) for term in floor_phase),
    )


def test_a_tiled_point_the_untiled_lines_headroom_covers_is_not_left_to_the_floor() -> None:
    """A tiled point's floor is what its estimate reserves whatever the lines price its tile and what tiling adds: the
    counted buffers and the untiled line's base, without the slope that prices the tile. The untiled line runs from the
    256 px point to the 1024 px one, above the 1000 MiB measured at the 512 px tile, so the tile phase alone covers a
    4096 px encode at that tile of up to its price there less the margin, though the encode lies above that 1000 MiB.
    Read with the slope beneath, its floor would be that tile phase and would leave the point to no line; read without,
    it is the 320 MiB of buffers and the base, and the point stays in the tiled line's hull."""
    spec = EstimatorSpec(
        id="tiler",
        version=1,
        coefficients=frozenset(),
        platform_keys=("conv_backend",),
        terms=_floored_terms,
        line=convolution_line,
    )
    values = {"k_encode_bf16": 0.0, "b_encode_bf16": 0.0, "k_tiled_encode_bf16": 0.0, "b_tiled_encode_bf16": 0.0}
    key = ("tiler", 1, (("conv_backend", "cudnn"),))
    untiled = [_point("tiler", "encode", px, mib) for px, mib in ((256, 400), (512, 1000), (1024, 4400))]

    def data(mib: int) -> WorkingMemoryData:
        """The untiled points, a 2048 px encode at a 1024 px tile, and a 4096 px one at a 512 px tile at `mib`."""
        tiled = [_point("tiler", "encode", 2048, 4500, tile=1024), _point("tiler", "encode", 4096, mib, tile=512)]
        return WorkingMemoryData(schema_version=1, coefficients=[_entry("tiler", values)], points=untiled + tiled)

    def fitted(mib: int) -> dict[str, int]:
        return dict(fitted_values([data(mib)], {"tiler": spec}, {})[key])

    shipped = fitted(1150)
    k, b = shipped["k_encode_bf16"], shipped["b_encode_bf16"]

    def at(tile: int) -> Fraction:
        return Fraction(k * tile * tile + b)

    groups = fit_points([data(1150)], {"tiler": spec})
    readings = [
        point_reading(point, base_values(data(1150).coefficients, groups, {}))
        for point in groups[(key, "tiled_encode_bf16")]
    ]
    assert [reading.floor for reading in readings] == [COUNTED_PER_PIXEL * px * px + b for px in (2048, 4096)]
    for mib in (1150, 1200):
        assert 1000 * MIB < mib * MIB and mib * MIB * (1 + SHIPPED_MARGIN) <= at(512), "within the headroom at the tile"
        assert mib * MIB * (1 + SHIPPED_MARGIN) > COUNTED_PER_PIXEL * 4096 * 4096 + b, "above the floor"

    tiled = (shipped["k_tiled_encode_bf16"], shipped["b_tiled_encode_bf16"])
    assert tiled == shipped_pair(
        [
            (2048 * 2048, 4500 * MIB, Fraction(4400 * MIB), at(1024)),
            (4096 * 4096, 1150 * MIB, Fraction(1000 * MIB), at(512)),
        ]
    )
    raised = fitted(1200)
    assert (raised["k_tiled_encode_bf16"], raised["b_tiled_encode_bf16"]) != tiled, "the 4096 px point moves the line"


def test_a_tile_below_the_smallest_the_app_sets_is_covered_but_caps_nothing() -> None:
    """A 256 px tile is set by hand in the workflow editor only. Its point joins the tiled line's fit: above the line it
    raises it to cover itself, below it moves nothing, as a probe window would not; and its estimate is held from below
    only, however far above it the line lies."""
    encode = {"operation": "encode", "dtype": "bf16", "height": 4096, "width": 4096}
    assert convolution_line(encode | {"tile": SMALLEST_APP_SIDE}).caps
    assert not convolution_line(encode | {"tile": SMALLEST_APP_SIDE // 2}).caps

    def fitted(*extra: MeasuredPoint) -> dict[str, int]:
        data = WorkingMemoryData(
            schema_version=1,
            coefficients=[
                _entry("lender", {"k_encode_bf16": 0.0, "b_encode_bf16": 0.0}),
                _entry("borrower", {"k_tiled_encode_bf16": 0.0, "b_tiled_encode_bf16": 0.0}, lender="lender"),
            ],
            points=[*_untiled("encode"), *TILED_POINTS, *extra],
        )
        return dict(fitted_values([data], SPECS, {})[("borrower", 1, (("conv_backend", "cudnn"),))])

    low, high = (_point("borrower", "encode", 4096, mib, tile=256) for mib in (100, 6000))
    assert fitted(low) == fitted()
    raised = fitted(high)
    k, b = shipped_pair(UNTILED)
    assert k * 256 * 256 + b + raised["k_tiled_encode_bf16"] * 4096 * 4096 + raised["b_tiled_encode_bf16"] >= 6000 * MIB

    def estimate(reserved: int) -> WorkingMemoryEstimate:
        terms = (WorkingMemoryTerm.fitted("tiled", "k", 1, {"k": reserved}),)
        return WorkingMemoryEstimate(estimator="borrower", version=1, features=low.features, terms=terms, source="-")

    assert bound_violation(low, SPECS["borrower"], estimate(100 * low.measured_bytes)) is None
    assert bound_violation(low, SPECS["borrower"], estimate(low.measured_bytes - 1)) is not None
    capping = _point("borrower", "encode", 4096, 100, tile=512)
    assert bound_violation(capping, SPECS["borrower"], estimate(100 * capping.measured_bytes)) is not None


def test_an_image_below_the_smallest_side_the_app_sets_is_covered_but_rates_nothing() -> None:
    """An untiled image of fewer pixels than a 512 px square bounds its estimate from below only. Its point lies on the
    hull here, below the line the larger images take on their own (3733 bytes per pixel from 667 MiB through 512 and
    1024 px), which reserves 2x its demand: rated, it would pick the edge through it and 512 px instead (6133 bytes per
    pixel from 67 MiB), 1.4x over the 1024 px point. Covered only, it leaves the larger images' line; a point above
    that line raises it."""
    encode = {"operation": "encode", "dtype": "bf16", "tile": None}
    assert convolution_line(encode | {"height": SMALLEST_APP_SIDE, "width": SMALLEST_APP_SIDE}).caps
    assert not convolution_line(encode | {"height": SMALLEST_APP_SIDE // 2, "width": SMALLEST_APP_SIDE // 2}).caps
    larger = [_point("lender", "encode", 512, 1600), _point("lender", "encode", 1024, 4400)]

    def fitted(*extra: MeasuredPoint) -> dict[str, int]:
        data = WorkingMemoryData(
            schema_version=1,
            coefficients=[_entry("lender", {"k_encode_bf16": 0.0, "b_encode_bf16": 0.0})],
            points=[*larger, *extra],
        )
        return dict(fitted_values([data], SPECS, {})[("lender", 1, (("conv_backend", "cudnn"),))])

    on_hull, above = (_point("lender", "encode", 256, mib) for mib in (450, 2000))
    readings = [(512 * 512, 1600 * MIB), (1024 * 1024, 4400 * MIB)]
    k, b = shipped_pair(readings)
    assert fitted(on_hull) == fitted() == {"k_encode_bf16": k, "b_encode_bf16": b}
    assert (k, b) != shipped_pair([(256 * 256, 450 * MIB), *readings]), "rated, it would move the line"
    raised = fitted(above)
    assert raised["k_encode_bf16"] * 256 * 256 + raised["b_encode_bf16"] >= 2000 * MIB

    def estimate(point: MeasuredPoint, reserved: int) -> WorkingMemoryEstimate:
        terms = (WorkingMemoryTerm.fitted("untiled", "k", 1, {"k": reserved}),)
        return WorkingMemoryEstimate(estimator="lender", version=1, features=point.features, terms=terms, source="-")

    assert bound_violation(on_hull, SPECS["lender"], estimate(on_hull, 100 * on_hull.measured_bytes)) is None
    assert bound_violation(on_hull, SPECS["lender"], estimate(on_hull, on_hull.measured_bytes - 1)) is not None
    capping = _point("lender", "encode", SMALLEST_APP_SIDE, 700)
    assert bound_violation(capping, SPECS["lender"], estimate(capping, 100 * capping.measured_bytes)) is not None


def test_a_tiled_line_on_an_untiled_line_no_entry_carries_is_refused() -> None:
    data = WorkingMemoryData(
        schema_version=1,
        coefficients=[_entry("borrower", {"k_tiled_encode_bf16": 0.0, "b_tiled_encode_bf16": 0.0})],
        points=TILED_POINTS,
    )
    with pytest.raises(ValueError, match="tiled_encode_bf16 sits on encode_bf16, not carried"):
        fitted_values([data], SPECS, {})


def _decodes(*readings: tuple[int, int, int]) -> list[tuple[int, int, Fraction]]:
    """Tiled decodes as the search reads them, from (Mi pixels, MiB measured, MiB the tile is priced at): the search
    rates the lines as they ship, slopes in whole bytes per pixel and bases in whole MiB."""
    return [(x * MIB, y * MIB, Fraction(o * MIB)) for x, y, o in readings]


def _shipped_ratios(points) -> list[Fraction]:
    """Each point's estimate under the lines `shipped_two_phase` ships for `points`, over its reading."""
    (k_t, b_t), (k_a, b_a) = shipped_two_phase(points)
    return [max(o + k_t * x + b_t, Fraction(k_a * x + b_a)) / y for x, y, o in points]


FLAT_THEN_STEEP = _decodes((1, 110, 100), (2, 110, 100), (4, 110, 100), (6, 200, 100), (8, 300, 100))
"""One tile's decodes: 10 MiB above the tile's price up to 4 Mi pixels, then the assembly, 200 and 300 MiB."""


def test_an_assembly_line_takes_the_points_the_tile_cannot_price_without_rising_under_the_others() -> None:
    """The tiled line alone has to rise to 300 MiB at 8 Mi pixels and ships 1.89x over 4 Mi pixels (25 bytes per
    pixel, from the origin). The assembly line through the readings at 6 and 8, 50 bytes per pixel from -100 MiB,
    prices both exactly and lies under the tile's price up to 4, where the tiled line is the flat 10 MiB the points
    there measured above it: before the margin every estimate is its reading, as shipped the worst 1.065x."""
    line, assembly = two_phase_envelope(FLAT_THEN_STEEP)

    assert (line, assembly) == ((0, 10 * MIB), (50, -100 * MIB))
    assert upper_envelope(FLAT_THEN_STEEP) == (25, 0), "the tiled line alone"
    assert all(max(o + line[0] * x + line[1], assembly[0] * x + assembly[1]) == y for x, y, o in FLAT_THEN_STEEP)
    assert max(_shipped_ratios(FLAT_THEN_STEEP)) == Fraction(213, 200)


def test_a_point_its_floor_covers_needs_no_assembly_line() -> None:
    """The tile's decodes measure 10 MiB above its price up to 4 Mi pixels, and 300 MiB at 8, which its floor reserves
    anyway, with the shipped margin: no assembly line is needed for it, and none passes through it."""
    flat = _decodes((1, 110, 100), (2, 110, 100), (4, 110, 100))
    x, y, offset = 8 * MIB, 300 * MIB, Fraction(100 * MIB)

    assert two_phase_envelope([*flat, (x, y, offset, offset, Fraction(0))])[1] != (0, 0)
    assert two_phase_envelope([*flat, (x, y, offset, offset, Fraction(315 * MIB))]) == ((0, 10 * MIB), (0, 0))


def test_a_linear_set_ships_no_assembly_line() -> None:
    """Two tiles whose decodes lie 10 bytes per pixel above their tile's price: the tiled line alone prices every point
    exactly. So does the assembly line through the small tile's readings, before the margin, with the large tile on the
    tiled line. As they ship it is worse: the margin scales the assembly line over the small tile's whole demand,
    and the tiled line over only the part above the tile."""
    points = _decodes((1, 110, 100), (2, 120, 100), (4, 140, 100), (8, 180, 100),
                      (2, 420, 400), (4, 440, 400), (8, 480, 400))  # fmt: skip

    assert two_phase_envelope(points) == ((10, 0), (0, 0))


def test_the_candidates_are_rated_as_they_ship() -> None:
    """Before the margin the assembly line through the readings at 2 and 8 Mi pixels would win: the tiled line under
    it covers the 4 and 20 MiB measured above the tile at 1 and 4, and its worst estimate is 110/104 against 112.5/104
    for the tiled line alone (6.25 bytes per pixel). As they ship, the assembly line, 9 bytes per pixel from 94 MiB,
    reserves 166 MiB for the 150 at 8 Mi pixels, 1.107x: the margin scales its whole demand, but the tiled line's (7
    bytes per pixel) only the 50 MiB above the tile, 156 MiB there, and its worst estimate is 114/104, 1.096x. So no
    assembly line ships."""
    points = _decodes((1, 104, 100), (2, 104, 100), (4, 120, 100), (8, 150, 100))

    assert two_phase_envelope(points) == ((Fraction(25, 4), 0), (0, 0))
    assert max(_shipped_ratios(points)) == Fraction(114, 104)


def test_an_assembly_line_no_point_needs_ties_with_none_and_ships_as_none() -> None:
    """The assembly line through the two probe readings, which lie below the tile's price and cap nothing, covers no
    capping point and leaves the tiled line as it is: it ships under the tile phase everywhere, so every estimate is
    the same as without it. A tie, and it goes to the zero assembly line."""
    capping = _decodes((1, 110, 100), (2, 120, 100), (4, 140, 100), (8, 180, 100))
    probes = _decodes((2, 50, 100), (4, 60, 100))

    assert two_phase_envelope(capping + probes, capping) == ((10, 0), (0, 0))


def test_only_the_small_tiles_large_points_take_the_assembly_line() -> None:
    """The small tile measures 10 MiB above its price up to 4 Mi pixels, then 200 and 300 MiB; the large tile 5 to 10
    MiB above its own, to 410 MiB at 8 Mi pixels. The assembly line through the small tile's readings at 6 and 8 prices
    them, as shipped (53 bytes per pixel from -105 MiB) still under every other point's tile phase, the large tile's
    at 8 Mi pixels too (319 against 411 MiB). The tiled line covers the 10 MiB the rest measured above their tile."""
    small = _decodes((1, 110, 100), (2, 110, 100), (4, 110, 100), (6, 200, 100), (8, 300, 100))
    large = _decodes((2, 405, 400), (4, 405, 400), (8, 410, 400))

    line, assembly = two_phase_envelope(small + large)

    assert (line, assembly) == ((0, 10 * MIB), (50, -100 * MIB))
    (k_t, b_t), (k_a, b_a) = shipped_two_phase(small + large)
    priced_by_assembly = [(x // MIB, o / MIB) for x, _, o in small + large if k_a * x + b_a > o + k_t * x + b_t]
    assert priced_by_assembly == [(6, 100), (8, 100)]


def test_a_tie_between_two_assembly_lines_goes_to_the_steeper_one() -> None:
    """Two assembly lines through the reading at 8 Mi pixels ship to the same 315 MiB there, 1.05 times its 300: 60
    bytes per pixel through a probe reading at 10, and 80 through one at 6. Each covers the other's probe and leaves the
    flat 10 MiB to the tiled line, and neither reaches the tile phase below 8, so every estimate ties. The steeper
    wins, the more conservative past the largest point."""
    capping = _decodes((1, 110, 100), (2, 110, 100), (4, 110, 100), (8, 300, 100))
    probes = _decodes((6, 140, 100), (10, 420, 100))

    line, assembly = two_phase_envelope(capping + probes, capping)

    assert (line, assembly) == ((0, 10 * MIB), (80, -340 * MIB))
    assert two_phase_envelope(list(reversed(capping + probes)), list(reversed(capping))) == (line, assembly)


def test_a_tie_between_two_parallel_assembly_lines_goes_to_the_higher_one() -> None:
    """Two parallel assembly lines, 50.5 bytes per pixel, through two probe readings each: from -103.5 MiB through 5
    and 6 Mi pixels, from -103 MiB through 2.5 and 3. Both cover the probe reading of 300 MiB at 8 Mi pixels and ship
    as the same line, 54 bytes per pixel from -108 MiB, which stays under the tile phase at every capping point, so
    every estimate ties; every other assembly line that ties is flatter. The higher one wins, the more conservative
    everywhere."""
    capping = _decodes((1, 110, 100), (2, 110, 100), (4, 110, 100))
    probes = [
        (int(Fraction(x) * MIB), int(Fraction(y) * MIB), Fraction(100 * MIB))
        for x, y in (("2.5", "23.25"), ("3", "48.5"), ("5", "149"), ("6", "199.5"), ("8", "300"))
    ]

    line, assembly = two_phase_envelope(capping + probes, capping)

    assert (line, assembly) == ((0, 10 * MIB), (Fraction(101, 2), -103 * MIB))


def test_a_probe_reading_above_the_assembly_line_is_covered_but_neither_rates_nor_shapes_the_tiled_line() -> None:
    """The probe reading of 4 MiB at 1 Mi pixels caps nothing. Left to the tiled line, it is covered by the line the
    capping readings choose (2.5 bytes per pixel from 10 MiB, 14 MiB there as shipped), and no assembly line ships.
    Rated, it would hand the decode to the steep assembly line through it (11 bytes per pixel from -7 MiB, 1.25x over
    it against 3.5x); rating the tiled line's edges, to the edge through the origin (7.5 bytes per pixel)."""
    capping = _decodes((2, 15, 0), (4, 20, 0))
    probe = _decodes((1, 4, 0))

    line, assembly = two_phase_envelope(capping + probe, capping)

    assert (line, assembly) == ((Fraction(5, 2), 10 * MIB), (0, 0))
    (k_t, b_t), _ = shipped_two_phase(capping + probe, capping)
    assert k_t * MIB + b_t >= 4 * MIB


def test_a_line_that_sits_on_a_line_that_sits_on_another_is_refused() -> None:
    """The fit runs in two steps, the lines that sit on none, then those that sit on one of them: a chain has no order
    in it."""

    def chained(features):
        size = features["height"] * features["width"]
        sits_on = {"middle": ("bottom", size), "top": ("middle", size)}.get(features["operation"])
        return Line(features["operation"], size, True, features["operation"], sits_on=sits_on)

    spec = EstimatorSpec(
        id="chain", version=1, coefficients=frozenset(), platform_keys=("conv_backend",), terms=_no_terms, line=chained
    )
    lines = {f"{part}_{line}": 0.0 for line in ("bottom", "middle", "top") for part in ("k", "b")}
    data = WorkingMemoryData(
        schema_version=1,
        coefficients=[_entry("chain", lines)],
        points=[_point("chain", operation, 1024, 4000) for operation in ("bottom", "middle", "top")],
    )

    with pytest.raises(ValueError, match=r"top sits on \['middle'\], which sit on others"):
        fit_points([data], {"chain": spec})


def test_without_the_fit_beneath_a_line_that_sits_on_another_is_a_problem_not_a_gap() -> None:
    """Fitted without the values of the line beneath, which `base_values` gives, a tiled line is named as a problem:
    leaving it out quietly would read as a fit."""
    data = WorkingMemoryData(
        schema_version=1,
        coefficients=[
            _entry("lender", {"k_encode_bf16": 0.0, "b_encode_bf16": 0.0}),
            _entry("borrower", {"k_tiled_encode_bf16": 0.0, "b_tiled_encode_bf16": 0.0}, lender="lender"),
        ],
        points=[*_untiled("encode"), *TILED_POINTS],
    )
    groups = fit_points([data], SPECS)
    borrower = data.coefficients[1]

    values, problems, conflicts = entry_fit(borrower, groups, None, {})

    assert values == {} and set(problems) == {"tiled_encode_bf16"} and not conflicts
    values, problems, conflicts = entry_fit(borrower, groups, base_values(data.coefficients, groups, {}), {})
    assert set(values) == {"k_tiled_encode_bf16", "b_tiled_encode_bf16"} and not problems and not conflicts


def test_the_margin_scales_the_whole_assembly_line_a_negative_intercept_included() -> None:
    """The assembly line of the flat-then-steep decodes, 50 bytes per pixel from -100 MiB, ships as 52.5 rounded up
    to 53 and -105 MiB, so it grows by at least 5% wherever it is above zero; the tiled line's 10 MiB as 11."""
    line, assembly = shipped_two_phase(FLAT_THEN_STEEP)

    assert line == (0, 11 * MIB)
    assert assembly == (53, -105 * MIB)
    for x in (3 * MIB, 6 * MIB, 8 * MIB):
        assert assembly[0] * x + assembly[1] >= (1 + SHIPPED_MARGIN) * (50 * x - 100 * MIB) > 0


def test_a_tiled_decode_line_and_its_assembly_line_are_fitted_together_on_the_tiled_decodes() -> None:
    """The decode counterpart of the encode above: the untiled line first, then the tiled decodes on what they measured
    above it at their tile, the tiled line and its assembly line from them together."""
    tiled = [
        _point("borrower", "decode", 1024, 1200, tile=512),
        _point("borrower", "decode", 2048, 1250, tile=512),
        _point("borrower", "decode", 4096, 1300, tile=512),
        _point("borrower", "decode", 8192, 6000, tile=512),
        _point("borrower", "decode", 2048, 4500, tile=1024),
    ]
    data = WorkingMemoryData(
        schema_version=1,
        coefficients=[
            _entry("lender", {"k_decode_bf16": 0.0, "b_decode_bf16": 0.0}),
            _entry(
                "borrower",
                {f"{part}_{line}_bf16": 0.0 for line in ("tiled_decode", "assembled_decode") for part in ("k", "b")},
                lender="lender",
            ),
        ],
        points=[*_untiled("decode"), *tiled],
    )

    fitted = fitted_values([data], SPECS, {})

    k, b = shipped_pair(UNTILED)
    measured = {1024: Fraction(4400 * MIB)}  # The borrower's untiled decode at the 1024px tile; none at 512px.

    def reading(point: MeasuredPoint) -> tuple[int, int, Fraction, Fraction]:
        tile = point.features["tile"]
        priced = Fraction(k * tile * tile + b)
        return (point.features["height"] ** 2, point.measured_bytes, measured.get(tile, priced), priced)

    (k_tiled, b_tiled), (k_assembled, b_assembled) = shipped_two_phase([reading(point) for point in tiled])
    assert k_assembled > 0, "the 8192px decode needs the assembly phase"
    assert fitted[("borrower", 1, (("conv_backend", "cudnn"),))] == {
        "k_tiled_decode_bf16": k_tiled,
        "b_tiled_decode_bf16": b_tiled,
        "k_assembled_decode_bf16": k_assembled,
        "b_assembled_decode_bf16": b_assembled,
    }


def test_a_tiled_decode_whose_entry_carries_no_assembly_line_is_refused() -> None:
    data = WorkingMemoryData(
        schema_version=1,
        coefficients=[
            _entry("lender", {"k_decode_bf16": 0.0, "b_decode_bf16": 0.0}),
            _entry("borrower", {"k_tiled_decode_bf16": 0.0, "b_tiled_decode_bf16": 0.0}, lender="lender"),
        ],
        points=[*_untiled("decode"), _point("borrower", "decode", 2048, 1250, tile=512)],
    )
    with pytest.raises(ValueError, match="tiled_decode_bf16 pairs with assembled_decode_bf16, not carried"):
        fitted_values([data], SPECS, {})


def test_a_line_without_enough_points_names_its_entry() -> None:
    data = WorkingMemoryData(
        schema_version=1,
        coefficients=[_entry("lender", {"k_decode_bf16": 0.0, "b_decode_bf16": 0.0})],
        points=[_point("lender", "decode", 512, 1100)],
    )
    with pytest.raises(ValueError, match=r"lender on \{'conv_backend': 'cudnn'\}, decode_bf16"):
        fitted_values([data], SPECS, {})


def _bounded_by(points: list[MeasuredPoint]) -> dict:
    """The lender's decode line held to 1.3 times each of `points`, untiled single-sample decodes."""
    return {
        (("lender", 1, (("conv_backend", "cudnn"),)), "decode_bf16"): [
            LineBound(
                Fraction(p.features["height"] * p.features["width"]),
                Fraction(1),
                Fraction(13, 10) * p.measured_bytes,
                f"fresh {p.features['height']}x{p.features['width']}",
            )
            for p in points
        ]
    }


def _probe(px: int, mib: int) -> MeasuredPoint:
    return MeasuredPoint.model_validate(_point("lender", "decode", px, mib).model_dump() | {"method": "probe_demand"})


@pytest.mark.parametrize(
    "points",
    [
        pytest.param([_probe(512, 500), _probe(1024, 2000)], id="probes-only"),
        # Together they span 4x, but the one fresh size pins nothing at 512px, where a cold run's base can lie far
        # above the warm window.
        pytest.param([_probe(512, 500), _point("lender", "decode", 1024, 3072)], id="one-fresh-size"),
    ],
)
def test_a_line_its_fresh_process_points_do_not_span_is_not_fitted(points) -> None:
    data = WorkingMemoryData(
        schema_version=1, coefficients=[_entry("lender", {"k_decode_bf16": 0.0, "b_decode_bf16": 0.0})], points=points
    )
    with pytest.raises(ValueError, match=r"decode_bf16: its capping points measure .* keep it a stand-in"):
        fitted_values([data], SPECS, {})


def test_a_probe_point_never_lowers_a_half_of_the_fresh_process_fit() -> None:
    """A probe at 2048px above the fresh fit of the points at 512px and 1024px: the edge from 512px to it would be
    steeper, but with a lower base than the fresh fit's 1004 MiB. The raise keeps the base and lifts the slope."""
    fresh = [_point("lender", "decode", 512, 1024), _point("lender", "decode", 1024, 1229)]
    data = WorkingMemoryData(
        schema_version=1,
        coefficients=[_entry("lender", {"k_decode_bf16": 0.0, "b_decode_bf16": 0.0})],
        points=[*fresh, _probe(2048, 2560)],
    )
    line = fitted_values([data], SPECS, _bounded_by(fresh))[("lender", 1, (("conv_backend", "cudnn"),))]

    fresh_k, fresh_b = shipped_pair([(512 * 512, 1024 * MIB), (1024 * 1024, 1229 * MIB)])
    assert fresh_b == 1004 * MIB
    assert line["k_decode_bf16"] > fresh_k
    assert line["b_decode_bf16"] == fresh_b
    assert line["k_decode_bf16"] * 2048 * 2048 + line["b_decode_bf16"] >= 2560 * MIB


def test_a_declared_stand_in_line_is_not_fitted() -> None:
    entry = _entry("lender", {"k_decode_bf16": 7200.0, "b_decode_bf16": 0.0}).model_copy(
        update={"stand_ins": ["k_decode_bf16", "b_decode_bf16"]}
    )
    data = WorkingMemoryData(schema_version=1, coefficients=[entry], points=[])
    assert fitted_values([data], SPECS, {}) == {}


@pytest.mark.parametrize(
    "platform_keys, token_base, exempt",
    [(("conv_backend",), False, set()), ((), False, {"k_token"}), ((), True, set())],
    ids=["keyed", "unkeyed", "unkeyed-carried-line"],
)
def test_only_an_estimator_keyed_by_no_platform_trait_carries_a_value_off_a_line(
    platform_keys, token_base, exempt
) -> None:
    """A keyed estimator's value off a line counts as unfitted; an unkeyed one's is hand-set. Lines are fitted either
    way, and a slope whose base ships with it is a carried line, hand-set nowhere."""
    values = {"k_decode_bf16": 1.0, "b_decode_bf16": 0.0, "k_token": 220000.0} | (
        {"b_token": 0.0} if token_base else {}
    )
    entry = _entry("estimator", values)
    spec = EstimatorSpec(
        id="estimator",
        version=1,
        coefficients=frozenset(entry.values or {}),
        platform_keys=platform_keys,
        terms=lambda features, values: (),
        line=convolution_line,
    )
    assert hand_set_values(entry, spec) == exempt


def test_a_probe_point_raises_a_line_but_does_not_choose_it() -> None:
    """The fresh points at 512x1024 and 1024x1024 px make the flatter edge win: 500 B/px past the first, a 1000 MiB
    base. A probe reading of 400 MiB at 512x512 px lies below that line; rating it would make the edge through the
    origin win, 2.04x over it instead of 3.1x."""
    values = {"k_decode_bf16": 0.0, "b_decode_bf16": 0.0}
    fresh = [_point("lender", "decode", 512, 1500, width=1024), _point("lender", "decode", 1024, 2000)]

    def fitted(*probes: MeasuredPoint) -> tuple[int, int]:
        data = WorkingMemoryData(schema_version=1, coefficients=[_entry("lender", values)], points=[*fresh, *probes])
        line = fitted_values([data], SPECS, _bounded_by(fresh))[("lender", 1, (("conv_backend", "cudnn"),))]
        return line["k_decode_bf16"], line["b_decode_bf16"]

    def probe(px: int, mib: int) -> MeasuredPoint:
        return MeasuredPoint.model_validate(
            _point("lender", "decode", px, mib).model_dump() | {"method": "probe_demand"}
        )

    shipped = shipped_pair([(512 * 1024, 1500 * MIB), (1024 * 1024, 2000 * MIB)])
    assert shipped[1] > 0, "the flatter edge, with a base"
    # Below the line, at a size the fresh points do not measure, it changes nothing.
    assert fitted(probe(512, 400)) == shipped
    # Above the line at 1536px it raises the line to cover it.
    raised = fitted(probe(1536, 3600))
    assert raised != shipped and raised[0] * 1536 * 1536 + raised[1] >= 1.05 * 3600 * MIB


DENOISE_PLATFORM = {"conv_backend": "cudnn", "allocator": "native"}
SLOPE, BASE = 225280, 100 * MIB
"""The line the Z-Image denoise forwards below are measured on: 220 KiB per attended token plus 100 MiB."""


def _forward(image_tokens: int, text_tokens: int = 512, **off_line: object) -> dict:
    """A Z-Image denoise forward's features as its estimator records them: the plain forward, unless `off_line` sets
    one of its analytic features."""
    plain = {"image_tokens": image_tokens, "text_tokens": text_tokens, "tokens": image_tokens + text_tokens,
             "dtype": "bf16", "control_blocks": 0, "has_attention_mask": False, "regional_bias_bytes": 0,
             "score_matrix_bytes": 0, "lora_count": 0, "dequant_transient_bytes": 0, "fp8_cast_transient_bytes": 0}  # fmt: skip
    return plain | off_line


def _measured_forward(features: dict, above_the_line: int = 0) -> MeasuredPoint:
    return MeasuredPoint(
        estimator=Z_IMAGE_DENOISE_SPEC.id,
        version=Z_IMAGE_DENOISE_SPEC.version,
        platform=DENOISE_PLATFORM,
        features=features,
        measured_bytes=SLOPE * features["tokens"] + BASE + above_the_line,
        method="demand_peak",
        device="NVIDIA GeForce RTX 4090",
        provenance="-",
    )


DENOISE_ENTRY = CoefficientEntry(
    estimator=Z_IMAGE_DENOISE_SPEC.id,
    version=Z_IMAGE_DENOISE_SPEC.version,
    platform=DENOISE_PLATFORM,
    values=dict.fromkeys(Z_IMAGE_DENOISE_SPEC.coefficients, 0.0),
    stand_ins=[f"{part}_denoise_{dtype}" for part in ("k", "b") for dtype in ("fp16", "fp32")],
    provenance="-",
)
"""An entry whose bf16 line the forwards below fit."""


def test_a_token_line_is_fitted_on_the_plain_forwards_and_bounds_every_forward() -> None:
    """Plain forwards of 1024 to 16384 image tokens and two caption lengths lie on one line in their image and caption
    tokens together. A ControlNet forward, one whose score matrix is materialized and one of a quantized model measure
    more, by what their own terms price; at 4096 image tokens, fitted on, each would raise the line. The fit is the
    plain forwards' line plus 5%, and under it every forward holds its bounds, the others with their terms on top."""
    hints = (6 + 4) * 4608 * 3840 * 2
    forwards = [
        *(_measured_forward(_forward(image)) for image in (1024, 4096, 16384)),
        _measured_forward(_forward(4096, text_tokens=96)),
        _measured_forward(_forward(4096, control_blocks=6, has_attention_mask=True), above_the_line=hints * 9 // 10),
        _measured_forward(_forward(4096, score_matrix_bytes=GIB), above_the_line=768 * MIB),
        _measured_forward(_forward(4096, dequant_transient_bytes=200 * MIB), above_the_line=200 * MIB),
    ]
    data = WorkingMemoryData(schema_version=1, coefficients=[DENOISE_ENTRY], points=forwards)

    [fitted] = fitted_values(
        [data],
        {Z_IMAGE_DENOISE_SPEC.id: Z_IMAGE_DENOISE_SPEC},
        line_bounds([data], {Z_IMAGE_DENOISE_SPEC.id: Z_IMAGE_DENOISE_SPEC}),
    ).values()

    assert fitted == {"k_denoise_bf16": 236544, "b_denoise_bf16": 105 * MIB}
    entry = DENOISE_ENTRY.model_copy(update={"values": (DENOISE_ENTRY.values or {}) | fitted})
    shipped = data.model_copy(update={"coefficients": [entry]})
    for forward in forwards:
        estimate = evaluate_point(forward, Z_IMAGE_DENOISE_SPEC, [shipped])
        assert bound_violation(forward, Z_IMAGE_DENOISE_SPEC, estimate) is None, forward.features


def test_any_analytic_term_takes_the_forwards_it_prices_out_of_the_token_line_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The line takes a forward only where every analytic term of the estimator comes to nothing, read off the terms
    themselves: a term added to the estimator takes the forwards it prices out of the fit, and their ids say so. A
    forward that passes an attention mask leaves it too, though no term prices the mask: it narrows the kernels SDPA
    can pick."""
    terms = denoise_working_memory._terms

    def with_a_new_term(features, values):
        return (*terms(features, values), WorkingMemoryTerm("new_transient", features.get("new_bytes", 0), True))

    monkeypatch.setattr(denoise_working_memory, "_terms", with_a_new_term)
    priced = _measured_forward(_forward(4096, new_bytes=GIB), above_the_line=GIB)
    masked = _measured_forward(_forward(4096, has_attention_mask=True), above_the_line=GIB)
    plain = [_measured_forward(_forward(image)) for image in (1024, 16384)]
    data = WorkingMemoryData(schema_version=1, coefficients=[DENOISE_ENTRY], points=[*plain, priced, masked])

    [line] = fit_points([data], {Z_IMAGE_DENOISE_SPEC.id: Z_IMAGE_DENOISE_SPEC}).values()

    assert [point.size for point in line] == [1024 + 512, 16384 + 512]
    assert Z_IMAGE_DENOISE_SPEC.line(priced.features) == Line(
        "denoise_bf16", 4608, False, "denoise-bf16-4096+512tok-new_transient=1073741824"
    )
    assert Z_IMAGE_DENOISE_SPEC.line(masked.features) == Line(
        "denoise_bf16", 4608, False, "denoise-bf16-4096+512tok-attention_mask"
    )


def _probe_forward(features: dict, above_the_line: int = 0) -> MeasuredPoint:
    return _measured_forward(features, above_the_line).model_copy(update={"method": "probe_demand"})


def test_a_token_line_its_fresh_forwards_do_not_span_is_not_fitted() -> None:
    """One fresh forward at 4096 image tokens and warm windows at 1024 and 16384: together they span 11x, but the one
    fresh size pins nothing at the others, where a cold first run can lie above the windows."""
    forwards = [_probe_forward(_forward(1024)), _measured_forward(_forward(4096)), _probe_forward(_forward(16384))]
    data = WorkingMemoryData(schema_version=1, coefficients=[DENOISE_ENTRY], points=forwards)

    with pytest.raises(ValueError, match=r"denoise_bf16: its capping points measure .* keep it a stand-in"):
        fitted_values(
            [data],
            {Z_IMAGE_DENOISE_SPEC.id: Z_IMAGE_DENOISE_SPEC},
            line_bounds([data], {Z_IMAGE_DENOISE_SPEC.id: Z_IMAGE_DENOISE_SPEC}),
        )


def test_a_probe_forward_never_lowers_a_half_of_the_fresh_token_fit() -> None:
    """A warm window at 16384 image tokens, 512 MiB above the fresh forwards' line (4242 MiB), takes no part in the
    fresh fit, 236544 bytes per token and a 105 MiB base, which reserves 3916.5 MiB at its 16896 tokens, short of the
    4454.1 MiB it needs with 5%. The raise starts from that fit and keeps both halves at least there. Averaged over the
    fresh sizes, whose middle is 3072 tokens, a slope reaching a size past it costs less than a base: so the base stays
    105 MiB and the slope rises to 269908 bytes per token, which keeps the fresh forwards' upper bounds in tokens."""
    fresh = [_measured_forward(_forward(image)) for image in (1024, 4096)]
    window = _probe_forward(_forward(16384), above_the_line=512 * MIB)
    data = WorkingMemoryData(schema_version=1, coefficients=[DENOISE_ENTRY], points=[*fresh, window])

    [fitted] = fitted_values(
        [data],
        {Z_IMAGE_DENOISE_SPEC.id: Z_IMAGE_DENOISE_SPEC},
        line_bounds([data], {Z_IMAGE_DENOISE_SPEC.id: Z_IMAGE_DENOISE_SPEC}),
    ).values()

    fresh_k, fresh_b = shipped_pair([(p.features["tokens"], p.measured_bytes) for p in fresh])
    assert (fresh_k, fresh_b) == (236544, 105 * MIB)
    assert fitted["b_denoise_bf16"] == fresh_b
    assert fitted["k_denoise_bf16"] == 269908 > fresh_k
    assert fitted["k_denoise_bf16"] * window.features["tokens"] + fitted["b_denoise_bf16"] >= window.measured_bytes


def _tiled_probe(operation: str, px: int, mib: int) -> MeasuredPoint:
    point = _point("borrower", operation, px, mib, tile=512)
    return point.model_copy(update={"method": "probe_demand"})


AT_512 = 1155
"""About what the untiled line `UNTILED` fits prices a 512px tile at, in MiB: the offset the tiled points below sit on,
since the borrower measured no untiled 512px point (`_above_the_tile`)."""


def _tiled_fit(operation: str, *points: MeasuredPoint) -> dict[str, int]:
    """The borrower's fitted tiled `operation` lines (with the assembly line for a decode) on `points`, over the
    lender's untiled line fitted on `UNTILED`."""
    tiled = ["tiled_encode"] if operation == "encode" else ["tiled_decode", "assembled_decode"]
    data = WorkingMemoryData(
        schema_version=1,
        coefficients=[
            _entry("lender", {f"k_{operation}_bf16": 0.0, f"b_{operation}_bf16": 0.0}),
            _entry("borrower", {f"{part}_{line}_bf16": 0.0 for line in tiled for part in ("k", "b")}, lender="lender"),
        ],
        points=[*_untiled(operation), *points],
    )
    return dict(fitted_values([data], SPECS, {})[("borrower", 1, (("conv_backend", "cudnn"),))])


def _above_the_tile(px: int, mib: int) -> tuple[int, int, Fraction, Fraction]:
    """A tiled point at a 512px tile as the fit reads it: above what the untiled line prices the tile at."""
    k, b = shipped_pair(UNTILED)
    priced = Fraction(k * 512 * 512 + b)
    return (px * px, mib * MIB, priced, priced)


def test_a_line_that_sits_on_another_its_fresh_points_do_not_span_is_not_fitted() -> None:
    """One fresh tiled encode at 2048px and a warm window at 4096px: together they span 4x, but the one fresh size pins
    nothing at the other, where a cold run can lie above the window."""
    points = [_point("borrower", "encode", 2048, AT_512 + 1000, tile=512), _tiled_probe("encode", 4096, AT_512 + 900)]

    with pytest.raises(ValueError, match=r"tiled_encode_bf16: its capping points measure .* keep it a stand-in"):
        _tiled_fit("encode", *points)


def test_a_probe_never_lowers_a_half_of_a_line_that_sits_on_another() -> None:
    """Fresh tiled encodes at 2048px and 4096px read 1000 and 1300 MiB above the tile. A warm window at 8192px, 2750 MiB
    above it, drops the 4096px vertex: rated on the fresh points, the edge from 2048px to the window wins, with a base
    below the fresh fit's. The shipped line keeps the larger of each half, read above the tile in both fits."""
    fresh = [(2048, AT_512 + 1000), (4096, AT_512 + 1300)]
    window = (8192, AT_512 + 2750)
    fitted = _tiled_fit(
        "encode",
        *(_point("borrower", "encode", px, mib, tile=512) for px, mib in fresh),
        _tiled_probe("encode", *window),
    )

    fresh_k, fresh_b = shipped_pair([_above_the_tile(*point) for point in fresh])
    all_k, all_b = shipped_pair(
        [_above_the_tile(*point) for point in (*fresh, window)], [_above_the_tile(*p) for p in fresh]
    )
    assert all_b < fresh_b, "the window alone would lower the base"
    assert (fitted["k_tiled_encode_bf16"], fitted["b_tiled_encode_bf16"]) == (max(all_k, fresh_k), fresh_b)


def test_a_probe_never_lowers_either_line_of_a_two_phase_pair() -> None:
    """Fresh tiled decodes from 1024px to 8192px, the largest needing the assembly phase. A warm window at 6144px reads
    1000 MiB above the tile: fitted with it, the tiled line's slope and the assembly line's intercept both drop below the
    fresh two-phase fit's. Each ships at the larger of the two fits, per line and coefficient."""
    fresh = [(1024, AT_512 + 45), (2048, AT_512 + 95), (4096, AT_512 + 145), (8192, AT_512 + 4845)]
    window = (6144, AT_512 + 1000)
    fitted = _tiled_fit(
        "decode",
        *(_point("borrower", "decode", px, mib, tile=512) for px, mib in fresh),
        _tiled_probe("decode", *window),
    )

    fresh_readings = [_above_the_tile(*point) for point in fresh]
    fresh_line, fresh_assembly = shipped_two_phase(fresh_readings)
    all_line, all_assembly = shipped_two_phase([*fresh_readings, _above_the_tile(*window)], fresh_readings)
    assert all_line[0] < fresh_line[0] and all_assembly[1] < fresh_assembly[1], "the window alone would lower both"
    assert (fitted["k_tiled_decode_bf16"], fitted["b_tiled_decode_bf16"]) == tuple(map(max, all_line, fresh_line))
    assert (fitted["k_assembled_decode_bf16"], fitted["b_assembled_decode_bf16"]) == tuple(
        map(max, all_assembly, fresh_assembly)
    )


def test_a_fresh_point_below_the_smallest_app_side_still_shapes_the_line() -> None:
    """A fresh 256px encode above the line the 512px and 1024px encodes take alone is a cold measurement: it caps
    nothing, but the line covers it as the hull through it gives, flatter than the capping points' own fit. Held to
    the capping points' fit instead, as if the small point were a warm window, the line would ship steeper."""
    capping = [(512, 1600), (1024, 4400)]
    small = (256, 1000)
    data = WorkingMemoryData(
        schema_version=1,
        coefficients=[_entry("lender", {"k_encode_bf16": 0.0, "b_encode_bf16": 0.0})],
        points=[_point("lender", "encode", px, mib) for px, mib in (*capping, small)],
    )

    fitted = fitted_values([data], SPECS, {})[("lender", 1, (("conv_backend", "cudnn"),))]

    readings = [(px * px, mib * MIB) for px, mib in capping]
    shaped = shipped_pair([*readings, (small[0] ** 2, small[1] * MIB)], readings)
    assert (fitted["k_encode_bf16"], fitted["b_encode_bf16"]) == shaped
    assert shipped_pair(readings)[0] > shaped[0], "the capping points alone fit a steeper line"


def test_a_token_window_no_line_covers_within_the_fresh_bounds_is_refused() -> None:
    """2 GiB above the line at 16384 image tokens, the window (5778 MiB) needs 6066.9 MiB at 16896 attended tokens.
    With the fresh fit's 105 MiB base that takes 369999 bytes per token, which puts the fresh 4096-token forward at
    1731 MiB, over 1.3x its 1090 MiB (1417 MiB): the bound, in tokens, conflicts with the window."""
    fresh = [_measured_forward(_forward(image)) for image in (1024, 4096)]
    window = _probe_forward(_forward(16384), above_the_line=2 * GIB)
    data = WorkingMemoryData(schema_version=1, coefficients=[DENOISE_ENTRY], points=[*fresh, window])
    specs = {Z_IMAGE_DENOISE_SPEC.id: Z_IMAGE_DENOISE_SPEC}

    with pytest.raises(LineConflict) as refused:
        fitted_values([data], specs, line_bounds([data], specs))

    message = str(refused.value)
    assert "denoise_bf16: no line at or above k=236544, b=105 MiB covers 5778 MiB at size 16896 by 5%" in message
    assert message.endswith(
        "-denoise-bf16-4096+512tok-NVIDIAGeForceRTX4090-demand_peak (measured 1090 MiB, at most 1417 MiB)"
    )


LENDER_DECODE = (("lender", 1, (("conv_backend", "cudnn"),)), "decode_bf16")
FRESH = [_point("lender", "decode", 512, 1000), _point("lender", "decode", 1024, 2500)]
"""Fresh-process decodes whose fit is 2000 bytes per pixel and a 500 MiB base, shipped as 2100 and 525 MiB."""


def _capped_at(ratio_512: float) -> dict:
    """Upper bounds of `FRESH` on the line: `ratio_512` times the 512px decode, 1.3 times the 1024px one."""
    return {
        LENDER_DECODE: [
            LineBound(Fraction(512 * 512), Fraction(1), Fraction(ratio_512) * 1000 * MIB, "fresh 512px"),
            LineBound(Fraction(1024 * 1024), Fraction(1), Fraction(13, 10) * 2500 * MIB, "fresh 1024px"),
        ]
    }


def _with_probe(mib: int) -> WorkingMemoryData:
    """`FRESH` and a probe decode of `mib` at 512x1024 px, between their sizes."""
    probe = MeasuredPoint.model_validate(
        _point("lender", "decode", 512, mib, width=1024).model_dump() | {"method": "probe_demand"}
    )
    values = {"k_decode_bf16": 0.0, "b_decode_bf16": 0.0}
    return WorkingMemoryData(schema_version=1, coefficients=[_entry("lender", values)], points=[*FRESH, probe])


def test_a_probe_no_hull_edge_covers_within_the_bounds_raises_the_line_between_them() -> None:
    """The fresh fit reserves 1575 MiB at 512x1024 px, short of 1785 MiB, the 1700 MiB probe plus 5%. The hull through
    the probe has two edges: 2800 bytes per pixel from 512px with a 300 MiB base (with the fresh base, 3465 MiB at
    1024px, over 1.3x its 2500 MiB) and 1600 from the probe with a 900 MiB base (1320 MiB at 512px, over 1.3x its
    1000 MiB). Between them, the slope floor with a 735 MiB base covers the probe exactly and keeps both bounds; of
    the lines that do, it reserves least averaged over 512px to 1024px."""
    assert shipped_pair([(512 * 512, 1000 * MIB), (1024 * 1024, 2500 * MIB)]) == (2100, 525 * MIB)

    line = fitted_values([_with_probe(1700)], SPECS, _capped_at(1.3))[LENDER_DECODE[0]]

    assert (line["k_decode_bf16"], line["b_decode_bf16"]) == (2100, 735 * MIB)


def test_a_probe_no_line_covers_within_the_bounds_is_refused_naming_the_conflict() -> None:
    """Held to 1.1x the 512px decode, 1100 MiB, a line at or above the fresh fit's 2100 bytes per pixel and 525 MiB
    cannot reach 1785 MiB at 512x1024 px: the probe and the 512px bound conflict, the 1024px bound takes no part."""
    with pytest.raises(LineConflict) as refused:
        fitted_values([_with_probe(1700)], SPECS, _capped_at(1.1))

    message = str(refused.value)
    assert "no line at or above k=2100, b=525 MiB covers 1700 MiB at size 524288 by 5%" in message
    assert message.endswith("within the upper bound of fresh 512px")


def test_a_line_that_covers_every_point_by_the_margin_is_kept_whatever_their_own_fit() -> None:
    """A stand-in of 4000 bytes per pixel and no base covers probes of 900 MiB at 512px and 3000 MiB at 1024px by more
    than 5%. Their own fit has a base of its own; taking it on top would raise the line at every size."""
    probes = [FitPoint(512 * 512, 900 * MIB, False), FitPoint(1024 * 1024, 3000 * MIB, False)]
    assert shipped_pair([(p.size, p.bytes) for p in probes])[1] > 0

    assert raised_pair((4000, 0), probes, []) == (4000, 0)


def test_a_line_is_bounded_by_every_point_priced_on_it_where_it_decides() -> None:
    """SD decodes on the cuDNN fp16 line SD3 lends: an upcast decode is bounded with its upcast's bytes taken off the
    ceiling, one whose score matrix decides at the shipped values as if the line did, and an out-of-fit one only by its
    own `max_ratio`. A probe and an out-of-fit point without one set no bound, nor does a fresh tiled decode, on the
    untiled line beneath or on its tiled line: the tiled line is fitted after the untiled one and not raised."""
    specs = import_estimator_modules()
    version = specs["vae.sd"].version
    platform = {"conv_backend": "cudnn", "allocator": "native"}
    names = [
        f"{part}_{op}_{dtype}" for part in "kb" for op in ("decode", "encode") for dtype in ("fp16", "bf16", "fp32")
    ]
    added = [
        f"{part}_{op}_{dtype}"
        for part in "kb"
        for op in ("tiled_decode", "tiled_encode", "assembled_decode")
        for dtype in ("fp16", "fp32")
    ]
    lender = CoefficientEntry(
        estimator="vae.sd3",
        version=specs["vae.sd3"].version,
        platform=platform,
        values=dict.fromkeys(names, 1000.0),
        provenance="-",
    )
    borrower = CoefficientEntry(
        estimator="vae.sd",
        version=version,
        platform=platform,
        values=dict.fromkeys(added, 0.0),
        values_from=ValuesFrom(estimator="vae.sd3", version=specs["vae.sd3"].version),
        provenance="-",
    )

    def decode(px: int, mib: int, **changes) -> MeasuredPoint:
        features = {"operation": "decode", "height": px, "width": px, "dtype": "fp16", "tile": None,
                    "vae_class": "AutoencoderKL", "upcast_weight_bytes": 0, "score_matrix_bytes": 0}  # fmt: skip
        features |= changes.pop("features", {})
        changes.setdefault("method", "demand_peak")
        return MeasuredPoint(
            estimator="vae.sd",
            version=version,
            platform=platform,
            features=features,
            measured_bytes=mib * MIB,
            device="-",
            provenance="-",
            **changes,
        )

    upcast = decode(1024, 4000, features={"upcast_weight_bytes": 160 * MIB})
    scores = decode(1024, 9000, features={"score_matrix_bytes": 8 * 2**30})
    out_of_fit = decode(512, 1200, in_fit=False, max_ratio=1.5)
    unbounded = [
        decode(768, 2000, method="probe_demand"),
        decode(768, 2000, in_fit=False),
        decode(2048, 3000, features={"tile": 512}),
    ]
    data = WorkingMemoryData(
        schema_version=1,
        coefficients=[lender, borrower],
        points=[upcast, scores, out_of_fit, *unbounded],
    )

    bounds = line_bounds([data], specs)

    assert set(bounds) == {(entry_key(lender), "decode_fp16")}
    by_point = {bound.point: bound[:3] for bound in bounds[(entry_key(lender), "decode_fp16")]}
    # Named by id, measured bytes and the most the estimate may reserve: two points can share an id.
    assert by_point == {
        f"{point_id(upcast, specs['vae.sd'])} (measured 4000 MiB, at most 5200 MiB)": (
            Fraction(1024**2),
            Fraction(1),
            Fraction(1.3) * 4000 * MIB - 160 * MIB,
        ),
        f"{point_id(scores, specs['vae.sd'])} (measured 9000 MiB, at most 11700 MiB)": (
            Fraction(1024**2),
            Fraction(1),
            Fraction(1.3) * 9000 * MIB,
        ),
        f"{point_id(out_of_fit, specs['vae.sd'])} (measured 1200 MiB, at most 1800 MiB)": (
            Fraction(512**2),
            Fraction(1),
            Fraction(1.5) * 1200 * MIB,
        ),
    }


def test_a_probe_the_raised_line_covers_leaves_it_where_it_is() -> None:
    """A 2200 MiB probe at 512x1536 px lies above the fresh fit's 2100 MiB there. Past the middle of the fresh sizes,
    the slope is the cheaper half to raise: 2380 bytes per pixel. A 4000 MiB probe at 2048x1024 px, which that line
    covers by 5%, leaves it there, though measured from its size, the base would now be the cheaper half."""
    values = {"k_decode_bf16": 0.0, "b_decode_bf16": 0.0}

    def probe(height: int, width: int, mib: int) -> MeasuredPoint:
        point = _point("lender", "decode", height, mib, width=width).model_dump() | {"method": "probe_demand"}
        return MeasuredPoint.model_validate(point)

    def raised(*probes: MeasuredPoint) -> tuple[float, float]:
        data = WorkingMemoryData(schema_version=1, coefficients=[_entry("lender", values)], points=[*FRESH, *probes])
        line = fitted_values([data], SPECS, _capped_at(1.3))[LENDER_DECODE[0]]
        return line["k_decode_bf16"], line["b_decode_bf16"]

    first = probe(512, 1536, 2200)
    assert raised(first) == (2380, 525 * MIB)
    assert raised(first, probe(2048, 1024, 4000)) == (2380, 525 * MIB)


def test_a_raise_every_line_on_the_probe_prices_alike_goes_to_the_steeper_one() -> None:
    """One probe size is the whole measured range, so every line through the probe plus 5% reserves the same there:
    the slope rises, 4200 bytes per pixel with no base rather than 1000 with 800 MiB, the more conservative past it."""
    assert raised_pair((1000, 0), [FitPoint(512 * 512, 1000 * MIB, False)], []) == (4200, 0)


def test_a_floor_that_already_breaks_a_bound_is_named_as_such() -> None:
    """A stand-in of 5000 bytes per pixel reserves 1250 MiB at 512px, over a bound of 1200 MiB there, and falls short
    of a 6000 MiB probe at 1024px. No covering takes part in the conflict: the floor alone breaks the bound."""
    probe = [FitPoint(1024 * 1024, 6000 * MIB, False)]
    bound = LineBound(Fraction(512 * 512), Fraction(1), Fraction(1200 * MIB), "fresh 512px (measured 923 MiB)")

    with pytest.raises(
        LineConflict,
        match=r"^the line's floor k=5000, b=0 MiB already exceeds the upper bound of fresh "
        r"512px \(measured 923 MiB\)$",
    ):
        raised_pair((5000, 0), probe, [bound])


def test_a_raise_without_the_bounds_of_its_fresh_process_points_is_refused() -> None:
    """Fresh-process points always bound their line (`bounds.line_bounds`); a raise given none of them would be free
    to break all of them."""
    with pytest.raises(ValueError, match="no upper bounds are given"):
        fitted_values([_with_probe(1700)], SPECS, {})


def test_a_raise_onto_a_bound_exactly_is_kept() -> None:
    """Fresh decodes of 250 MiB at 512px and 1000 MiB at 1024px fit 1000 bytes per pixel and no base, shipped as 1050.
    A 1238 MiB probe at 1024px needs 1299.9 MiB there, and both fresh decodes hold the line to 1.3x: 325 and 1300 MiB.
    1300 bytes per pixel with no base reserves exactly that at both sizes, so it ships, though no line keeps a whole
    byte per pixel and a MiB of room below them."""
    fresh = [_point("lender", "decode", 512, 250), _point("lender", "decode", 1024, 1000)]
    data = WorkingMemoryData(
        schema_version=1,
        coefficients=[_entry("lender", {"k_decode_bf16": 0.0, "b_decode_bf16": 0.0})],
        points=[*fresh, _probe(1024, 1238)],
    )

    line = fitted_values([data], SPECS, _bounded_by(fresh))[("lender", 1, (("conv_backend", "cudnn"),))]

    assert (line["k_decode_bf16"], line["b_decode_bf16"]) == (1300, 0)


ODD_SIZE = 1024 * 1536
"""1.5 MiB of pixels: at this size whole bytes per pixel step the reservation by 1.5 MiB, whole MiB of base by 1."""


def test_a_raise_whose_real_optimum_rounds_onto_a_bound_takes_the_rounded_line_next_to_it() -> None:
    """A probe at 1024x1536 px needs 6291980289 bytes with 5%, 4000 slope steps of 1.5 MiB and 0.5 MiB more. Averaged
    over the fresh sizes, slope is the cheaper half, so the lowest real line is 4000.33 bytes per pixel and no base;
    rounded up, 4001 reserves 1 MiB above the need, over the bound 0.6 MiB above it. 4000 bytes per pixel with a 1 MiB
    base reserves 0.5 MiB above the need, within the bound: the lowest rounded line."""
    fresh = [FitPoint(512 * 512, 100 * MIB, True), FitPoint(ODD_SIZE, 100 * MIB, True)]
    need = 1572864 * 4000 + 524289
    probe = FitPoint(ODD_SIZE, need * 20 // 21, False)
    assert Fraction(21, 20) * probe.bytes == need
    bound = LineBound(Fraction(ODD_SIZE), Fraction(1), Fraction(need + 629146), "fresh 1024x1536")

    assert raised_pair((2000, 0), [*fresh, probe], [bound]) == (4000, MIB)


def test_a_raise_only_a_line_between_whole_mib_could_make_is_refused() -> None:
    """At 1024px a whole byte per pixel is a whole MiB, so every rounded line reserves a whole number of MiB there. A
    probe that needs 1300.2 MiB under a bound of 1300.7 MiB leaves a slit of half a MiB: real lines fit it (1300.2 bytes
    per pixel, say), rounded lines do not."""
    fresh = [FitPoint(512 * 512, 100 * MIB, True), FitPoint(1024 * 1024, 100 * MIB, True)]
    probe = FitPoint(1024 * 1024, 20 * 64921834, False)
    need, ceiling = Fraction(21, 20) * probe.bytes, Fraction(13007, 10) * MIB
    assert 1300 * MIB < need < ceiling < 1301 * MIB
    bound = LineBound(Fraction(1024 * 1024), Fraction(1), ceiling, "fresh 1024px")

    with pytest.raises(
        LineConflict, match=r"covers 1238 MiB at size 1048576 by 5% within the upper bound of fresh 1024px$"
    ):
        raised_pair((1000, 0), [*fresh, probe], [bound])
