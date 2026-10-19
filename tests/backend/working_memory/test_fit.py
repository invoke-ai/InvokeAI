"""The shipped fit: an upper envelope of the points, with a non-negative base, plus the shipped margin."""

from fractions import Fraction

import pytest

from invokeai.backend.working_memory.bounds import bound_violation, evaluate_point
from invokeai.backend.working_memory.coefficients import CoefficientEntry, MeasuredPoint, ValuesFrom, WorkingMemoryData
from invokeai.backend.working_memory.estimate import WorkingMemoryEstimate, WorkingMemoryTerm
from invokeai.backend.working_memory.fit import (
    SHIPPED_MARGIN,
    SMALLEST_APP_SIDE,
    Line,
    base_values,
    convolution_line,
    entry_fit,
    fit_points,
    fitted_values,
    shipped_pair,
    shipped_two_phase,
    two_phase_envelope,
    upper_envelope,
)
from invokeai.backend.working_memory.spec import EstimatorSpec
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
    fitted = fitted_values([data], SPECS)[("lender", 1, (("conv_backend", "cudnn"),))]
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

    fitted = fitted_values([data], SPECS)

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
        return dict(fitted_values([data], SPECS)[("borrower", 1, (("conv_backend", "cudnn"),))])

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
        return dict(fitted_values([data], SPECS)[("lender", 1, (("conv_backend", "cudnn"),))])

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
        fitted_values([data], SPECS)


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

    values, problems = entry_fit(borrower, groups, None)

    assert values == {} and set(problems) == {"tiled_encode_bf16"}
    values, problems = entry_fit(borrower, groups, base_values(data.coefficients, groups))
    assert set(values) == {"k_tiled_encode_bf16", "b_tiled_encode_bf16"} and not problems


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

    fitted = fitted_values([data], SPECS)

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
        fitted_values([data], SPECS)


def test_a_line_without_enough_points_names_its_entry() -> None:
    data = WorkingMemoryData(
        schema_version=1,
        coefficients=[_entry("lender", {"k_decode_bf16": 0.0, "b_decode_bf16": 0.0})],
        points=[_point("lender", "decode", 512, 1100)],
    )
    with pytest.raises(ValueError, match=r"lender on \{'conv_backend': 'cudnn'\}, decode_bf16"):
        fitted_values([data], SPECS)


def test_a_declared_stand_in_line_is_not_fitted() -> None:
    entry = _entry("lender", {"k_decode_bf16": 7200.0, "b_decode_bf16": 0.0}).model_copy(
        update={"stand_ins": ["k_decode_bf16", "b_decode_bf16"]}
    )
    data = WorkingMemoryData(schema_version=1, coefficients=[entry], points=[])
    assert fitted_values([data], SPECS) == {}


def test_points_that_cap_nothing_are_covered_but_do_not_choose_the_line() -> None:
    """Rated on every point, the low reading at size 1 makes the edge through the origin win; rated on the capping
    points, the flatter edge does, and it still covers the low reading."""
    capping = [(2, 15), (4, 20)]
    points = [*capping, (1, 4)]

    assert upper_envelope(points) == (Fraction(15, 2), 0)
    k, b = upper_envelope(points, capping)
    assert (k, b) == (Fraction(5, 2), 10)
    assert all(k * x + b >= y for x, y in points)


def test_a_probe_point_raises_a_line_but_does_not_choose_it() -> None:
    """The fresh points at 512x1024 and 1024x1024 px make the flatter edge win: 500 B/px past the first, a 1000 MiB
    base. A probe reading of 400 MiB at 512x512 px lies below that line; rating it would make the edge through the
    origin win, 2.04x over it instead of 3.1x."""
    values = {"k_decode_bf16": 0.0, "b_decode_bf16": 0.0}
    fresh = [_point("lender", "decode", 512, 1500, width=1024), _point("lender", "decode", 1024, 2000)]

    def fitted(*probes: MeasuredPoint) -> tuple[int, int]:
        data = WorkingMemoryData(schema_version=1, coefficients=[_entry("lender", values)], points=[*fresh, *probes])
        line = fitted_values([data], SPECS)[("lender", 1, (("conv_backend", "cudnn"),))]
        return line["k_decode_bf16"], line["b_decode_bf16"]

    def probe(px: int, mib: int) -> MeasuredPoint:
        return MeasuredPoint.model_validate(
            _point("lender", "decode", px, mib).model_dump() | {"method": "probe_demand"}
        )

    shipped = shipped_pair([(512 * 1024, 1500 * MIB), (1024 * 1024, 2000 * MIB)])
    assert shipped[1] > 0, "the flatter edge, with a base"
    # Below the line, at a size the fresh points do not measure, it changes nothing.
    assert fitted(probe(512, 400)) == shipped
    # Above the fresh points it raises the line to cover it.
    raised = fitted(probe(1024, 5000))
    assert raised != shipped and raised[0] * 1024 * 1024 + raised[1] >= 5000 * MIB


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

    [fitted] = fitted_values([data], {Z_IMAGE_DENOISE_SPEC.id: Z_IMAGE_DENOISE_SPEC}).values()

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
