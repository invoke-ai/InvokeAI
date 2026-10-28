"""The shipped fit: an upper envelope of the points, with a non-negative base, plus the shipped margin."""

from fractions import Fraction

import pytest

from invokeai.backend.working_memory.coefficients import CoefficientEntry, MeasuredPoint, ValuesFrom, WorkingMemoryData
from invokeai.backend.working_memory.fit import (
    SHIPPED_MARGIN,
    fitted_values,
    hand_set_values,
    shipped_pair,
    upper_envelope,
)
from invokeai.backend.working_memory.spec import EstimatorSpec

MIB = 2**20

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


def test_a_falling_edge_is_never_the_line() -> None:
    """A line that falls with the size would price larger images below smaller ones."""
    assert upper_envelope([(1, 10), (4, 5)]) == (10, 0)


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
    fitted = fitted_values([data])[("lender", 1, (("conv_backend", "cudnn"),))]
    assert (fitted["k_decode_bf16"], fitted["b_decode_bf16"]) == shipped_pair(
        [(512 * 512, 1100 * MIB), (1024 * 1024, 4400 * MIB)]
    )


def test_a_line_without_enough_points_names_its_entry() -> None:
    data = WorkingMemoryData(
        schema_version=1,
        coefficients=[_entry("lender", {"k_decode_bf16": 0.0, "b_decode_bf16": 0.0})],
        points=[_point("lender", "decode", 512, 1100)],
    )
    with pytest.raises(ValueError, match=r"lender on \{'conv_backend': 'cudnn'\}, decode_bf16"):
        fitted_values([data])


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
    with pytest.raises(ValueError, match=r"decode_bf16: its fresh-process points measure .* keep it a stand-in"):
        fitted_values([data])


def test_a_probe_point_never_lowers_a_half_of_the_fresh_process_fit() -> None:
    """A probe at 2048px above the hull of the fresh points at 512px and 1024px drops the 1024px vertex: rated on the
    fresh points, the edge from 512px to the probe wins, steeper but with a lower base than the fresh fit's 1004 MiB.
    The shipped line keeps the larger of each half."""
    fresh = [_point("lender", "decode", 512, 1024), _point("lender", "decode", 1024, 1229)]
    data = WorkingMemoryData(
        schema_version=1,
        coefficients=[_entry("lender", {"k_decode_bf16": 0.0, "b_decode_bf16": 0.0})],
        points=[*fresh, _probe(2048, 2560)],
    )
    line = fitted_values([data])[("lender", 1, (("conv_backend", "cudnn"),))]

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
    assert fitted_values([data]) == {}


@pytest.mark.parametrize(
    "platform_keys, exempt", [(("conv_backend",), set()), ((), {"k_token"})], ids=["keyed", "unkeyed"]
)
def test_only_an_estimator_keyed_by_no_platform_trait_carries_a_value_off_a_pixel_line(platform_keys, exempt) -> None:
    """A keyed estimator's value off a line counts as unfitted; an unkeyed one's is hand-set. Lines are fitted either
    way."""
    entry = _entry("estimator", {"k_decode_bf16": 1.0, "b_decode_bf16": 0.0, "k_token": 220000.0})
    spec = EstimatorSpec(
        id="estimator",
        version=1,
        coefficients=frozenset(entry.values or {}),
        platform_keys=platform_keys,
        terms=lambda features, values: (),
    )
    assert hand_set_values(entry, spec) == exempt


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
        line = fitted_values([data])[("lender", 1, (("conv_backend", "cudnn"),))]
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
