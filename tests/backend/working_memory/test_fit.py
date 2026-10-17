"""The shipped fit: an upper envelope of the points, with a non-negative base, plus the shipped margin."""

from fractions import Fraction

import pytest

from invokeai.backend.working_memory.coefficients import CoefficientEntry, MeasuredPoint, ValuesFrom, WorkingMemoryData
from invokeai.backend.working_memory.fit import SHIPPED_MARGIN, fitted_values, shipped_pair, upper_envelope

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
            # None of these is the line's: another operation, a tile, a batch, a point out of regime.
            _point("borrower", "encode", 1024, 9000),
            _point("borrower", "decode", 2048, 1000, tile=512),
            _point("borrower", "decode", 1024, 9000, batch_size=2),
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


def test_a_declared_stand_in_line_is_not_fitted() -> None:
    entry = _entry("lender", {"k_decode_bf16": 7200.0, "b_decode_bf16": 0.0}).model_copy(
        update={"stand_ins": ["k_decode_bf16", "b_decode_bf16"]}
    )
    data = WorkingMemoryData(schema_version=1, coefficients=[entry], points=[])
    assert fitted_values([data]) == {}
