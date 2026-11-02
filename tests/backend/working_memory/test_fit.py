"""The shipped fit: an upper envelope of the points, with a non-negative base, plus the shipped margin, raised where
other points lie above it by the least the upper bounds allow."""

from fractions import Fraction

import pytest

from invokeai.backend.working_memory.bounds import bound_violation, evaluate_point, line_bounds, point_id
from invokeai.backend.working_memory.coefficients import CoefficientEntry, MeasuredPoint, ValuesFrom, WorkingMemoryData
from invokeai.backend.working_memory.estimate import WorkingMemoryTerm
from invokeai.backend.working_memory.fit import (
    SHIPPED_MARGIN,
    FitPoint,
    Line,
    LineBound,
    LineConflict,
    convolution_line,
    entry_key,
    fit_points,
    fitted_values,
    hand_set_values,
    raised_pair,
    shipped_pair,
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
    fitted = fitted_values([data], SPECS, {})[("lender", 1, (("conv_backend", "cudnn"),))]
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
    with pytest.raises(ValueError, match=r"decode_bf16: its fresh-process points measure .* keep it a stand-in"):
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

    with pytest.raises(ValueError, match=r"denoise_bf16: its fresh-process points measure .* keep it a stand-in"):
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
    ceiling, a tiled one at the tile's 1.25 times its pixels, one whose score matrix decides at the shipped values as
    if the line did, and an out-of-fit one only by its own `max_ratio`. A probe and an out-of-fit point without one set
    no bound."""
    platform = {"conv_backend": "cudnn", "allocator": "native"}
    names = [
        f"{part}_{op}_{dtype}" for part in "kb" for op in ("decode", "encode") for dtype in ("fp16", "bf16", "fp32")
    ]
    lender = CoefficientEntry(
        estimator="vae.sd3",
        version=2,
        platform=platform,
        values=dict.fromkeys(names, 1000.0),
        provenance="-",
    )
    borrower = CoefficientEntry(
        estimator="vae.sd",
        version=2,
        platform=platform,
        values_from=ValuesFrom(estimator="vae.sd3", version=2),
        provenance="-",
    )

    def decode(px: int, mib: int, **changes) -> MeasuredPoint:
        features = {"operation": "decode", "height": px, "width": px, "dtype": "fp16", "tile": None,
                    "vae_class": "AutoencoderKL", "upcast_weight_bytes": 0, "score_matrix_bytes": 0}  # fmt: skip
        features |= changes.pop("features", {})
        changes.setdefault("method", "demand_peak")
        return MeasuredPoint(
            estimator="vae.sd",
            version=2,
            platform=platform,
            features=features,
            measured_bytes=mib * MIB,
            device="-",
            provenance="-",
            **changes,
        )

    upcast = decode(1024, 4000, features={"upcast_weight_bytes": 160 * MIB})
    tiled = decode(2048, 3000, features={"tile": 512})
    scores = decode(1024, 9000, features={"score_matrix_bytes": 8 * 2**30})
    out_of_fit = decode(512, 1200, in_fit=False, max_ratio=1.5)
    unbounded = [decode(768, 2000, method="probe_demand"), decode(768, 2000, in_fit=False)]
    data = WorkingMemoryData(
        schema_version=1,
        coefficients=[lender, borrower],
        points=[upcast, tiled, scores, out_of_fit, *unbounded],
    )

    specs = import_estimator_modules()
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
        f"{point_id(tiled, specs['vae.sd'])} (measured 3000 MiB, at most 3900 MiB)": (
            Fraction(512**2 * 5, 4),
            Fraction(1),
            Fraction(1.3) * 3000 * MIB,
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
