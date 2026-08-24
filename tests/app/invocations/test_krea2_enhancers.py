"""Tests for the optional Krea-2 conditioning enhancers (rebalance + seed variance).

Both operate on the 4D ``prompt_embeds (B, seq, 12, hidden)`` conditioning between the text encoder and
denoise. The load-bearing logic - the per-layer gain broadcast, the exact-count weight validation, and the
seeded-noise determinism / out-of-place property - is exercised here with a stub conditioning context.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from invokeai.app.invocations.fields import Krea2ConditioningField, TensorField
from invokeai.app.invocations.krea2_conditioning_rebalance import Krea2ConditioningRebalanceInvocation
from invokeai.app.invocations.krea2_seed_variance import Krea2SeedVarianceInvocation
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, Krea2ConditioningInfo


def _make_context(embeds: torch.Tensor, saved: dict) -> SimpleNamespace:
    def load(_name: str) -> ConditioningFieldData:
        return ConditioningFieldData(
            conditionings=[Krea2ConditioningInfo(prompt_embeds=embeds, prompt_embeds_mask=None)]
        )

    def save(data: ConditioningFieldData) -> str:
        saved["data"] = data
        return "saved-name"

    return SimpleNamespace(conditioning=SimpleNamespace(load=load, save=save))


def _saved_embeds(saved: dict) -> torch.Tensor:
    conditioning = saved["data"].conditionings[0]
    assert isinstance(conditioning, Krea2ConditioningInfo)
    return conditioning.prompt_embeds


class TestRebalanceParseWeights:
    def test_accepts_exactly_twelve_values(self) -> None:
        invocation = Krea2ConditioningRebalanceInvocation.model_construct(
            per_layer_weights="1,2,3,4,5,6,7,8,9,10,11,12"
        )
        assert invocation._parse_weights() == [float(i) for i in range(1, 13)]

    def test_accepts_decimal_scientific_notation(self) -> None:
        invocation = Krea2ConditioningRebalanceInvocation.model_construct(
            per_layer_weights="1e2,-2.5e-1,3E+1,4,5,6,7,8,9,10,11,12"
        )
        assert invocation._parse_weights() == [100.0, -0.25, 30.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

    @pytest.mark.parametrize("weights", ["1,2,3", "1,2,3,4,5,6,7,8,9,10,11,12,13"])
    def test_rejects_wrong_count(self, weights: str) -> None:
        invocation = Krea2ConditioningRebalanceInvocation.model_construct(per_layer_weights=weights)
        with pytest.raises(ValueError, match="exactly 12 values"):
            invocation._parse_weights()

    def test_rejects_non_numeric(self) -> None:
        invocation = Krea2ConditioningRebalanceInvocation.model_construct(per_layer_weights="a,b,c,d,e,f,g,h,i,j,k,l")
        with pytest.raises(ValueError, match="comma-separated numbers"):
            invocation._parse_weights()

    @pytest.mark.parametrize("value", ["0x10", "0b10", "0o10"])
    def test_rejects_non_decimal_numeric_syntax(self, value: str) -> None:
        values = ["1"] * 11 + [value]
        invocation = Krea2ConditioningRebalanceInvocation.model_construct(per_layer_weights=",".join(values))
        with pytest.raises(ValueError, match="comma-separated numbers"):
            invocation._parse_weights()

    @pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
    def test_rejects_non_finite_weights(self, value: str) -> None:
        values = ["1"] * 11 + [value]
        invocation = Krea2ConditioningRebalanceInvocation.model_construct(per_layer_weights=",".join(values))
        with pytest.raises(ValueError, match="finite"):
            invocation._parse_weights()


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_rebalance_rejects_non_finite_multiplier(value: float) -> None:
    with pytest.raises(ValueError):
        Krea2ConditioningRebalanceInvocation(
            conditioning=Krea2ConditioningField(conditioning_name="c"),
            multiplier=value,
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_seed_variance_rejects_non_finite_strength(value: float) -> None:
    with pytest.raises(ValueError):
        Krea2SeedVarianceInvocation(
            conditioning=Krea2ConditioningField(conditioning_name="c"),
            strength=value,
        )


def test_rebalance_applies_per_layer_gains_on_the_layer_axis() -> None:
    # embeds is (B=1, seq=2, 12 layers, hidden=4); gains must apply along the layer axis (dim=2).
    embeds = torch.ones(1, 2, 12, 4)
    saved: dict = {}
    invocation = Krea2ConditioningRebalanceInvocation.model_construct(
        conditioning=Krea2ConditioningField(conditioning_name="c"),
        per_layer_weights="1,2,3,4,5,6,7,8,9,10,11,12",
        multiplier=1.0,
    )

    invocation.invoke(_make_context(embeds, saved))

    out = _saved_embeds(saved)
    assert out.shape == (1, 2, 12, 4)
    for layer_index in range(12):
        assert torch.allclose(out[:, :, layer_index, :], torch.full((1, 2, 4), float(layer_index + 1)))


def test_rebalance_applies_overall_multiplier() -> None:
    embeds = torch.ones(1, 1, 12, 2)
    saved: dict = {}
    invocation = Krea2ConditioningRebalanceInvocation.model_construct(
        conditioning=Krea2ConditioningField(conditioning_name="c"),
        per_layer_weights=",".join(["1.0"] * 12),
        multiplier=3.0,
    )

    invocation.invoke(_make_context(embeds, saved))

    assert torch.allclose(_saved_embeds(saved), torch.full((1, 1, 12, 2), 3.0))


def test_rebalance_preserves_the_regional_mask() -> None:
    regional_mask = TensorField(tensor_name="regional-mask")
    invocation = Krea2ConditioningRebalanceInvocation.model_construct(
        conditioning=Krea2ConditioningField(conditioning_name="c", mask=regional_mask),
        per_layer_weights=",".join(["1.0"] * 12),
        multiplier=1.0,
    )

    output = invocation.invoke(_make_context(torch.ones(1, 1, 12, 2), {}))

    assert output.conditioning.mask == regional_mask


# The noise magnitude is auto-calibrated to the embedding std, so tests must use a non-constant tensor
# (a constant tensor has std 0 and would produce no noise at all).
def _ramp_embeds() -> torch.Tensor:
    return torch.arange(1 * 3 * 12 * 4, dtype=torch.float32).reshape(1, 3, 12, 4)


def test_seed_variance_is_deterministic_for_a_fixed_seed() -> None:
    embeds = _ramp_embeds()
    saved_a: dict = {}
    saved_b: dict = {}
    invocation = Krea2SeedVarianceInvocation.model_construct(
        conditioning=Krea2ConditioningField(conditioning_name="c"),
        strength=0.5,
        randomize_percent=50.0,
        variance_seed=42,
    )

    invocation.invoke(_make_context(embeds.clone(), saved_a))
    invocation.invoke(_make_context(embeds.clone(), saved_b))

    assert torch.equal(_saved_embeds(saved_a), _saved_embeds(saved_b))


def test_seed_variance_differs_across_seeds() -> None:
    embeds = _ramp_embeds()
    saved_a: dict = {}
    saved_b: dict = {}

    def _run(seed: int, saved: dict) -> None:
        Krea2SeedVarianceInvocation.model_construct(
            conditioning=Krea2ConditioningField(conditioning_name="c"),
            strength=0.5,
            randomize_percent=50.0,
            variance_seed=seed,
        ).invoke(_make_context(embeds.clone(), saved))

    _run(42, saved_a)
    _run(43, saved_b)

    assert not torch.equal(_saved_embeds(saved_a), _saved_embeds(saved_b))


def test_seed_variance_does_not_mutate_the_input_conditioning() -> None:
    embeds = _ramp_embeds()
    original = embeds.clone()
    saved: dict = {}
    invocation = Krea2SeedVarianceInvocation.model_construct(
        conditioning=Krea2ConditioningField(conditioning_name="c"),
        strength=0.5,
        randomize_percent=50.0,
        variance_seed=7,
    )

    invocation.invoke(_make_context(embeds, saved))

    # The invocation must produce a new tensor, not perturb the caller's embeds in place.
    assert torch.equal(embeds, original)
    assert not torch.equal(_saved_embeds(saved), original)


def test_seed_variance_scales_noise_with_embedding_std() -> None:
    # Auto-calibration: doubling the embedding scale (hence its std) must double the injected noise, so a
    # fixed `strength` behaves the same relative to the signal regardless of the upstream embedding scale.
    base = _ramp_embeds()
    saved_small: dict = {}
    saved_large: dict = {}

    def _run(embeds: torch.Tensor, saved: dict) -> None:
        Krea2SeedVarianceInvocation.model_construct(
            conditioning=Krea2ConditioningField(conditioning_name="c"),
            strength=0.5,
            randomize_percent=100.0,
            variance_seed=1,
        ).invoke(_make_context(embeds, saved))

    _run(base.clone(), saved_small)
    _run(base.clone() * 2.0, saved_large)

    # Per-element noise = out - in. With randomize_percent=100 every element is perturbed, so the large-scale
    # run's noise should be ~2x the small-scale run's (same seed → same underlying uniform draw and mask).
    noise_small = _saved_embeds(saved_small) - base
    noise_large = _saved_embeds(saved_large) - base * 2.0
    assert torch.allclose(noise_large, noise_small * 2.0, atol=1e-4)


def test_seed_variance_is_a_noop_when_disabled() -> None:
    embeds = _ramp_embeds()
    regional_mask = TensorField(tensor_name="regional-mask")
    for strength, percent in ((0.0, 50.0), (0.5, 0.0)):
        saved: dict = {}
        out = Krea2SeedVarianceInvocation.model_construct(
            conditioning=Krea2ConditioningField(conditioning_name="c", mask=regional_mask),
            strength=strength,
            randomize_percent=percent,
            variance_seed=3,
        ).invoke(_make_context(embeds.clone(), saved))
        # Nothing saved, and the output points back at the untouched input conditioning.
        assert saved == {}
        assert out.conditioning.conditioning_name == "c"
        assert out.conditioning.mask == regional_mask


def test_seed_variance_preserves_the_regional_mask_when_enabled() -> None:
    regional_mask = TensorField(tensor_name="regional-mask")
    invocation = Krea2SeedVarianceInvocation.model_construct(
        conditioning=Krea2ConditioningField(conditioning_name="c", mask=regional_mask),
        strength=0.5,
        randomize_percent=50.0,
        variance_seed=3,
    )

    output = invocation.invoke(_make_context(_ramp_embeds(), {}))

    assert output.conditioning.mask == regional_mask
