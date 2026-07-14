"""Tests for the optional Krea-2 conditioning enhancers (rebalance + seed variance).

Both operate on the 4D ``prompt_embeds (B, seq, 12, hidden)`` conditioning between the text encoder and
denoise. The load-bearing logic - the per-layer gain broadcast, the exact-count weight validation, and the
seeded-noise determinism / out-of-place property - is exercised here with a stub conditioning context.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from invokeai.app.invocations.fields import Krea2ConditioningField
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

    @pytest.mark.parametrize("weights", ["1,2,3", "1,2,3,4,5,6,7,8,9,10,11,12,13"])
    def test_rejects_wrong_count(self, weights: str) -> None:
        invocation = Krea2ConditioningRebalanceInvocation.model_construct(per_layer_weights=weights)
        with pytest.raises(ValueError, match="exactly 12 values"):
            invocation._parse_weights()

    def test_rejects_non_numeric(self) -> None:
        invocation = Krea2ConditioningRebalanceInvocation.model_construct(per_layer_weights="a,b,c,d,e,f,g,h,i,j,k,l")
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


def test_seed_variance_is_deterministic_for_a_fixed_seed() -> None:
    embeds = torch.ones(1, 3, 12, 4)
    saved_a: dict = {}
    saved_b: dict = {}
    invocation = Krea2SeedVarianceInvocation.model_construct(
        conditioning=Krea2ConditioningField(conditioning_name="c"),
        strength=20.0,
        randomize_percent=50.0,
        variance_seed=42,
    )

    invocation.invoke(_make_context(embeds.clone(), saved_a))
    invocation.invoke(_make_context(embeds.clone(), saved_b))

    assert torch.equal(_saved_embeds(saved_a), _saved_embeds(saved_b))


def test_seed_variance_differs_across_seeds() -> None:
    embeds = torch.ones(1, 3, 12, 4)
    saved_a: dict = {}
    saved_b: dict = {}

    def _run(seed: int, saved: dict) -> None:
        Krea2SeedVarianceInvocation.model_construct(
            conditioning=Krea2ConditioningField(conditioning_name="c"),
            strength=20.0,
            randomize_percent=50.0,
            variance_seed=seed,
        ).invoke(_make_context(embeds.clone(), saved))

    _run(42, saved_a)
    _run(43, saved_b)

    assert not torch.equal(_saved_embeds(saved_a), _saved_embeds(saved_b))


def test_seed_variance_does_not_mutate_the_input_conditioning() -> None:
    embeds = torch.ones(1, 3, 12, 4)
    original = embeds.clone()
    saved: dict = {}
    invocation = Krea2SeedVarianceInvocation.model_construct(
        conditioning=Krea2ConditioningField(conditioning_name="c"),
        strength=20.0,
        randomize_percent=50.0,
        variance_seed=7,
    )

    invocation.invoke(_make_context(embeds, saved))

    # The invocation must produce a new tensor, not perturb the caller's embeds in place.
    assert torch.equal(embeds, original)
    assert not torch.equal(_saved_embeds(saved), original)
