"""Tests for ``WanLoRALoaderInvocation`` target resolution and routing."""

from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, WanTransformerField
from invokeai.app.invocations.wan_lora_loader import (
    WanLoRACollectionLoader,
    WanLoRALoaderInvocation,
    _resolve_target,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


# --------------------------------------------------------------------------
# _resolve_target — pure function, no mocks needed.
# --------------------------------------------------------------------------


class TestResolveTarget:
    @pytest.mark.parametrize(
        "target, expert, expected",
        [
            ("auto", None, (True, True)),
            ("auto", "high", (True, False)),
            ("auto", "low", (False, True)),
            ("both", None, (True, True)),
            ("both", "high", (True, True)),
            ("both", "low", (True, True)),
            ("high", None, (True, False)),
            ("high", "low", (True, False)),  # explicit target overrides config
            ("low", None, (False, True)),
            ("low", "high", (False, True)),
        ],
    )
    def test_target_resolution(self, target, expert, expected):
        assert _resolve_target(target, expert) == expected


# --------------------------------------------------------------------------
# WanLoRALoaderInvocation — routing into primary vs low-noise lists.
# --------------------------------------------------------------------------


def _make_lora_field(key: str = "lora-1") -> ModelIdentifierField:
    return ModelIdentifierField(
        key=key,
        hash=f"hash-{key}",
        name=f"name-{key}",
        base=BaseModelType.Wan,
        type=ModelType.LoRA,
    )


def _make_transformer_field() -> WanTransformerField:
    transformer_id = ModelIdentifierField(
        key="wan-main",
        hash="wan-main-hash",
        name="wan-main",
        base=BaseModelType.Wan,
        type=ModelType.Main,
    )
    return WanTransformerField(transformer=transformer_id)


def _make_context(lora_expert: str | None) -> MagicMock:
    """Mock context where context.models.get_config(lora).expert == lora_expert
    and context.models.exists returns True for any lora key."""
    ctx = MagicMock()
    ctx.models.exists.return_value = True
    config = MagicMock()
    config.expert = lora_expert
    ctx.models.get_config.return_value = config
    return ctx


class TestSingleLoaderRouting:
    def test_auto_untagged_goes_to_both(self):
        inv = WanLoRALoaderInvocation(
            id="inv-1", lora=_make_lora_field(), transformer=_make_transformer_field()
        )
        out = inv.invoke(_make_context(lora_expert=None))
        assert out.transformer is not None
        assert len(out.transformer.loras) == 1
        assert len(out.transformer.loras_low_noise) == 1

    def test_auto_high_tag_goes_to_primary_only(self):
        inv = WanLoRALoaderInvocation(
            id="inv-1", lora=_make_lora_field(), transformer=_make_transformer_field()
        )
        out = inv.invoke(_make_context(lora_expert="high"))
        assert out.transformer is not None
        assert len(out.transformer.loras) == 1
        assert len(out.transformer.loras_low_noise) == 0

    def test_auto_low_tag_goes_to_low_only(self):
        inv = WanLoRALoaderInvocation(
            id="inv-1", lora=_make_lora_field(), transformer=_make_transformer_field()
        )
        out = inv.invoke(_make_context(lora_expert="low"))
        assert out.transformer is not None
        assert len(out.transformer.loras) == 0
        assert len(out.transformer.loras_low_noise) == 1

    def test_explicit_target_overrides_tag(self):
        inv = WanLoRALoaderInvocation(
            id="inv-1",
            lora=_make_lora_field(),
            target="high",
            transformer=_make_transformer_field(),
        )
        out = inv.invoke(_make_context(lora_expert="low"))
        assert out.transformer is not None
        assert len(out.transformer.loras) == 1
        assert len(out.transformer.loras_low_noise) == 0

    def test_weight_propagates(self):
        inv = WanLoRALoaderInvocation(
            id="inv-1",
            lora=_make_lora_field(),
            weight=0.42,
            transformer=_make_transformer_field(),
        )
        out = inv.invoke(_make_context(lora_expert=None))
        assert out.transformer is not None
        assert out.transformer.loras[0].weight == pytest.approx(0.42)

    def test_unknown_lora_raises(self):
        ctx = _make_context(lora_expert=None)
        ctx.models.exists.return_value = False
        inv = WanLoRALoaderInvocation(
            id="inv-1", lora=_make_lora_field(), transformer=_make_transformer_field()
        )
        with pytest.raises(ValueError, match="Unknown lora"):
            inv.invoke(ctx)

    def test_duplicate_on_primary_raises(self):
        existing = LoRAField(lora=_make_lora_field(key="dup"), weight=0.5)
        transformer = _make_transformer_field()
        transformer.loras.append(existing)

        inv = WanLoRALoaderInvocation(
            id="inv-1", lora=_make_lora_field(key="dup"), transformer=transformer
        )
        with pytest.raises(ValueError, match="already applied to primary"):
            inv.invoke(_make_context(lora_expert="high"))

    def test_duplicate_on_low_noise_raises(self):
        existing = LoRAField(lora=_make_lora_field(key="dup"), weight=0.5)
        transformer = _make_transformer_field()
        transformer.loras_low_noise.append(existing)

        inv = WanLoRALoaderInvocation(
            id="inv-1", lora=_make_lora_field(key="dup"), transformer=transformer
        )
        with pytest.raises(ValueError, match="already applied to low-noise"):
            inv.invoke(_make_context(lora_expert="low"))

    def test_no_transformer_returns_empty_output(self):
        inv = WanLoRALoaderInvocation(id="inv-1", lora=_make_lora_field(), transformer=None)
        out = inv.invoke(_make_context(lora_expert=None))
        assert out.transformer is None


# --------------------------------------------------------------------------
# Collection loader — list routing + base validation.
# --------------------------------------------------------------------------


class TestCollectionLoaderRouting:
    def test_routes_mixed_tagged_loras(self):
        """A collection of three LoRAs (high, low, untagged) routes correctly."""
        high_lora = LoRAField(lora=_make_lora_field(key="lora-high"), weight=0.5)
        low_lora = LoRAField(lora=_make_lora_field(key="lora-low"), weight=0.6)
        untagged_lora = LoRAField(lora=_make_lora_field(key="lora-any"), weight=0.7)

        inv = WanLoRACollectionLoader(
            id="inv-1",
            loras=[high_lora, low_lora, untagged_lora],
            transformer=_make_transformer_field(),
        )

        # The collection loader queries each LoRA's config separately. Mock
        # get_config to return different expert tags by lora key.
        expert_by_key = {"lora-high": "high", "lora-low": "low", "lora-any": None}
        ctx = MagicMock()
        ctx.models.exists.return_value = True

        def get_config(field: ModelIdentifierField) -> MagicMock:
            config = MagicMock()
            config.expert = expert_by_key[field.key]
            return config

        ctx.models.get_config.side_effect = get_config
        out = inv.invoke(ctx)
        assert out.transformer is not None

        primary_keys = {item.lora.key for item in out.transformer.loras}
        low_keys = {item.lora.key for item in out.transformer.loras_low_noise}
        # high -> primary only; low -> low only; untagged -> both
        assert primary_keys == {"lora-high", "lora-any"}
        assert low_keys == {"lora-low", "lora-any"}

    def test_rejects_non_wan_base(self):
        wrong_base_lora = LoRAField(
            lora=ModelIdentifierField(
                key="not-wan", hash="h", name="n", base=BaseModelType.Flux, type=ModelType.LoRA
            ),
            weight=0.5,
        )
        inv = WanLoRACollectionLoader(
            id="inv-1", loras=[wrong_base_lora], transformer=_make_transformer_field()
        )
        ctx = MagicMock()
        ctx.models.exists.return_value = True
        with pytest.raises(ValueError, match="not Wan 2.2"):
            inv.invoke(ctx)

    def test_skips_duplicates(self):
        dup_lora = LoRAField(lora=_make_lora_field(key="dup"), weight=0.5)
        inv = WanLoRACollectionLoader(
            id="inv-1",
            loras=[dup_lora, dup_lora],
            transformer=_make_transformer_field(),
        )
        ctx = MagicMock()
        ctx.models.exists.return_value = True
        config = MagicMock()
        config.expert = None
        ctx.models.get_config.return_value = config

        out = inv.invoke(ctx)
        assert out.transformer is not None
        assert len(out.transformer.loras) == 1

    def test_no_loras_returns_clean_copy(self):
        inv = WanLoRACollectionLoader(
            id="inv-1", loras=None, transformer=_make_transformer_field()
        )
        out = inv.invoke(MagicMock())
        assert out.transformer is not None
        assert len(out.transformer.loras) == 0
        assert len(out.transformer.loras_low_noise) == 0
