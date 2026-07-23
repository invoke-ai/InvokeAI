"""Tests for ``WanLoRALoaderInvocation`` target resolution and routing."""

from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, WanTransformerField
from invokeai.app.invocations.wan_lora_loader import (
    WanLoRACollectionLoader,
    WanLoRALoaderInvocation,
    _resolve_target,
)
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelType,
    WanLoRAVariantType,
    WanVariantType,
)

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


def _make_lora_config(
    expert: str | None = None,
    variant=None,
    base: BaseModelType = BaseModelType.Wan,
    model_type: ModelType = ModelType.LoRA,
) -> MagicMock:
    """A resolved LoRA config: the loaders validate type/base on the *config*, not the
    client-supplied identifier, so mocks must carry real taxonomy values."""
    config = MagicMock()
    config.expert = expert
    config.variant = variant
    config.base = base
    config.type = model_type
    return config


def _make_main_config(variant=None) -> MagicMock:
    config = MagicMock()
    config.variant = variant
    config.base = BaseModelType.Wan
    config.type = ModelType.Main
    return config


def _make_context(
    lora_expert: str | None,
    lora_config: MagicMock | None = None,
    main_variant=None,
) -> MagicMock:
    """Mock context resolving LoRA keys to a Wan LoRA config (with the given expert
    tag) and the 'wan-main' transformer key to a Wan main config."""
    ctx = MagicMock()
    ctx.models.exists.return_value = True
    resolved_lora = lora_config if lora_config is not None else _make_lora_config(expert=lora_expert)
    main_config = _make_main_config(variant=main_variant)

    def _get_config(model_id: ModelIdentifierField) -> MagicMock:
        return main_config if model_id.key == "wan-main" else resolved_lora

    ctx.models.get_config.side_effect = _get_config
    return ctx


class TestSingleLoaderRouting:
    def test_auto_untagged_goes_to_both(self):
        inv = WanLoRALoaderInvocation(id="inv-1", lora=_make_lora_field(), transformer=_make_transformer_field())
        out = inv.invoke(_make_context(lora_expert=None))
        assert out.transformer is not None
        assert len(out.transformer.loras) == 1
        assert len(out.transformer.loras_low_noise) == 1

    def test_auto_high_tag_goes_to_primary_only(self):
        inv = WanLoRALoaderInvocation(id="inv-1", lora=_make_lora_field(), transformer=_make_transformer_field())
        out = inv.invoke(_make_context(lora_expert="high"))
        assert out.transformer is not None
        assert len(out.transformer.loras) == 1
        assert len(out.transformer.loras_low_noise) == 0

    def test_auto_low_tag_goes_to_low_only(self):
        inv = WanLoRALoaderInvocation(id="inv-1", lora=_make_lora_field(), transformer=_make_transformer_field())
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
        inv = WanLoRALoaderInvocation(id="inv-1", lora=_make_lora_field(), transformer=_make_transformer_field())
        with pytest.raises(ValueError, match="Unknown lora"):
            inv.invoke(ctx)

    def test_duplicate_on_primary_raises(self):
        existing = LoRAField(lora=_make_lora_field(key="dup"), weight=0.5)
        transformer = _make_transformer_field()
        transformer.loras.append(existing)

        inv = WanLoRALoaderInvocation(id="inv-1", lora=_make_lora_field(key="dup"), transformer=transformer)
        with pytest.raises(ValueError, match="already applied to primary"):
            inv.invoke(_make_context(lora_expert="high"))

    def test_duplicate_on_low_noise_raises(self):
        existing = LoRAField(lora=_make_lora_field(key="dup"), weight=0.5)
        transformer = _make_transformer_field()
        transformer.loras_low_noise.append(existing)

        inv = WanLoRALoaderInvocation(id="inv-1", lora=_make_lora_field(key="dup"), transformer=transformer)
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
            if field.key == "wan-main":
                return _make_main_config()
            return _make_lora_config(expert=expert_by_key[field.key])

        ctx.models.get_config.side_effect = get_config
        out = inv.invoke(ctx)
        assert out.transformer is not None

        primary_keys = {item.lora.key for item in out.transformer.loras}
        low_keys = {item.lora.key for item in out.transformer.loras_low_noise}
        # high -> primary only; low -> low only; untagged -> both
        assert primary_keys == {"lora-high", "lora-any"}
        assert low_keys == {"lora-low", "lora-any"}

    def test_rejects_non_wan_base(self):
        """The identifier claims Wan, but the key resolves to a Flux LoRA — the
        resolved config is authoritative."""
        mislabeled_lora = LoRAField(lora=_make_lora_field(key="not-wan"), weight=0.5)
        inv = WanLoRACollectionLoader(id="inv-1", loras=[mislabeled_lora], transformer=_make_transformer_field())
        ctx = _make_context(lora_expert=None, lora_config=_make_lora_config(base=BaseModelType.Flux))
        with pytest.raises(ValueError, match="not a Wan LoRA"):
            inv.invoke(ctx)

    def test_skips_duplicates(self):
        dup_lora = LoRAField(lora=_make_lora_field(key="dup"), weight=0.5)
        inv = WanLoRACollectionLoader(
            id="inv-1",
            loras=[dup_lora, dup_lora],
            transformer=_make_transformer_field(),
        )
        out = inv.invoke(_make_context(lora_expert=None))
        assert out.transformer is not None
        assert len(out.transformer.loras) == 1

    def test_no_loras_returns_clean_copy(self):
        inv = WanLoRACollectionLoader(id="inv-1", loras=None, transformer=_make_transformer_field())
        out = inv.invoke(MagicMock())
        assert out.transformer is not None
        assert len(out.transformer.loras) == 0
        assert len(out.transformer.loras_low_noise) == 0


# --------------------------------------------------------------------------
# LoRA-variant vs transformer-variant validation (A14B vs 5B).
# --------------------------------------------------------------------------


def _make_variant_context(lora_variant, main_variant) -> MagicMock:
    """Context whose get_config returns a LoRA config (with variant + expert) for LoRA
    keys and a main config (with variant) for the transformer key."""
    return _make_context(
        lora_expert=None,
        lora_config=_make_lora_config(variant=lora_variant),
        main_variant=main_variant,
    )


class TestVariantValidation:
    """An A14B LoRA against a 5B main (or vice versa) crashes deep in the layer patcher
    with an opaque shape error — the loaders must reject the mismatch up front."""

    @pytest.mark.parametrize(
        "lora_variant, main_variant",
        [
            (WanLoRAVariantType.A14B, WanVariantType.TI2V_5B),
            (WanLoRAVariantType.Wan5B, WanVariantType.T2V_A14B),
            (WanLoRAVariantType.Wan5B, WanVariantType.I2V_A14B),
        ],
    )
    def test_single_loader_rejects_mismatch(self, lora_variant, main_variant):
        inv = WanLoRALoaderInvocation(id="inv-1", lora=_make_lora_field(), transformer=_make_transformer_field())
        with pytest.raises(ValueError, match="not interchangeable"):
            inv.invoke(_make_variant_context(lora_variant, main_variant))

    @pytest.mark.parametrize(
        "lora_variant, main_variant",
        [
            (WanLoRAVariantType.A14B, WanVariantType.T2V_A14B),
            (WanLoRAVariantType.A14B, WanVariantType.I2V_A14B),
            (WanLoRAVariantType.Wan5B, WanVariantType.TI2V_5B),
            (None, WanVariantType.T2V_A14B),  # unrecorded LoRA variant -> skip check
            (WanLoRAVariantType.A14B, None),  # unrecorded main variant -> skip check
        ],
    )
    def test_single_loader_accepts_match_or_unknown(self, lora_variant, main_variant):
        inv = WanLoRALoaderInvocation(id="inv-1", lora=_make_lora_field(), transformer=_make_transformer_field())
        out = inv.invoke(_make_variant_context(lora_variant, main_variant))
        assert out.transformer is not None
        assert len(out.transformer.loras) == 1

    def test_collection_loader_rejects_mismatch(self):
        lora = LoRAField(lora=_make_lora_field(), weight=1.0)
        inv = WanLoRACollectionLoader(id="inv-1", loras=[lora], transformer=_make_transformer_field())
        with pytest.raises(ValueError, match="not interchangeable"):
            inv.invoke(_make_variant_context(WanLoRAVariantType.Wan5B, WanVariantType.T2V_A14B))

    def test_collection_loader_accepts_match(self):
        lora = LoRAField(lora=_make_lora_field(), weight=1.0)
        inv = WanLoRACollectionLoader(id="inv-1", loras=[lora], transformer=_make_transformer_field())
        out = inv.invoke(_make_variant_context(WanLoRAVariantType.A14B, WanVariantType.T2V_A14B))
        assert out.transformer is not None
        assert len(out.transformer.loras) == 1


# --------------------------------------------------------------------------
# Resolved-config validation — the identifier's base/type are client-supplied
# and untrusted; only the config the key resolves to counts (JPPhoto review
# 2026-07-21).
# --------------------------------------------------------------------------


class TestResolvedConfigValidation:
    @pytest.mark.parametrize(
        "bad_config",
        [
            _make_lora_config(base=BaseModelType.Flux),  # Flux LoRA labeled as Wan
            _make_lora_config(base=BaseModelType.StableDiffusionXL),  # SDXL LoRA labeled as Wan
            _make_lora_config(model_type=ModelType.Main),  # a main-model key labeled as a LoRA
        ],
        ids=["flux-lora", "sdxl-lora", "main-model"],
    )
    def test_single_loader_rejects_mislabeled_key(self, bad_config):
        inv = WanLoRALoaderInvocation(id="inv-1", lora=_make_lora_field(), transformer=_make_transformer_field())
        with pytest.raises(ValueError, match="not a Wan LoRA"):
            inv.invoke(_make_context(lora_expert=None, lora_config=bad_config))

    def test_single_loader_rejects_mislabeled_key_even_without_transformer(self):
        """The check must not depend on a transformer being wired."""
        inv = WanLoRALoaderInvocation(id="inv-1", lora=_make_lora_field(), transformer=None)
        with pytest.raises(ValueError, match="not a Wan LoRA"):
            inv.invoke(_make_context(lora_expert=None, lora_config=_make_lora_config(model_type=ModelType.Main)))

    def test_collection_loader_rejects_mislabeled_key(self):
        lora = LoRAField(lora=_make_lora_field(), weight=1.0)
        inv = WanLoRACollectionLoader(id="inv-1", loras=[lora], transformer=_make_transformer_field())
        with pytest.raises(ValueError, match="not a Wan LoRA"):
            inv.invoke(_make_context(lora_expert=None, lora_config=_make_lora_config(model_type=ModelType.Main)))

    def test_real_wan_lora_still_accepted(self):
        inv = WanLoRALoaderInvocation(id="inv-1", lora=_make_lora_field(), transformer=_make_transformer_field())
        out = inv.invoke(_make_context(lora_expert=None))
        assert out.transformer is not None
        assert len(out.transformer.loras) == 1


# --------------------------------------------------------------------------
# Collection loader vs already-applied LoRAs — chaining loaders must not
# silently double a LoRA's effective weight (JPPhoto review 2026-07-21).
# --------------------------------------------------------------------------


class TestCollectionLoaderUpstreamDuplicates:
    def test_rejects_lora_already_on_primary_list(self):
        transformer = _make_transformer_field()
        transformer.loras.append(LoRAField(lora=_make_lora_field(key="dup"), weight=0.5))

        inv = WanLoRACollectionLoader(
            id="inv-1",
            loras=[LoRAField(lora=_make_lora_field(key="dup"), weight=0.5)],
            transformer=transformer,
        )
        with pytest.raises(ValueError, match="already applied to primary"):
            inv.invoke(_make_context(lora_expert=None))

    def test_rejects_lora_already_on_low_noise_list(self):
        transformer = _make_transformer_field()
        transformer.loras_low_noise.append(LoRAField(lora=_make_lora_field(key="dup"), weight=0.5))

        inv = WanLoRACollectionLoader(
            id="inv-1",
            loras=[LoRAField(lora=_make_lora_field(key="dup"), weight=0.5)],
            transformer=transformer,
        )
        with pytest.raises(ValueError, match="already applied to low-noise"):
            inv.invoke(_make_context(lora_expert="low"))

    def test_new_key_appends_alongside_existing(self):
        transformer = _make_transformer_field()
        transformer.loras.append(LoRAField(lora=_make_lora_field(key="existing"), weight=0.5))

        inv = WanLoRACollectionLoader(
            id="inv-1",
            loras=[LoRAField(lora=_make_lora_field(key="new"), weight=0.5)],
            transformer=transformer,
        )
        out = inv.invoke(_make_context(lora_expert="high"))
        assert out.transformer is not None
        assert [item.lora.key for item in out.transformer.loras] == ["existing", "new"]


# --------------------------------------------------------------------------
# TI2V-5B inert low routing — the single-transformer path never consumes the
# low-noise list, so low-only routing must warn (JPPhoto review 2026-07-21).
# --------------------------------------------------------------------------


class TestInertLowRoutingWarning:
    def test_single_loader_warns_for_low_only_routing_on_ti2v(self):
        inv = WanLoRALoaderInvocation(
            id="inv-1", lora=_make_lora_field(), target="low", transformer=_make_transformer_field()
        )
        ctx = _make_context(
            lora_expert=None,
            lora_config=_make_lora_config(variant=WanLoRAVariantType.Wan5B),
            main_variant=WanVariantType.TI2V_5B,
        )
        out = inv.invoke(ctx)
        assert out.transformer is not None
        ctx.logger.warning.assert_called_once()
        assert "no effect" in ctx.logger.warning.call_args.args[0]

    def test_collection_loader_warns_for_low_tagged_lora_on_ti2v(self):
        lora = LoRAField(lora=_make_lora_field(), weight=1.0)
        inv = WanLoRACollectionLoader(id="inv-1", loras=[lora], transformer=_make_transformer_field())
        ctx = _make_context(
            lora_expert="low",
            lora_config=_make_lora_config(expert="low", variant=WanLoRAVariantType.Wan5B),
            main_variant=WanVariantType.TI2V_5B,
        )
        out = inv.invoke(ctx)
        assert out.transformer is not None
        ctx.logger.warning.assert_called_once()

    @pytest.mark.parametrize("target", ["auto", "both", "high"])
    def test_no_warning_when_primary_list_is_reached(self, target):
        """Routing that touches the primary list is consumed on TI2V — no warning."""
        inv = WanLoRALoaderInvocation(
            id="inv-1", lora=_make_lora_field(), target=target, transformer=_make_transformer_field()
        )
        ctx = _make_context(
            lora_expert=None,
            lora_config=_make_lora_config(variant=WanLoRAVariantType.Wan5B),
            main_variant=WanVariantType.TI2V_5B,
        )
        inv.invoke(ctx)
        ctx.logger.warning.assert_not_called()

    def test_no_warning_for_low_routing_on_a14b(self):
        inv = WanLoRALoaderInvocation(
            id="inv-1", lora=_make_lora_field(), target="low", transformer=_make_transformer_field()
        )
        ctx = _make_context(
            lora_expert=None,
            lora_config=_make_lora_config(variant=WanLoRAVariantType.A14B),
            main_variant=WanVariantType.T2V_A14B,
        )
        inv.invoke(ctx)
        ctx.logger.warning.assert_not_called()
