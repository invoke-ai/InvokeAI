from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.wan_model_loader import WanModelLoaderInvocation
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, WanVariantType


def _model(key: str) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash="hash", name=key, base="wan", type="main")


def _config(
    name: str,
    variant: WanVariantType,
    expert: str,
    *,
    format: ModelFormat = ModelFormat.GGUFQuantized,
    has_dual_expert: bool = False,
    boundary_ratio: float | None = None,
    base: BaseModelType = BaseModelType.Wan,
    type: ModelType = ModelType.Main,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        format=format,
        variant=variant,
        expert=expert,
        has_dual_expert=has_dual_expert,
        boundary_ratio=boundary_ratio,
        base=base,
        type=type,
    )


def _prepare(
    main_config: SimpleNamespace,
    low_config: SimpleNamespace | None = None,
    component_config: SimpleNamespace | None = None,
    *,
    use_component_vae: bool = False,
    vae_latent_channels: int | None = None,
    vae_config: SimpleNamespace | None = None,
    t5_config: SimpleNamespace | None = None,
    low_key: str = "low",
) -> tuple[WanModelLoaderInvocation, MagicMock]:
    main = _model("main")
    low = _model(low_key) if low_config is not None else None
    context = MagicMock()
    configs = {"main": main_config}
    if low_config is not None:
        configs[low_key] = low_config
    component = _model("component") if component_config is not None else None
    if component_config is not None:
        configs["component"] = component_config
    if not use_component_vae:
        if vae_latent_channels is None:
            vae_latent_channels = 48 if getattr(main_config, "variant", None) == WanVariantType.TI2V_5B else 16
        configs["vae"] = vae_config or SimpleNamespace(
            name="vae",
            latent_channels=vae_latent_channels,
            base=BaseModelType.Wan,
            type=ModelType.VAE,
        )
    configs["t5"] = t5_config or SimpleNamespace(
        name="t5",
        base=BaseModelType.Any,
        type=ModelType.WanT5Encoder,
        format=ModelFormat.WanT5Encoder,
    )
    context.models.get_config.side_effect = lambda model: configs[model.key]
    invocation = WanModelLoaderInvocation(
        id="test",
        model=main,
        transformer_low_noise_model=low,
        vae_model=None if use_component_vae else _model("vae"),
        wan_t5_encoder_model=_model("t5"),
        component_source=component,
    )
    return invocation, context


def _invoke(*args, **kwargs):
    invocation, context = _prepare(*args, **kwargs)
    return invocation.invoke(context)


def _warnings(context: MagicMock) -> list[str]:
    return [call.args[0] for call in context.logger.warning.call_args_list]


@pytest.mark.parametrize("variant", [WanVariantType.T2V_A14B, WanVariantType.I2V_A14B])
@pytest.mark.parametrize(
    "main_expert,low_expert",
    [("high", "low"), ("low", "high"), ("none", "none")],
)
def test_gguf_loader_preserves_explicit_slot_wiring_regardless_of_filename_tags(
    variant: WanVariantType, main_expert: str, low_expert: str
) -> None:
    """Filename-derived expert tags are advisory; explicit graph wiring defines the roles."""
    output = _invoke(
        _config("main", variant, main_expert),
        _config("low", variant, low_expert),
    )

    assert output.transformer.transformer.key == "main"
    assert output.transformer.transformer_low_noise is not None
    assert output.transformer.transformer_low_noise.key == "low"


@pytest.mark.parametrize("expert", ["high", "low"])
def test_gguf_loader_rejects_same_phase_expert_pairs(expert: str) -> None:
    with pytest.raises(ValueError, match="one high and one low"):
        _invoke(
            _config("main", WanVariantType.T2V_A14B, expert),
            _config("low", WanVariantType.T2V_A14B, expert),
        )


@pytest.mark.parametrize(
    "main_config,low_config",
    [
        (
            _config("main", WanVariantType.T2V_A14B, "high"),
            _config("low", WanVariantType.I2V_A14B, "low"),
        ),
    ],
)
def test_gguf_loader_rejects_mismatched_expert_variants(
    main_config: SimpleNamespace, low_config: SimpleNamespace
) -> None:
    with pytest.raises(ValueError, match="variant"):
        _invoke(main_config, low_config)


@pytest.mark.parametrize(
    "main_expert,low_expert",
    [
        # The expert tag comes from a filename heuristic, so untagged community
        # finetunes are common. The explicit slot wiring remains authoritative
        # whether one, both, or neither filename supplies an expert hint.
        ("none", "none"),
        ("high", "none"),
        ("none", "low"),
        ("none", "high"),
        ("low", "none"),
    ],
)
def test_gguf_loader_uses_wiring_for_untagged_experts(main_expert: str, low_expert: str) -> None:
    output = _invoke(
        _config("main", WanVariantType.I2V_A14B, main_expert),
        _config("low", WanVariantType.I2V_A14B, low_expert),
    )

    assert output.transformer.transformer.key == "main"
    assert output.transformer.transformer_low_noise is not None
    assert output.transformer.transformer_low_noise.key == "low"


@pytest.mark.parametrize("low_variant", [WanVariantType.TI2V_5B, WanVariantType.T2V_A14B])
def test_ti2v_5b_main_ignores_wired_low_noise_model(low_variant: WanVariantType) -> None:
    """The field docs promise 'Transformer (Low Noise)' is ignored for the single-expert
    TI2V-5B — e.g. a wire left over from an A14B session must not abort the run."""
    output = _invoke(
        _config("main", WanVariantType.TI2V_5B, "none"),
        _config("low", low_variant, "low"),
    )

    assert output.transformer.transformer.key == "main"
    assert output.transformer.transformer_low_noise is None


@pytest.mark.parametrize("expert", ["high", "low", "none"])
def test_gguf_loader_runs_unpaired_primary_whatever_its_tag(expert: str) -> None:
    """A single wired transformer is explicit intent just like a pair is, and the tag is
    only a filename guess — so an unpaired A14B runs with a warning rather than aborting."""
    invocation, context = _prepare(_config("main", WanVariantType.T2V_A14B, expert))
    output = invocation.invoke(context)

    assert output.transformer.transformer.key == "main"
    assert output.transformer.transformer_low_noise is None
    assert any("only this one expert will run" in warning.lower() for warning in _warnings(context))


def test_gguf_loader_hints_at_the_expert_swap_for_an_unpaired_low_noise_model() -> None:
    invocation, context = _prepare(_config("main", WanVariantType.T2V_A14B, "low"))
    invocation.invoke(context)

    assert any("high-noise one is usually the better choice" in warning for warning in _warnings(context))


def test_gguf_loader_rejects_the_same_model_in_both_transformer_slots() -> None:
    """Wiring one model twice used to fail the {high, low} pair check. It must stay an error:
    the denoiser would unload and reload the same multi-GB expert at every boundary crossing."""
    main_config = _config("main", WanVariantType.T2V_A14B, "high")
    with pytest.raises(ValueError, match="same model"):
        _invoke(main_config, main_config, low_key="main")


@pytest.mark.parametrize("main_expert,low_expert", [("low", "high"), ("low", "none"), ("none", "high")])
def test_gguf_loader_warns_when_filename_tags_disagree_with_wiring(main_expert: str, low_expert: str) -> None:
    """A suspicious filename can produce a warning, but must not override explicit wiring."""
    invocation, context = _prepare(
        _config("main", WanVariantType.I2V_A14B, main_expert),
        _config("low", WanVariantType.I2V_A14B, low_expert),
    )
    output = invocation.invoke(context)

    assert output.transformer.transformer.key == "main"
    assert output.transformer.transformer_low_noise is not None
    assert output.transformer.transformer_low_noise.key == "low"
    assert any("wiring" in warning.lower() for warning in _warnings(context))


@pytest.mark.parametrize("main_expert,low_expert", [("high", "low"), ("high", "none"), ("none", "low")])
def test_gguf_loader_is_quiet_when_the_wiring_stands(main_expert: str, low_expert: str) -> None:
    invocation, context = _prepare(
        _config("main", WanVariantType.I2V_A14B, main_expert),
        _config("low", WanVariantType.I2V_A14B, low_expert),
    )
    invocation.invoke(context)

    assert _warnings(context) == []


def test_gguf_loader_warns_when_neither_expert_is_tagged() -> None:
    invocation, context = _prepare(
        _config("main", WanVariantType.I2V_A14B, "none"),
        _config("low", WanVariantType.I2V_A14B, "none"),
    )
    invocation.invoke(context)

    assert any("Neither Wan A14B filename identifies its expert" in warning for warning in _warnings(context))


@pytest.mark.parametrize(
    "variant,expected",
    [(WanVariantType.T2V_A14B, 0.875), (WanVariantType.I2V_A14B, 0.9)],
)
def test_gguf_loader_uses_variant_boundary_default(variant: WanVariantType, expected: float) -> None:
    output = _invoke(_config("main", variant, "high"))

    assert output.transformer.boundary_ratio == expected


def test_diffusers_i2v_loader_uses_variant_boundary_default_when_metadata_missing() -> None:
    output = _invoke(
        _config(
            "main",
            WanVariantType.I2V_A14B,
            "none",
            format=ModelFormat.Diffusers,
            has_dual_expert=True,
        )
    )

    assert output.transformer.boundary_ratio == 0.9


def test_gguf_loader_uses_matching_component_source_boundary() -> None:
    output = _invoke(
        _config("main", WanVariantType.I2V_A14B, "high"),
        component_config=_config(
            "component", WanVariantType.I2V_A14B, "none", format=ModelFormat.Diffusers, boundary_ratio=0.91
        ),
    )

    assert output.transformer.boundary_ratio == 0.91


def test_gguf_loader_ignores_mismatched_component_source_boundary() -> None:
    output = _invoke(
        _config("main", WanVariantType.I2V_A14B, "high"),
        component_config=_config(
            "component", WanVariantType.T2V_A14B, "none", format=ModelFormat.Diffusers, boundary_ratio=0.875
        ),
    )

    assert output.transformer.boundary_ratio == 0.9


@pytest.mark.parametrize(
    "main_variant,component_variant",
    [
        (WanVariantType.TI2V_5B, WanVariantType.T2V_A14B),
        (WanVariantType.T2V_A14B, WanVariantType.TI2V_5B),
    ],
)
def test_gguf_loader_rejects_component_source_with_incompatible_vae_family(
    main_variant: WanVariantType, component_variant: WanVariantType
) -> None:
    with pytest.raises(ValueError, match="VAE"):
        _invoke(
            _config("main", main_variant, "none" if main_variant == WanVariantType.TI2V_5B else "high"),
            component_config=_config("component", component_variant, "none", format=ModelFormat.Diffusers),
            use_component_vae=True,
        )


@pytest.mark.parametrize(
    "main_variant,component_variant",
    [
        (WanVariantType.TI2V_5B, WanVariantType.TI2V_5B),
        (WanVariantType.T2V_A14B, WanVariantType.I2V_A14B),
        (WanVariantType.I2V_A14B, WanVariantType.T2V_A14B),
    ],
)
def test_gguf_loader_accepts_component_source_with_compatible_vae_family(
    main_variant: WanVariantType, component_variant: WanVariantType
) -> None:
    output = _invoke(
        _config("main", main_variant, "none" if main_variant == WanVariantType.TI2V_5B else "high"),
        component_config=_config("component", component_variant, "none", format=ModelFormat.Diffusers),
        use_component_vae=True,
    )

    assert output.vae.vae.key == "component"


@pytest.mark.parametrize(
    "main_variant,vae_latent_channels",
    [
        (WanVariantType.TI2V_5B, 16),
        (WanVariantType.T2V_A14B, 48),
        (WanVariantType.I2V_A14B, 48),
    ],
)
def test_loader_rejects_incompatible_standalone_vae(main_variant: WanVariantType, vae_latent_channels: int) -> None:
    with pytest.raises(ValueError, match="VAE"):
        _invoke(
            _config("main", main_variant, "none" if main_variant == WanVariantType.TI2V_5B else "high"),
            vae_latent_channels=vae_latent_channels,
        )


@pytest.mark.parametrize(
    "main_variant,vae_latent_channels",
    [
        (WanVariantType.TI2V_5B, 48),
        (WanVariantType.T2V_A14B, 16),
        (WanVariantType.I2V_A14B, 16),
    ],
)
def test_loader_accepts_compatible_standalone_vae(main_variant: WanVariantType, vae_latent_channels: int) -> None:
    output = _invoke(
        _config("main", main_variant, "none" if main_variant == WanVariantType.TI2V_5B else "high"),
        vae_latent_channels=vae_latent_channels,
    )

    assert output.vae.vae.key == "vae"


def test_loader_rejects_forged_non_wan_main_identifier() -> None:
    with pytest.raises(ValueError, match="Wan main"):
        _invoke(
            SimpleNamespace(
                name="not-wan",
                format=ModelFormat.Diffusers,
                base=BaseModelType.StableDiffusionXL,
                type=ModelType.Main,
            )
        )


def test_loader_rejects_forged_non_wan_t5_identifier() -> None:
    with pytest.raises(ValueError, match="Wan T5"):
        _invoke(
            _config("main", WanVariantType.T2V_A14B, "high"),
            t5_config=SimpleNamespace(
                name="not-t5",
                base=BaseModelType.StableDiffusionXL,
                type=ModelType.Main,
                format=ModelFormat.Diffusers,
            ),
        )


def test_loader_rejects_forged_non_wan_vae_identifier() -> None:
    with pytest.raises(ValueError, match="Wan VAE"):
        _invoke(
            _config("main", WanVariantType.T2V_A14B, "high"),
            vae_config=SimpleNamespace(
                name="not-vae",
                latent_channels=16,
                base=BaseModelType.StableDiffusionXL,
                type=ModelType.Main,
            ),
        )


def test_loader_rejects_forged_non_wan_component_source_identifier() -> None:
    with pytest.raises(ValueError, match="Wan.*Component Source|Component Source.*Wan"):
        _invoke(
            _config("main", WanVariantType.T2V_A14B, "high"),
            component_config=SimpleNamespace(
                name="not-wan",
                format=ModelFormat.Diffusers,
                base=BaseModelType.StableDiffusionXL,
                type=ModelType.Main,
            ),
            use_component_vae=True,
        )


def test_loader_rejects_forged_component_source_even_with_standalone_components() -> None:
    with pytest.raises(ValueError, match="Wan.*Component Source|Component Source.*Wan"):
        _invoke(
            _config("main", WanVariantType.T2V_A14B, "high"),
            component_config=SimpleNamespace(
                name="not-wan",
                format=ModelFormat.Diffusers,
                variant=WanVariantType.T2V_A14B,
                boundary_ratio=0.5,
                base=BaseModelType.StableDiffusionXL,
                type=ModelType.Main,
            ),
        )


# --- Single-file safetensors checkpoints (#9463) ---------------------------------


@pytest.mark.parametrize("variant", [WanVariantType.T2V_A14B, WanVariantType.I2V_A14B])
def test_checkpoint_main_accepts_expert_pair(variant: WanVariantType) -> None:
    output = _invoke(
        _config("main", variant, "high", format=ModelFormat.Checkpoint),
        _config("low", variant, "low", format=ModelFormat.Checkpoint),
    )

    assert output.transformer.transformer.key == "main"
    assert output.transformer.transformer_low_noise is not None
    assert output.transformer.transformer_low_noise.key == "low"


@pytest.mark.parametrize(
    "main_format,low_format",
    [
        (ModelFormat.Checkpoint, ModelFormat.GGUFQuantized),
        (ModelFormat.GGUFQuantized, ModelFormat.Checkpoint),
    ],
)
def test_experts_may_mix_single_file_formats(main_format: ModelFormat, low_format: ModelFormat) -> None:
    """Both single-file loaders produce a plain WanTransformer3DModel, so a GGUF
    high-noise expert pairs fine with a safetensors low-noise one."""
    output = _invoke(
        _config("main", WanVariantType.T2V_A14B, "high", format=main_format),
        _config("low", WanVariantType.T2V_A14B, "low", format=low_format),
    )

    assert output.transformer.transformer.key == "main"
    assert output.transformer.transformer_low_noise is not None
    assert output.transformer.transformer_low_noise.key == "low"


def test_checkpoint_ti2v_5b_runs_unpaired() -> None:
    output = _invoke(_config("main", WanVariantType.TI2V_5B, "none", format=ModelFormat.Checkpoint))

    assert output.transformer.transformer.key == "main"
    assert output.transformer.transformer_low_noise is None


@pytest.mark.parametrize("expert", ["high", "low", "none"])
def test_checkpoint_main_runs_unpaired_whatever_its_tag(expert: str) -> None:
    """The checkpoint path takes the same wiring-first rule #9505 gave the GGUF path: a
    single wired transformer is explicit intent, and the tag is only a filename guess.
    Untagged community checkpoints are the common case this branch exists to support, so
    they must not be fatal here when the paired path accepts them."""
    invocation, context = _prepare(_config("main", WanVariantType.T2V_A14B, expert, format=ModelFormat.Checkpoint))
    output = invocation.invoke(context)

    assert output.transformer.transformer.key == "main"
    assert output.transformer.transformer_low_noise is None
    assert any("only this one expert will run" in warning.lower() for warning in _warnings(context))


def test_low_noise_slot_rejects_a_diffusers_model() -> None:
    with pytest.raises(ValueError, match="single-file"):
        _invoke(
            _config("main", WanVariantType.T2V_A14B, "high", format=ModelFormat.Checkpoint),
            _config("low", WanVariantType.T2V_A14B, "low", format=ModelFormat.Diffusers),
        )


def test_checkpoint_main_uses_component_source_boundary() -> None:
    output = _invoke(
        _config("main", WanVariantType.T2V_A14B, "high", format=ModelFormat.Checkpoint),
        _config("low", WanVariantType.T2V_A14B, "low", format=ModelFormat.Checkpoint),
        _config(
            "component",
            WanVariantType.T2V_A14B,
            "none",
            format=ModelFormat.Diffusers,
            has_dual_expert=True,
            boundary_ratio=0.5,
        ),
    )

    assert output.transformer.boundary_ratio == 0.5
