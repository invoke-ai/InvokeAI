from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.wan_model_loader import WanModelLoaderInvocation
from invokeai.backend.model_manager.taxonomy import ModelFormat, WanVariantType


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
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        format=format,
        variant=variant,
        expert=expert,
        has_dual_expert=has_dual_expert,
        boundary_ratio=boundary_ratio,
    )


def _invoke(main_config: SimpleNamespace, low_config: SimpleNamespace | None = None):
    main = _model("main")
    low = _model("low") if low_config is not None else None
    context = MagicMock()
    configs = {"main": main_config}
    if low_config is not None:
        configs["low"] = low_config
    context.models.get_config.side_effect = lambda model: configs[model.key]
    invocation = WanModelLoaderInvocation(
        id="test",
        model=main,
        transformer_low_noise_model=low,
        vae_model=_model("vae"),
        wan_t5_encoder_model=_model("t5"),
    )
    return invocation.invoke(context)


@pytest.mark.parametrize("variant", [WanVariantType.T2V_A14B, WanVariantType.I2V_A14B])
@pytest.mark.parametrize("main_expert,low_expert", [("high", "low"), ("low", "high")])
def test_gguf_loader_accepts_valid_expert_pair_in_either_order(
    variant: WanVariantType, main_expert: str, low_expert: str
) -> None:
    output = _invoke(
        _config("main", variant, main_expert),
        _config("low", variant, low_expert),
    )

    assert output.transformer.transformer.key == ("main" if main_expert == "high" else "low")
    assert output.transformer.transformer_low_noise is not None
    assert output.transformer.transformer_low_noise.key == ("low" if low_expert == "low" else "main")


@pytest.mark.parametrize(
    "main_config,low_config",
    [
        (
            _config("main", WanVariantType.T2V_A14B, "high"),
            _config("low", WanVariantType.I2V_A14B, "low"),
        ),
        (
            _config("main", WanVariantType.T2V_A14B, "high"),
            _config("low", WanVariantType.T2V_A14B, "high"),
        ),
        (
            _config("main", WanVariantType.T2V_A14B, "high"),
            _config("low", WanVariantType.T2V_A14B, "none"),
        ),
        (
            _config("main", WanVariantType.TI2V_5B, "none"),
            _config("low", WanVariantType.T2V_A14B, "low"),
        ),
    ],
)
def test_gguf_loader_rejects_invalid_expert_pair(main_config: SimpleNamespace, low_config: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="expert|variant"):
        _invoke(main_config, low_config)


@pytest.mark.parametrize("expert", ["low", "none"])
def test_gguf_loader_rejects_non_high_primary_without_pair(expert: str) -> None:
    with pytest.raises(ValueError, match="high-noise"):
        _invoke(_config("main", WanVariantType.T2V_A14B, expert))


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
