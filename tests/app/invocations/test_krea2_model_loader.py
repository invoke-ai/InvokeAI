from types import SimpleNamespace

import pytest

from invokeai.app.invocations.krea2.krea2_model_loader import Krea2ModelLoaderInvocation
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, Qwen3VLVariantType


def _model(key: str, base: BaseModelType, model_type: ModelType) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash=f"hash-{key}", name=key, base=base, type=model_type)


def _context(configs: dict[str, SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(models=SimpleNamespace(get_config=lambda identifier: configs[identifier.key]))


def _config(
    base: BaseModelType,
    model_type: ModelType,
    model_format: ModelFormat,
    variant: Qwen3VLVariantType | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        base=base,
        type=model_type,
        format=model_format,
        name=f"{base.value}-{model_type.value}",
        variant=variant,
    )


def _encoder_config(variant: Qwen3VLVariantType | None = Qwen3VLVariantType.Qwen3VL_4B) -> SimpleNamespace:
    return _config(BaseModelType.Any, ModelType.Qwen3VLEncoder, ModelFormat.Qwen3VLEncoder, variant)


@pytest.mark.parametrize("vae_base", [BaseModelType.QwenImage, BaseModelType.Anima])
def test_loader_accepts_supported_standalone_components(vae_base: BaseModelType) -> None:
    main = _model("main", BaseModelType.Krea2, ModelType.Main)
    vae = _model("vae", vae_base, ModelType.VAE)
    encoder = _model("encoder", BaseModelType.Any, ModelType.Qwen3VLEncoder)
    context = _context(
        {
            "main": _config(BaseModelType.Krea2, ModelType.Main, ModelFormat.Checkpoint),
            "vae": _config(vae_base, ModelType.VAE, ModelFormat.Checkpoint),
            "encoder": _encoder_config(),
        }
    )

    output = Krea2ModelLoaderInvocation(model=main, vae_model=vae, qwen3_vl_encoder_model=encoder).invoke(context)

    assert output.vae.vae.key == "vae"
    assert output.qwen3_vl_encoder.text_encoder.key == "encoder"


@pytest.mark.parametrize(
    ("target", "stored_config"),
    [
        ("main", _config(BaseModelType.Flux, ModelType.Main, ModelFormat.Checkpoint)),
        ("vae", _config(BaseModelType.StableDiffusionXL, ModelType.VAE, ModelFormat.Checkpoint)),
        ("encoder", _config(BaseModelType.Any, ModelType.Qwen3Encoder, ModelFormat.Checkpoint)),
    ],
)
def test_loader_rejects_incompatible_stored_component(target: str, stored_config: SimpleNamespace) -> None:
    main = _model("main", BaseModelType.Krea2, ModelType.Main)
    vae = _model("vae", BaseModelType.QwenImage, ModelType.VAE)
    encoder = _model("encoder", BaseModelType.Any, ModelType.Qwen3VLEncoder)
    configs = {
        "main": _config(BaseModelType.Krea2, ModelType.Main, ModelFormat.Checkpoint),
        "vae": _config(BaseModelType.QwenImage, ModelType.VAE, ModelFormat.Checkpoint),
        "encoder": _encoder_config(),
    }
    configs[target] = stored_config

    with pytest.raises(ValueError, match="Krea-2|VAE|Qwen3-VL"):
        Krea2ModelLoaderInvocation(model=main, vae_model=vae, qwen3_vl_encoder_model=encoder).invoke(_context(configs))


def test_loader_vae_ui_filter_includes_qwen_image_and_anima() -> None:
    field = Krea2ModelLoaderInvocation.model_fields["vae_model"]
    assert field.json_schema_extra is not None
    assert set(field.json_schema_extra["ui_model_base"]) == {
        BaseModelType.QwenImage.value,
        BaseModelType.Anima.value,
    }


@pytest.mark.parametrize(
    ("variant", "match"),
    [
        (Qwen3VLVariantType.Qwen3VL_8B, "is the Qwen3-VL 8B encoder"),
        (None, "of an unrecorded size"),
    ],
)
def test_loader_rejects_a_qwen3_vl_encoder_of_the_wrong_size(variant, match: str) -> None:
    """Ideogram 4's 8B installs under the same model type and is not interchangeable.

    Without this the 8B loads (~10 GB) and Krea-2 then fails on a 4096-against-2560 shape error
    naming a tensor, which says nothing about what the user picked. `None` covers MiniMax H3's
    Qwen3-VL-32B, which shares the type and records no variant at all.
    """
    main = _model("main", BaseModelType.Krea2, ModelType.Main)
    vae = _model("vae", BaseModelType.QwenImage, ModelType.VAE)
    encoder = _model("encoder", BaseModelType.Any, ModelType.Qwen3VLEncoder)
    configs = {
        "main": _config(BaseModelType.Krea2, ModelType.Main, ModelFormat.Checkpoint),
        "vae": _config(BaseModelType.QwenImage, ModelType.VAE, ModelFormat.Checkpoint),
        "encoder": _encoder_config(variant),
    }

    with pytest.raises(ValueError, match=match):
        Krea2ModelLoaderInvocation(model=main, vae_model=vae, qwen3_vl_encoder_model=encoder).invoke(_context(configs))


def _tower_configs(
    encoder_format: ModelFormat, tower_type: ModelType, tower_variant: Qwen3VLVariantType
) -> dict[str, SimpleNamespace]:
    return {
        "main": _config(BaseModelType.Krea2, ModelType.Main, ModelFormat.Checkpoint),
        "vae": _config(BaseModelType.QwenImage, ModelType.VAE, ModelFormat.Checkpoint),
        "encoder": _config(BaseModelType.Any, ModelType.Qwen3VLEncoder, encoder_format, Qwen3VLVariantType.Qwen3VL_4B),
        "tower": _config(BaseModelType.Any, tower_type, ModelFormat.GGUFQuantized, tower_variant),
    }


def _tower_loader(with_tower: bool = True) -> Krea2ModelLoaderInvocation:
    return Krea2ModelLoaderInvocation(
        model=_model("main", BaseModelType.Krea2, ModelType.Main),
        vae_model=_model("vae", BaseModelType.QwenImage, ModelType.VAE),
        qwen3_vl_encoder_model=_model("encoder", BaseModelType.Any, ModelType.Qwen3VLEncoder),
        qwen3_vl_vision_tower_model=(
            _model("tower", BaseModelType.Any, ModelType.Qwen3VLVisionTower) if with_tower else None
        ),
    )


def test_loader_hands_the_vision_tower_beside_a_gguf_encoder_to_the_field() -> None:
    configs = _tower_configs(ModelFormat.GGUFQuantized, ModelType.Qwen3VLVisionTower, Qwen3VLVariantType.Qwen3VL_4B)

    field = _tower_loader().invoke(_context(configs)).qwen3_vl_encoder

    assert field.vision_tower is not None
    assert field.vision_tower.key == "tower"


@pytest.mark.parametrize(
    ("encoder_format", "tower_type", "tower_variant", "match"),
    [
        (
            ModelFormat.GGUFQuantized,
            ModelType.Qwen3VLVisionTower,
            Qwen3VLVariantType.Qwen3VL_8B,
            "Krea-2 reads references with the 4B one",
        ),
        (
            ModelFormat.GGUFQuantized,
            ModelType.Qwen3VLEncoder,
            Qwen3VLVariantType.Qwen3VL_4B,
            "is not a Qwen3-VL vision tower",
        ),
        (
            ModelFormat.Qwen3VLEncoder,
            ModelType.Qwen3VLVisionTower,
            Qwen3VLVariantType.Qwen3VL_4B,
            "pairs only with a GGUF Qwen3-VL encoder",
        ),
    ],
)
def test_loader_refuses_a_vision_tower_that_does_not_fit(
    encoder_format: ModelFormat, tower_type: ModelType, tower_variant: Qwen3VLVariantType, match: str
) -> None:
    configs = _tower_configs(encoder_format, tower_type, tower_variant)

    with pytest.raises(ValueError, match=match):
        _tower_loader().invoke(_context(configs))


def test_loader_refuses_a_vision_tower_without_a_standalone_encoder() -> None:
    """The Diffusers pipeline's own encoder brings a tower; a stale selection must not pass silently."""
    configs = _tower_configs(ModelFormat.GGUFQuantized, ModelType.Qwen3VLVisionTower, Qwen3VLVariantType.Qwen3VL_4B)
    configs["main"] = _config(BaseModelType.Krea2, ModelType.Main, ModelFormat.Diffusers)
    loader = _tower_loader().model_copy(update={"qwen3_vl_encoder_model": None})

    with pytest.raises(ValueError, match="pairs only with a GGUF Qwen3-VL encoder"):
        loader.invoke(_context(configs))


def test_loader_leaves_the_vision_tower_empty_when_none_is_selected() -> None:
    configs = _tower_configs(ModelFormat.GGUFQuantized, ModelType.Qwen3VLVisionTower, Qwen3VLVariantType.Qwen3VL_4B)

    assert _tower_loader(with_tower=False).invoke(_context(configs)).qwen3_vl_encoder.vision_tower is None
