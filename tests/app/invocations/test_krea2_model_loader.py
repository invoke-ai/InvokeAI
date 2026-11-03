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
