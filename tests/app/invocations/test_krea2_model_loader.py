from types import SimpleNamespace

import pytest

from invokeai.app.invocations.krea2_model_loader import Krea2ModelLoaderInvocation
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType


def _model(key: str, base: BaseModelType, model_type: ModelType) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash=f"hash-{key}", name=key, base=base, type=model_type)


def _context(configs: dict[str, SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(models=SimpleNamespace(get_config=lambda identifier: configs[identifier.key]))


def _config(base: BaseModelType, model_type: ModelType, model_format: ModelFormat) -> SimpleNamespace:
    return SimpleNamespace(base=base, type=model_type, format=model_format, name=f"{base.value}-{model_type.value}")


@pytest.mark.parametrize("vae_base", [BaseModelType.QwenImage, BaseModelType.Anima])
def test_loader_accepts_supported_standalone_components(vae_base: BaseModelType) -> None:
    main = _model("main", BaseModelType.Krea2, ModelType.Main)
    vae = _model("vae", vae_base, ModelType.VAE)
    encoder = _model("encoder", BaseModelType.Any, ModelType.Qwen3VLEncoder)
    context = _context(
        {
            "main": _config(BaseModelType.Krea2, ModelType.Main, ModelFormat.Checkpoint),
            "vae": _config(vae_base, ModelType.VAE, ModelFormat.Checkpoint),
            "encoder": _config(BaseModelType.Any, ModelType.Qwen3VLEncoder, ModelFormat.Qwen3VLEncoder),
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
        "encoder": _config(BaseModelType.Any, ModelType.Qwen3VLEncoder, ModelFormat.Qwen3VLEncoder),
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
