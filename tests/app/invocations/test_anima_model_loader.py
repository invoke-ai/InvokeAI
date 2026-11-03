from types import SimpleNamespace

import pytest

from invokeai.app.invocations.anima.anima_model_loader import AnimaModelLoaderInvocation, AnimaModelLoaderOutput
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.backend.model_manager.taxonomy import AnimaVariantType, BaseModelType, ModelType, Qwen35VariantType


def _model(key: str, base: BaseModelType, model_type: ModelType) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash=f"hash-{key}", name=key, base=base, type=model_type)


def _load_anima_3_8b(encoder_variant: Qwen35VariantType | None) -> AnimaModelLoaderOutput:
    configs = {
        "anima": SimpleNamespace(name="Anima-3.8B", variant=AnimaVariantType.Qwen35),
        "qwen35": SimpleNamespace(name="My Qwen3.5 encoder", variant=encoder_variant),
    }
    context = SimpleNamespace(models=SimpleNamespace(get_config=lambda identifier: configs[identifier.key]))
    return AnimaModelLoaderInvocation(
        model=_model("anima", BaseModelType.Anima, ModelType.Main),
        vae_model=_model("vae", BaseModelType.QwenImage, ModelType.VAE),
        qwen3_encoder_model=_model("qwen3", BaseModelType.Any, ModelType.Qwen3Encoder),
        qwen3_5_encoder_model=_model("qwen35", BaseModelType.Any, ModelType.Qwen35Encoder),
    ).invoke(context)


def test_anima_3_8b_reads_the_selected_qwen3_5_4b_encoder() -> None:
    output = _load_anima_3_8b(Qwen35VariantType.Qwen35_4B)

    assert output.qwen3_5_encoder is not None
    assert output.qwen3_5_encoder.text_encoder.key == "qwen35"


@pytest.mark.parametrize("variant", [Qwen35VariantType.Qwen35_9B, None])
def test_anima_3_8b_refuses_a_qwen3_5_encoder_of_another_size(variant: Qwen35VariantType | None) -> None:
    """MOSS-Image's 9B installs under the same model type; the connector was trained on the 4B.

    Without this the 9B loads (~16 GB) and denoising fails on a shape mismatch that names no model.
    `None` covers a record with no variant at all.
    """
    with pytest.raises(ValueError, match="'My Qwen3.5 encoder' is not a Qwen3.5 4B encoder"):
        _load_anima_3_8b(variant)
