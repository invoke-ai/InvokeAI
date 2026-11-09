from types import SimpleNamespace

import pytest

from invokeai.app.invocations.anima.anima_model_loader import AnimaModelLoaderInvocation, AnimaModelLoaderOutput
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.backend.model_manager.taxonomy import (
    AnimaVariantType,
    BaseModelType,
    ModelType,
    Qwen3VariantType,
    Qwen35VariantType,
    SubModelType,
)
from tests.fixtures.model_roles import node_context, text_llm_file, text_llm_qwen3_folder


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


def _load_with_text_llms(qwen3_5_variant: Qwen35VariantType | None) -> AnimaModelLoaderOutput:
    configs = {
        "anima": SimpleNamespace(name="Anima-3.8B", type=ModelType.Main, variant=AnimaVariantType.Qwen35),
        "qwen3": text_llm_qwen3_folder("qwen3", Qwen3VariantType.Qwen3_06B),
        "qwen35": text_llm_file("qwen35", qwen3_5_variant),
    }
    return AnimaModelLoaderInvocation(
        model=_model("anima", BaseModelType.Anima, ModelType.Main),
        vae_model=_model("vae", BaseModelType.QwenImage, ModelType.VAE),
        qwen3_encoder_model=_model("qwen3", BaseModelType.Any, ModelType.TextLLM),
        qwen3_5_encoder_model=_model("qwen35", BaseModelType.Any, ModelType.TextLLM),
    ).invoke(node_context(configs))  # type: ignore[arg-type]


def test_text_llms_serving_as_the_encoders_are_emitted_in_those_roles() -> None:
    output = _load_with_text_llms(Qwen35VariantType.Qwen35_4B)

    assert output.qwen3_5_encoder is not None
    qwen3 = output.qwen3_encoder
    qwen3_5 = output.qwen3_5_encoder
    assert [(f.key, f.type, f.submodel_type) for f in (qwen3.text_encoder, qwen3.tokenizer)] == [
        ("qwen3", ModelType.Qwen3Encoder, SubModelType.TextEncoder),
        ("qwen3", ModelType.Qwen3Encoder, SubModelType.Tokenizer),
    ]
    assert [(f.key, f.type, f.submodel_type) for f in (qwen3_5.text_encoder, qwen3_5.tokenizer)] == [
        ("qwen35", ModelType.Qwen35Encoder, SubModelType.TextEncoder),
        ("qwen35", ModelType.Qwen35Encoder, SubModelType.Tokenizer),
    ]


def test_a_text_llm_without_the_4b_encoder_role_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="'qwen35 llm' is not a Qwen3.5 4B encoder"):
        _load_with_text_llms(None)
