"""`moss_image_model_loader`: where each submodel comes from, and the selections it refuses."""

from types import SimpleNamespace

import pytest

from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.moss_image.moss_image_model_loader import (
    MossImageModelLoaderInvocation,
    MossImageModelLoaderOutput,
)
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
    Qwen35VariantType,
    SubModelType,
)
from tests.fixtures.model_roles import node_context, text_llm_file


def _model(key: str, base: BaseModelType, model_type: ModelType) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash=f"hash-{key}", name=key, base=base, type=model_type)


_CONFIGS = {
    "moss-diffusers": SimpleNamespace(
        name="MOSS-Image Base", base=BaseModelType.MossImage, type=ModelType.Main, format=ModelFormat.Diffusers
    ),
    "moss-single": SimpleNamespace(
        name="moss-image-dit", base=BaseModelType.MossImage, type=ModelType.Main, format=ModelFormat.Checkpoint
    ),
    "flux2-vae": SimpleNamespace(name="FLUX.2 VAE", base=BaseModelType.Flux2, type=ModelType.VAE),
    "sdxl-vae": SimpleNamespace(name="SDXL VAE", base=BaseModelType.StableDiffusionXL, type=ModelType.VAE),
    "qwen35-9b": SimpleNamespace(
        name="qwen35_9b", base=BaseModelType.Any, type=ModelType.Qwen35Encoder, variant=Qwen35VariantType.Qwen35_9B
    ),
    "qwen35-4b": SimpleNamespace(
        name="qwen35_4b", base=BaseModelType.Any, type=ModelType.Qwen35Encoder, variant=Qwen35VariantType.Qwen35_4B
    ),
    "krea2": SimpleNamespace(
        name="Krea-2", base=BaseModelType.Krea2, type=ModelType.Main, format=ModelFormat.Diffusers
    ),
}


def _load(main: str, vae: str | None = None, encoder: str | None = None) -> MossImageModelLoaderOutput:
    context = SimpleNamespace(models=SimpleNamespace(get_config=lambda identifier: _CONFIGS[identifier.key]))
    return MossImageModelLoaderInvocation(
        model=_model(main, _CONFIGS[main].base, ModelType.Main),
        vae_model=_model(vae, _CONFIGS[vae].base, ModelType.VAE) if vae else None,
        qwen3_5_encoder_model=_model(encoder, BaseModelType.Any, ModelType.Qwen35Encoder) if encoder else None,
    ).invoke(context)  # type: ignore[arg-type]


def test_a_diffusers_pipeline_supplies_every_submodel_itself() -> None:
    output = _load("moss-diffusers")

    submodels = {
        "transformer": output.transformer.transformer,
        "text_encoder": output.qwen3_5_encoder.text_encoder,
        "tokenizer": output.qwen3_5_encoder.tokenizer,
        "vae": output.vae.vae,
    }
    assert {name: (field.key, field.submodel_type) for name, field in submodels.items()} == {
        "transformer": ("moss-diffusers", SubModelType.Transformer),
        "text_encoder": ("moss-diffusers", SubModelType.TextEncoder),
        "tokenizer": ("moss-diffusers", SubModelType.Tokenizer),
        "vae": ("moss-diffusers", SubModelType.VAE),
    }


def test_standalone_components_replace_the_pipeline_s_own() -> None:
    output = _load("moss-single", vae="flux2-vae", encoder="qwen35-9b")

    assert output.transformer.transformer.key == "moss-single"
    assert (output.vae.vae.key, output.vae.vae.submodel_type) == ("flux2-vae", SubModelType.VAE)
    assert (output.qwen3_5_encoder.text_encoder.key, output.qwen3_5_encoder.text_encoder.submodel_type) == (
        "qwen35-9b",
        SubModelType.TextEncoder,
    )
    assert output.qwen3_5_encoder.tokenizer.submodel_type is SubModelType.Tokenizer


@pytest.mark.parametrize(
    ("vae", "encoder", "missing"),
    [
        (None, None, "VAE and Qwen3.5 Encoder"),
        ("flux2-vae", None, "Qwen3.5 Encoder"),
        (None, "qwen35-9b", "VAE"),
    ],
)
def test_a_single_file_transformer_names_the_components_it_lacks(
    vae: str | None, encoder: str | None, missing: str
) -> None:
    with pytest.raises(ValueError, match=f"'moss-image-dit' is a single-file .* Select a standalone {missing}, or"):
        _load("moss-single", vae=vae, encoder=encoder)


def test_the_anima_qwen3_5_4b_encoder_is_refused_by_name() -> None:
    # Both sizes install under one model type; MOSS-Image reads the 9B's 4096-wide states.
    with pytest.raises(ValueError, match="'qwen35_4b' is not a Qwen3.5 9B encoder"):
        _load("moss-diffusers", encoder="qwen35-4b")


def test_a_vae_from_another_latent_space_is_refused() -> None:
    with pytest.raises(ValueError, match="'SDXL VAE' is not compatible with MOSS-Image"):
        _load("moss-diffusers", vae="sdxl-vae")


def test_a_main_model_of_another_architecture_is_refused() -> None:
    with pytest.raises(ValueError, match="'Krea-2' is not a MOSS-Image main model"):
        _load("krea2")


def _load_with_roles(encoder_key: str) -> MossImageModelLoaderOutput:
    configs = {
        **_CONFIGS,
        "llm-9b": text_llm_file("llm-9b", Qwen35VariantType.Qwen35_9B),
        "llm-4b": text_llm_file("llm-4b", Qwen35VariantType.Qwen35_4B),
        "llm-plain": text_llm_file("llm-plain", None),
    }
    return MossImageModelLoaderInvocation(
        model=_model("moss-diffusers", BaseModelType.MossImage, ModelType.Main),
        # The UI puts the record's own type on the node.
        qwen3_5_encoder_model=_model(encoder_key, BaseModelType.Any, ModelType.TextLLM),
    ).invoke(node_context(configs))  # type: ignore[arg-type]


def test_a_text_llm_serving_as_the_9b_encoder_is_accepted_and_emitted_as_an_encoder() -> None:
    output = _load_with_roles("llm-9b")

    for field in (output.qwen3_5_encoder.text_encoder, output.qwen3_5_encoder.tokenizer):
        assert (field.key, field.type) == ("llm-9b", ModelType.Qwen35Encoder)
    assert output.qwen3_5_encoder.text_encoder.submodel_type is SubModelType.TextEncoder


@pytest.mark.parametrize("key", ["llm-4b", "llm-plain"])
def test_a_text_llm_without_the_9b_encoder_role_is_refused_by_name(key: str) -> None:
    with pytest.raises(ValueError, match=f"'{key} llm' is not a Qwen3.5 9B encoder.*or a Text LLM that serves as one"):
        _load_with_roles(key)
