"""The ERNIE-Image loader node with a single-file transformer.

A diffusers pipeline carries every submodel; a single file carries the transformer alone, so its
text encoder and VAE are chosen on the node. Selecting one and forgetting the other has to fail
with something a user can act on, rather than deep inside the loader at generate time.
"""

from types import SimpleNamespace

import pytest

from invokeai.app.invocations.ernie_image.ernie_image_model_loader import ErnieImageModelLoaderInvocation
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    MistralVariantType,
    ModelFormat,
    ModelType,
    SubModelType,
)


def _identifier(key: str, type: ModelType = ModelType.Main) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash=f"hash:{key}", name=key, base=BaseModelType.ErnieImage, type=type)


def _context(
    format: ModelFormat, encoder_variant: MistralVariantType | None = MistralVariantType.Ministral3B
) -> SimpleNamespace:
    """A context whose `get_config` answers per model, so the encoder can carry its own variant."""

    def get_config(model: ModelIdentifierField) -> SimpleNamespace:
        if model.type is ModelType.MistralEncoder:
            return SimpleNamespace(format=ModelFormat.Checkpoint, variant=encoder_variant)
        return SimpleNamespace(format=format)

    return SimpleNamespace(models=SimpleNamespace(get_config=get_config))


def _node(**fields) -> ErnieImageModelLoaderInvocation:
    defaults = {
        "model": _identifier("transformer"),
        "text_encoder_model": None,
        "vae_model": None,
        "use_prompt_enhancer": True,
    }
    return ErnieImageModelLoaderInvocation.model_construct(**{**defaults, **fields})


def test_a_single_file_emits_the_chosen_encoder_and_vae() -> None:
    node = _node(
        text_encoder_model=_identifier("mistral", ModelType.MistralEncoder),
        vae_model=_identifier("vae", ModelType.VAE),
    )

    output = node.invoke(_context(ModelFormat.Checkpoint))

    assert output.transformer.transformer.key == "transformer"
    assert output.transformer.transformer.submodel_type is SubModelType.Transformer
    assert output.text_encoder.text_encoder.key == "mistral"
    assert output.text_encoder.text_encoder.submodel_type is SubModelType.TextEncoder
    assert output.text_encoder.tokenizer.submodel_type is SubModelType.Tokenizer
    assert output.vae.vae.key == "vae"
    assert output.vae.vae.submodel_type is SubModelType.VAE
    # The prompt enhancer ships only with the pipeline, so a single file has none.
    assert output.prompt_enhancer is None


@pytest.mark.parametrize(
    "fields, missing",
    [
        ({"vae_model": _identifier("vae", ModelType.VAE)}, "Text Encoder"),
        ({"text_encoder_model": _identifier("mistral", ModelType.MistralEncoder)}, "VAE"),
        ({}, "Text Encoder and VAE"),
    ],
)
def test_a_single_file_without_its_companions_says_which_one_is_missing(fields: dict, missing: str) -> None:
    node = _node(**fields)

    with pytest.raises(ValueError, match=missing):
        node.invoke(_context(ModelFormat.Checkpoint))


def test_an_encoder_from_the_wrong_family_is_refused_by_name(monkeypatch) -> None:
    """Every Mistral encoder installs under one type, but FLUX.2's are 5120 wide against ERNIE's
    3072. Without this the mismatch surfaces as a shape error inside denoising, which tells the
    user nothing about the model they picked."""
    node = _node(
        text_encoder_model=_identifier("cow-mistral", ModelType.MistralEncoder),
        vae_model=_identifier("vae", ModelType.VAE),
    )

    with pytest.raises(ValueError, match="Ministral 3B"):
        node.invoke(_context(ModelFormat.Checkpoint, encoder_variant=MistralVariantType.Cow))


def test_a_diffusers_pipeline_still_takes_every_submodel_from_itself(monkeypatch) -> None:
    """The pipeline path must not start demanding the new fields."""
    node = _node()
    monkeypatch.setattr(ErnieImageModelLoaderInvocation, "_pipeline_has_prompt_enhancer", lambda self, context: False)

    output = node.invoke(_context(ModelFormat.Diffusers))

    assert output.transformer.transformer.key == "transformer"
    assert output.text_encoder.text_encoder.key == "transformer"
    assert output.vae.vae.key == "transformer"
