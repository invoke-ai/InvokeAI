"""The FLUX.2 loaders' cross-variant guard must gate the *encoder* extraction path only.

Klein and [dev] share the same 32-channel `AutoencoderKLFlux2`, and the linear UI relies on that:
`buildFLUXGraph` falls back to *any* FLUX.2 diffusers pipeline when only the VAE is needed, and
neither `isFlux2DiffusersMainModelConfig` nor readiness filter by variant. A variant guard on the
VAE-extraction call site therefore turns a working configuration (GGUF main + standalone encoder +
a cross-variant diffusers pipeline as the only VAE source) into a runtime `ValueError` behind an
enabled Invoke button. The encoder path must still reject cross-variant sources, because a
mismatched tokenizer/encoder only surfaces as an opaque matmul error deep in denoise.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.flux2_dev_model_loader import Flux2DevModelLoaderInvocation
from invokeai.app.invocations.flux2_klein_model_loader import Flux2KleinModelLoaderInvocation
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    Flux2VariantType,
    ModelFormat,
    ModelType,
    Qwen3VariantType,
    SubModelType,
)


def _identifier(key: str, model_type: ModelType = ModelType.Main) -> ModelIdentifierField:
    return ModelIdentifierField(
        key=key,
        hash=f"hash-{key}",
        name=key,
        base=BaseModelType.Flux2,
        type=model_type,
    )


def _context(configs: dict[str, SimpleNamespace]) -> MagicMock:
    """An InvocationContext stand-in resolving `get_config(identifier)` by key."""
    context = MagicMock()
    context.models.get_config.side_effect = lambda identifier: configs[identifier.key]
    return context


def test_klein_loader_extracts_vae_from_a_dev_pipeline() -> None:
    """Klein GGUF main + standalone Qwen3 encoder + a [dev] pipeline as the only VAE source."""
    main = _identifier("klein-gguf")
    encoder = _identifier("qwen3-4b", ModelType.Qwen3Encoder)
    source = _identifier("flux2-dev-diffusers")

    invocation = Flux2KleinModelLoaderInvocation.model_construct(
        model=main,
        vae_model=None,
        qwen3_encoder_model=encoder,
        qwen3_source_model=source,
        max_seq_len=512,
    )
    context = _context(
        {
            "klein-gguf": SimpleNamespace(
                name="Klein 4B", format=ModelFormat.GGUFQuantized, variant=Flux2VariantType.Klein4B
            ),
            "qwen3-4b": SimpleNamespace(
                name="Qwen3 4B", format=ModelFormat.Diffusers, variant=Qwen3VariantType.Qwen3_4B
            ),
            "flux2-dev-diffusers": SimpleNamespace(
                name="FLUX.2 [dev]", format=ModelFormat.Diffusers, variant=Flux2VariantType.Dev
            ),
        }
    )

    output = invocation.invoke(context)

    # The VAE comes from the [dev] pipeline; the encoder comes from the standalone model.
    assert output.vae.vae.key == "flux2-dev-diffusers"
    assert output.vae.vae.submodel_type == SubModelType.VAE
    assert output.qwen3_encoder.text_encoder.key == "qwen3-4b"


def test_klein_loader_rejects_a_dev_pipeline_as_encoder_source() -> None:
    main = _identifier("klein-gguf")
    source = _identifier("flux2-dev-diffusers")

    invocation = Flux2KleinModelLoaderInvocation.model_construct(
        model=main,
        vae_model=None,
        qwen3_encoder_model=None,
        qwen3_source_model=source,
        max_seq_len=512,
    )
    context = _context(
        {
            "klein-gguf": SimpleNamespace(
                name="Klein 4B", format=ModelFormat.GGUFQuantized, variant=Flux2VariantType.Klein4B
            ),
            "flux2-dev-diffusers": SimpleNamespace(
                name="FLUX.2 [dev]", format=ModelFormat.Diffusers, variant=Flux2VariantType.Dev
            ),
        }
    )

    with pytest.raises(ValueError, match="must be a FLUX.2 Klein pipeline"):
        invocation.invoke(context)


def test_dev_loader_extracts_vae_from_a_klein_pipeline() -> None:
    """Mirror: [dev] GGUF main + standalone Mistral encoder + a Klein pipeline as the VAE source."""
    main = _identifier("dev-gguf")
    encoder = _identifier("mistral-small", ModelType.MistralEncoder)
    source = _identifier("klein-diffusers")

    invocation = Flux2DevModelLoaderInvocation.model_construct(
        model=main,
        vae_model=None,
        mistral_encoder_model=encoder,
        mistral_source_model=source,
        max_seq_len=512,
    )
    context = _context(
        {
            "dev-gguf": SimpleNamespace(
                name="FLUX.2 [dev]", format=ModelFormat.GGUFQuantized, variant=Flux2VariantType.Dev
            ),
            "klein-diffusers": SimpleNamespace(
                name="Klein 9B", format=ModelFormat.Diffusers, variant=Flux2VariantType.Klein9B
            ),
        }
    )

    output = invocation.invoke(context)

    assert output.vae.vae.key == "klein-diffusers"
    assert output.vae.vae.submodel_type == SubModelType.VAE
    assert output.mistral_encoder.text_encoder.key == "mistral-small"


def test_dev_loader_rejects_a_klein_pipeline_as_encoder_source() -> None:
    main = _identifier("dev-gguf")
    source = _identifier("klein-diffusers")

    invocation = Flux2DevModelLoaderInvocation.model_construct(
        model=main,
        vae_model=None,
        mistral_encoder_model=None,
        mistral_source_model=source,
        max_seq_len=512,
    )
    context = _context(
        {
            "dev-gguf": SimpleNamespace(
                name="FLUX.2 [dev]", format=ModelFormat.GGUFQuantized, variant=Flux2VariantType.Dev
            ),
            "klein-diffusers": SimpleNamespace(
                name="Klein 9B", format=ModelFormat.Diffusers, variant=Flux2VariantType.Klein9B
            ),
        }
    )

    with pytest.raises(ValueError, match=r"must be a FLUX.2 \[dev\] pipeline"):
        invocation.invoke(context)


@pytest.mark.parametrize("source_format", [ModelFormat.GGUFQuantized, ModelFormat.Checkpoint])
def test_vae_source_must_still_be_diffusers(source_format: ModelFormat) -> None:
    """Dropping the variant check from the VAE path must not drop the format check with it."""
    main = _identifier("klein-gguf")
    encoder = _identifier("qwen3-4b", ModelType.Qwen3Encoder)
    source = _identifier("not-diffusers")

    invocation = Flux2KleinModelLoaderInvocation.model_construct(
        model=main,
        vae_model=None,
        qwen3_encoder_model=encoder,
        qwen3_source_model=source,
        max_seq_len=512,
    )
    context = _context(
        {
            "klein-gguf": SimpleNamespace(
                name="Klein 4B", format=ModelFormat.GGUFQuantized, variant=Flux2VariantType.Klein4B
            ),
            "qwen3-4b": SimpleNamespace(
                name="Qwen3 4B", format=ModelFormat.Diffusers, variant=Qwen3VariantType.Qwen3_4B
            ),
            "not-diffusers": SimpleNamespace(
                name="Klein single-file", format=source_format, variant=Flux2VariantType.Klein4B
            ),
        }
    )

    with pytest.raises(ValueError, match="must be a Diffusers format model"):
        invocation.invoke(context)
