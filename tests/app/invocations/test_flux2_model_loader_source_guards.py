"""The FLUX.2 loaders' cross-variant guard must gate the *encoder* extraction path only.

Klein and [dev] share the same 32-channel `AutoencoderKLFlux2` — the repo ships the Klein-sourced
`flux2_vae` as a dependency of every [dev] GGUF starter model — so a cross-variant pipeline is a
legitimate VAE source. In the *Klein* linear UI that is load-bearing: `buildFLUXGraph` falls back to
any FLUX.2 diffusers pipeline when only the VAE is needed, and `isFlux2DiffusersMainModelConfig`
does not filter by variant, so a variant guard on the VAE-extraction call site would turn a working
configuration (GGUF main + standalone encoder + a cross-variant diffusers pipeline as the only VAE
source) into a runtime `ValueError` behind an enabled Invoke button. The [dev] linear UI is
stricter — it sources from dev-only pipelines and readiness gates on one — so there the same
configuration is reachable through the workflow editor rather than the linear UI. Either way the
backend stays permissive on the VAE path.

The encoder path must still reject cross-variant sources, because a mismatched tokenizer/encoder
only surfaces as an opaque matmul error deep in denoise. That includes Klein-vs-Klein mismatches:
`qwen3_source_model` is not variant-filtered in the workflow editor, so a 9B pipeline can be wired
in as the encoder source for a 4B transformer.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.flux2.flux2_dev_model_loader import Flux2DevModelLoaderInvocation
from invokeai.app.invocations.flux2.flux2_klein_model_loader import Flux2KleinModelLoaderInvocation
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


@pytest.mark.parametrize(
    ("main_variant", "source_variant"),
    [
        (Flux2VariantType.Klein4B, Flux2VariantType.Klein9B),
        (Flux2VariantType.Klein9B, Flux2VariantType.Klein4BBase),
    ],
)
def test_klein_loader_rejects_a_mismatched_klein_pipeline_as_encoder_source(
    main_variant: Flux2VariantType, source_variant: Flux2VariantType
) -> None:
    """Rejecting only [dev] left the Klein-vs-Klein case open: a 9B pipeline carries a Qwen3 8B
    encoder, which produces wrong-width conditioning against a 4B transformer and fails as an
    opaque matmul error deep in denoise. The frontend and the standalone-encoder path both enforce
    the family match, so the workflow editor's `qwen3_source_model` was the only way in."""
    main = _identifier("klein-gguf")
    source = _identifier("klein-diffusers")

    invocation = Flux2KleinModelLoaderInvocation.model_construct(
        model=main,
        vae_model=None,
        qwen3_encoder_model=None,
        qwen3_source_model=source,
        max_seq_len=512,
    )
    context = _context(
        {
            "klein-gguf": SimpleNamespace(name="Klein main", format=ModelFormat.GGUFQuantized, variant=main_variant),
            "klein-diffusers": SimpleNamespace(
                name="Klein source", format=ModelFormat.Diffusers, variant=source_variant
            ),
        }
    )

    with pytest.raises(ValueError, match="Qwen3 encoder variant mismatch"):
        invocation.invoke(context)


def test_klein_loader_accepts_a_same_family_klein_pipeline_as_encoder_source() -> None:
    """The distilled and Base variants of one size share a Qwen3 encoder, so they remain valid
    sources for each other — the guard matches on the Qwen3 family, not on the exact variant."""
    main = _identifier("klein-gguf")
    source = _identifier("klein-diffusers")

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
                name="Klein 9B", format=ModelFormat.GGUFQuantized, variant=Flux2VariantType.Klein9B
            ),
            "klein-diffusers": SimpleNamespace(
                name="Klein 9B Base", format=ModelFormat.Diffusers, variant=Flux2VariantType.Klein9BBase
            ),
        }
    )

    output = invocation.invoke(context)

    assert output.qwen3_encoder.text_encoder.key == "klein-diffusers"
    assert output.qwen3_encoder.text_encoder.submodel_type == SubModelType.TextEncoder
    assert output.vae.vae.key == "klein-diffusers"


def test_klein_loader_rejects_a_standalone_encoder_from_the_wrong_family() -> None:
    """The standalone-encoder guard shares the Klein->Qwen3 map with the source guard; it had no
    negative-path coverage at all, so a regression there would have been silent."""
    main = _identifier("klein-gguf")
    encoder = _identifier("qwen3-8b", ModelType.Qwen3Encoder)
    source = _identifier("klein-diffusers")

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
            "qwen3-8b": SimpleNamespace(
                name="Qwen3 8B", format=ModelFormat.Diffusers, variant=Qwen3VariantType.Qwen3_8B
            ),
            "klein-diffusers": SimpleNamespace(
                name="Klein 4B", format=ModelFormat.Diffusers, variant=Flux2VariantType.Klein4B
            ),
        }
    )

    with pytest.raises(ValueError, match="Qwen3 encoder variant mismatch"):
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

    # The Klein loader also accepts a self-contained SDNQ pipeline as a source, so its message names
    # the accepted shape ("Diffusers-style FLUX.2 pipeline") rather than the diffusers format alone.
    # Single-file / GGUF sources are still rejected, which is what this test is about.
    with pytest.raises(ValueError, match="must be a Diffusers-style FLUX.2 pipeline"):
        invocation.invoke(context)
