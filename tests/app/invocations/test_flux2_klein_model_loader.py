"""Tests for Flux2KleinModelLoaderInvocation submodel-source resolution.

Focus: a partial SDNQ FLUX.2 pipeline (format=sdnq_quantized whose submodels contains only the
transformer) must NOT be treated as a self-contained source. Otherwise the invocation sets
main_is_diffusers=True, skips the standalone VAE/Qwen3 requirement, and emits VAE/Tokenizer/
TextEncoder requests against the main model, moving the failure to runtime submodel loading.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

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
from tests.fixtures.model_roles import node_context, text_llm_qwen3_folder


def _make_main_model() -> ModelIdentifierField:
    return ModelIdentifierField(
        key="flux2-sdnq-key",
        hash="blake3:fakehash",
        name="FLUX.2 Klein SDNQ",
        base=BaseModelType.Flux2,
        type=ModelType.Main,
    )


def _make_context(config) -> MagicMock:
    context = MagicMock()
    context.models.get_config.return_value = config
    return context


class TestFlux2KleinSelfContainedSDNQ:
    def test_full_sdnq_pipeline_falls_back_to_main_model(self):
        invocation = Flux2KleinModelLoaderInvocation(model=_make_main_model())
        config = SimpleNamespace(
            format=ModelFormat.SDNQQuantized,
            submodels={
                SubModelType.Transformer: object(),
                SubModelType.VAE: object(),
                SubModelType.TextEncoder: object(),
                SubModelType.Tokenizer: object(),
            },
            name="FLUX.2 Klein SDNQ",
        )
        output = invocation.invoke(_make_context(config))
        assert output.vae.vae.submodel_type is SubModelType.VAE
        assert output.qwen3_encoder.text_encoder.submodel_type is SubModelType.TextEncoder
        assert output.qwen3_encoder.tokenizer.submodel_type is SubModelType.Tokenizer

    def test_partial_sdnq_pipeline_requires_explicit_source(self):
        """Only the transformer is recognized — not self-contained, so a VAE source is required."""
        invocation = Flux2KleinModelLoaderInvocation(model=_make_main_model())
        config = SimpleNamespace(
            format=ModelFormat.SDNQQuantized,
            submodels={SubModelType.Transformer: object()},  # missing VAE / TextEncoder / Tokenizer
            name="FLUX.2 Klein partial pipeline",
        )
        with pytest.raises(ValueError, match="No VAE source"):
            invocation.invoke(_make_context(config))

    def test_single_file_sdnq_requires_explicit_source(self):
        invocation = Flux2KleinModelLoaderInvocation(model=_make_main_model())
        config = SimpleNamespace(format=ModelFormat.SDNQQuantized, submodels=None, name="FLUX.2 single file")
        with pytest.raises(ValueError, match="No VAE source"):
            invocation.invoke(_make_context(config))


class TestQwen3TextLLMServingAsEncoder:
    """A Qwen3 folder installed as a Text LLM serves as the Klein encoder; the node asks for that role."""

    @staticmethod
    def _invoke(llm_variant: Qwen3VariantType | None):
        encoder = ModelIdentifierField(
            key="llm", hash="h-llm", name="llm folder", base=BaseModelType.Any, type=ModelType.TextLLM
        )
        vae = ModelIdentifierField(
            key="vae", hash="h-vae", name="FLUX.2 VAE", base=BaseModelType.Flux2, type=ModelType.VAE
        )
        configs = {
            "flux2-sdnq-key": SimpleNamespace(
                name="Klein 4B",
                type=ModelType.Main,
                format=ModelFormat.GGUFQuantized,
                variant=Flux2VariantType.Klein4B,
                submodels=None,
            ),
            "llm": text_llm_qwen3_folder("llm", llm_variant),
        }
        invocation = Flux2KleinModelLoaderInvocation(
            model=_make_main_model(), vae_model=vae, qwen3_encoder_model=encoder
        )
        return invocation.invoke(node_context(configs))  # type: ignore[arg-type]

    def test_the_matching_folder_is_emitted_as_a_qwen3_encoder(self):
        output = self._invoke(Qwen3VariantType.Qwen3_4B)

        for field in (output.qwen3_encoder.text_encoder, output.qwen3_encoder.tokenizer):
            assert (field.key, field.type) == ("llm", ModelType.Qwen3Encoder)

    def test_a_folder_of_the_wrong_size_is_refused_by_its_role_variant(self):
        with pytest.raises(ValueError, match="requires qwen3_4b encoder, but qwen3_8b was selected"):
            self._invoke(Qwen3VariantType.Qwen3_8B)

    def test_a_folder_without_the_role_is_refused_naming_the_selection(self):
        with pytest.raises(ValueError, match="'llm folder' is not a Qwen3 encoder.*or a Text LLM that serves as one"):
            self._invoke(None)
