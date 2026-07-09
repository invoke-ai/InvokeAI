"""Tests for Flux2KleinModelLoaderInvocation submodel-source resolution.

Focus: a partial SDNQ FLUX.2 pipeline (format=sdnq_quantized whose submodels contains only the
transformer) must NOT be treated as a self-contained source. Otherwise the invocation sets
main_is_diffusers=True, skips the standalone VAE/Qwen3 requirement, and emits VAE/Tokenizer/
TextEncoder requests against the main model, moving the failure to runtime submodel loading.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.flux2_klein_model_loader import Flux2KleinModelLoaderInvocation
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, SubModelType


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
