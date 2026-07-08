"""Tests for ZImageModelLoaderInvocation submodel-source resolution.

Focus: a freshly installed SDNQ Z-Image pipeline (format=sdnq_quantized with submodels) is
self-contained and must generate without the user manually selecting a VAE / Qwen3 component
source. In that case the loader must fall back to the main model itself for the VAE and Qwen3
encoder/tokenizer submodels.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.z_image_model_loader import ZImageModelLoaderInvocation
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, SubModelType


def _make_main_model() -> ModelIdentifierField:
    return ModelIdentifierField(
        key="z-image-sdnq-key",
        hash="blake3:fakehash",
        name="Z-Image Turbo SDNQ",
        base=BaseModelType.ZImage,
        type=ModelType.Main,
    )


def _make_context(config) -> MagicMock:
    context = MagicMock()
    context.models.get_config.return_value = config
    return context


class TestZImageSelfContainedSDNQ:
    def test_sdnq_pipeline_with_submodels_falls_back_to_main_model(self):
        main_model = _make_main_model()
        invocation = ZImageModelLoaderInvocation(model=main_model)
        # SDNQ pipeline config exposing submodels — self-contained.
        config = SimpleNamespace(
            format=ModelFormat.SDNQQuantized,
            submodels={SubModelType.VAE: object(), SubModelType.TextEncoder: object()},
            name="Z-Image Turbo SDNQ",
        )
        context = _make_context(config)

        output = invocation.invoke(context)

        # Transformer, VAE and Qwen3 encoder/tokenizer all resolve to the main model's key.
        assert output.transformer.transformer.key == main_model.key
        assert output.transformer.transformer.submodel_type is SubModelType.Transformer
        assert output.vae.vae.key == main_model.key
        assert output.vae.vae.submodel_type is SubModelType.VAE
        assert output.qwen3_encoder.text_encoder.key == main_model.key
        assert output.qwen3_encoder.text_encoder.submodel_type is SubModelType.TextEncoder
        assert output.qwen3_encoder.tokenizer.key == main_model.key
        assert output.qwen3_encoder.tokenizer.submodel_type is SubModelType.Tokenizer

    def test_sdnq_without_submodels_still_requires_source(self):
        """A single-file SDNQ Z-Image model (no submodels) is not self-contained."""
        invocation = ZImageModelLoaderInvocation(model=_make_main_model())
        config = SimpleNamespace(format=ModelFormat.SDNQQuantized, submodels=None, name="Z-Image single file")
        context = _make_context(config)

        with pytest.raises(ValueError, match="No VAE source"):
            invocation.invoke(context)

    def test_non_sdnq_model_still_requires_source(self):
        """A GGUF Z-Image model must still require an explicit VAE / Qwen3 source."""
        invocation = ZImageModelLoaderInvocation(model=_make_main_model())
        config = SimpleNamespace(format=ModelFormat.GGUFQuantized, submodels=None, name="Z-Image GGUF")
        context = _make_context(config)

        with pytest.raises(ValueError, match="No VAE source"):
            invocation.invoke(context)


class TestQwen3SourceFieldTemplate:
    """The generic node/workflow model pickers filter candidates by the field template's
    ui_model_base / ui_model_type / ui_model_format hints. Since _validate_diffusers_format now
    accepts SDNQ pipeline configs (with submodels) as a Qwen3 source, the qwen3_source_model field
    must NOT pin ui_model_format=diffusers — otherwise SDNQ Z-Image pipelines are filtered out of
    the picker even though the backend supports them.
    """

    def test_qwen3_source_field_has_no_format_filter(self):
        schema = ZImageModelLoaderInvocation.model_json_schema()
        field = schema["properties"]["qwen3_source_model"]

        # No format constraint: an SDNQ z-image main model (format=sdnq_quantized) is not filtered.
        assert "ui_model_format" not in field
        # Still scoped to Z-Image main models so unrelated models don't show up.
        assert field["ui_model_base"] == [BaseModelType.ZImage.value]
        assert field["ui_model_type"] == [ModelType.Main.value]

    def test_sdnq_pipeline_passes_field_template_filter(self):
        """Reproduce the picker's filter predicate and confirm an SDNQ pipeline is kept."""
        schema = ZImageModelLoaderInvocation.model_json_schema()
        field = schema["properties"]["qwen3_source_model"]

        sdnq_pipeline_config = {
            "base": BaseModelType.ZImage.value,
            "type": ModelType.Main.value,
            "format": ModelFormat.SDNQQuantized.value,
        }

        # Mirror the filter in ModelIdentifierFieldInputComponent / WorkflowFieldRenderer.
        def is_filtered_out(config: dict) -> bool:
            if "ui_model_base" in field and config["base"] not in field["ui_model_base"]:
                return True
            if "ui_model_type" in field and config["type"] not in field["ui_model_type"]:
                return True
            if "ui_model_format" in field and config["format"] not in field["ui_model_format"]:
                return True
            return False

        assert not is_filtered_out(sdnq_pipeline_config)
