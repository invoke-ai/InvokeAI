"""Tests for ZImageModelLoaderInvocation submodel-source resolution.

Focus: a freshly installed SDNQ Z-Image pipeline (format=sdnq_quantized with submodels) is
self-contained and must generate without the user manually selecting a VAE / Qwen3 component
source. In that case the loader must fall back to the main model itself for the VAE and Qwen3
encoder/tokenizer submodels.
"""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.z_image.z_image_model_loader import ZImageModelLoaderInvocation
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
    Qwen3VariantType,
    SubModelType,
)


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
        # SDNQ pipeline config exposing all required submodels — self-contained.
        config = SimpleNamespace(
            format=ModelFormat.SDNQQuantized,
            submodels={
                SubModelType.Transformer: object(),
                SubModelType.VAE: object(),
                SubModelType.TextEncoder: object(),
                SubModelType.Tokenizer: object(),
            },
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

    def test_partial_sdnq_pipeline_is_not_self_contained(self):
        """A pipeline that only exposes some submodels (e.g. just the transformer) must not be
        treated as self-contained — the loader would fail on the missing vae / text_encoder folders."""
        invocation = ZImageModelLoaderInvocation(model=_make_main_model())
        config = SimpleNamespace(
            format=ModelFormat.SDNQQuantized,
            submodels={SubModelType.Transformer: object()},  # missing VAE / TextEncoder / Tokenizer
            name="Z-Image partial pipeline",
        )
        context = _make_context(config)

        with pytest.raises(ValueError, match="No VAE source"):
            invocation.invoke(context)

    def test_validate_diffusers_format_rejects_partial_sdnq_source(self):
        """A partial SDNQ pipeline selected as the explicit Qwen3 Source must be rejected — otherwise
        the invocation would emit VAE/text_encoder/tokenizer requests against missing subfolders."""
        invocation = ZImageModelLoaderInvocation(model=_make_main_model())
        source = ModelIdentifierField(
            key="src", hash="h", name="partial", base=BaseModelType.ZImage, type=ModelType.Main
        )
        partial = SimpleNamespace(
            format=ModelFormat.SDNQQuantized,
            submodels={SubModelType.Transformer: object()},
            name="Z-Image partial pipeline",
        )
        context = _make_context(partial)

        with pytest.raises(ValueError, match="must be a Diffusers-style Z-Image pipeline"):
            invocation._validate_diffusers_format(context, source, "Qwen3 Source")

    def test_non_sdnq_model_still_requires_source(self):
        """A GGUF Z-Image model must still require an explicit VAE / Qwen3 source."""
        invocation = ZImageModelLoaderInvocation(model=_make_main_model())
        config = SimpleNamespace(format=ModelFormat.GGUFQuantized, submodels=None, name="Z-Image GGUF")
        context = _make_context(config)

        with pytest.raises(ValueError, match="No VAE source"):
            invocation.invoke(context)


class TestStandaloneQwen3EncoderVariant:
    """Every Qwen3 size installs as one model type, but Z-Image's caption embedder only takes the 4B
    width. A workflow can wire any of them, so the loader refuses the others by name."""

    @staticmethod
    def _invoke(encoder_config: SimpleNamespace):
        encoder = ModelIdentifierField(
            key="encoder",
            hash="h-encoder",
            name=encoder_config.name,
            base=BaseModelType.Any,
            type=ModelType.Qwen3Encoder,
        )
        vae = ModelIdentifierField(
            key="vae", hash="h-vae", name="FLUX VAE", base=BaseModelType.Flux, type=ModelType.VAE
        )
        configs = {
            "z-image-sdnq-key": SimpleNamespace(format=ModelFormat.GGUFQuantized, submodels=None, name="Z-Image GGUF"),
            "encoder": encoder_config,
        }
        context = MagicMock()
        context.models.get_config.side_effect = lambda identifier: configs[identifier.key]
        invocation = ZImageModelLoaderInvocation(model=_make_main_model(), vae_model=vae, qwen3_encoder_model=encoder)
        return invocation.invoke(context)

    def test_qwen3_4b_encoder_is_passed_through(self):
        output = self._invoke(
            SimpleNamespace(type=ModelType.Qwen3Encoder, variant=Qwen3VariantType.Qwen3_4B, name="Qwen3 4B")
        )

        assert output.qwen3_encoder.text_encoder.key == "encoder"
        assert output.qwen3_encoder.text_encoder.submodel_type is SubModelType.TextEncoder
        assert output.qwen3_encoder.tokenizer.key == "encoder"

    @pytest.mark.parametrize("variant", [Qwen3VariantType.Qwen3_8B, Qwen3VariantType.Qwen3_06B])
    def test_other_qwen3_sizes_are_refused_by_name(self, variant: Qwen3VariantType):
        config = SimpleNamespace(type=ModelType.Qwen3Encoder, variant=variant, name="Picked Encoder")
        expected = (
            f"'Picked Encoder' is a Qwen3 encoder of variant '{variant.value}'. Z-Image conditions on the Qwen3 4B "
            "encoder (hidden size 2560): select a Qwen3 4B encoder, or clear it and set 'Qwen3 Source' to a "
            "Z-Image pipeline."
        )

        with pytest.raises(ValueError, match=re.escape(expected)):
            self._invoke(config)

    def test_a_non_qwen3_model_in_the_encoder_slot_is_refused(self):
        config = SimpleNamespace(type=ModelType.Qwen3VLEncoder, variant="qwen3_vl_4b", name="Qwen3-VL 4B")

        with pytest.raises(ValueError, match=re.escape("'Qwen3-VL 4B' is not a Qwen3 encoder. Z-Image conditions")):
            self._invoke(config)


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
