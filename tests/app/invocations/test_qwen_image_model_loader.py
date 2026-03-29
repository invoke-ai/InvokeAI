"""Tests for the Qwen Image model loader invocation."""

from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.qwen_image_model_loader import QwenImageModelLoaderInvocation
from invokeai.backend.model_manager.taxonomy import ModelFormat, SubModelType


def _make_model_id(**kwargs) -> ModelIdentifierField:
    defaults = {"key": "test-key", "hash": "test-hash", "name": "test", "base": "qwen-image", "type": "main"}
    defaults.update(kwargs)
    return ModelIdentifierField(**defaults)


def _make_mock_context(main_format: ModelFormat = ModelFormat.Diffusers, source_format: ModelFormat = ModelFormat.Diffusers):
    """Create a mock InvocationContext that returns configs with the given formats."""
    context = MagicMock()

    def get_config(model_id):
        config = MagicMock()
        if model_id.key == "main-key":
            config.format = main_format
            config.name = "Main Model"
        elif model_id.key == "source-key":
            config.format = source_format
            config.name = "Source Model"
        return config

    context.models.get_config = get_config
    context.models.exists = MagicMock(return_value=True)
    return context


class TestDiffusersModel:
    """Tests for loading a Diffusers-format Qwen Image model."""

    def test_diffusers_model_extracts_all_components(self):
        """A Diffusers model should extract transformer, VAE, tokenizer, and text encoder from itself."""
        model_id = _make_model_id(key="main-key")
        inv = QwenImageModelLoaderInvocation.model_construct(model=model_id, component_source=None)
        context = _make_mock_context(main_format=ModelFormat.Diffusers)

        result = inv.invoke(context)

        assert result.transformer.transformer.submodel_type == SubModelType.Transformer
        assert result.vae.vae.submodel_type == SubModelType.VAE
        assert result.qwen_vl_encoder.tokenizer.submodel_type == SubModelType.Tokenizer
        assert result.qwen_vl_encoder.text_encoder.submodel_type == SubModelType.TextEncoder

        # All should reference the main model key
        assert result.transformer.transformer.key == "main-key"
        assert result.vae.vae.key == "main-key"
        assert result.qwen_vl_encoder.tokenizer.key == "main-key"
        assert result.qwen_vl_encoder.text_encoder.key == "main-key"

    def test_diffusers_model_ignores_component_source(self):
        """A Diffusers model should ignore the component_source even if provided."""
        model_id = _make_model_id(key="main-key")
        source_id = _make_model_id(key="source-key")
        inv = QwenImageModelLoaderInvocation.model_construct(model=model_id, component_source=source_id)
        context = _make_mock_context(main_format=ModelFormat.Diffusers)

        result = inv.invoke(context)

        # All components should come from main, not source
        assert result.vae.vae.key == "main-key"
        assert result.qwen_vl_encoder.tokenizer.key == "main-key"


class TestGGUFModel:
    """Tests for loading a GGUF-format Qwen Image model."""

    def test_gguf_with_component_source_succeeds(self):
        """A GGUF model with a Diffusers component source should load successfully."""
        model_id = _make_model_id(key="main-key")
        source_id = _make_model_id(key="source-key")
        inv = QwenImageModelLoaderInvocation.model_construct(model=model_id, component_source=source_id)
        context = _make_mock_context(main_format=ModelFormat.GGUFQuantized, source_format=ModelFormat.Diffusers)

        result = inv.invoke(context)

        # Transformer from main model
        assert result.transformer.transformer.key == "main-key"
        assert result.transformer.transformer.submodel_type == SubModelType.Transformer

        # VAE and encoder from component source
        assert result.vae.vae.key == "source-key"
        assert result.qwen_vl_encoder.tokenizer.key == "source-key"
        assert result.qwen_vl_encoder.text_encoder.key == "source-key"

    def test_gguf_without_component_source_raises(self):
        """A GGUF model without a component source should raise ValueError."""
        model_id = _make_model_id(key="main-key")
        inv = QwenImageModelLoaderInvocation.model_construct(model=model_id, component_source=None)
        context = _make_mock_context(main_format=ModelFormat.GGUFQuantized)

        with pytest.raises(ValueError, match="No source for VAE"):
            inv.invoke(context)

    def test_gguf_with_non_diffusers_source_raises(self):
        """A GGUF model with a non-Diffusers component source should raise ValueError."""
        model_id = _make_model_id(key="main-key")
        source_id = _make_model_id(key="source-key")
        inv = QwenImageModelLoaderInvocation.model_construct(model=model_id, component_source=source_id)
        context = _make_mock_context(main_format=ModelFormat.GGUFQuantized, source_format=ModelFormat.GGUFQuantized)

        with pytest.raises(ValueError, match="Component Source model must be in Diffusers format"):
            inv.invoke(context)
