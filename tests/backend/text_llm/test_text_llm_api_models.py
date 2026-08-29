"""Tests for TextLLM API request/response models and validation."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from pydantic import ValidationError

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers.utilities import (
    ExpandPromptRequest,
    ExpandPromptResponse,
    ImageToPromptRequest,
    _run_expand_prompt,
)
from invokeai.backend.model_manager.taxonomy import ModelType


class TestExpandPromptRequest:
    def test_defaults(self):
        req = ExpandPromptRequest(prompt="a cat", model_key="abc-123")
        assert req.max_tokens == 300
        assert req.system_prompt is None
        assert req.seed is None

    def test_max_tokens_upper_bound(self):
        """max_tokens should be capped at 2048."""
        with pytest.raises(ValidationError):
            ExpandPromptRequest(prompt="a cat", model_key="abc-123", max_tokens=2049)

    def test_max_tokens_lower_bound(self):
        """max_tokens must be at least 1."""
        with pytest.raises(ValidationError):
            ExpandPromptRequest(prompt="a cat", model_key="abc-123", max_tokens=0)

    def test_max_tokens_valid_range(self):
        req = ExpandPromptRequest(prompt="a cat", model_key="abc-123", max_tokens=2048)
        assert req.max_tokens == 2048
        req2 = ExpandPromptRequest(prompt="a cat", model_key="abc-123", max_tokens=1)
        assert req2.max_tokens == 1

    def test_custom_system_prompt(self):
        req = ExpandPromptRequest(prompt="a cat", model_key="abc-123", system_prompt="Be brief.")
        assert req.system_prompt == "Be brief."

    def test_seed_range(self):
        assert ExpandPromptRequest(prompt="a cat", model_key="abc-123", seed=42).seed == 42
        with pytest.raises(ValidationError):
            ExpandPromptRequest(prompt="a cat", model_key="abc-123", seed=-1)


class TestImageToPromptRequest:
    def test_defaults(self):
        req = ImageToPromptRequest(image_name="img.png", model_key="abc-123")
        assert "Describe" in req.instruction

    def test_custom_instruction(self):
        req = ImageToPromptRequest(image_name="img.png", model_key="abc-123", instruction="What is this?")
        assert req.instruction == "What is this?"


class TestExpandPromptResponse:
    def test_success_response(self):
        resp = ExpandPromptResponse(expanded_prompt="A detailed scene", seed=42)
        assert resp.expanded_prompt == "A detailed scene"
        assert resp.seed == 42
        assert resp.error is None

    def test_error_response(self):
        resp = ExpandPromptResponse(expanded_prompt="", seed=42, error="Model failed")
        assert resp.error == "Model failed"


def test_expand_prompt_uses_fresh_seed() -> None:
    model_config = MagicMock(type=ModelType.TextLLM, path="model")
    model = MagicMock()
    model.parameters.side_effect = lambda: iter([torch.nn.Parameter(torch.zeros(1))])
    loaded_model = MagicMock()
    loaded_model.model_on_device.return_value.__enter__.return_value = (None, model)
    services = MagicMock()
    services.model_manager.store.get_model.return_value = model_config
    services.model_manager.load.load_model.return_value = loaded_model

    with (
        patch.object(ApiDependencies, "invoker", MagicMock(services=services), create=True),
        patch("invokeai.app.api.routers.utilities._resolve_model_path", return_value="model"),
        patch("invokeai.app.api.routers.utilities.AutoTokenizer.from_pretrained"),
        patch("invokeai.app.api.routers.utilities.get_random_seed", return_value=123),
        patch("invokeai.app.api.routers.utilities.TextLLMPipeline") as pipeline_class,
    ):
        pipeline_class.return_value.run.return_value = "expanded"
        assert _run_expand_prompt("cat", "model", 10, None, None, None, "user") == ("expanded", 123)
        assert pipeline_class.return_value.run.call_args.kwargs["seed"] == 123

        pipeline_class.return_value.run.reset_mock()
        assert _run_expand_prompt("cat", "model", 10, None, 456, None, "user") == ("expanded", 456)
        assert pipeline_class.return_value.run.call_args.kwargs["seed"] == 456
