"""Tests for TextLLM API request/response models and validation."""

import pytest
from pydantic import ValidationError

from invokeai.app.api.routers.utilities import ExpandPromptRequest, ExpandPromptResponse, ImageToPromptRequest


class TestExpandPromptRequest:
    def test_defaults(self):
        req = ExpandPromptRequest(prompt="a cat", model_key="abc-123")
        assert req.max_tokens == 300
        assert req.system_prompt is None

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


class TestImageToPromptRequest:
    def test_defaults(self):
        req = ImageToPromptRequest(image_name="img.png", model_key="abc-123")
        assert "Describe" in req.instruction

    def test_custom_instruction(self):
        req = ImageToPromptRequest(image_name="img.png", model_key="abc-123", instruction="What is this?")
        assert req.instruction == "What is this?"


class TestExpandPromptResponse:
    def test_success_response(self):
        resp = ExpandPromptResponse(expanded_prompt="A detailed scene")
        assert resp.expanded_prompt == "A detailed scene"
        assert resp.error is None

    def test_error_response(self):
        resp = ExpandPromptResponse(expanded_prompt="", error="Model failed")
        assert resp.error == "Model failed"
