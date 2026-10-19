"""Validation tests for the `device` config field.

Note these construct the config rather than assigning to an existing instance: the model does
not enable `validate_assignment`, so `config.device = ...` bypasses the pattern entirely.
"""

import pytest
from pydantic import ValidationError

from invokeai.app.services.config.config_default import InvokeAIAppConfig


@pytest.mark.parametrize(
    "value",
    ["auto", "cpu", "mps", "cuda", "cuda:0", "cuda:1", "xpu", "xpu:0", "xpu:1"],
)
def test_valid_device(value):
    assert InvokeAIAppConfig(device=value).device == value


@pytest.mark.parametrize(
    "value",
    ["gpu0", "xpu:", "xpu:x", "xpu0", "cuda:", "cuda:x", "XPU:0", "xpu:0:1"],
)
def test_invalid_device_is_rejected(value):
    with pytest.raises(ValidationError):
        InvokeAIAppConfig(device=value)
