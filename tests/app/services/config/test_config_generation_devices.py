"""Validation tests for the multi-GPU `generation_devices` config field."""

import pytest
from pydantic import ValidationError

from invokeai.app.services.config.config_default import InvokeAIAppConfig


@pytest.mark.parametrize(
    "value",
    [
        "auto",
        ["cuda:0"],
        ["cuda:0", "cuda:1"],
        ["cpu"],
        ["mps"],
        ["cuda"],
        ["xpu"],
        ["xpu:0"],
        ["xpu:0", "xpu:1"],
    ],
)
def test_valid_generation_devices(value):
    cfg = InvokeAIAppConfig(generation_devices=value)
    assert cfg.generation_devices == value


def test_non_auto_string_is_rejected():
    # A bare string (other than "auto") would otherwise be iterated character-by-character.
    with pytest.raises(ValidationError):
        InvokeAIAppConfig(generation_devices="cuda:0")


def test_empty_list_is_rejected():
    with pytest.raises(ValidationError):
        InvokeAIAppConfig(generation_devices=[])


def test_invalid_device_name_is_rejected():
    with pytest.raises(ValidationError):
        InvokeAIAppConfig(generation_devices=["gpu0"])


@pytest.mark.parametrize("value", [["xpu:x"], ["xpu:"], ["xpu0"]])
def test_malformed_xpu_device_is_rejected(value):
    with pytest.raises(ValidationError):
        InvokeAIAppConfig(generation_devices=value)
