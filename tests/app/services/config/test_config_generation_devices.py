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


def test_auto_copy_documents_legacy_device_precedence():
    """`generation_devices: auto` resolves to the single pinned legacy `device` when one is set
    (see TorchDevice.get_generation_devices), so the schema description must disclose that
    exception instead of promising "every available GPU" unconditionally. The API docs and the
    generated settings docs derive from this description; the docs workflow's check-docs-data
    step fails when the generated copy is stale. UI and guide copy belong to their own packages."""
    field_description = InvokeAIAppConfig.model_fields["generation_devices"].description
    assert field_description is not None
    assert "legacy `device`" in field_description


@pytest.mark.parametrize("value", [["xpu:x"], ["xpu:"], ["xpu0"]])
def test_malformed_xpu_device_is_rejected(value):
    with pytest.raises(ValidationError):
        InvokeAIAppConfig(generation_devices=value)
