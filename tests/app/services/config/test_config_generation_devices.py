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
    (see TorchDevice.get_generation_devices), so every user-facing description of `auto` must
    disclose that exception instead of promising "every available GPU" unconditionally. Checks the
    schema description (source of the API docs and generated settings docs), the generated docs
    settings.json, the Settings UI copy, and the configuration guide's behavior table."""
    import json
    from pathlib import Path

    repo_root = Path(__file__).parents[4]

    field_description = InvokeAIAppConfig.model_fields["generation_devices"].description
    assert field_description is not None
    assert "legacy `device`" in field_description

    generated_settings = json.loads((repo_root / "docs/src/generated/settings.json").read_text())
    generated_description = next(
        s["description"] for s in generated_settings["settings"] if s["name"] == "generation_devices"
    )
    assert "legacy `device`" in generated_description

    locales = json.loads((repo_root / "invokeai/frontend/web/public/locales/en.json").read_text(encoding="utf-8"))
    assert locales["settings"]["generationDevicesAuto"] == "Auto"  # not "Auto (all GPUs)"
    assert "'device'" in locales["settings"]["generationDevicesHelp"]

    guide = (repo_root / "docs/src/content/docs/configuration/invokeai-yaml.mdx").read_text(encoding="utf-8")
    auto_row = next(line for line in guide.splitlines() if line.startswith("| `auto`"))
    assert "legacy `device`" in auto_row


@pytest.mark.parametrize("value", [["xpu:x"], ["xpu:"], ["xpu0"]])
def test_malformed_xpu_device_is_rejected(value):
    with pytest.raises(ValidationError):
        InvokeAIAppConfig(generation_devices=value)
