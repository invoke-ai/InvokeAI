import pytest
from pydantic import ValidationError

from invokeai.backend.model_manager.configs.external_api import (
    ExternalApiModelConfig,
    ExternalApiModelDefaultSettings,
    ExternalImageSize,
    ExternalModelCapabilities,
)


def test_external_api_model_config_defaults() -> None:
    capabilities = ExternalModelCapabilities(modes=["txt2img"], supports_seed=True)

    config = ExternalApiModelConfig(
        name="Test External",
        provider_id="openai",
        provider_model_id="gpt-image-1",
        capabilities=capabilities,
    )

    assert config.path == "external://openai/gpt-image-1"
    assert config.source == "external://openai/gpt-image-1"
    assert config.hash == "external:openai:gpt-image-1"
    assert config.file_size == 0
    assert config.default_settings is None
    assert config.capabilities.supports_seed is True


def test_external_api_model_capabilities_allows_aspect_ratio_sizes() -> None:
    capabilities = ExternalModelCapabilities(
        modes=["txt2img"],
        allowed_aspect_ratios=["1:1"],
        aspect_ratio_sizes={"1:1": ExternalImageSize(width=1024, height=1024)},
    )

    assert capabilities.aspect_ratio_sizes is not None
    assert capabilities.aspect_ratio_sizes["1:1"].width == 1024


def test_external_api_model_config_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ExternalModelCapabilities(modes=["txt2img"], supports_seed=True, extra_field=True)  # type: ignore

    with pytest.raises(ValidationError):
        ExternalApiModelDefaultSettings(width=512, extra_field=True)  # type: ignore


def test_external_api_model_config_validates_limits() -> None:
    with pytest.raises(ValidationError):
        ExternalModelCapabilities(modes=["txt2img"], max_images_per_request=0)

    with pytest.raises(ValidationError):
        ExternalApiModelDefaultSettings(width=0)
