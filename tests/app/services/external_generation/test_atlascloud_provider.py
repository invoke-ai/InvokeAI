import io
import logging
from typing import Any, Iterator

import pytest
from PIL import Image

from invokeai.app.services.config.config_default import EXTERNAL_PROVIDER_CONFIG_FIELDS, InvokeAIAppConfig
from invokeai.app.services.external_generation.errors import (
    ExternalProviderRateLimitError,
    ExternalProviderRequestError,
)
from invokeai.app.services.external_generation.external_generation_common import ExternalGenerationRequest
from invokeai.app.services.external_generation.providers.atlascloud import AtlasCloudProvider
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig, ExternalModelCapabilities
from invokeai.backend.model_manager.starter_models import STARTER_MODELS
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType


class DummyResponse:
    def __init__(
        self,
        *,
        ok: bool,
        status_code: int = 200,
        json_data: dict[str, Any] | None = None,
        text: str = "",
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.ok = ok
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
        self.content = content
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        return self._json_data

    def iter_content(self, chunk_size: int = 65536) -> Iterator[bytes]:
        for index in range(0, len(self.content), chunk_size):
            yield self.content[index : index + chunk_size]

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color="blue").save(buffer, format="PNG")
    return buffer.getvalue()


def _model() -> ExternalApiModelConfig:
    return ExternalApiModelConfig(
        key="atlascloud_flux_schnell",
        name="Atlas Cloud FLUX.1 Schnell",
        provider_id="atlascloud",
        provider_model_id="black-forest-labs/flux-schnell",
        capabilities=ExternalModelCapabilities(
            modes=["txt2img"],
            supports_negative_prompt=False,
            supports_seed=True,
            max_images_per_request=4,
        ),
    )


def _starter_config(provider_model_id: str) -> ExternalApiModelConfig:
    """Build a model config from the registered starter model, so payloads are checked
    against the capabilities the app actually ships."""
    source = f"external://atlascloud/{provider_model_id}"
    starter = next(model for model in STARTER_MODELS if model.source == source)
    assert starter.capabilities is not None
    return ExternalApiModelConfig(
        key=provider_model_id,
        name=starter.name,
        provider_id="atlascloud",
        provider_model_id=provider_model_id,
        capabilities=starter.capabilities,
    )


def _request() -> ExternalGenerationRequest:
    return ExternalGenerationRequest(
        model=_model(),
        mode="txt2img",
        prompt="a blue square",
        seed=42,
        num_images=2,
        width=1024,
        height=768,
        image_size=None,
        init_image=None,
        mask_image=None,
        reference_images=[],
        metadata=None,
    )


def test_atlascloud_registration() -> None:
    assert "external_atlascloud_api_key" in EXTERNAL_PROVIDER_CONFIG_FIELDS
    assert "external_atlascloud_base_url" in EXTERNAL_PROVIDER_CONFIG_FIELDS
    starter = next(model for model in STARTER_MODELS if model.source.startswith("external://atlascloud/"))
    assert starter.source == "external://atlascloud/black-forest-labs/flux-schnell"
    assert starter.capabilities is not None
    assert starter.capabilities.max_images_per_request == 4


def test_atlascloud_is_configured() -> None:
    assert AtlasCloudProvider(
        InvokeAIAppConfig(external_atlascloud_api_key="test-key"), logging.getLogger("test")
    ).is_configured()
    assert not AtlasCloudProvider(InvokeAIAppConfig(), logging.getLogger("test")).is_configured()


def test_atlascloud_submit_poll_and_download(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = AtlasCloudProvider(InvokeAIAppConfig(external_atlascloud_api_key="atlas-key"), logging.getLogger("test"))
    captured: dict[str, object] = {}
    image_url = "https://cdn.atlascloud.ai/output.png"

    def fake_post(url: str, headers: dict[str, str], json: dict[str, object], timeout: int) -> DummyResponse:
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return DummyResponse(
            ok=True,
            json_data={
                "id": "prediction-1",
                "status": "queued",
                "urls": {"result": "/api/v1/model/result/prediction-1"},
            },
        )

    def fake_get(url: str, **kwargs: Any) -> DummyResponse:
        if url.endswith("/api/v1/model/result/prediction-1"):
            captured["poll_headers"] = kwargs["headers"]
            return DummyResponse(
                ok=True,
                json_data={
                    "code": 200,
                    "data": {"id": "prediction-1", "status": "completed", "outputs": [image_url]},
                },
            )
        assert url == image_url
        assert kwargs["stream"] is True
        return DummyResponse(ok=True, content=_png_bytes())

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.get", fake_get)

    result = provider.generate(_request())

    assert captured["url"] == "https://api.atlascloud.ai/api/v1/model/generateImage"
    assert captured["headers"] == {
        "Authorization": "Bearer atlas-key",
        "Content-Type": "application/json",
    }
    assert captured["json"] == {
        "model": "black-forest-labs/flux-schnell",
        "prompt": "a blue square",
        "size": "1024*768",
        "num_images": 2,
        "seed": 42,
    }
    assert captured["poll_headers"] == captured["headers"]
    assert len(result.images) == 1
    assert result.images[0].image.size == (16, 16)
    assert result.images[0].seed == 42
    assert result.provider_request_id == "prediction-1"


def test_atlascloud_uses_custom_base_url_and_fallback_poll_url(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = AtlasCloudProvider(
        InvokeAIAppConfig(
            external_atlascloud_api_key="atlas-key",
            external_atlascloud_base_url="https://proxy.atlas.test/",
        ),
        logging.getLogger("test"),
    )
    requested_urls: list[str] = []

    def fake_post(url: str, **_kwargs: Any) -> DummyResponse:
        requested_urls.append(url)
        return DummyResponse(ok=True, json_data={"code": 200, "data": {"id": "prediction-2", "status": "queued"}})

    def fake_get(url: str, **_kwargs: Any) -> DummyResponse:
        requested_urls.append(url)
        if url.endswith("/result/prediction-2"):
            return DummyResponse(
                ok=True,
                json_data={"id": "prediction-2", "status": "succeeded", "output": ["https://cdn.test/image.png"]},
            )
        return DummyResponse(ok=True, content=_png_bytes())

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.get", fake_get)

    provider.generate(_request())

    assert requested_urls[:2] == [
        "https://proxy.atlas.test/api/v1/model/generateImage",
        "https://proxy.atlas.test/api/v1/model/result/prediction-2",
    ]


def test_atlascloud_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = AtlasCloudProvider(InvokeAIAppConfig(external_atlascloud_api_key="atlas-key"), logging.getLogger("test"))
    monkeypatch.setattr(
        "requests.post",
        lambda *_args, **_kwargs: DummyResponse(
            ok=False,
            status_code=429,
            text="rate limited",
            headers={"Retry-After": "2.5"},
        ),
    )

    with pytest.raises(ExternalProviderRateLimitError) as error:
        provider.generate(_request())

    assert error.value.retry_after == 2.5


def test_atlascloud_failed_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = AtlasCloudProvider(InvokeAIAppConfig(external_atlascloud_api_key="atlas-key"), logging.getLogger("test"))
    monkeypatch.setattr(
        "requests.post",
        lambda *_args, **_kwargs: DummyResponse(ok=True, json_data={"id": "prediction-3", "status": "queued"}),
    )
    monkeypatch.setattr(
        "requests.get",
        lambda *_args, **_kwargs: DummyResponse(
            ok=True,
            json_data={"id": "prediction-3", "status": "failed", "logs": "content rejected"},
        ),
    )

    with pytest.raises(ExternalProviderRequestError, match="content rejected"):
        provider.generate(_request())


def test_atlascloud_requires_api_key() -> None:
    provider = AtlasCloudProvider(InvokeAIAppConfig(), logging.getLogger("test"))

    with pytest.raises(ExternalProviderRequestError, match="API key is not configured"):
        provider.generate(_request())


def test_atlascloud_starter_models_cover_multiple_models() -> None:
    atlas_models = [model for model in STARTER_MODELS if model.source.startswith("external://atlascloud/")]
    assert len(atlas_models) > 1
    sources = [model.source for model in atlas_models]
    assert len(sources) == len(set(sources))
    for model in atlas_models:
        assert model.base is BaseModelType.External
        assert model.type is ModelType.ExternalImageGenerator
        assert model.format is ModelFormat.ExternalApi
        assert model.capabilities is not None
        assert model.capabilities.modes == ["txt2img"]


@pytest.mark.parametrize(
    "provider_model_id, num_images, expected_extra",
    [
        # Explicit "<width>*<height>" size string
        ("black-forest-labs/flux-schnell", 2, {"size": "1024*768", "num_images": 2, "seed": 42}),
        ("black-forest-labs/flux-dev", 2, {"size": "1024*768", "num_images": 2, "seed": 42}),
        ("black-forest-labs/flux-2-pro/text-to-image", 1, {"size": "1024*768", "seed": 42}),
        # Batch size is spelled "n" here
        ("qwen-image-3.0/text-to-image", 2, {"size": "1024*768", "n": 2, "seed": 42}),
        ("z-image/turbo", 1, {"size": "1024*768", "seed": 42}),
        # No seed support upstream
        ("microsoft/mai-image-2.5/text-to-image", 1, {"size": "1024*768"}),
        # Named "image_size" preset
        ("ideogram/v4/turbo/text-to-image", 1, {"image_size": "landscape_4_3", "seed": 42}),
        ("ideogram/v4/quality/text-to-image", 1, {"image_size": "landscape_4_3", "seed": 42}),
        ("krea-2-turbo/text-to-image", 2, {"image_size": "landscape_4_3", "num_images": 2, "seed": 42}),
        ("hidream-o1-1.5/text-to-image", 1, {"image_size": "landscape_4_3"}),
        # "aspect_ratio" string
        ("xai/grok-imagine-image-2.0/text-to-image", 2, {"aspect_ratio": "4:3", "num_images": 2}),
        ("google/nano-banana-2/text-to-image", 1, {"aspect_ratio": "4:3", "seed": 42}),
    ],
)
def test_atlascloud_payload_matches_model_request_schema(
    provider_model_id: str,
    num_images: int,
    expected_extra: dict[str, object],
) -> None:
    """Each model only receives the request fields its upstream schema accepts."""
    provider = AtlasCloudProvider(InvokeAIAppConfig(external_atlascloud_api_key="atlas-key"), logging.getLogger("test"))
    request = ExternalGenerationRequest(
        model=_starter_config(provider_model_id),
        mode="txt2img",
        prompt="a blue square",
        seed=42,
        num_images=num_images,
        width=1024,
        height=768,
        image_size=None,
        init_image=None,
        mask_image=None,
        reference_images=[],
        metadata=None,
    )

    payload = provider._build_payload(request)

    assert payload == {"model": provider_model_id, "prompt": "a blue square", **expected_extra}


def test_atlascloud_resolution_preset_is_forwarded_lowercase() -> None:
    provider = AtlasCloudProvider(InvokeAIAppConfig(external_atlascloud_api_key="atlas-key"), logging.getLogger("test"))
    request = ExternalGenerationRequest(
        model=_starter_config("google/nano-banana-2/text-to-image"),
        mode="txt2img",
        prompt="a blue square",
        seed=None,
        num_images=1,
        width=1024,
        height=1024,
        image_size="2K",
        init_image=None,
        mask_image=None,
        reference_images=[],
        metadata=None,
    )

    payload = provider._build_payload(request)

    assert payload["aspect_ratio"] == "1:1"
    assert payload["resolution"] == "2k"


@pytest.mark.parametrize(
    "provider_model_id, width, height, expected_preset",
    [
        ("ideogram/v4/turbo/text-to-image", 1024, 1024, "square_hd"),
        ("ideogram/v4/turbo/text-to-image", 1920, 1080, "landscape_16_9"),
        # `portrait_9_16` and `portrait_16_9` both mean a 9:16 portrait, so the digit
        # order in the preset name must not flip the orientation.
        ("ideogram/v4/turbo/text-to-image", 1080, 1920, "portrait_9_16"),
        ("hidream-o1-1.5/text-to-image", 1080, 1920, "portrait_16_9"),
        ("hidream-o1-1.5/text-to-image", 1920, 1080, "landscape_16_9"),
    ],
)
def test_atlascloud_selects_closest_size_preset(
    provider_model_id: str,
    width: int,
    height: int,
    expected_preset: str,
) -> None:
    provider = AtlasCloudProvider(InvokeAIAppConfig(external_atlascloud_api_key="atlas-key"), logging.getLogger("test"))
    request = ExternalGenerationRequest(
        model=_starter_config(provider_model_id),
        mode="txt2img",
        prompt="a blue square",
        seed=None,
        num_images=1,
        width=width,
        height=height,
        image_size=None,
        init_image=None,
        mask_image=None,
        reference_images=[],
        metadata=None,
    )

    assert provider._build_payload(request)["image_size"] == expected_preset


def test_atlascloud_unknown_model_uses_explicit_size_schema() -> None:
    """Custom installs via `external://atlascloud/<model_id>` keep working."""
    provider = AtlasCloudProvider(InvokeAIAppConfig(external_atlascloud_api_key="atlas-key"), logging.getLogger("test"))
    model = ExternalApiModelConfig(
        key="custom",
        name="Custom Atlas Cloud Model",
        provider_id="atlascloud",
        provider_model_id="vendor/some-unlisted-model",
        capabilities=ExternalModelCapabilities(modes=["txt2img"], supports_seed=True),
    )
    request = ExternalGenerationRequest(
        model=model,
        mode="txt2img",
        prompt="a blue square",
        seed=7,
        num_images=1,
        width=512,
        height=512,
        image_size=None,
        init_image=None,
        mask_image=None,
        reference_images=[],
        metadata=None,
    )

    assert provider._build_payload(request) == {
        "model": "vendor/some-unlisted-model",
        "prompt": "a blue square",
        "size": "512*512",
        "num_images": 1,
        "seed": 7,
    }
