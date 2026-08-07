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
