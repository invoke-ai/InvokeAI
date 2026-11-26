import base64
import io
import logging
from typing import Any

import pytest
from PIL import Image

from invokeai.app.services.config.config_default import EXTERNAL_PROVIDER_CONFIG_FIELDS, InvokeAIAppConfig
from invokeai.app.services.external_generation.errors import (
    ExternalProviderRateLimitError,
    ExternalProviderRequestError,
)
from invokeai.app.services.external_generation.external_generation_common import ExternalGenerationRequest
from invokeai.app.services.external_generation.external_generation_default import ExternalGenerationService
from invokeai.app.services.external_generation.providers.martllm import MartLLMProvider
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig
from invokeai.backend.model_manager.starter_models import STARTER_MODELS

# The `size` values every MartLLM image model accepts (from each model's request schema).
MARTLLM_ACCEPTED_SIZES = {"1024x1024", "1024x768", "768x1024", "1536x1024", "1024x1536"}


class DummyResponse:
    def __init__(
        self,
        *,
        ok: bool,
        status_code: int = 200,
        json_data: Any = None,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.ok = ok
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.headers = headers or {}

    def json(self) -> Any:
        if self._json_data is None:
            raise ValueError("no JSON body")
        return self._json_data


def _png_base64(width: int = 16, height: int = 16) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color="red").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _starter_config(provider_model_id: str) -> ExternalApiModelConfig:
    source = f"external://martllm/{provider_model_id}"
    starter = next(model for model in STARTER_MODELS if model.source == source)
    assert starter.capabilities is not None
    return ExternalApiModelConfig(
        key=provider_model_id,
        name=starter.name,
        provider_id="martllm",
        provider_model_id=provider_model_id,
        capabilities=starter.capabilities,
    )


def _request(width: int = 1024, height: int = 768, num_images: int = 2) -> ExternalGenerationRequest:
    return ExternalGenerationRequest(
        model=_starter_config("venice/z-image-turbo"),
        mode="txt2img",
        prompt="a red apple",
        seed=None,
        num_images=num_images,
        width=width,
        height=height,
        image_size=None,
        init_image=None,
        mask_image=None,
        reference_images=[],
        metadata=None,
    )


def _provider(**config: str) -> MartLLMProvider:
    return MartLLMProvider(
        InvokeAIAppConfig(external_martllm_api_key="ull-key", **config),
        logging.getLogger("test"),
    )


def test_martllm_registration() -> None:
    assert "external_martllm_api_key" in EXTERNAL_PROVIDER_CONFIG_FIELDS
    assert "external_martllm_base_url" in EXTERNAL_PROVIDER_CONFIG_FIELDS
    assert MartLLMProvider(
        InvokeAIAppConfig(external_martllm_api_key="ull-key"), logging.getLogger("test")
    ).is_configured()
    assert not MartLLMProvider(InvokeAIAppConfig(), logging.getLogger("test")).is_configured()


def test_martllm_starter_models_only_offer_sizes_martllm_accepts() -> None:
    martllm_models = [model for model in STARTER_MODELS if model.source.startswith("external://martllm/")]
    assert len(martllm_models) > 1
    for model in martllm_models:
        assert model.capabilities is not None
        assert model.capabilities.modes == ["txt2img"]
        assert model.capabilities.aspect_ratio_sizes is not None
        offered = {f"{size.width}x{size.height}" for size in model.capabilities.aspect_ratio_sizes.values()}
        assert offered <= MARTLLM_ACCEPTED_SIZES


def test_martllm_generates_and_decodes_images(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, headers: dict[str, str], json: dict[str, object], timeout: int) -> DummyResponse:
        captured.update(url=url, headers=headers, json=json)
        return DummyResponse(
            ok=True,
            json_data={
                "created": 1,
                "data": [{"b64_json": _png_base64()}, {"b64_json": _png_base64()}],
                "usage": {"cost_usd": 0.008},
            },
            headers={"x-request-id": "req-1"},
        )

    monkeypatch.setattr("requests.post", fake_post)

    result = _provider().generate(_request())

    assert captured["url"] == "https://api.martllm.com/v1/images/generations"
    assert captured["headers"] == {"Authorization": "Bearer ull-key", "Content-Type": "application/json"}
    assert captured["json"] == {
        "model": "venice/z-image-turbo",
        "prompt": "a red apple",
        "n": 2,
        "size": "1024x768",
        "response_format": "b64_json",
    }
    assert [image.image.size for image in result.images] == [(16, 16), (16, 16)]
    assert result.provider_request_id == "req-1"
    assert result.provider_metadata == {"model": "venice/z-image-turbo", "cost_usd": 0.008}


def test_martllm_uses_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    requested_urls: list[str] = []

    def fake_post(url: str, **_kwargs: Any) -> DummyResponse:
        requested_urls.append(url)
        return DummyResponse(ok=True, json_data={"data": [{"b64_json": _png_base64()}]})

    monkeypatch.setattr("requests.post", fake_post)

    _provider(external_martllm_base_url="https://proxy.martllm.test/").generate(_request(num_images=1))

    assert requested_urls == ["https://proxy.martllm.test/v1/images/generations"]


def test_martllm_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "requests.post",
        lambda *_args, **_kwargs: DummyResponse(ok=False, status_code=429, headers={"Retry-After": "3"}),
    )

    with pytest.raises(ExternalProviderRateLimitError) as error:
        _provider().generate(_request())

    assert error.value.retry_after == 3.0


def test_martllm_error_message_is_surfaced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "requests.post",
        lambda *_args, **_kwargs: DummyResponse(
            ok=False,
            status_code=401,
            json_data={"error": {"message": "Invalid API key.", "type": "invalid_request_error"}},
        ),
    )

    with pytest.raises(ExternalProviderRequestError, match="status 401: Invalid API key."):
        _provider().generate(_request())


def test_martllm_response_without_images_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "requests.post",
        lambda *_args, **_kwargs: DummyResponse(ok=True, json_data={"data": [{"url": "https://example.test/0"}]}),
    )

    with pytest.raises(ExternalProviderRequestError, match="contained no images"):
        _provider().generate(_request())


def test_martllm_requires_api_key() -> None:
    provider = MartLLMProvider(InvokeAIAppConfig(), logging.getLogger("test"))

    with pytest.raises(ExternalProviderRequestError, match="API key is not configured"):
        provider.generate(_request())


@pytest.mark.parametrize(
    "width, height, expected_size",
    [
        (1024, 1024, "1024x1024"),
        (1024, 768, "1024x768"),
        (768, 1024, "768x1024"),
        (1536, 1024, "1536x1024"),
        (1024, 1536, "1024x1536"),
        # Sizes off the listed ratios snap to the closest size MartLLM accepts.
        (1368, 768, "1536x1024"),
        (512, 512, "1024x1024"),
        (832, 1216, "1024x1536"),
    ],
)
def test_martllm_service_requests_land_on_accepted_sizes(
    monkeypatch: pytest.MonkeyPatch, width: int, height: int, expected_size: str
) -> None:
    """The generation service buckets each request to a size the model takes before MartLLM sees it."""
    submitted: list[dict[str, object]] = []

    def fake_post(url: str, json: dict[str, object], **_kwargs: Any) -> DummyResponse:
        submitted.append(json)
        return DummyResponse(ok=True, json_data={"data": [{"b64_json": _png_base64()}]})

    monkeypatch.setattr("requests.post", fake_post)
    service = ExternalGenerationService({"martllm": _provider()}, logging.getLogger("test"))

    service.generate(_request(width=width, height=height, num_images=1))

    assert submitted[0]["size"] == expected_size
