import base64
import io
import logging
from typing import Any, Iterator

import pytest
from PIL import Image

from invokeai.app.services.config.config_default import EXTERNAL_PROVIDER_CONFIG_FIELDS, InvokeAIAppConfig
from invokeai.app.services.external_generation.errors import (
    ExternalProviderCapabilityError,
    ExternalProviderRateLimitError,
    ExternalProviderRequestError,
)
from invokeai.app.services.external_generation.external_generation_common import ExternalGenerationRequest
from invokeai.app.services.external_generation.external_generation_default import ExternalGenerationService
from invokeai.app.services.external_generation.providers.worldrouter import WorldRouterProvider
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig
from invokeai.backend.model_manager.starter_models import STARTER_MODELS


class DummyResponse:
    def __init__(
        self,
        *,
        ok: bool = True,
        status_code: int = 200,
        json_data: Any = None,
        text: str = "",
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.ok = ok
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.content = content
        self.headers = headers or {}

    def json(self) -> Any:
        if self._json_data is None:
            raise ValueError("no JSON body")
        return self._json_data

    def iter_content(self, chunk_size: int = 65536) -> Iterator[bytes]:
        for index in range(0, len(self.content), chunk_size):
            yield self.content[index : index + chunk_size]

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


def _png_bytes(color: str = "blue") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _starter_config(provider_model_id: str) -> ExternalApiModelConfig:
    """Build a model config from the registered starter model, so requests are checked
    against the capabilities the app actually ships."""
    starter = next(model for model in STARTER_MODELS if model.source == f"external://worldrouter/{provider_model_id}")
    assert starter.capabilities is not None
    return ExternalApiModelConfig(
        key=provider_model_id,
        name=starter.name,
        provider_id="worldrouter",
        provider_model_id=provider_model_id,
        capabilities=starter.capabilities,
    )


def _request(provider_model_id: str = "gpt-image-2", **overrides: Any) -> ExternalGenerationRequest:
    fields: dict[str, Any] = {
        "model": _starter_config(provider_model_id),
        "mode": "txt2img",
        "prompt": "a blue square",
        "seed": None,
        "num_images": 1,
        "width": 1024,
        "height": 1024,
        "image_size": None,
        "init_image": None,
        "mask_image": None,
        "reference_images": [],
        "metadata": None,
    }
    fields.update(overrides)
    return ExternalGenerationRequest(**fields)


def _provider(**config: str) -> WorldRouterProvider:
    return WorldRouterProvider(
        InvokeAIAppConfig(external_worldrouter_api_key="sk-worldrouter-test", **config), logging.getLogger("test")
    )


def test_worldrouter_registration() -> None:
    assert "external_worldrouter_api_key" in EXTERNAL_PROVIDER_CONFIG_FIELDS
    assert "external_worldrouter_base_url" in EXTERNAL_PROVIDER_CONFIG_FIELDS
    sources = {model.source for model in STARTER_MODELS if model.source.startswith("external://worldrouter/")}
    assert sources == {
        "external://worldrouter/gpt-image-2",
        "external://worldrouter/gpt-image-2.5-flare",
        "external://worldrouter/gpt-image-2.5-sunburst",
        "external://worldrouter/gpt-image-2.6-flare",
        "external://worldrouter/gpt-image-2.6-sunburst",
    }
    assert WorldRouterProvider(InvokeAIAppConfig(), logging.getLogger("test")).is_configured() is False
    assert _provider().is_configured() is True


def test_worldrouter_posts_images_generation_and_decodes_base64(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> DummyResponse:
        captured.update(url=url, **kwargs)
        encoded = base64.b64encode(_png_bytes()).decode()
        return DummyResponse(json_data={"data": [{"b64_json": encoded}, {"b64_json": encoded}]})

    monkeypatch.setattr("requests.post", fake_post)

    result = _provider().generate(_request("gpt-image-2.6-sunburst", num_images=2, width=1536, height=1024))

    assert captured["url"] == "https://inference-api.worldrouter.ai/v1/images/generations"
    assert captured["headers"] == {"Authorization": "Bearer sk-worldrouter-test"}
    assert captured["json"] == {
        "model": "gpt-image-2.6-sunburst",
        "prompt": "a blue square",
        "n": 2,
        "size": "1536x1024",
    }
    assert [image.image.size for image in result.images] == [(16, 16), (16, 16)]
    assert result.provider_metadata == {"model": "gpt-image-2.6-sunburst"}


def test_worldrouter_downloads_url_results_from_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[str] = []
    image_url = "https://cdn.example.test/output.png"

    def fake_post(url: str, **_kwargs: Any) -> DummyResponse:
        requested.append(url)
        return DummyResponse(json_data={"data": [{"url": image_url}]})

    def fake_get(url: str, **kwargs: Any) -> DummyResponse:
        requested.append(url)
        assert kwargs["stream"] is True
        return DummyResponse(content=_png_bytes("red"))

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.get", fake_get)

    result = _provider(external_worldrouter_base_url="https://gateway.example.test/").generate(
        _request("gpt-image-2.6-flare")
    )

    assert requested == ["https://gateway.example.test/v1/images/generations", image_url]
    assert result.images[0].image.getpixel((0, 0)) == (255, 0, 0)


def test_worldrouter_rate_limit_reports_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "requests.post",
        lambda *_args, **_kwargs: DummyResponse(ok=False, status_code=429, headers={"retry-after": "12"}),
    )

    with pytest.raises(ExternalProviderRateLimitError) as exc_info:
        _provider().generate(_request())

    assert exc_info.value.retry_after == 12.0


@pytest.mark.parametrize(
    "response",
    [
        DummyResponse(ok=False, status_code=401, text="invalid api key"),
        DummyResponse(text="<html>bad gateway</html>"),
        DummyResponse(json_data=None),
        DummyResponse(json_data=[]),
        DummyResponse(json_data={"data": None}),
        DummyResponse(json_data={"data": [{"revised_prompt": "no image"}]}),
    ],
    ids=["http-error", "not-json", "empty-body", "not-an-object", "data-null", "no-image-fields"],
)
def test_worldrouter_rejects_failed_or_malformed_responses(
    monkeypatch: pytest.MonkeyPatch, response: DummyResponse
) -> None:
    monkeypatch.setattr("requests.post", lambda *_args, **_kwargs: response)

    with pytest.raises(ExternalProviderRequestError, match="WorldRouter"):
        _provider().generate(_request())


@pytest.mark.parametrize(
    ("model_id", "width", "height", "expected_size"),
    [
        ("gpt-image-2.5-sunburst", 1200, 800, "1536x1024"),
        ("gpt-image-2.5-sunburst", 900, 1000, "1024x1024"),
        # The gpt-image-2 route only returns 1536x1024, so every canvas is sent as that shape.
        ("gpt-image-2", 1024, 1024, "1536x1024"),
    ],
)
def test_worldrouter_starter_models_reach_provider_with_supported_size(
    monkeypatch: pytest.MonkeyPatch, model_id: str, width: int, height: int, expected_size: str
) -> None:
    """An off-ratio canvas is bucketed to a size the model returns before the request is sent."""
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> DummyResponse:
        captured.update(kwargs["json"])
        return DummyResponse(json_data={"data": [{"b64_json": base64.b64encode(_png_bytes()).decode()}]})

    monkeypatch.setattr("requests.post", fake_post)
    service = ExternalGenerationService({"worldrouter": _provider()}, logging.getLogger("test"))

    service.generate(_request(model_id, width=width, height=height))

    assert captured["model"] == model_id
    assert captured["size"] == expected_size


def test_worldrouter_gpt_image_2_refuses_batches_its_route_ignores() -> None:
    service = ExternalGenerationService({"worldrouter": _provider()}, logging.getLogger("test"))

    with pytest.raises(ExternalProviderCapabilityError):
        service.generate(_request("gpt-image-2", num_images=2, width=1536, height=1024))
