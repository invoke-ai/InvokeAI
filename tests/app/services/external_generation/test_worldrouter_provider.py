import base64
import http.server
import io
import logging
import threading
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


def _rgba_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (16, 16), color=(255, 0, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def _guarded_downloads(monkeypatch: pytest.MonkeyPatch, responses: dict[str, DummyResponse]) -> list[str]:
    """Serve result-image URLs through the guarded session; the plain `requests.get` path must not be used."""
    fetched: list[str] = []

    def fail_unguarded_get(*_args: Any, **_kwargs: Any) -> DummyResponse:
        pytest.fail("result image URLs must not use the unguarded requests.get path")

    def session_get(_session: Any, url: str, **kwargs: Any) -> DummyResponse:
        assert kwargs["stream"] is True
        fetched.append(url)
        return responses[url]

    monkeypatch.setattr("requests.get", fail_unguarded_get)
    monkeypatch.setattr("requests.Session.get", session_get)
    return fetched


def _respond_with(monkeypatch: pytest.MonkeyPatch, data: list[dict[str, str]]) -> None:
    monkeypatch.setattr("requests.post", lambda *_args, **_kwargs: DummyResponse(json_data={"data": data}))


def test_worldrouter_downloads_url_results_from_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    posted: list[str] = []
    image_url = "https://cdn.example.test/output.png"

    def fake_post(url: str, **_kwargs: Any) -> DummyResponse:
        posted.append(url)
        return DummyResponse(json_data={"data": [{"url": image_url}]})

    monkeypatch.setattr("requests.post", fake_post)
    fetched = _guarded_downloads(monkeypatch, {image_url: DummyResponse(content=_png_bytes("red"))})

    result = _provider(external_worldrouter_base_url="https://gateway.example.test/").generate(
        _request("gpt-image-2.6-flare")
    )

    assert posted == ["https://gateway.example.test/v1/images/generations"]
    assert fetched == [image_url]
    assert result.images[0].image.getpixel((0, 0)) == (255, 0, 0)


def test_worldrouter_url_results_keep_transparency(monkeypatch: pytest.MonkeyPatch) -> None:
    image_url = "https://cdn.example.test/transparent.png"
    _respond_with(monkeypatch, [{"url": image_url}])
    _guarded_downloads(monkeypatch, {image_url: DummyResponse(content=_rgba_png_bytes())})

    result = _provider().generate(_request("gpt-image-2.6-flare"))

    assert result.images[0].image.mode == "RGBA"
    assert result.images[0].image.getpixel((0, 0)) == (255, 0, 0, 0)


def test_worldrouter_refuses_result_urls_on_private_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercises the real guard: a listener on loopback must never be reached."""
    hits: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            hits.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(_png_bytes())

        def log_message(self, *_args: Any) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _respond_with(monkeypatch, [{"url": f"http://127.0.0.1:{server.server_port}/latest/meta-data"}])
        with pytest.raises(ExternalProviderRequestError, match="unsafe image URL"):
            _provider().generate(_request("gpt-image-2.6-flare"))
    finally:
        server.shutdown()
        server.server_close()

    assert hits == []


@pytest.mark.parametrize(
    "response",
    [
        DummyResponse(content=b"x", headers={"Content-Length": str(64 * 1024 * 1024)}),
        DummyResponse(content=b"x" * (32 * 1024 * 1024 + 1)),
    ],
    ids=["declared-length", "streamed-bytes"],
)
def test_worldrouter_caps_downloaded_image_size(monkeypatch: pytest.MonkeyPatch, response: DummyResponse) -> None:
    image_url = "https://cdn.example.test/huge.png"
    _respond_with(monkeypatch, [{"url": image_url}])
    _guarded_downloads(monkeypatch, {image_url: response})

    with pytest.raises(ExternalProviderRequestError, match="byte cap"):
        _provider().generate(_request("gpt-image-2.6-flare"))


@pytest.mark.parametrize(
    ("item", "download", "message"),
    [
        ({"url": "https://cdn.example.test/missing.png"}, DummyResponse(ok=False, status_code=404), "status 404"),
        ({"url": "https://cdn.example.test/broken.png"}, DummyResponse(content=b"not an image"), "not a valid image"),
        ({"b64_json": "!!!not-base64!!!"}, None, "invalid base64"),
        ({"b64_json": base64.b64encode(b"not an image").decode()}, None, "invalid base64"),
    ],
    ids=["download-failed", "invalid-bytes", "bad-base64", "base64-not-an-image"],
)
def test_worldrouter_reports_unusable_images(
    monkeypatch: pytest.MonkeyPatch, item: dict[str, str], download: DummyResponse | None, message: str
) -> None:
    _respond_with(monkeypatch, [item])
    _guarded_downloads(monkeypatch, {item["url"]: download} if download is not None else {})

    with pytest.raises(ExternalProviderRequestError, match=message):
        _provider().generate(_request("gpt-image-2.6-flare", num_images=1))


def test_worldrouter_keeps_billed_images_when_one_download_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    good_url, bad_url = "https://cdn.example.test/good.png", "https://cdn.example.test/bad.png"
    _respond_with(monkeypatch, [{"url": bad_url}, {"url": good_url}])
    _guarded_downloads(
        monkeypatch,
        {bad_url: DummyResponse(ok=False, status_code=500), good_url: DummyResponse(content=_png_bytes("red"))},
    )

    result = _provider().generate(_request("gpt-image-2.6-flare", num_images=2))

    assert [image.image.getpixel((0, 0)) for image in result.images] == [(255, 0, 0)]


def test_worldrouter_truncates_provider_error_bodies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "requests.post", lambda *_args, **_kwargs: DummyResponse(ok=False, status_code=502, text="x" * 5000)
    )

    with pytest.raises(ExternalProviderRequestError) as exc_info:
        _provider().generate(_request())

    assert len(str(exc_info.value)) < 600


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
