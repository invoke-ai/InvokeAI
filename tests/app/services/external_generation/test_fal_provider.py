import io
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from PIL import Image

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.external_generation.errors import ExternalProviderRequestError
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGenerationRequest,
)
from invokeai.app.services.external_generation.providers.fal import FalProvider
from invokeai.backend.model_manager.configs.external_api import (
    ExternalApiModelConfig,
    ExternalImageSize,
    ExternalModelCapabilities,
)


class DummyResponse:
    def __init__(
        self,
        *,
        ok: bool = True,
        status_code: int = 200,
        json_data: dict[str, Any] | None = None,
        content: bytes = b"",
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.ok = ok
        self.status_code = status_code
        self._json_data = json_data or {}
        self.content = content
        self.text = text
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        return self._json_data

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        del chunk_size
        yield self.content

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _png_bytes(color: str = "red") -> bytes:
    image = Image.new("RGB", (2, 2), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _model(model_id: str, *, modes: list[str]) -> ExternalApiModelConfig:
    return ExternalApiModelConfig(
        key="fal-test",
        name="fal test",
        provider_id="fal",
        provider_model_id=model_id,
        capabilities=ExternalModelCapabilities(
            modes=modes,  # type: ignore[arg-type]
            supports_seed=True,
            max_images_per_request=4,
            allowed_aspect_ratios=["1:1", "4:3", "3:4", "16:9", "9:16"],
            aspect_ratio_sizes={
                "1:1": ExternalImageSize(width=1024, height=1024),
                "4:3": ExternalImageSize(width=1152, height=864),
                "3:4": ExternalImageSize(width=864, height=1152),
                "16:9": ExternalImageSize(width=1280, height=720),
                "9:16": ExternalImageSize(width=720, height=1280),
            },
        ),
    )


def _request(
    model: ExternalApiModelConfig,
    *,
    mode: str = "txt2img",
    init_image: Image.Image | None = None,
    mask_image: Image.Image | None = None,
) -> ExternalGenerationRequest:
    return ExternalGenerationRequest(
        model=model,
        mode=mode,  # type: ignore[arg-type]
        prompt="A test prompt",
        seed=123,
        num_images=2,
        width=1024,
        height=1024,
        image_size=None,
        init_image=init_image,
        mask_image=mask_image,
        reference_images=[],
        metadata=None,
    )


def test_fal_provider_reports_configuration_from_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("FAL_API_KEY", raising=False)
    configured = FalProvider(InvokeAIAppConfig(external_fal_api_key="test-key"), logging.getLogger("test"))
    unconfigured = FalProvider(InvokeAIAppConfig(), logging.getLogger("test"))

    assert configured.is_configured() is True
    assert unconfigured.is_configured() is False


def test_fal_provider_accepts_official_environment_key_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setenv("FAL_API_KEY", "legacy-key")
    assert FalProvider(InvokeAIAppConfig(), logging.getLogger("test")).is_configured() is True

    monkeypatch.setenv("FAL_KEY", "official-key")
    assert FalProvider(InvokeAIAppConfig(), logging.getLogger("test")).is_configured() is True


def test_fal_provider_submits_queue_polls_and_downloads_images(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InvokeAIAppConfig(external_fal_api_key="fal-key", external_fal_base_url="https://queue.test")
    provider = FalProvider(config, logging.getLogger("test"))
    request = _request(_model("fal-ai/flux/schnell", modes=["txt2img"]))
    captured: dict[str, Any] = {}
    output_bytes = _png_bytes("green")

    def fake_post(url: str, headers: dict[str, str], json: dict[str, Any], timeout: int) -> DummyResponse:
        captured["post_url"] = url
        captured["post_headers"] = headers
        captured["post_json"] = json
        captured["post_timeout"] = timeout
        return DummyResponse(json_data={"request_id": "request-1"})

    def fake_get(url: str, headers: dict[str, str], timeout: int, stream: bool = False) -> DummyResponse:
        del headers, timeout
        if url.endswith("/status"):
            return DummyResponse(json_data={"status": "COMPLETED"})
        if "/requests/request-1" in url:
            return DummyResponse(json_data={"images": [{"url": "https://cdn.test/result.png"}], "seed": 777})
        assert stream is True
        return DummyResponse(content=output_bytes)

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.get", fake_get)

    result = provider.generate(request)

    assert captured["post_url"] == "https://queue.test/fal-ai/flux/schnell"
    assert captured["post_headers"] == {"Authorization": "Key fal-key", "Content-Type": "application/json"}
    assert captured["post_json"] == {
        "prompt": request.prompt,
        "image_size": "square_hd",
        "num_images": 2,
        "seed": 123,
    }
    assert result.provider_request_id == "request-1"
    assert result.seed_used == 777
    assert result.images[0].image.size == (2, 2)
    assert result.images[0].seed == 777


def test_fal_provider_uploads_kontext_input_and_uses_image_url(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InvokeAIAppConfig(external_fal_api_key="fal-key", external_fal_base_url="https://queue.test")
    provider = FalProvider(config, logging.getLogger("test"))
    request = _request(
        _model("fal-ai/flux-pro/kontext", modes=["img2img"]),
        mode="img2img",
        init_image=Image.new("RGB", (2, 2), color="blue"),
    )
    post_calls: list[dict[str, Any]] = []
    put_calls: list[dict[str, Any]] = []

    def fake_post(url: str, headers: dict[str, str], json: dict[str, Any], timeout: int) -> DummyResponse:
        post_calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        if "/storage/upload/initiate" in url:
            return DummyResponse(
                json_data={
                    "file_url": "https://cdn.test/input.png",
                    "upload_url": "https://upload.test/input",
                }
            )
        return DummyResponse(json_data={"request_id": "request-2"})

    def fake_put(url: str, data: bytes, headers: dict[str, str], timeout: int) -> DummyResponse:
        put_calls.append({"url": url, "data": data, "headers": headers, "timeout": timeout})
        return DummyResponse()

    def fake_get(url: str, headers: dict[str, str], timeout: int, stream: bool = False) -> DummyResponse:
        del headers, timeout
        if url.endswith("/status"):
            return DummyResponse(json_data={"status": "COMPLETED"})
        if "/requests/request-2" in url:
            return DummyResponse(json_data={"images": [{"url": "https://cdn.test/result.png"}], "seed": 123})
        assert stream is True
        return DummyResponse(content=_png_bytes("green"))

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.put", fake_put)
    monkeypatch.setattr("requests.get", fake_get)

    provider.generate(request)

    assert post_calls[0]["url"] == "https://rest.fal.ai/storage/upload/initiate?storage_type=fal-cdn-v3"
    assert post_calls[0]["json"] == {"file_name": "image.png", "content_type": "image/png"}
    assert put_calls[0]["url"] == "https://upload.test/input"
    assert put_calls[0]["headers"] == {"Content-Type": "image/png"}
    assert post_calls[1]["url"] == "https://queue.test/fal-ai/flux-pro/kontext"
    assert post_calls[1]["json"] == {
        "prompt": request.prompt,
        "image_url": "https://cdn.test/input.png",
        "aspect_ratio": "1:1",
        "num_images": 2,
        "seed": 123,
    }


def test_fal_provider_inverts_invoke_mask_for_flux_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InvokeAIAppConfig(external_fal_api_key="fal-key", external_fal_base_url="https://queue.test")
    provider = FalProvider(config, logging.getLogger("test"))
    mask = Image.new("L", (2, 1))
    mask.putdata([0, 255])
    request = _request(
        _model("fal-ai/flux-lora-fill", modes=["inpaint"]),
        mode="inpaint",
        init_image=Image.new("RGB", (2, 1), color="blue"),
        mask_image=mask,
    )
    uploads: dict[str, bytes] = {}
    queue_payload: dict[str, Any] = {}
    upload_index = 0

    def fake_post(url: str, headers: dict[str, str], json: dict[str, Any], timeout: int) -> DummyResponse:
        nonlocal upload_index
        del headers, timeout
        if "/storage/upload/initiate" in url:
            upload_index += 1
            name = "image" if upload_index == 1 else "mask"
            return DummyResponse(
                json_data={
                    "file_url": f"https://cdn.test/{name}.png",
                    "upload_url": f"https://upload.test/{name}",
                }
            )
        queue_payload.update(json)
        return DummyResponse(json_data={"request_id": "request-3"})

    def fake_put(url: str, data: bytes, headers: dict[str, str], timeout: int) -> DummyResponse:
        del headers, timeout
        uploads[url.rsplit("/", 1)[-1]] = data
        return DummyResponse()

    def fake_get(url: str, headers: dict[str, str], timeout: int, stream: bool = False) -> DummyResponse:
        del headers, timeout
        if url.endswith("/status"):
            return DummyResponse(json_data={"status": "COMPLETED"})
        if "/requests/request-3" in url:
            return DummyResponse(json_data={"images": [{"url": "https://cdn.test/result.png"}], "seed": 123})
        assert stream is True
        return DummyResponse(content=_png_bytes("green"))

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.put", fake_put)
    monkeypatch.setattr("requests.get", fake_get)

    provider.generate(request)

    uploaded_mask = Image.open(io.BytesIO(uploads["mask"])).convert("L")
    assert list(uploaded_mask.getdata()) == [255, 0]
    assert queue_payload == {
        "prompt": request.prompt,
        "image_size": "square_hd",
        "num_images": 2,
        "seed": 123,
        "image_url": "https://cdn.test/image.png",
        "mask_url": "https://cdn.test/mask.png",
        "paste_back": True,
        "resize_to_original": True,
    }


def test_fal_provider_retries_poll_rate_limit_without_resubmitting(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InvokeAIAppConfig(external_fal_api_key="fal-key", external_fal_base_url="https://queue.test")
    provider = FalProvider(config, logging.getLogger("test"))
    request = _request(_model("fal-ai/flux/schnell", modes=["txt2img"]))
    submit_count = 0
    status_count = 0

    def fake_post(*args: Any, **kwargs: Any) -> DummyResponse:
        nonlocal submit_count
        del args, kwargs
        submit_count += 1
        return DummyResponse(json_data={"request_id": "request-4"})

    def fake_get(url: str, headers: dict[str, str], timeout: int, stream: bool = False) -> DummyResponse:
        nonlocal status_count
        del headers, timeout
        if url.endswith("/status"):
            status_count += 1
            if status_count == 1:
                return DummyResponse(ok=False, status_code=429, headers={"Retry-After": "1"})
            return DummyResponse(json_data={"status": "COMPLETED"})
        if "/requests/request-4" in url:
            return DummyResponse(json_data={"images": [{"url": "https://cdn.test/result.png"}], "seed": 123})
        assert stream is True
        return DummyResponse(content=_png_bytes("green"))

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("invokeai.app.services.external_generation.providers.fal.time.sleep", lambda _: None)

    provider.generate(request)

    assert submit_count == 1
    assert status_count == 2


def test_fal_provider_reports_queue_error(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InvokeAIAppConfig(external_fal_api_key="fal-key")
    provider = FalProvider(config, logging.getLogger("test"))
    request = _request(_model("fal-ai/flux/schnell", modes=["txt2img"]))

    def fake_post(*args: Any, **kwargs: Any) -> DummyResponse:
        del args, kwargs
        return DummyResponse(ok=False, status_code=400, text="invalid model input")

    monkeypatch.setattr("requests.post", fake_post)

    with pytest.raises(ExternalProviderRequestError, match="fal.ai request failed"):
        provider.generate(request)
