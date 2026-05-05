import io
import logging
from typing import Any, Iterator

import pytest
from PIL import Image

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.external_generation.errors import ExternalProviderRequestError
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGenerationRequest,
    ExternalReferenceImage,
)
from invokeai.app.services.external_generation.image_utils import encode_image_base64
from invokeai.app.services.external_generation.providers import alibabacloud as alibabacloud_module
from invokeai.app.services.external_generation.providers.alibabacloud import AlibabaCloudProvider
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig, ExternalModelCapabilities


class DummyResponse:
    def __init__(
        self,
        ok: bool,
        status_code: int = 200,
        json_data: dict | None = None,
        text: str = "",
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.ok = ok
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
        self.content = content
        self.headers: dict[str, str] = headers or {}

    def json(self) -> dict:
        return self._json_data

    def iter_content(self, chunk_size: int = 65536) -> Iterator[bytes]:
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


def _make_image(color: str = "blue") -> Image.Image:
    return Image.new("RGB", (16, 16), color=color)


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _build_model(provider_model_id: str) -> ExternalApiModelConfig:
    return ExternalApiModelConfig(
        key=f"alibabacloud_{provider_model_id}",
        name=provider_model_id,
        provider_id="alibabacloud",
        provider_model_id=provider_model_id,
        capabilities=ExternalModelCapabilities(
            modes=["txt2img"],
            supports_reference_images=True,
            supports_seed=True,
        ),
    )


def _build_request(
    model: ExternalApiModelConfig,
    reference_images: list[ExternalReferenceImage] | None = None,
) -> ExternalGenerationRequest:
    return ExternalGenerationRequest(
        model=model,
        mode="txt2img",  # type: ignore[arg-type]
        prompt="a cat",
        seed=42,
        num_images=1,
        width=1024,
        height=1024,
        image_size=None,
        init_image=None,
        mask_image=None,
        reference_images=reference_images or [],
        metadata=None,
    )


def _provider() -> AlibabaCloudProvider:
    config = InvokeAIAppConfig(external_alibabacloud_api_key="test-key")
    return AlibabaCloudProvider(config, logging.getLogger("test"))


def test_unknown_model_id_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    request = _build_request(_build_model("not-a-real-model"))

    def fail_post(*_args: Any, **_kwargs: Any) -> DummyResponse:  # pragma: no cover - should not be called
        raise AssertionError("network must not be touched for unknown model")

    monkeypatch.setattr("requests.post", fail_post)

    with pytest.raises(ExternalProviderRequestError, match="Unknown DashScope model_id"):
        provider.generate(request)


def test_sync_routes_qwen_edit_max_with_reference_images(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    ref = _make_image("red")
    request = _build_request(
        _build_model("qwen-image-edit-max"),
        reference_images=[ExternalReferenceImage(image=ref)],
    )
    captured: dict[str, Any] = {}

    image_url = "https://example.invalid/result.png"
    image_bytes = _png_bytes(_make_image("green"))

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> DummyResponse:
        captured["url"] = url
        captured["json"] = json
        return DummyResponse(
            ok=True,
            json_data={
                "request_id": "req-1",
                "output": {
                    "choices": [
                        {"message": {"content": [{"image": image_url}]}},
                    ]
                },
            },
        )

    def fake_get(url: str, timeout: int, stream: bool = False) -> DummyResponse:
        assert url == image_url
        return DummyResponse(
            ok=True,
            content=image_bytes,
            headers={"Content-Length": str(len(image_bytes))},
        )

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.get", fake_get)

    result = provider.generate(request)

    assert "multimodal-generation" in captured["url"]
    payload = captured["json"]
    messages = payload["input"]["messages"]
    content = messages[0]["content"]
    # Reference image first, then prompt text — and no init_image entry.
    assert content[0]["image"].startswith("data:image/png;base64,")
    assert content[0]["image"].endswith(encode_image_base64(ref))
    assert content[1] == {"text": request.prompt}
    assert len(content) == 2
    assert payload["model"] == "qwen-image-edit-max"
    assert payload["parameters"]["seed"] == request.seed
    assert result.provider_request_id == "req-1"
    assert len(result.images) == 1


def test_sync_error_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    request = _build_request(_build_model("qwen-image-2.0-pro"))

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> DummyResponse:
        return DummyResponse(ok=False, status_code=400, text="bad request")

    monkeypatch.setattr("requests.post", fake_post)

    with pytest.raises(ExternalProviderRequestError, match="DashScope request failed"):
        provider.generate(request)


def test_sync_retries_on_429_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    request = _build_request(_build_model("qwen-image-2.0-pro"))
    image_bytes = _png_bytes(_make_image("yellow"))
    image_url = "https://example.invalid/r.png"

    calls = {"n": 0}

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> DummyResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return DummyResponse(ok=False, status_code=429, text="rate limited", headers={"Retry-After": "0"})
        return DummyResponse(
            ok=True,
            json_data={
                "request_id": "req-2",
                "output": {"choices": [{"message": {"content": [{"image": image_url}]}}]},
            },
        )

    def fake_get(url: str, timeout: int, stream: bool = False) -> DummyResponse:
        return DummyResponse(ok=True, content=image_bytes, headers={"Content-Length": str(len(image_bytes))})

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    result = provider.generate(request)
    assert calls["n"] == 2
    assert len(result.images) == 1


def test_async_parser_does_not_double_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """A result with both `url` and `b64_image` must yield one image, not two."""
    provider = _provider()
    request = _build_request(_build_model("qwen-image-2.0-pro"))
    image_bytes = _png_bytes(_make_image("magenta"))
    image_url = "https://example.invalid/x.png"

    def fake_get(url: str, timeout: int, stream: bool = False) -> DummyResponse:
        return DummyResponse(ok=True, content=image_bytes, headers={"Content-Length": str(len(image_bytes))})

    monkeypatch.setattr("requests.get", fake_get)

    output: dict[str, Any] = {
        "results": [
            {
                "url": image_url,
                "b64_image": encode_image_base64(_make_image("cyan")),
            }
        ]
    }
    result = provider._parse_async_response(output, request, request_id="rid")
    assert len(result.images) == 1


def test_async_parser_accepts_b64_only(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    request = _build_request(_build_model("qwen-image-2.0-pro"))
    output: dict[str, Any] = {
        "results": [
            {"b64_image": encode_image_base64(_make_image("cyan"))},
        ]
    }
    result = provider._parse_async_response(output, request, request_id="rid")
    assert len(result.images) == 1


def test_download_image_size_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    too_big = alibabacloud_module._DOWNLOAD_MAX_BYTES + 1

    def fake_get(url: str, timeout: int, stream: bool = False) -> DummyResponse:
        return DummyResponse(
            ok=True,
            content=b"\x00" * 16,  # body itself is small; we trip the Content-Length check first
            headers={"Content-Length": str(too_big)},
        )

    monkeypatch.setattr("requests.get", fake_get)

    with pytest.raises(ExternalProviderRequestError, match="exceeds"):
        provider._download_image("https://example.invalid/big.png")


def test_poll_task_first_call_no_initial_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """First poll must not be preceded by a sleep — fast tasks should not pay the poll interval."""
    provider = _provider()
    request = _build_request(_build_model("qwen-image-2.0-pro"))
    image_bytes = _png_bytes(_make_image("teal"))
    image_url = "https://example.invalid/y.png"

    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    def fake_get(url: str, headers: dict, timeout: int) -> DummyResponse:
        return DummyResponse(
            ok=True,
            json_data={
                "output": {
                    "task_status": "SUCCEEDED",
                    "results": [{"url": image_url}],
                }
            },
        )

    def fake_download_get(url: str, timeout: int, stream: bool = False) -> DummyResponse:
        return DummyResponse(ok=True, content=image_bytes, headers={"Content-Length": str(len(image_bytes))})

    # Single requests.get is shared by polling (with headers kwarg) and download (no kwarg).
    def dispatch_get(*args: Any, **kwargs: Any) -> DummyResponse:
        if "headers" in kwargs and "task" in args[0]:
            return fake_get(*args, **kwargs)
        return fake_download_get(*args, **kwargs)

    monkeypatch.setattr("requests.get", dispatch_get)
    monkeypatch.setattr("time.sleep", fake_sleep)

    result = provider._poll_task(
        base_url="https://dashscope.invalid",
        headers={"Authorization": "Bearer test", "Content-Type": "application/json"},
        task_id="task-xyz",
        request=request,
        request_id="rid",
    )

    assert len(result.images) == 1
    # No sleep should have been recorded — task succeeded on the first poll.
    assert sleeps == []
