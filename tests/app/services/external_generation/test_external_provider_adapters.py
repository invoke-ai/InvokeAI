import io
import logging

import pytest
from PIL import Image

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.external_generation.errors import ExternalProviderRequestError
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGenerationRequest,
    ExternalReferenceImage,
)
from invokeai.app.services.external_generation.image_utils import decode_image_base64, encode_image_base64
from invokeai.app.services.external_generation.providers.gemini import GeminiProvider
from invokeai.app.services.external_generation.providers.openai import OpenAIProvider
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig, ExternalModelCapabilities


class DummyResponse:
    def __init__(self, ok: bool, status_code: int = 200, json_data: dict | None = None, text: str = "") -> None:
        self.ok = ok
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
        self.headers: dict[str, str] = {}

    def json(self) -> dict:
        return self._json_data


def _make_image(color: str = "black") -> Image.Image:
    return Image.new("RGB", (32, 32), color=color)


def _build_model(provider_id: str, provider_model_id: str) -> ExternalApiModelConfig:
    return ExternalApiModelConfig(
        key=f"{provider_id}_test",
        name=f"{provider_id.title()} Test",
        provider_id=provider_id,
        provider_model_id=provider_model_id,
        capabilities=ExternalModelCapabilities(
            modes=["txt2img", "img2img", "inpaint"],
            supports_negative_prompt=True,
            supports_reference_images=True,
            supports_seed=True,
            supports_guidance=True,
        ),
    )


def _build_request(
    model: ExternalApiModelConfig,
    mode: str = "txt2img",
    init_image: Image.Image | None = None,
    mask_image: Image.Image | None = None,
    reference_images: list[ExternalReferenceImage] | None = None,
) -> ExternalGenerationRequest:
    return ExternalGenerationRequest(
        model=model,
        mode=mode,  # type: ignore[arg-type]
        prompt="A test prompt",
        negative_prompt="",
        seed=123,
        num_images=1,
        width=256,
        height=256,
        steps=20,
        guidance=5.5,
        init_image=init_image,
        mask_image=mask_image,
        reference_images=reference_images or [],
        metadata=None,
    )


def test_gemini_generate_success(monkeypatch: pytest.MonkeyPatch) -> None:
    api_key = "gemini-key"
    config = InvokeAIAppConfig(external_gemini_api_key=api_key)
    provider = GeminiProvider(config, logging.getLogger("test"))
    model = _build_model("gemini", "gemini-2.5-flash-image")
    init_image = _make_image("blue")
    ref_image = _make_image("red")
    request = _build_request(
        model,
        init_image=init_image,
        reference_images=[ExternalReferenceImage(image=ref_image, weight=0.6)],
    )
    encoded = encode_image_base64(_make_image("green"))
    captured: dict[str, object] = {}

    def fake_post(url: str, params: dict, json: dict, timeout: int) -> DummyResponse:
        captured["url"] = url
        captured["params"] = params
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse(
            ok=True,
            json_data={
                "candidates": [
                    {"content": {"parts": [{"inlineData": {"data": encoded}}]}},
                ]
            },
        )

    monkeypatch.setattr("requests.post", fake_post)

    result = provider.generate(request)

    assert (
        captured["url"]
        == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent"
    )
    assert captured["params"] == {"key": api_key}
    payload = captured["json"]
    assert isinstance(payload, dict)
    system_instruction = payload.get("systemInstruction")
    assert isinstance(system_instruction, dict)
    system_parts = system_instruction.get("parts")
    assert isinstance(system_parts, list)
    system_text = str(system_parts[0]).lower()
    assert "image" in system_text
    generation_config = payload.get("generationConfig")
    assert isinstance(generation_config, dict)
    assert generation_config["candidateCount"] == 1
    assert generation_config["responseModalities"] == ["IMAGE"]
    contents = payload.get("contents")
    assert isinstance(contents, list)
    first_content = contents[0]
    assert isinstance(first_content, dict)
    parts = first_content.get("parts")
    assert isinstance(parts, list)
    assert len(parts) >= 3
    part0 = parts[0]
    part1 = parts[1]
    part2 = parts[2]
    assert isinstance(part0, dict)
    assert isinstance(part1, dict)
    assert isinstance(part2, dict)
    inline0 = part0.get("inlineData")
    assert isinstance(inline0, dict)
    assert part1["text"] == request.prompt
    inline1 = part2.get("inlineData")
    assert isinstance(inline1, dict)
    assert inline0["data"] == encode_image_base64(init_image)
    assert inline1["data"] == encode_image_base64(ref_image)
    assert result.images[0].seed == request.seed
    assert result.provider_metadata == {"model": request.model.provider_model_id}


def test_gemini_generate_error_response(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InvokeAIAppConfig(external_gemini_api_key="gemini-key")
    provider = GeminiProvider(config, logging.getLogger("test"))
    model = _build_model("gemini", "gemini-2.5-flash-image")
    request = _build_request(model)

    def fake_post(url: str, params: dict, json: dict, timeout: int) -> DummyResponse:
        return DummyResponse(ok=False, status_code=400, text="bad request")

    monkeypatch.setattr("requests.post", fake_post)

    with pytest.raises(ExternalProviderRequestError, match="Gemini request failed"):
        provider.generate(request)


def test_gemini_generate_uses_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InvokeAIAppConfig(
        external_gemini_api_key="gemini-key",
        external_gemini_base_url="https://proxy.gemini",
    )
    provider = GeminiProvider(config, logging.getLogger("test"))
    model = _build_model("gemini", "gemini-2.5-flash-image")
    request = _build_request(model)
    encoded = encode_image_base64(_make_image("green"))
    captured: dict[str, object] = {}

    def fake_post(url: str, params: dict, json: dict, timeout: int) -> DummyResponse:
        captured["url"] = url
        return DummyResponse(
            ok=True,
            json_data={"candidates": [{"content": {"parts": [{"inlineData": {"data": encoded}}]}}]},
        )

    monkeypatch.setattr("requests.post", fake_post)

    provider.generate(request)

    assert captured["url"] == "https://proxy.gemini/v1beta/models/gemini-2.5-flash-image:generateContent"


def test_gemini_generate_keeps_base_url_version(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InvokeAIAppConfig(
        external_gemini_api_key="gemini-key",
        external_gemini_base_url="https://proxy.gemini/v1",
    )
    provider = GeminiProvider(config, logging.getLogger("test"))
    model = _build_model("gemini", "gemini-2.5-flash-image")
    request = _build_request(model)
    encoded = encode_image_base64(_make_image("green"))
    captured: dict[str, object] = {}

    def fake_post(url: str, params: dict, json: dict, timeout: int) -> DummyResponse:
        captured["url"] = url
        return DummyResponse(
            ok=True,
            json_data={"candidates": [{"content": {"parts": [{"inlineData": {"data": encoded}}]}}]},
        )

    monkeypatch.setattr("requests.post", fake_post)

    provider.generate(request)

    assert captured["url"] == "https://proxy.gemini/v1/models/gemini-2.5-flash-image:generateContent"


def test_gemini_generate_strips_models_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InvokeAIAppConfig(external_gemini_api_key="gemini-key")
    provider = GeminiProvider(config, logging.getLogger("test"))
    model = _build_model("gemini", "models/gemini-2.5-flash-image")
    request = _build_request(model)
    encoded = encode_image_base64(_make_image("green"))
    captured: dict[str, object] = {}

    def fake_post(url: str, params: dict, json: dict, timeout: int) -> DummyResponse:
        captured["url"] = url
        return DummyResponse(
            ok=True,
            json_data={"candidates": [{"content": {"parts": [{"inlineData": {"data": encoded}}]}}]},
        )

    monkeypatch.setattr("requests.post", fake_post)

    provider.generate(request)

    assert (
        captured["url"]
        == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent"
    )


def test_openai_generate_txt2img_success(monkeypatch: pytest.MonkeyPatch) -> None:
    api_key = "openai-key"
    config = InvokeAIAppConfig(external_openai_api_key=api_key)
    provider = OpenAIProvider(config, logging.getLogger("test"))
    model = _build_model("openai", "gpt-image-1")
    request = _build_request(model)
    encoded = encode_image_base64(_make_image("purple"))
    captured: dict[str, object] = {}

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> DummyResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        response = DummyResponse(ok=True, json_data={"data": [{"b64_json": encoded}]})
        response.headers["x-request-id"] = "req-123"
        return response

    monkeypatch.setattr("requests.post", fake_post)

    result = provider.generate(request)

    assert captured["url"] == "https://api.openai.com/v1/images/generations"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == f"Bearer {api_key}"
    json_payload = captured["json"]
    assert isinstance(json_payload, dict)
    assert json_payload["prompt"] == request.prompt
    assert result.provider_request_id == "req-123"
    assert result.images[0].seed == request.seed
    assert decode_image_base64(encoded).size == result.images[0].image.size


def test_openai_generate_uses_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InvokeAIAppConfig(
        external_openai_api_key="openai-key",
        external_openai_base_url="https://proxy.openai/",
    )
    provider = OpenAIProvider(config, logging.getLogger("test"))
    model = _build_model("openai", "gpt-image-1")
    request = _build_request(model)
    encoded = encode_image_base64(_make_image("purple"))
    captured: dict[str, object] = {}

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> DummyResponse:
        captured["url"] = url
        return DummyResponse(ok=True, json_data={"data": [{"b64_json": encoded}]})

    monkeypatch.setattr("requests.post", fake_post)

    provider.generate(request)

    assert captured["url"] == "https://proxy.openai/v1/images/generations"


def test_openai_generate_txt2img_error_response(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InvokeAIAppConfig(external_openai_api_key="openai-key")
    provider = OpenAIProvider(config, logging.getLogger("test"))
    model = _build_model("openai", "gpt-image-1")
    request = _build_request(model)

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> DummyResponse:
        return DummyResponse(ok=False, status_code=500, text="server error")

    monkeypatch.setattr("requests.post", fake_post)

    with pytest.raises(ExternalProviderRequestError, match="OpenAI request failed"):
        provider.generate(request)


def test_openai_generate_inpaint_uses_edit_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InvokeAIAppConfig(external_openai_api_key="openai-key")
    provider = OpenAIProvider(config, logging.getLogger("test"))
    model = _build_model("openai", "gpt-image-1")
    request = _build_request(
        model,
        mode="inpaint",
        init_image=_make_image("white"),
        mask_image=_make_image("black"),
    )
    encoded = encode_image_base64(_make_image("orange"))
    captured: dict[str, object] = {}

    def fake_post(url: str, headers: dict, data: dict, files: dict, timeout: int) -> DummyResponse:
        captured["url"] = url
        captured["data"] = data
        captured["files"] = files
        response = DummyResponse(ok=True, json_data={"data": [{"b64_json": encoded}]})
        return response

    monkeypatch.setattr("requests.post", fake_post)

    result = provider.generate(request)

    assert captured["url"] == "https://api.openai.com/v1/images/edits"
    data_payload = captured["data"]
    assert isinstance(data_payload, dict)
    assert data_payload["prompt"] == request.prompt
    files = captured["files"]
    assert isinstance(files, dict)
    assert "image" in files
    assert "mask" in files
    image_tuple = files["image"]
    assert isinstance(image_tuple, tuple)
    assert image_tuple[0] == "image.png"
    assert isinstance(image_tuple[1], io.BytesIO)
    assert result.images
