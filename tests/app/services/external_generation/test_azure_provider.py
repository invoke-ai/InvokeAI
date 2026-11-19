import dataclasses
import io
import logging

import pytest
from PIL import Image

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.external_generation.errors import (
    ExternalProviderRateLimitError,
    ExternalProviderRequestError,
)
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGenerationRequest,
    ExternalReferenceImage,
)
from invokeai.app.services.external_generation.image_utils import encode_image_base64
from invokeai.app.services.external_generation.providers.azure import AzureAIFoundryProvider, is_mai_model
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig, ExternalModelCapabilities

BASE_URL = "https://example-resource.services.ai.azure.com"


class DummyResponse:
    def __init__(self, ok: bool = True, status_code: int = 200, json_data: dict | None = None, text: str = "") -> None:
        self.ok = ok
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
        self.headers: dict[str, str] = {"apim-request-id": "req-123"}

    def json(self) -> dict:
        return self._json_data


def _provider(api_key: str | None = "azure-key", base_url: str | None = BASE_URL + "/") -> AzureAIFoundryProvider:
    config = InvokeAIAppConfig(external_azure_api_key=api_key, external_azure_base_url=base_url)
    return AzureAIFoundryProvider(config, logging.getLogger("test"))


def _request(
    model_id: str,
    *,
    references: int = 0,
    num_images: int = 1,
    width: int = 1024,
    height: int = 1024,
    provider_options: dict | None = None,
) -> ExternalGenerationRequest:
    model = ExternalApiModelConfig(
        key=f"azure_{model_id}",
        name=model_id,
        provider_id="azure",
        provider_model_id=model_id,
        capabilities=ExternalModelCapabilities(modes=["txt2img", "img2img"], supports_reference_images=True),
    )
    return ExternalGenerationRequest(
        model=model,
        mode="txt2img",
        prompt="A brass robot cat",
        seed=None,
        num_images=num_images,
        width=width,
        height=height,
        image_size=None,
        init_image=None,
        mask_image=None,
        reference_images=[ExternalReferenceImage(image=Image.new("RGB", (16, 16), "red")) for _ in range(references)],
        metadata=None,
        provider_options=provider_options,
    )


def _image_payload(mode: str = "RGB") -> dict:
    color = (0, 0, 0, 0) if mode == "RGBA" else "orange"
    return {"data": [{"b64_json": encode_image_base64(Image.new(mode, (16, 16), color))}]}


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    recorded: list[dict] = []

    def fake_post(url: str, headers: dict, timeout: int, json: dict | None = None, data=None, files=None):
        recorded.append({"url": url, "headers": headers, "json": json, "data": data, "files": files})
        return DummyResponse(json_data=_image_payload())

    monkeypatch.setattr("requests.post", fake_post)
    return recorded


def test_is_configured_requires_key_and_endpoint() -> None:
    assert _provider().is_configured()
    assert not _provider(api_key=None).is_configured()
    assert not _provider(base_url=None).is_configured()
    with pytest.raises(ExternalProviderRequestError, match="endpoint"):
        _provider(base_url=None).generate(_request("gpt-image-2"))


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [("MAI-Image-2.7", True), ("mai-image-2.6-flash", True), ("gpt-image-2", False), ("my-mai-deployment", False)],
)
def test_is_mai_model(model_id: str, expected: bool) -> None:
    assert is_mai_model(model_id) is expected


def test_gpt_image_txt2img_uses_openai_v1_generations(calls: list[dict]) -> None:
    result = _provider().generate(
        _request("gpt-image-2", num_images=2, provider_options={"quality": "high", "background": "auto"})
    )

    (call,) = calls
    assert call["url"] == f"{BASE_URL}/openai/v1/images/generations"
    assert call["headers"] == {"api-key": "azure-key"}
    assert call["json"] == {
        "model": "gpt-image-2",
        "prompt": "A brass robot cat",
        "n": 2,
        "size": "1024x1024",
        "output_format": "png",
        "quality": "high",
    }
    assert result.provider_request_id == "req-123"
    assert result.images


def test_gpt_image_references_use_edits_with_image_array(calls: list[dict]) -> None:
    _provider().generate(_request("gpt-image-2.5-sunburst", references=2, provider_options={"input_fidelity": "high"}))

    (call,) = calls
    assert call["url"] == f"{BASE_URL}/openai/v1/images/edits"
    assert "input_fidelity" not in call["data"]
    assert [name for name, _ in call["files"]] == ["image[]", "image[]"]
    assert isinstance(call["files"][0][1][1], io.BytesIO)


def test_mai_txt2img_uses_mai_generations_with_width_and_height(calls: list[dict]) -> None:
    _provider().generate(
        _request(
            "MAI-Image-2.7", width=1536, height=1024, provider_options={"quality": "high", "background": "transparent"}
        )
    )

    (call,) = calls
    assert call["url"] == f"{BASE_URL}/mai/v1/images/generations"
    assert call["json"] == {
        "model": "MAI-Image-2.7",
        "prompt": "A brass robot cat",
        "width": 1536,
        "height": 1024,
        "background": "transparent",
    }


def test_mai_edits_repeat_the_image_field(calls: list[dict]) -> None:
    _provider().generate(_request("MAI-Image-2.7", references=3))

    (call,) = calls
    assert call["url"] == f"{BASE_URL}/mai/v1/images/edits"
    assert [name for name, _ in call["files"]] == ["image", "image", "image"]
    assert call["data"]["width"] == 1024


def test_mai_rejects_more_than_five_images(calls: list[dict]) -> None:
    with pytest.raises(ExternalProviderRequestError, match="at most 5"):
        _provider().generate(_request("MAI-Image-2.7", references=6))
    assert calls == []


def test_mai_sends_one_request_per_image(calls: list[dict]) -> None:
    result = _provider().generate(_request("MAI-Image-2.6", num_images=2))

    assert len(calls) == 2
    assert len(result.images) == 2


def test_transparent_result_keeps_alpha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: DummyResponse(json_data=_image_payload("RGBA")))

    result = _provider().generate(_request("MAI-Image-2.7", provider_options={"background": "transparent"}))

    assert result.images[0].image.mode == "RGBA"


def test_rate_limit_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    limited = DummyResponse(ok=False, status_code=429)
    limited.headers["retry-after"] = "7"
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: limited)
    with pytest.raises(ExternalProviderRateLimitError) as rate_error:
        _provider().generate(_request("gpt-image-2"))
    assert rate_error.value.retry_after == 7

    bad = DummyResponse(ok=False, status_code=400, text="Invalid dimensions")
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: bad)
    with pytest.raises(ExternalProviderRequestError, match="status 400: Invalid dimensions"):
        _provider().generate(_request("MAI-Image-2.7"))


def test_empty_response_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: DummyResponse(json_data={"data": []}))
    with pytest.raises(ExternalProviderRequestError, match="no images"):
        _provider().generate(dataclasses.replace(_request("gpt-image-2")))
