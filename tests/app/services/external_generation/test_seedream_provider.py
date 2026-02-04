import logging

import pytest
from PIL import Image

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.external_generation.errors import ExternalProviderRequestError
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGenerationRequest,
    ExternalReferenceImage,
)
from invokeai.app.services.external_generation.image_utils import encode_image_base64
from invokeai.app.services.external_generation.providers.seedream import SeedreamProvider
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


def _build_model(provider_model_id: str, modes: list[str] | None = None) -> ExternalApiModelConfig:
    return ExternalApiModelConfig(
        key="seedream_test",
        name="Seedream Test",
        provider_id="seedream",
        provider_model_id=provider_model_id,
        capabilities=ExternalModelCapabilities(
            modes=modes or ["txt2img", "img2img"],
            supports_negative_prompt=False,
            supports_reference_images=True,
            supports_seed=True,
            supports_guidance=True,
        ),
    )


def _build_request(
    model: ExternalApiModelConfig,
    mode: str = "txt2img",
    init_image: Image.Image | None = None,
    reference_images: list[ExternalReferenceImage] | None = None,
    num_images: int = 1,
    guidance: float | None = None,
    seed: int | None = 123,
) -> ExternalGenerationRequest:
    return ExternalGenerationRequest(
        model=model,
        mode=mode,  # type: ignore[arg-type]
        prompt="A test prompt",
        negative_prompt="",
        seed=seed,
        num_images=num_images,
        width=2048,
        height=2048,
        steps=20,
        guidance=guidance,
        init_image=init_image,
        mask_image=None,
        reference_images=reference_images or [],
        metadata=None,
    )


def test_seedream_is_configured() -> None:
    config = InvokeAIAppConfig(external_seedream_api_key="test-key")
    provider = SeedreamProvider(config, logging.getLogger("test"))
    assert provider.is_configured() is True


def test_seedream_not_configured() -> None:
    config = InvokeAIAppConfig()
    provider = SeedreamProvider(config, logging.getLogger("test"))
    assert provider.is_configured() is False


def test_seedream_txt2img_success(monkeypatch: pytest.MonkeyPatch) -> None:
    api_key = "seedream-key"
    config = InvokeAIAppConfig(external_seedream_api_key=api_key)
    provider = SeedreamProvider(config, logging.getLogger("test"))
    model = _build_model("seedream-4-5-251128")
    request = _build_request(model)
    encoded = encode_image_base64(_make_image("green"))
    captured: dict[str, object] = {}

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> DummyResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return DummyResponse(ok=True, json_data={"data": [{"b64_json": encoded}]})

    monkeypatch.setattr("requests.post", fake_post)

    result = provider.generate(request)

    assert captured["url"] == "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == f"Bearer {api_key}"
    json_payload = captured["json"]
    assert isinstance(json_payload, dict)
    assert json_payload["model"] == "seedream-4-5-251128"
    assert json_payload["prompt"] == "A test prompt"
    assert json_payload["size"] == "2048x2048"
    assert json_payload["response_format"] == "b64_json"
    assert json_payload["watermark"] is False
    assert json_payload["sequential_image_generation"] == "disabled"
    # Seed should not be sent for 4.x models
    assert "seed" not in json_payload
    # Guidance should not be sent for 4.x models
    assert "guidance_scale" not in json_payload
    assert len(result.images) == 1
    assert result.images[0].seed == 123


def test_seedream_3_0_t2i_sends_seed_and_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InvokeAIAppConfig(external_seedream_api_key="seedream-key")
    provider = SeedreamProvider(config, logging.getLogger("test"))
    model = _build_model("seedream-3-0-t2i-250415", modes=["txt2img"])
    request = _build_request(model, seed=42, guidance=2.5)
    encoded = encode_image_base64(_make_image("green"))
    captured: dict[str, object] = {}

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> DummyResponse:
        captured["json"] = json
        return DummyResponse(ok=True, json_data={"data": [{"b64_json": encoded}]})

    monkeypatch.setattr("requests.post", fake_post)

    provider.generate(request)

    json_payload = captured["json"]
    assert isinstance(json_payload, dict)
    assert json_payload["seed"] == 42
    assert json_payload["guidance_scale"] == 2.5
    # 3.0 models should not have sequential_image_generation
    assert "sequential_image_generation" not in json_payload


def test_seedream_batch_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InvokeAIAppConfig(external_seedream_api_key="seedream-key")
    provider = SeedreamProvider(config, logging.getLogger("test"))
    model = _build_model("seedream-4-5-251128")
    request = _build_request(model, num_images=3)
    encoded = encode_image_base64(_make_image("green"))
    captured: dict[str, object] = {}

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> DummyResponse:
        captured["json"] = json
        return DummyResponse(
            ok=True,
            json_data={"data": [{"b64_json": encoded}, {"b64_json": encoded}, {"b64_json": encoded}]},
        )

    monkeypatch.setattr("requests.post", fake_post)

    result = provider.generate(request)

    json_payload = captured["json"]
    assert isinstance(json_payload, dict)
    assert json_payload["sequential_image_generation"] == "auto"
    assert json_payload["sequential_image_generation_options"] == {"max_images": 3}
    assert len(result.images) == 3


def test_seedream_img2img_with_reference_images(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InvokeAIAppConfig(external_seedream_api_key="seedream-key")
    provider = SeedreamProvider(config, logging.getLogger("test"))
    model = _build_model("seedream-4-5-251128")
    init_image = _make_image("blue")
    ref_image = _make_image("red")
    request = _build_request(
        model,
        mode="img2img",
        init_image=init_image,
        reference_images=[ExternalReferenceImage(image=ref_image, weight=0.5)],
    )
    encoded = encode_image_base64(_make_image("green"))
    captured: dict[str, object] = {}

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> DummyResponse:
        captured["json"] = json
        return DummyResponse(ok=True, json_data={"data": [{"b64_json": encoded}]})

    monkeypatch.setattr("requests.post", fake_post)

    result = provider.generate(request)

    json_payload = captured["json"]
    assert isinstance(json_payload, dict)
    images = json_payload["image"]
    assert isinstance(images, list)
    assert len(images) == 2  # init_image + reference
    assert images[0].startswith("data:image/png;base64,")
    assert images[1].startswith("data:image/png;base64,")
    assert len(result.images) == 1


def test_seedream_single_image_not_array(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InvokeAIAppConfig(external_seedream_api_key="seedream-key")
    provider = SeedreamProvider(config, logging.getLogger("test"))
    model = _build_model("seedream-3-0-t2i-250415", modes=["txt2img"])
    init_image = _make_image("blue")
    request = _build_request(model, mode="txt2img", init_image=init_image, guidance=5.5)
    encoded = encode_image_base64(_make_image("green"))
    captured: dict[str, object] = {}

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> DummyResponse:
        captured["json"] = json
        return DummyResponse(ok=True, json_data={"data": [{"b64_json": encoded}]})

    monkeypatch.setattr("requests.post", fake_post)

    provider.generate(request)

    json_payload = captured["json"]
    assert isinstance(json_payload, dict)
    # Single image should be a string, not an array
    image = json_payload["image"]
    assert isinstance(image, str)
    assert image.startswith("data:image/png;base64,")


def test_seedream_error_response(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InvokeAIAppConfig(external_seedream_api_key="seedream-key")
    provider = SeedreamProvider(config, logging.getLogger("test"))
    model = _build_model("seedream-4-5-251128")
    request = _build_request(model)

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> DummyResponse:
        return DummyResponse(ok=False, status_code=400, text="bad request")

    monkeypatch.setattr("requests.post", fake_post)

    with pytest.raises(ExternalProviderRequestError, match="Seedream request failed"):
        provider.generate(request)


def test_seedream_no_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InvokeAIAppConfig()
    provider = SeedreamProvider(config, logging.getLogger("test"))
    model = _build_model("seedream-4-5-251128")
    request = _build_request(model)

    with pytest.raises(ExternalProviderRequestError, match="API key is not configured"):
        provider.generate(request)


def test_seedream_uses_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InvokeAIAppConfig(
        external_seedream_api_key="seedream-key",
        external_seedream_base_url="https://proxy.seedream/",
    )
    provider = SeedreamProvider(config, logging.getLogger("test"))
    model = _build_model("seedream-4-5-251128")
    request = _build_request(model)
    encoded = encode_image_base64(_make_image("green"))
    captured: dict[str, object] = {}

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> DummyResponse:
        captured["url"] = url
        return DummyResponse(ok=True, json_data={"data": [{"b64_json": encoded}]})

    monkeypatch.setattr("requests.post", fake_post)

    provider.generate(request)

    assert captured["url"] == "https://proxy.seedream/api/v3/images/generations"


def test_seedream_batch_skips_error_items(monkeypatch: pytest.MonkeyPatch) -> None:
    config = InvokeAIAppConfig(external_seedream_api_key="seedream-key")
    provider = SeedreamProvider(config, logging.getLogger("test"))
    model = _build_model("seedream-4-5-251128")
    request = _build_request(model, num_images=3)
    encoded = encode_image_base64(_make_image("green"))

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> DummyResponse:
        return DummyResponse(
            ok=True,
            json_data={
                "data": [
                    {"b64_json": encoded},
                    {"error": {"code": "content_filter", "message": "filtered"}},
                    {"b64_json": encoded},
                ]
            },
        )

    monkeypatch.setattr("requests.post", fake_post)

    result = provider.generate(request)

    assert len(result.images) == 2
