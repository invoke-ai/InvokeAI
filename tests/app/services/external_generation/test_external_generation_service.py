import logging

import pytest
from PIL import Image

from invokeai.app.services.external_generation.errors import (
    ExternalProviderCapabilityError,
    ExternalProviderNotConfiguredError,
    ExternalProviderNotFoundError,
)
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGeneratedImage,
    ExternalGenerationRequest,
    ExternalGenerationResult,
    ExternalReferenceImage,
)
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.external_generation.external_generation_base import ExternalProvider
from invokeai.app.services.external_generation.external_generation_default import ExternalGenerationService
from invokeai.backend.model_manager.configs.external_api import (
    ExternalApiModelConfig,
    ExternalImageSize,
    ExternalModelCapabilities,
)


class DummyProvider(ExternalProvider):
    def __init__(self, provider_id: str, configured: bool, result: ExternalGenerationResult | None = None) -> None:
        super().__init__(InvokeAIAppConfig(), logging.getLogger("test"))
        self.provider_id = provider_id
        self._configured = configured
        self._result = result
        self.last_request: ExternalGenerationRequest | None = None

    def is_configured(self) -> bool:
        return self._configured

    def generate(self, request: ExternalGenerationRequest) -> ExternalGenerationResult:
        self.last_request = request
        assert self._result is not None
        return self._result


def _build_model(capabilities: ExternalModelCapabilities) -> ExternalApiModelConfig:
    return ExternalApiModelConfig(
        key="external_test",
        name="External Test",
        provider_id="openai",
        provider_model_id="gpt-image-1",
        capabilities=capabilities,
    )


def _build_request(
    *,
    model: ExternalApiModelConfig,
    mode: str = "txt2img",
    negative_prompt: str | None = None,
    seed: int | None = None,
    num_images: int = 1,
    guidance: float | None = None,
    width: int = 64,
    height: int = 64,
    init_image: Image.Image | None = None,
    mask_image: Image.Image | None = None,
    reference_images: list[ExternalReferenceImage] | None = None,
) -> ExternalGenerationRequest:
    return ExternalGenerationRequest(
        model=model,
        mode=mode,  # type: ignore[arg-type]
        prompt="A test prompt",
        negative_prompt=negative_prompt,
        seed=seed,
        num_images=num_images,
        width=width,
        height=height,
        steps=10,
        guidance=guidance,
        init_image=init_image,
        mask_image=mask_image,
        reference_images=reference_images or [],
        metadata=None,
    )


def _make_image() -> Image.Image:
    return Image.new("RGB", (64, 64), color="black")


def test_generate_requires_registered_provider() -> None:
    model = _build_model(ExternalModelCapabilities(modes=["txt2img"]))
    request = _build_request(model=model)
    service = ExternalGenerationService({}, logging.getLogger("test"))

    with pytest.raises(ExternalProviderNotFoundError):
        service.generate(request)


def test_generate_requires_configured_provider() -> None:
    model = _build_model(ExternalModelCapabilities(modes=["txt2img"]))
    request = _build_request(model=model)
    provider = DummyProvider("openai", configured=False)
    service = ExternalGenerationService({"openai": provider}, logging.getLogger("test"))

    with pytest.raises(ExternalProviderNotConfiguredError):
        service.generate(request)


def test_generate_validates_mode_support() -> None:
    model = _build_model(ExternalModelCapabilities(modes=["txt2img"]))
    request = _build_request(model=model, mode="img2img", init_image=_make_image())
    provider = DummyProvider("openai", configured=True, result=ExternalGenerationResult(images=[]))
    service = ExternalGenerationService({"openai": provider}, logging.getLogger("test"))

    with pytest.raises(ExternalProviderCapabilityError, match="Mode 'img2img'"):
        service.generate(request)


def test_generate_validates_negative_prompt_support() -> None:
    model = _build_model(ExternalModelCapabilities(modes=["txt2img"], supports_negative_prompt=False))
    request = _build_request(model=model, negative_prompt="bad")
    provider = DummyProvider("openai", configured=True, result=ExternalGenerationResult(images=[]))
    service = ExternalGenerationService({"openai": provider}, logging.getLogger("test"))

    with pytest.raises(ExternalProviderCapabilityError, match="Negative prompts"):
        service.generate(request)


def test_generate_requires_init_image_for_img2img() -> None:
    model = _build_model(ExternalModelCapabilities(modes=["img2img"]))
    request = _build_request(model=model, mode="img2img")
    provider = DummyProvider("openai", configured=True, result=ExternalGenerationResult(images=[]))
    service = ExternalGenerationService({"openai": provider}, logging.getLogger("test"))

    with pytest.raises(ExternalProviderCapabilityError, match="requires an init image"):
        service.generate(request)


def test_generate_requires_mask_for_inpaint() -> None:
    model = _build_model(ExternalModelCapabilities(modes=["inpaint"]))
    request = _build_request(model=model, mode="inpaint", init_image=_make_image())
    provider = DummyProvider("openai", configured=True, result=ExternalGenerationResult(images=[]))
    service = ExternalGenerationService({"openai": provider}, logging.getLogger("test"))

    with pytest.raises(ExternalProviderCapabilityError, match="requires a mask"):
        service.generate(request)


def test_generate_validates_reference_images() -> None:
    model = _build_model(ExternalModelCapabilities(modes=["txt2img"], supports_reference_images=False))
    request = _build_request(
        model=model,
        reference_images=[ExternalReferenceImage(image=_make_image(), weight=0.8)],
    )
    provider = DummyProvider("openai", configured=True, result=ExternalGenerationResult(images=[]))
    service = ExternalGenerationService({"openai": provider}, logging.getLogger("test"))

    with pytest.raises(ExternalProviderCapabilityError, match="Reference images"):
        service.generate(request)


def test_generate_validates_limits() -> None:
    model = _build_model(
        ExternalModelCapabilities(
            modes=["txt2img"],
            supports_reference_images=True,
            max_reference_images=1,
            max_images_per_request=1,
        )
    )
    request = _build_request(
        model=model,
        num_images=2,
        reference_images=[
            ExternalReferenceImage(image=_make_image()),
            ExternalReferenceImage(image=_make_image()),
        ],
    )
    provider = DummyProvider("openai", configured=True, result=ExternalGenerationResult(images=[]))
    service = ExternalGenerationService({"openai": provider}, logging.getLogger("test"))

    with pytest.raises(ExternalProviderCapabilityError, match="supports at most"):
        service.generate(request)


def test_generate_validates_allowed_aspect_ratios() -> None:
    model = _build_model(
        ExternalModelCapabilities(
            modes=["txt2img"],
            allowed_aspect_ratios=["1:1", "16:9"],
            aspect_ratio_sizes={
                "1:1": ExternalImageSize(width=1024, height=1024),
                "16:9": ExternalImageSize(width=1344, height=768),
            },
        )
    )
    request = _build_request(model=model)
    provider = DummyProvider("openai", configured=True, result=ExternalGenerationResult(images=[]))
    service = ExternalGenerationService({"openai": provider}, logging.getLogger("test"))

    response = service.generate(request)
    assert response.images == []
    assert provider.last_request is not None
    assert provider.last_request.width == 1024
    assert provider.last_request.height == 1024


def test_generate_validates_allowed_aspect_ratios_with_bucket_sizes() -> None:
    model = _build_model(
        ExternalModelCapabilities(
            modes=["txt2img"],
            allowed_aspect_ratios=["1:1", "16:9"],
            aspect_ratio_sizes={
                "1:1": ExternalImageSize(width=1024, height=1024),
                "16:9": ExternalImageSize(width=1344, height=768),
            },
        )
    )
    request = _build_request(model=model, width=160, height=90)
    provider = DummyProvider("openai", configured=True, result=ExternalGenerationResult(images=[]))
    service = ExternalGenerationService({"openai": provider}, logging.getLogger("test"))

    response = service.generate(request)

    assert response.images == []
    assert provider.last_request is not None
    assert provider.last_request.width == 1344
    assert provider.last_request.height == 768


def test_generate_happy_path() -> None:
    model = _build_model(
        ExternalModelCapabilities(modes=["txt2img"], supports_negative_prompt=True, supports_seed=True)
    )
    request = _build_request(model=model, negative_prompt="", seed=42)
    result = ExternalGenerationResult(images=[ExternalGeneratedImage(image=_make_image(), seed=42)])
    provider = DummyProvider("openai", configured=True, result=result)
    service = ExternalGenerationService({"openai": provider}, logging.getLogger("test"))

    response = service.generate(request)

    assert response is result
    assert provider.last_request == request


def test_generate_resizes_inpaint_result_to_original_init_size() -> None:
    model = _build_model(ExternalModelCapabilities(modes=["inpaint"]))
    request = _build_request(
        model=model,
        mode="inpaint",
        width=128,
        height=128,
        init_image=_make_image(),
        mask_image=_make_image(),
    )
    generated_large = Image.new("RGB", (128, 128), color="black")
    result = ExternalGenerationResult(images=[ExternalGeneratedImage(image=generated_large, seed=1)])
    provider = DummyProvider("openai", configured=True, result=result)
    service = ExternalGenerationService({"openai": provider}, logging.getLogger("test"))

    response = service.generate(request)

    assert request.init_image is not None
    assert response.images[0].image.width == request.init_image.width
    assert response.images[0].image.height == request.init_image.height
    assert response.images[0].seed == 1
