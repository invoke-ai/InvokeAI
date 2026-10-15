import logging

import pytest
from PIL import Image

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.external_generation.errors import (
    ExternalProviderCapabilityError,
    ExternalProviderNotConfiguredError,
    ExternalProviderNotFoundError,
)
from invokeai.app.services.external_generation.external_generation_base import ExternalProvider
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGeneratedImage,
    ExternalGenerationRequest,
    ExternalGenerationResult,
    ExternalReferenceImage,
)
from invokeai.app.services.external_generation.external_generation_default import ExternalGenerationService
from invokeai.backend.model_manager.configs.external_api import (
    ExternalApiModelConfig,
    ExternalImageSize,
    ExternalModelCapabilities,
)
from invokeai.backend.model_manager.starter_models import STARTER_MODELS


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
    seed: int | None = None,
    num_images: int = 1,
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
        seed=seed,
        num_images=num_images,
        width=width,
        height=height,
        image_size=None,
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
        reference_images=[ExternalReferenceImage(image=_make_image())],
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
    model = _build_model(ExternalModelCapabilities(modes=["txt2img"], supports_seed=True))
    request = _build_request(model=model, seed=42)
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


def test_qwen_image_edit_max_enforces_three_reference_images() -> None:
    from invokeai.backend.model_manager.starter_models.external import alibabacloud_qwen_image_edit_max

    capabilities = alibabacloud_qwen_image_edit_max.capabilities
    assert capabilities is not None
    assert capabilities.max_reference_images == 3

    model = ExternalApiModelConfig(
        key="qwen_image_edit_max",
        name="Qwen Image Edit Max",
        provider_id="alibabacloud",
        provider_model_id="qwen-image-edit-max",
        capabilities=capabilities,
    )
    request = _build_request(
        model=model,
        reference_images=[ExternalReferenceImage(image=_make_image()) for _ in range(4)],
    )
    provider = DummyProvider("alibabacloud", configured=True, result=ExternalGenerationResult(images=[]))
    service = ExternalGenerationService({"alibabacloud": provider}, logging.getLogger("test"))

    with pytest.raises(ExternalProviderCapabilityError, match="supports at most 3 reference images"):
        service.generate(request)

    assert provider.last_request is None


def test_generate_snaps_unlisted_ratio_when_model_has_no_bucket_sizes() -> None:
    """Models sized by resolution preset list ratios but no sizes; an off-ratio request keeps its area."""
    model = _build_model(ExternalModelCapabilities(modes=["txt2img"], allowed_aspect_ratios=["1:1", "16:9"]))
    request = _build_request(model=model, width=1368, height=768)
    provider = DummyProvider("openai", configured=True, result=ExternalGenerationResult(images=[]))
    service = ExternalGenerationService({"openai": provider}, logging.getLogger("test"))

    service.generate(request)

    assert provider.last_request is not None
    assert (provider.last_request.width, provider.last_request.height) == (1360, 765)


def _starter_sizes() -> list[tuple[str, str, int, int]]:
    """Every fixed size a txt2img external starter model offers: its resolution presets and ratio buckets."""
    cases: list[tuple[str, str, int, int]] = []
    for starter in STARTER_MODELS:
        capabilities = starter.capabilities
        if capabilities is None or "txt2img" not in capabilities.modes:
            continue
        cases.extend(
            (starter.source, preset.label, preset.width, preset.height)
            for preset in capabilities.resolution_presets or []
        )
        cases.extend(
            (starter.source, ratio, size.width, size.height)
            for ratio, size in (capabilities.aspect_ratio_sizes or {}).items()
        )
    return cases


@pytest.mark.parametrize("source, label, width, height", _starter_sizes())
def test_starter_model_offered_sizes_are_accepted(source: str, label: str, width: int, height: int) -> None:
    """A size a model offers in its own presets must be accepted at (nearly) that size, even when its declared ratio
    is not in lowest terms ("21:9") and the preset is only approximately that ratio (1024x439)."""
    starter = next(model for model in STARTER_MODELS if model.source == source)
    assert starter.capabilities is not None
    provider_id, provider_model_id = source.removeprefix("external://").split("/", 1)
    model = ExternalApiModelConfig(
        key=source,
        name=starter.name,
        provider_id=provider_id,
        provider_model_id=provider_model_id,
        capabilities=starter.capabilities,
    )
    provider = DummyProvider(provider_id, configured=True, result=ExternalGenerationResult(images=[]))
    service = ExternalGenerationService({provider_id: provider}, logging.getLogger("test"))

    service.generate(_build_request(model=model, width=width, height=height))

    assert provider.last_request is not None
    # Snapping to an exact 21:9 moves in 21x9 px steps, so a small preset (512x219 -> 504x216) can shift ~2%.
    assert provider.last_request.width == pytest.approx(width, rel=0.02)
    assert provider.last_request.height == pytest.approx(height, rel=0.02)


def test_generate_matches_declared_ratio_not_in_lowest_terms() -> None:
    model = _build_model(ExternalModelCapabilities(modes=["txt2img"], allowed_aspect_ratios=["1:1", "21:9"]))
    request = _build_request(model=model, width=2100, height=900)
    provider = DummyProvider("openai", configured=True, result=ExternalGenerationResult(images=[]))
    service = ExternalGenerationService({"openai": provider}, logging.getLogger("test"))

    service.generate(request)

    assert provider.last_request == request


def test_generate_snapped_size_stays_within_max_image_size() -> None:
    model = _build_model(
        ExternalModelCapabilities(
            modes=["txt2img"],
            allowed_aspect_ratios=["1:1", "16:9"],
            max_image_size=ExternalImageSize(width=4096, height=4096),
        )
    )
    request = _build_request(model=model, width=4096, height=2400)
    provider = DummyProvider("openai", configured=True, result=ExternalGenerationResult(images=[]))
    service = ExternalGenerationService({"openai": provider}, logging.getLogger("test"))

    service.generate(request)

    assert provider.last_request is not None
    assert (provider.last_request.width, provider.last_request.height) == (4096, 2304)


def _starter_buckets() -> list[tuple[str, str, int, int]]:
    """Every ratio bucket a txt2img external starter model declares."""
    return [
        (starter.source, ratio, size.width, size.height)
        for starter in STARTER_MODELS
        if starter.capabilities is not None and "txt2img" in starter.capabilities.modes
        for ratio, size in (starter.capabilities.aspect_ratio_sizes or {}).items()
    ]


@pytest.mark.parametrize("source, ratio, width, height", _starter_buckets())
def test_starter_model_ratio_request_is_moved_to_its_bucket(source: str, ratio: str, width: int, height: int) -> None:
    """A request at a bucketed ratio but another size gets the bucket's size, even when the bucket's key is not in
    lowest terms (Seedream's "21:9" bucket for a 1568x672 request, which reduces to 7:3)."""
    starter = next(model for model in STARTER_MODELS if model.source == source)
    assert starter.capabilities is not None
    provider_id, provider_model_id = source.removeprefix("external://").split("/", 1)
    model = ExternalApiModelConfig(
        key=source,
        name=starter.name,
        provider_id=provider_id,
        provider_model_id=provider_model_id,
        capabilities=starter.capabilities,
    )
    provider = DummyProvider(provider_id, configured=True, result=ExternalGenerationResult(images=[]))
    service = ExternalGenerationService({provider_id: provider}, logging.getLogger("test"))
    left, right = (int(part) for part in ratio.split(":"))

    service.generate(_build_request(model=model, width=left * 16, height=right * 16))

    assert provider.last_request is not None
    assert (provider.last_request.width, provider.last_request.height) == (width, height)


@pytest.mark.parametrize(
    "allowed, sizes, width, height, expected",
    [
        # A literally allowed ratio keeps its size even when a bucket is keyed in other terms.
        (["7:3"], {"21:9": (1536, 672)}, 1400, 600, (1400, 600)),
        (["1:1"], {"2:2": (512, 512)}, 1024, 1024, (1024, 1024)),
        (["21:9", "7:3"], {"21:9": (1680, 720)}, 1400, 600, (1400, 600)),
        # An exact bucket key wins over an equivalent one declared first.
        (["21:9", "7:3"], {"21:9": (1680, 720), "7:3": (1400, 600)}, 700, 300, (1400, 600)),
        # A bucket whose ratio is not allowed is never used; the request snaps to an allowed bucket.
        (["1:1"], {"21:9": (1536, 672), "1:1": (1024, 1024)}, 735, 315, (1024, 1024)),
        # A ratio key that str.isdigit() accepts but int() rejects is ignored, not a crash.
        (["²:1", "1:1"], None, 600, 600, (600, 600)),
        # So is one too long for int() to parse.
        (["1:1", "9" * 5000 + ":1"], None, 600, 1400, (917, 917)),
    ],
)
def test_generate_custom_ratio_declarations_keep_previous_sizes(
    allowed: list[str],
    sizes: dict[str, tuple[int, int]] | None,
    width: int,
    height: int,
    expected: tuple[int, int],
) -> None:
    """User-installed models may declare ratios and buckets in any spelling; equivalence matching must not change
    the size a request was previously sent at."""
    model = _build_model(
        ExternalModelCapabilities(
            modes=["txt2img"],
            allowed_aspect_ratios=allowed,
            aspect_ratio_sizes=None
            if sizes is None
            else {ratio: ExternalImageSize(width=w, height=h) for ratio, (w, h) in sizes.items()},
        )
    )
    provider = DummyProvider("openai", configured=True, result=ExternalGenerationResult(images=[]))
    service = ExternalGenerationService({"openai": provider}, logging.getLogger("test"))

    service.generate(_build_request(model=model, width=width, height=height))

    assert provider.last_request is not None
    assert (provider.last_request.width, provider.last_request.height) == expected
