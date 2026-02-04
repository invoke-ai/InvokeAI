from __future__ import annotations

from logging import Logger

from PIL import Image
from PIL.Image import Image as PILImageType

from invokeai.app.services.external_generation.errors import (
    ExternalProviderCapabilityError,
    ExternalProviderNotConfiguredError,
    ExternalProviderNotFoundError,
)
from invokeai.app.services.external_generation.external_generation_base import (
    ExternalGenerationServiceBase,
    ExternalProvider,
)
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGenerationRequest,
    ExternalGenerationResult,
    ExternalProviderStatus,
)
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig, ExternalImageSize
from invokeai.backend.model_manager.starter_models import STARTER_MODELS


class ExternalGenerationService(ExternalGenerationServiceBase):
    def __init__(self, providers: dict[str, ExternalProvider], logger: Logger) -> None:
        self._providers = providers
        self._logger = logger

    def generate(self, request: ExternalGenerationRequest) -> ExternalGenerationResult:
        provider = self._providers.get(request.model.provider_id)
        if provider is None:
            raise ExternalProviderNotFoundError(f"No external provider registered for '{request.model.provider_id}'")

        if not provider.is_configured():
            raise ExternalProviderNotConfiguredError(f"Provider '{request.model.provider_id}' is missing credentials")

        request = self._refresh_model_capabilities(request)
        request = self._bucket_request(request)

        self._validate_request(request)
        return provider.generate(request)

    def get_provider_statuses(self) -> dict[str, ExternalProviderStatus]:
        return {provider_id: provider.get_status() for provider_id, provider in self._providers.items()}

    def _validate_request(self, request: ExternalGenerationRequest) -> None:
        capabilities = request.model.capabilities

        self._logger.debug(
            "Validating external request provider=%s model=%s mode=%s supported=%s",
            request.model.provider_id,
            request.model.provider_model_id,
            request.mode,
            capabilities.modes,
        )

        if request.mode not in capabilities.modes:
            raise ExternalProviderCapabilityError(f"Mode '{request.mode}' is not supported by {request.model.name}")

        if request.negative_prompt and not capabilities.supports_negative_prompt:
            raise ExternalProviderCapabilityError(f"Negative prompts are not supported by {request.model.name}")

        if request.seed is not None and not capabilities.supports_seed:
            raise ExternalProviderCapabilityError(f"Seed control is not supported by {request.model.name}")

        if request.guidance is not None and not capabilities.supports_guidance:
            raise ExternalProviderCapabilityError(f"Guidance is not supported by {request.model.name}")

        if request.reference_images and not capabilities.supports_reference_images:
            raise ExternalProviderCapabilityError(f"Reference images are not supported by {request.model.name}")

        if capabilities.max_reference_images is not None:
            if len(request.reference_images) > capabilities.max_reference_images:
                raise ExternalProviderCapabilityError(
                    f"{request.model.name} supports at most {capabilities.max_reference_images} reference images"
                )

        if capabilities.max_images_per_request is not None and request.num_images > capabilities.max_images_per_request:
            raise ExternalProviderCapabilityError(
                f"{request.model.name} supports at most {capabilities.max_images_per_request} images per request"
            )

        if capabilities.max_image_size is not None:
            if request.width > capabilities.max_image_size.width or request.height > capabilities.max_image_size.height:
                raise ExternalProviderCapabilityError(
                    f"{request.model.name} supports a maximum size of {capabilities.max_image_size.width}x{capabilities.max_image_size.height}"
                )

        if capabilities.allowed_aspect_ratios:
            aspect_ratio = _format_aspect_ratio(request.width, request.height)
            if aspect_ratio not in capabilities.allowed_aspect_ratios:
                size_ratio = None
                if capabilities.aspect_ratio_sizes:
                    size_ratio = _ratio_for_size(request.width, request.height, capabilities.aspect_ratio_sizes)
                if size_ratio is None or size_ratio not in capabilities.allowed_aspect_ratios:
                    ratio_label = size_ratio or aspect_ratio
                    raise ExternalProviderCapabilityError(
                        f"{request.model.name} does not support aspect ratio {ratio_label}"
                    )

        required_modes = capabilities.input_image_required_for or ["img2img", "inpaint"]
        if request.mode in required_modes and request.init_image is None:
            raise ExternalProviderCapabilityError(
                f"Mode '{request.mode}' requires an init image for {request.model.name}"
            )

        if request.mode == "inpaint" and request.mask_image is None:
            raise ExternalProviderCapabilityError(
                f"Mode '{request.mode}' requires a mask image for {request.model.name}"
            )

    def _refresh_model_capabilities(self, request: ExternalGenerationRequest) -> ExternalGenerationRequest:
        try:
            from invokeai.app.api.dependencies import ApiDependencies

            record = ApiDependencies.invoker.services.model_manager.store.get_model(request.model.key)
        except Exception:
            record = None

        if not isinstance(record, ExternalApiModelConfig):
            return request

        if record.key != request.model.key:
            return request

        if record.provider_id != request.model.provider_id:
            return request

        if record.provider_model_id != request.model.provider_model_id:
            return request

        record = _apply_starter_overrides(record)

        if record == request.model:
            return request

        return ExternalGenerationRequest(
            model=record,
            mode=request.mode,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            num_images=request.num_images,
            width=request.width,
            height=request.height,
            steps=request.steps,
            guidance=request.guidance,
            init_image=request.init_image,
            mask_image=request.mask_image,
            reference_images=request.reference_images,
            metadata=request.metadata,
        )

    def _bucket_request(self, request: ExternalGenerationRequest) -> ExternalGenerationRequest:
        capabilities = request.model.capabilities
        if not capabilities.allowed_aspect_ratios:
            return request

        aspect_ratio = _format_aspect_ratio(request.width, request.height)
        size = None
        if capabilities.aspect_ratio_sizes:
            size = capabilities.aspect_ratio_sizes.get(aspect_ratio)

        if size is not None:
            if request.width == size.width and request.height == size.height:
                return request
            return self._bucket_to_size(request, size.width, size.height, aspect_ratio)

        if aspect_ratio in capabilities.allowed_aspect_ratios:
            return request

        if not capabilities.aspect_ratio_sizes:
            return request

        closest = _select_closest_ratio(
            request.width,
            request.height,
            capabilities.allowed_aspect_ratios,
        )
        if closest is None:
            return request

        size = capabilities.aspect_ratio_sizes.get(closest)
        if size is None:
            return request

        return self._bucket_to_size(request, size.width, size.height, closest)

    def _bucket_to_size(
        self,
        request: ExternalGenerationRequest,
        width: int,
        height: int,
        ratio: str,
    ) -> ExternalGenerationRequest:
        self._logger.info(
            "Bucketing external request provider=%s model=%s %sx%s -> %sx%s (ratio %s)",
            request.model.provider_id,
            request.model.provider_model_id,
            request.width,
            request.height,
            width,
            height,
            ratio,
        )

        return ExternalGenerationRequest(
            model=request.model,
            mode=request.mode,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            num_images=request.num_images,
            width=width,
            height=height,
            steps=request.steps,
            guidance=request.guidance,
            init_image=_resize_image(request.init_image, width, height, "RGB"),
            mask_image=_resize_image(request.mask_image, width, height, "L"),
            reference_images=request.reference_images,
            metadata=request.metadata,
        )


def _format_aspect_ratio(width: int, height: int) -> str:
    divisor = _gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def _select_closest_ratio(width: int, height: int, ratios: list[str]) -> str | None:
    ratio = width / height
    parsed: list[tuple[str, float]] = []
    for value in ratios:
        parsed_ratio = _parse_ratio(value)
        if parsed_ratio is not None:
            parsed.append((value, parsed_ratio))
    if not parsed:
        return None
    return min(parsed, key=lambda item: abs(item[1] - ratio))[0]


def _ratio_for_size(width: int, height: int, sizes: dict[str, ExternalImageSize]) -> str | None:
    for ratio, size in sizes.items():
        if size.width == width and size.height == height:
            return ratio
    return None


def _parse_ratio(value: str) -> float | None:
    if ":" not in value:
        return None
    left, right = value.split(":", 1)
    try:
        numerator = float(left)
        denominator = float(right)
    except ValueError:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _resize_image(image: PILImageType | None, width: int, height: int, mode: str) -> PILImageType | None:
    if image is None:
        return None
    if image.width == width and image.height == height:
        return image
    return image.convert(mode).resize((width, height), Image.Resampling.LANCZOS)


def _apply_starter_overrides(model: ExternalApiModelConfig) -> ExternalApiModelConfig:
    source = model.source or f"external://{model.provider_id}/{model.provider_model_id}"
    starter_match = next((starter for starter in STARTER_MODELS if starter.source == source), None)
    if starter_match is None:
        return model
    updates: dict[str, object] = {}
    if starter_match.capabilities is not None:
        updates["capabilities"] = starter_match.capabilities
    if starter_match.default_settings is not None:
        updates["default_settings"] = starter_match.default_settings
    if not updates:
        return model
    return model.model_copy(update=updates)
