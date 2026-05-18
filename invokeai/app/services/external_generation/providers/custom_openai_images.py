from __future__ import annotations

import io

import requests
from PIL import Image
from PIL.Image import Image as PILImageType

from invokeai.app.services.external_generation.errors import (
    ExternalProviderRateLimitError,
    ExternalProviderRequestError,
)
from invokeai.app.services.external_generation.external_generation_base import ExternalProvider
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGeneratedImage,
    ExternalGenerationRequest,
    ExternalGenerationResult,
    ExternalProviderStatus,
)
from invokeai.app.services.external_generation.image_utils import decode_image_base64
from invokeai.backend.model_manager.configs.external_api import (
    ExternalApiModelDefaultSettings,
    ExternalImageSize,
    ExternalModelCapabilities,
    ExternalResolutionPreset,
)

CUSTOM_OPENAI_IMAGES_PROVIDER_ID = "custom_openai_images"
CUSTOM_OPENAI_IMAGES_ASPECT_RATIOS = [
    "1:1",
    "1:4",
    "1:8",
    "2:3",
    "3:2",
    "3:4",
    "4:1",
    "4:3",
    "4:5",
    "5:4",
    "8:1",
    "9:16",
    "16:9",
    "21:9",
]
CUSTOM_OPENAI_IMAGES_MAX_SIZE = ExternalImageSize(width=4096, height=4096)
_DIMENSION_MULTIPLE = 8


def _build_resolution_presets() -> list[ExternalResolutionPreset]:
    presets: list[ExternalResolutionPreset] = []
    for image_size, base in (("1K", 1024), ("2K", 2048), ("4K", 4096)):
        for aspect_ratio in CUSTOM_OPENAI_IMAGES_ASPECT_RATIOS:
            width_part, height_part = (int(part) for part in aspect_ratio.split(":"))
            scale = max(1, base // max(width_part, height_part))
            scale = max(1, (scale // _DIMENSION_MULTIPLE) * _DIMENSION_MULTIPLE)
            width = width_part * scale
            height = height_part * scale
            presets.append(
                ExternalResolutionPreset(
                    label=f"{aspect_ratio} ({image_size}) - {width}x{height}",
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    width=width,
                    height=height,
                )
            )
    return presets


CUSTOM_OPENAI_IMAGES_RESOLUTION_PRESETS = _build_resolution_presets()
CUSTOM_OPENAI_IMAGES_CAPABILITIES = ExternalModelCapabilities(
    modes=["txt2img", "img2img"],
    supports_reference_images=True,
    max_images_per_request=10,
    max_image_size=CUSTOM_OPENAI_IMAGES_MAX_SIZE,
    resolution_presets=CUSTOM_OPENAI_IMAGES_RESOLUTION_PRESETS,
)
CUSTOM_OPENAI_IMAGES_DEFAULT_SETTINGS = ExternalApiModelDefaultSettings(width=1024, height=1024, num_images=1)


class CustomOpenAIImagesProvider(ExternalProvider):
    provider_id = CUSTOM_OPENAI_IMAGES_PROVIDER_ID

    def is_configured(self) -> bool:
        return bool(self._app_config.external_custom_openai_images_api_key) and bool(
            self._app_config.external_custom_openai_images_base_url
        )

    def get_status(self) -> ExternalProviderStatus:
        api_key = self._app_config.external_custom_openai_images_api_key
        base_url = self._app_config.external_custom_openai_images_base_url
        configured = bool(api_key) and bool(base_url)
        message = None if configured else "Custom OpenAI Images-compatible provider requires an API key and base URL"
        return ExternalProviderStatus(provider_id=self.provider_id, configured=configured, message=message)

    def generate(self, request: ExternalGenerationRequest) -> ExternalGenerationResult:
        api_key = self._app_config.external_custom_openai_images_api_key
        if not api_key:
            raise ExternalProviderRequestError("Custom OpenAI Images-compatible API key is not configured")
        base_url = self._app_config.external_custom_openai_images_base_url
        if not base_url:
            raise ExternalProviderRequestError("Custom OpenAI Images-compatible base URL is not configured")

        model_id = request.model.provider_model_id
        size = f"{request.width}x{request.height}"
        headers = {"Authorization": f"Bearer {api_key}"}
        use_edits_endpoint = request.mode != "txt2img" or bool(request.reference_images)
        opts = request.provider_options or {}

        if not use_edits_endpoint:
            payload: dict[str, object] = {
                "model": model_id,
                "prompt": request.prompt,
                "n": request.num_images,
                "size": size,
            }
            if opts.get("quality") and opts["quality"] != "auto":
                payload["quality"] = opts["quality"]
            if opts.get("background") and opts["background"] != "auto":
                payload["background"] = opts["background"]
            response = requests.post(
                _build_images_url(base_url, "generations"),
                headers=headers,
                json=payload,
                timeout=120,
            )
        else:
            images: list[PILImageType] = []
            if request.init_image is not None:
                images.append(request.init_image)
            images.extend(reference.image for reference in request.reference_images)
            if not images:
                raise ExternalProviderRequestError("Custom OpenAI Images-compatible edits require at least one image")

            files: list[tuple[str, tuple[str, io.BytesIO, str]]] = []
            image_field_name = "image" if len(images) == 1 else "image[]"
            for index, image in enumerate(images):
                image_buffer = io.BytesIO()
                image.save(image_buffer, format="PNG")
                image_buffer.seek(0)
                files.append((image_field_name, (f"image_{index}.png", image_buffer, "image/png")))

            if request.mask_image is not None:
                mask_buffer = io.BytesIO()
                request.mask_image.save(mask_buffer, format="PNG")
                mask_buffer.seek(0)
                files.append(("mask", ("mask.png", mask_buffer, "image/png")))

            data: dict[str, object] = {
                "model": model_id,
                "prompt": request.prompt,
                "n": request.num_images,
                "size": size,
            }
            if opts.get("quality") and opts["quality"] != "auto":
                data["quality"] = opts["quality"]
            if opts.get("background") and opts["background"] != "auto":
                data["background"] = opts["background"]
            if opts.get("input_fidelity"):
                data["input_fidelity"] = opts["input_fidelity"]

            response = requests.post(
                _build_images_url(base_url, "edits"),
                headers=headers,
                data=data,
                files=files,
                timeout=120,
            )

        if not response.ok:
            if response.status_code == 429:
                retry_after = _parse_retry_after(response.headers.get("retry-after"))
                raise ExternalProviderRateLimitError(
                    f"Custom OpenAI Images-compatible provider rate limit exceeded. {f'Retry after {retry_after:.0f}s.' if retry_after else 'Please try again later.'}",
                    retry_after=retry_after,
                )
            raise ExternalProviderRequestError(
                f"Custom OpenAI Images-compatible request failed with status {response.status_code}: {response.text}"
            )

        response_payload = response.json()
        if not isinstance(response_payload, dict):
            raise ExternalProviderRequestError("Custom OpenAI Images-compatible response payload was not a JSON object")
        data_items = response_payload.get("data")
        if not isinstance(data_items, list):
            raise ExternalProviderRequestError("Custom OpenAI Images-compatible response payload missing image data")

        images = _extract_response_images(data_items, api_key, request.num_images)
        if not images:
            raise ExternalProviderRequestError("Custom OpenAI Images-compatible response contained no images")

        return ExternalGenerationResult(
            images=[ExternalGeneratedImage(image=image, seed=request.seed) for image in images],
            seed_used=request.seed,
            provider_request_id=response.headers.get("x-request-id"),
            provider_metadata={"model": model_id},
        )


def _build_images_url(base_url: str, operation: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/images"):
        return f"{normalized}/{operation}"
    if normalized.endswith("/v1"):
        return f"{normalized}/images/{operation}"
    return f"{normalized}/v1/images/{operation}"


def _image_from_response_item(item: dict[str, object], api_key: str) -> PILImageType | None:
    encoded = item.get("b64_json")
    if isinstance(encoded, str) and encoded:
        return decode_image_base64(encoded)

    image_url = item.get("url")
    if not isinstance(image_url, str) or not image_url:
        return None
    if image_url.startswith("data:image/"):
        _, _, encoded_data = image_url.partition(",")
        if not encoded_data:
            return None
        return decode_image_base64(encoded_data)

    response = requests.get(image_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=120)
    if not response.ok:
        raise ExternalProviderRequestError(
            f"Custom OpenAI Images-compatible image download failed with status {response.status_code}: {response.text}"
        )
    try:
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise ExternalProviderRequestError("Custom OpenAI Images-compatible image URL did not return an image") from e


def _extract_response_images(data_items: list[object], api_key: str, limit: int) -> list[PILImageType]:
    images: list[PILImageType] = []
    for item in data_items:
        if len(images) >= limit:
            break
        if not isinstance(item, dict):
            continue
        image = _image_from_response_item(item, api_key)
        if image is not None:
            images.append(image)
    return images


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
