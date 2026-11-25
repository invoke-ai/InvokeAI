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
)
from invokeai.app.services.external_generation.image_utils import decode_image_base64

_DEFAULT_BASE_URL = "https://inference-api.worldrouter.ai"
# GPT Image requests routinely take more than a minute; the OpenAI provider allows the same.
_REQUEST_TIMEOUT = 300
_DOWNLOAD_TIMEOUT = 60
_DOWNLOAD_MAX_BYTES = 32 * 1024 * 1024


class WorldRouterProvider(ExternalProvider):
    """Text-to-image generation through WorldRouter's OpenAI-compatible Images API."""

    provider_id = "worldrouter"

    def is_configured(self) -> bool:
        return bool(self._app_config.external_worldrouter_api_key)

    def generate(self, request: ExternalGenerationRequest) -> ExternalGenerationResult:
        api_key = self._app_config.external_worldrouter_api_key
        if not api_key:
            raise ExternalProviderRequestError("WorldRouter API key is not configured")

        model_id = request.model.provider_model_id
        base_url = (self._app_config.external_worldrouter_base_url or _DEFAULT_BASE_URL).rstrip("/")
        payload = {
            "model": model_id,
            "prompt": request.prompt,
            "n": request.num_images,
            "size": f"{request.width}x{request.height}",
        }
        try:
            response = requests.post(
                f"{base_url}/v1/images/generations",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise ExternalProviderRequestError(f"WorldRouter image request failed: {exc}") from exc

        if not response.ok:
            if response.status_code == 429:
                raise ExternalProviderRateLimitError(
                    "WorldRouter rate limit exceeded",
                    retry_after=_parse_retry_after(response.headers.get("retry-after")),
                )
            raise ExternalProviderRequestError(
                f"WorldRouter request failed with status {response.status_code}: {response.text}"
            )

        try:
            response_payload = response.json()
        except ValueError as exc:
            raise ExternalProviderRequestError("WorldRouter response was not valid JSON") from exc
        data_items = response_payload.get("data") if isinstance(response_payload, dict) else None
        if not isinstance(data_items, list):
            raise ExternalProviderRequestError("WorldRouter response payload missing image data")

        # The gateway forwards each upstream's Images API response, which carries either inline
        # base64 data or a temporary download URL depending on the model.
        images: list[ExternalGeneratedImage] = []
        for item in data_items:
            if not isinstance(item, dict):
                continue
            encoded = item.get("b64_json")
            url = item.get("url")
            if isinstance(encoded, str) and encoded:
                image = decode_image_base64(encoded)
            elif isinstance(url, str) and url:
                image = _download_image(url)
            else:
                continue
            images.append(ExternalGeneratedImage(image=image, seed=request.seed))

        if not images:
            raise ExternalProviderRequestError("WorldRouter response contained no images")

        return ExternalGenerationResult(
            images=images,
            seed_used=request.seed,
            provider_request_id=response.headers.get("x-request-id"),
            provider_metadata={"model": model_id},
        )


def _download_image(url: str) -> PILImageType:
    try:
        response = requests.get(url, timeout=_DOWNLOAD_TIMEOUT, stream=True)
    except requests.RequestException as exc:
        raise ExternalProviderRequestError(f"Failed to download image from WorldRouter: {exc}") from exc

    with response:
        if not response.ok:
            raise ExternalProviderRequestError(
                f"Failed to download image from WorldRouter (status {response.status_code})"
            )
        buffer = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            buffer.extend(chunk)
            if len(buffer) > _DOWNLOAD_MAX_BYTES:
                raise ExternalProviderRequestError(f"WorldRouter image exceeds {_DOWNLOAD_MAX_BYTES} byte cap")

    try:
        return Image.open(io.BytesIO(bytes(buffer))).convert("RGB")
    except Exception as exc:
        raise ExternalProviderRequestError("WorldRouter output was not a valid image") from exc


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
