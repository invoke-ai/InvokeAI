from __future__ import annotations

import binascii

import requests
from PIL import UnidentifiedImageError
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
from invokeai.app.services.external_generation.image_utils import decode_image_base64, decode_image_bytes
from invokeai.app.util.ssrf import UnsafeDownloadURLException, build_guarded_session

_DEFAULT_BASE_URL = "https://inference-api.worldrouter.ai"
# GPT Image requests routinely take more than a minute; the OpenAI provider allows the same.
_REQUEST_TIMEOUT = 300
_DOWNLOAD_TIMEOUT = 60
_DOWNLOAD_MAX_BYTES = 32 * 1024 * 1024
# Error bodies come from upstream services; keep enough to diagnose without echoing a whole page.
_ERROR_BODY_MAX_CHARS = 500


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
                f"WorldRouter request failed with status {response.status_code}: "
                f"{response.text[:_ERROR_BODY_MAX_CHARS]}"
            )

        try:
            response_payload = response.json()
        except ValueError as exc:
            raise ExternalProviderRequestError("WorldRouter response was not valid JSON") from exc
        data_items = response_payload.get("data") if isinstance(response_payload, dict) else None
        if not isinstance(data_items, list):
            raise ExternalProviderRequestError("WorldRouter response payload missing image data")

        # The gateway forwards each upstream's Images API response, which carries either inline
        # base64 data or a temporary download URL depending on the model. The images are already
        # billed, so one failed download does not discard the others.
        images: list[ExternalGeneratedImage] = []
        first_error: ExternalProviderRequestError | None = None
        for item in data_items:
            if not isinstance(item, dict):
                continue
            try:
                image = _decode_item(item)
            except ExternalProviderRequestError as exc:
                self._logger.warning("Skipping a WorldRouter image: %s", exc)
                first_error = first_error or exc
                continue
            if image is not None:
                images.append(ExternalGeneratedImage(image=image, seed=request.seed))

        if not images:
            if first_error is not None:
                raise first_error
            raise ExternalProviderRequestError("WorldRouter response contained no images")

        return ExternalGenerationResult(
            images=images,
            seed_used=request.seed,
            provider_request_id=response.headers.get("x-request-id"),
            provider_metadata={"model": model_id},
        )


def _decode_item(item: dict[str, object]) -> PILImageType | None:
    encoded = item.get("b64_json")
    url = item.get("url")
    if isinstance(encoded, str) and encoded:
        try:
            return decode_image_base64(encoded)
        except (binascii.Error, ValueError, UnidentifiedImageError) as exc:
            raise ExternalProviderRequestError("WorldRouter returned invalid base64 image data") from exc
    if isinstance(url, str) and url:
        return _download_image(url)
    return None


def _download_image(url: str) -> PILImageType:
    """Download a result image through the SSRF-guarded session, with a size cap."""
    try:
        with build_guarded_session() as session:
            response = session.get(url, timeout=_DOWNLOAD_TIMEOUT, stream=True)
            with response:
                if not response.ok:
                    raise ExternalProviderRequestError(
                        f"Failed to download image from WorldRouter (status {response.status_code})"
                    )
                content_length = response.headers.get("Content-Length")
                if content_length is not None and content_length.isdigit():
                    if int(content_length) > _DOWNLOAD_MAX_BYTES:
                        raise ExternalProviderRequestError(f"WorldRouter image exceeds {_DOWNLOAD_MAX_BYTES} byte cap")
                buffer = bytearray()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    buffer.extend(chunk)
                    if len(buffer) > _DOWNLOAD_MAX_BYTES:
                        raise ExternalProviderRequestError(f"WorldRouter image exceeds {_DOWNLOAD_MAX_BYTES} byte cap")
    except UnsafeDownloadURLException as exc:
        raise ExternalProviderRequestError("WorldRouter returned an unsafe image URL") from exc
    except requests.RequestException as exc:
        raise ExternalProviderRequestError("Failed to download image from WorldRouter") from exc

    try:
        return decode_image_bytes(bytes(buffer))
    except (UnidentifiedImageError, OSError) as exc:
        raise ExternalProviderRequestError("WorldRouter output was not a valid image") from exc


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
