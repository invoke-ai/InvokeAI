from __future__ import annotations

import io
import time
from dataclasses import dataclass
from math import gcd
from typing import Any, Literal

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

_DEFAULT_BASE_URL = "https://api.atlascloud.ai"
_REQUEST_TIMEOUT = 30
_POLL_INTERVAL = 3.0
_POLL_TIMEOUT = 300.0
_DOWNLOAD_TIMEOUT = 60
_DOWNLOAD_MAX_BYTES = 32 * 1024 * 1024
_SUCCESS_STATUSES = {"completed", "succeeded"}
_FAILURE_STATUSES = {"canceled", "cancelled", "failed"}

# Atlas Cloud fronts many image models behind one endpoint, but their request schemas
# differ: output dimensions arrive as an explicit "size" string, a named "image_size"
# preset, or an "aspect_ratio", and batch size is spelled "num_images", "n", or is not
# supported at all. Recording those differences per model keeps each starter model's
# payload valid for the model it targets. Models absent from this table fall back to
# the explicit-size schema used by most Atlas Cloud image models, so custom installs
# via `external://atlascloud/<model_id>` keep working.
_SizeStyle = Literal["size", "image_size", "aspect_ratio"]

# Size presets spelled the way the upstream models spell them. Note that `portrait_3_4`
# and `portrait_4_3` both describe a 3:4 portrait: the prefix fixes the orientation, not
# the order of the digits.
_STANDARD_SIZE_PRESETS = (
    "square_hd",
    "square",
    "portrait_3_4",
    "portrait_9_16",
    "landscape_4_3",
    "landscape_16_9",
)
_HIDREAM_SIZE_PRESETS = (
    "square_hd",
    "square",
    "portrait_4_3",
    "portrait_16_9",
    "landscape_4_3",
    "landscape_16_9",
)


@dataclass(frozen=True)
class _AtlasModelSchema:
    """Request fields accepted by a single Atlas Cloud image model."""

    size_style: _SizeStyle = "size"
    size_presets: tuple[str, ...] = ()
    num_images_field: str | None = "num_images"
    supports_seed: bool = True
    resolution_field: str | None = None


_DEFAULT_MODEL_SCHEMA = _AtlasModelSchema()

_MODEL_SCHEMAS: dict[str, _AtlasModelSchema] = {
    # Explicit "<width>*<height>" size string
    "black-forest-labs/flux-schnell": _AtlasModelSchema(),
    "black-forest-labs/flux-dev": _AtlasModelSchema(),
    "black-forest-labs/flux-2-pro/text-to-image": _AtlasModelSchema(num_images_field=None),
    "qwen-image-3.0/text-to-image": _AtlasModelSchema(num_images_field="n"),
    "z-image/turbo": _AtlasModelSchema(num_images_field=None),
    "microsoft/mai-image-2.5/text-to-image": _AtlasModelSchema(num_images_field=None, supports_seed=False),
    # Named "image_size" preset
    "ideogram/v4/turbo/text-to-image": _AtlasModelSchema(
        size_style="image_size", size_presets=_STANDARD_SIZE_PRESETS, num_images_field=None
    ),
    "ideogram/v4/quality/text-to-image": _AtlasModelSchema(
        size_style="image_size", size_presets=_STANDARD_SIZE_PRESETS, num_images_field=None
    ),
    "krea-2-turbo/text-to-image": _AtlasModelSchema(size_style="image_size", size_presets=_STANDARD_SIZE_PRESETS),
    "hidream-o1-1.5/text-to-image": _AtlasModelSchema(
        size_style="image_size",
        size_presets=_HIDREAM_SIZE_PRESETS,
        num_images_field=None,
        supports_seed=False,
    ),
    # "aspect_ratio" string
    "xai/grok-imagine-image-2.0/text-to-image": _AtlasModelSchema(size_style="aspect_ratio", supports_seed=False),
    "google/nano-banana-2/text-to-image": _AtlasModelSchema(
        size_style="aspect_ratio", num_images_field=None, resolution_field="resolution"
    ),
}


class AtlasCloudProvider(ExternalProvider):
    provider_id = "atlascloud"

    def is_configured(self) -> bool:
        return bool(self._app_config.external_atlascloud_api_key)

    def generate(self, request: ExternalGenerationRequest) -> ExternalGenerationResult:
        api_key = self._app_config.external_atlascloud_api_key
        if not api_key:
            raise ExternalProviderRequestError("Atlas Cloud API key is not configured")

        base_url = (self._app_config.external_atlascloud_base_url or _DEFAULT_BASE_URL).rstrip("/")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(request)

        submit_url = f"{base_url}/api/v1/model/generateImage"
        try:
            response = requests.post(submit_url, headers=headers, json=payload, timeout=_REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            raise ExternalProviderRequestError(f"Atlas Cloud image submission failed: {exc}") from exc

        self._raise_for_error(response, "image submission")
        prediction = self._response_data(response, "image submission")
        prediction_id = prediction.get("id")
        if not isinstance(prediction_id, str) or not prediction_id:
            raise ExternalProviderRequestError("Atlas Cloud image submission response missing prediction id")

        poll_url = self._get_poll_url(prediction, base_url, prediction_id)
        completed = self._poll_prediction(poll_url, headers, prediction_id)
        output_urls = completed.get("output", completed.get("outputs"))
        if not isinstance(output_urls, list):
            raise ExternalProviderRequestError("Atlas Cloud completed prediction contained no image outputs")

        images: list[ExternalGeneratedImage] = []
        for output_url in output_urls:
            if isinstance(output_url, str) and output_url:
                images.append(ExternalGeneratedImage(image=self._download_image(output_url), seed=request.seed))

        if not images:
            raise ExternalProviderRequestError("Atlas Cloud completed prediction contained no downloadable images")

        return ExternalGenerationResult(
            images=images,
            seed_used=request.seed,
            provider_request_id=prediction_id,
            provider_metadata={
                "model": request.model.provider_model_id,
                "status": str(completed.get("status", "succeeded")),
            },
        )

    def _build_payload(self, request: ExternalGenerationRequest) -> dict[str, object]:
        """Build a submission payload using the request fields the target model accepts."""
        model_id = request.model.provider_model_id
        schema = _MODEL_SCHEMAS.get(model_id, _DEFAULT_MODEL_SCHEMA)
        payload: dict[str, object] = {"model": model_id, "prompt": request.prompt}

        if schema.size_style == "size":
            payload["size"] = f"{request.width}*{request.height}"
        elif schema.size_style == "image_size":
            payload["image_size"] = _select_size_preset(request.width, request.height, schema.size_presets)
        else:
            aspect_ratio = _select_aspect_ratio(
                request.width, request.height, request.model.capabilities.allowed_aspect_ratios
            )
            if aspect_ratio is not None:
                payload["aspect_ratio"] = aspect_ratio

        # Resolution presets are named "1K"/"2K"/"4K" in Invoke and lowercase upstream.
        if schema.resolution_field is not None and request.image_size is not None:
            payload[schema.resolution_field] = request.image_size.lower()

        if schema.num_images_field is not None:
            payload[schema.num_images_field] = request.num_images

        if schema.supports_seed and request.seed is not None:
            payload["seed"] = request.seed

        return payload

    def _poll_prediction(
        self,
        poll_url: str,
        headers: dict[str, str],
        prediction_id: str,
    ) -> dict[str, Any]:
        started_at = time.monotonic()

        while True:
            if time.monotonic() - started_at > _POLL_TIMEOUT:
                raise ExternalProviderRequestError(
                    f"Atlas Cloud prediction {prediction_id} timed out after {_POLL_TIMEOUT:.0f}s"
                )

            try:
                response = requests.get(poll_url, headers=headers, timeout=_REQUEST_TIMEOUT)
            except requests.RequestException as exc:
                raise ExternalProviderRequestError(f"Atlas Cloud prediction polling failed: {exc}") from exc

            self._raise_for_error(response, "prediction polling")
            prediction = self._response_data(response, "prediction polling")
            status = str(prediction.get("status", "")).lower()

            if status in _SUCCESS_STATUSES:
                return prediction
            if status in _FAILURE_STATUSES:
                detail = prediction.get("error") or prediction.get("logs") or "Unknown provider error"
                raise ExternalProviderRequestError(f"Atlas Cloud prediction {prediction_id} failed: {detail}")

            self._logger.debug("Atlas Cloud prediction %s status: %s", prediction_id, status or "unknown")
            time.sleep(_POLL_INTERVAL)

    @staticmethod
    def _get_poll_url(prediction: dict[str, Any], base_url: str, prediction_id: str) -> str:
        urls = prediction.get("urls")
        if isinstance(urls, dict):
            result_url = urls.get("result")
            if isinstance(result_url, str) and result_url:
                if result_url.startswith("/"):
                    return f"{base_url}{result_url}"
                return result_url
        return f"{base_url}/api/v1/model/result/{prediction_id}"

    @staticmethod
    def _response_data(response: requests.Response, operation: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ExternalProviderRequestError(f"Atlas Cloud {operation} returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ExternalProviderRequestError(f"Atlas Cloud {operation} response was not a JSON object")

        if "data" in payload:
            code = payload.get("code")
            if code not in (None, 0, 200):
                detail = payload.get("message") or payload.get("msg") or "Unknown provider error"
                raise ExternalProviderRequestError(f"Atlas Cloud {operation} failed: {detail}")
            data = payload.get("data")
            if not isinstance(data, dict):
                raise ExternalProviderRequestError(f"Atlas Cloud {operation} response missing data")
            return data
        return payload

    @staticmethod
    def _raise_for_error(response: requests.Response, operation: str) -> None:
        if response.ok:
            return
        if response.status_code == 429:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            raise ExternalProviderRateLimitError(
                f"Atlas Cloud rate limit exceeded during {operation}",
                retry_after=retry_after,
            )
        raise ExternalProviderRequestError(
            f"Atlas Cloud {operation} failed with status {response.status_code}: {response.text}"
        )

    def _download_image(self, url: str) -> PILImageType:
        try:
            response = requests.get(url, timeout=_DOWNLOAD_TIMEOUT, stream=True)
        except requests.RequestException as exc:
            raise ExternalProviderRequestError(f"Failed to download image from Atlas Cloud: {exc}") from exc

        with response:
            if not response.ok:
                raise ExternalProviderRequestError(
                    f"Failed to download image from Atlas Cloud (status {response.status_code})"
                )

            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    if int(content_length) > _DOWNLOAD_MAX_BYTES:
                        raise ExternalProviderRequestError(f"Atlas Cloud image exceeds {_DOWNLOAD_MAX_BYTES} byte cap")
                except ValueError:
                    pass

            buffer = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                buffer.extend(chunk)
                if len(buffer) > _DOWNLOAD_MAX_BYTES:
                    raise ExternalProviderRequestError(f"Atlas Cloud image exceeds {_DOWNLOAD_MAX_BYTES} byte cap")

        try:
            return Image.open(io.BytesIO(bytes(buffer))).convert("RGB")
        except Exception as exc:
            raise ExternalProviderRequestError("Atlas Cloud output was not a valid image") from exc


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _select_size_preset(width: int, height: int, presets: tuple[str, ...]) -> str:
    """Pick the named size preset whose aspect ratio is closest to the requested one."""
    if not presets:
        presets = _STANDARD_SIZE_PRESETS
    if height <= 0 or width <= 0:
        return presets[0]
    ratio = width / height
    return min(presets, key=lambda preset: abs(_preset_ratio(preset) - ratio))


def _preset_ratio(preset: str) -> float:
    """Aspect ratio of a named size preset, e.g. `landscape_16_9` -> 16/9, `portrait_3_4` -> 3/4."""
    parts = preset.split("_")
    numbers = [int(part) for part in parts[1:] if part.isdigit()]
    if len(numbers) != 2:
        return 1.0  # `square` and `square_hd`
    longer, shorter = max(numbers), min(numbers)
    return longer / shorter if parts[0] == "landscape" else shorter / longer


def _select_aspect_ratio(width: int, height: int, allowed: list[str] | None) -> str | None:
    """Pick the closest allowed aspect ratio, falling back to the exact reduced ratio."""
    if width <= 0 or height <= 0:
        return None
    divisor = gcd(width, height)
    exact = f"{width // divisor}:{height // divisor}"
    if not allowed:
        return exact
    ratio = width / height
    candidates = [(value, _parse_ratio(value)) for value in allowed]
    parsed = [(value, parsed_ratio) for value, parsed_ratio in candidates if parsed_ratio is not None]
    if not parsed:
        return exact
    return min(parsed, key=lambda item: abs(item[1] - ratio))[0]


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
