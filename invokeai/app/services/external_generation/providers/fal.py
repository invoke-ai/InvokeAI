from __future__ import annotations

import base64
import io
import os
import time
from typing import Any

import requests
from PIL import Image, ImageOps
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

_DEFAULT_QUEUE_URL = "https://queue.fal.run"
_UPLOAD_URL = "https://rest.fal.ai/storage/upload/initiate?storage_type=fal-cdn-v3"
_POLL_INTERVAL = 1.0
_POLL_TIMEOUT = 300.0
_REQUEST_TIMEOUT = 60
_DOWNLOAD_TIMEOUT = 60
_DOWNLOAD_MAX_BYTES = 32 * 1024 * 1024
_RETRY_STATUS_CODES = {500, 502, 503, 504}

_IMAGE_SIZE_BY_RATIO = {
    "1:1": "square_hd",
    "4:3": "landscape_4_3",
    "3:4": "portrait_4_3",
    "16:9": "landscape_16_9",
    "9:16": "portrait_16_9",
}

_FLUX_FILL_MODELS = {"fal-ai/flux-lora-fill"}
_FLUX_KONTEXT_MODELS = {"fal-ai/flux-pro/kontext", "fal-ai/flux-pro/kontext/max"}
_FLUX_TEXT_MODELS = {"fal-ai/flux/schnell", "fal-ai/flux/dev"}


class FalProvider(ExternalProvider):
    """InvokeAI adapter for fal.ai's queue and CDN APIs."""

    provider_id = "fal"

    def is_configured(self) -> bool:
        return bool(self._api_key())

    def generate(self, request: ExternalGenerationRequest) -> ExternalGenerationResult:
        api_key = self._api_key()
        if not api_key:
            raise ExternalProviderRequestError("fal.ai API key is not configured")

        headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
        image_url = None
        mask_url = None
        if request.init_image is not None:
            image_url = self._upload_image(request.init_image, "image.png", headers)
        if request.mask_image is not None:
            mask = ImageOps.invert(request.mask_image.convert("L"))
            mask_url = self._upload_image(mask, "mask.png", headers)

        payload = self._build_payload(request, image_url=image_url, mask_url=mask_url)
        model_id = request.model.provider_model_id
        queue_url = f"{self._queue_base_url}/{model_id}"
        submit_response = self._request(
            "POST",
            queue_url,
            headers=headers,
            json=payload,
            timeout=_REQUEST_TIMEOUT,
        )
        self._raise_for_response(submit_response, "fal.ai request")
        submitted = self._parse_json(submit_response, "fal.ai queue response")

        request_id = submitted.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            if self._has_images(submitted):
                return self._parse_result(submitted, request, request_id=None)
            raise ExternalProviderRequestError("fal.ai queue response missing request_id")

        result_payload = self._wait_for_result(model_id, request_id, headers)
        return self._parse_result(result_payload, request, request_id=request_id)

    def _api_key(self) -> str | None:
        return self._app_config.external_fal_api_key or os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")

    @property
    def _queue_base_url(self) -> str:
        return (self._app_config.external_fal_base_url or _DEFAULT_QUEUE_URL).rstrip("/")

    def _build_payload(
        self,
        request: ExternalGenerationRequest,
        *,
        image_url: str | None,
        mask_url: str | None,
    ) -> dict[str, Any]:
        model_id = request.model.provider_model_id
        ratio = _select_aspect_ratio(request.width, request.height, request.model.capabilities.allowed_aspect_ratios)
        payload: dict[str, Any] = {"prompt": request.prompt}

        if request.seed is not None and request.model.capabilities.supports_seed:
            payload["seed"] = request.seed
        if request.num_images > 1 or request.model.capabilities.max_images_per_request is not None:
            payload["num_images"] = request.num_images

        if model_id in _FLUX_FILL_MODELS or request.mode == "inpaint":
            if image_url is None or mask_url is None:
                raise ExternalProviderRequestError("fal.ai inpainting requires both image and mask inputs")
            payload.update(
                {
                    "image_size": _image_size_for_ratio(ratio),
                    "image_url": image_url,
                    "mask_url": mask_url,
                    "paste_back": True,
                    "resize_to_original": True,
                }
            )
            return payload

        if model_id in _FLUX_KONTEXT_MODELS or request.mode == "img2img":
            if image_url is None:
                raise ExternalProviderRequestError("fal.ai image editing requires an input image")
            payload["image_url"] = image_url
            payload["aspect_ratio"] = ratio
            return payload

        if model_id in _FLUX_TEXT_MODELS or request.mode == "txt2img":
            payload["image_size"] = _image_size_for_ratio(ratio)
            return payload

        # Unknown external models get conservative common arguments. Curated models above
        # use exact schemas; custom external model records can still use txt2img safely.
        if image_url is not None:
            payload["image_url"] = image_url
        if mask_url is not None:
            payload["mask_url"] = mask_url
        payload["image_size"] = _image_size_for_ratio(ratio)
        return payload

    def _upload_image(self, image: PILImageType, filename: str, headers: dict[str, str]) -> str:
        upload_headers = {"Authorization": headers["Authorization"], "Content-Type": "application/json"}
        response = self._request(
            "POST",
            _UPLOAD_URL,
            headers=upload_headers,
            json={"file_name": filename, "content_type": "image/png"},
            timeout=_REQUEST_TIMEOUT,
        )
        self._raise_for_response(response, "fal.ai upload initiation")
        upload = self._parse_json(response, "fal.ai upload initiation response")
        file_url = upload.get("file_url")
        upload_url = upload.get("upload_url")
        if not isinstance(file_url, str) or not file_url or not isinstance(upload_url, str) or not upload_url:
            raise ExternalProviderRequestError("fal.ai upload response missing file_url or upload_url")

        image_bytes = _encode_png(image)
        put_response = self._request(
            "PUT",
            upload_url,
            headers={"Content-Type": "image/png"},
            data=image_bytes,
            timeout=_REQUEST_TIMEOUT,
        )
        self._raise_for_response(put_response, "fal.ai file upload")
        return file_url

    def _wait_for_result(self, model_id: str, request_id: str, headers: dict[str, str]) -> dict[str, Any]:
        status_url = f"{self._queue_base_url}/{model_id}/requests/{request_id}/status"
        result_url = f"{self._queue_base_url}/{model_id}/requests/{request_id}"
        deadline = time.monotonic() + _POLL_TIMEOUT
        while True:
            if time.monotonic() >= deadline:
                raise ExternalProviderRequestError(f"fal.ai request {request_id} timed out")

            try:
                status_response = self._request("GET", status_url, headers=headers, timeout=_REQUEST_TIMEOUT)
            except ExternalProviderRequestError:
                # The job already exists. Retrying polling avoids submitting a second billable job.
                time.sleep(_POLL_INTERVAL)
                continue
            if status_response.status_code == 429 or status_response.status_code in _RETRY_STATUS_CODES:
                time.sleep(_retry_delay(status_response))
                continue
            self._raise_for_response(status_response, "fal.ai status request")
            status_payload = self._parse_json(status_response, "fal.ai status response")
            status = status_payload.get("status")
            if status == "COMPLETED":
                try:
                    result_response = self._request("GET", result_url, headers=headers, timeout=_REQUEST_TIMEOUT)
                except ExternalProviderRequestError:
                    time.sleep(_POLL_INTERVAL)
                    continue
                if result_response.status_code == 429 or result_response.status_code in _RETRY_STATUS_CODES:
                    time.sleep(_retry_delay(result_response))
                    continue
                self._raise_for_response(result_response, "fal.ai result request")
                return self._parse_json(result_response, "fal.ai result response")
            if status in {"FAILED", "CANCELED", "CANCELLED"}:
                detail = status_payload.get("error") or status_payload.get("message") or status
                raise ExternalProviderRequestError(f"fal.ai request {request_id} failed: {detail}")
            if status not in {"IN_QUEUE", "IN_PROGRESS"}:
                raise ExternalProviderRequestError(f"fal.ai returned unknown request status: {status}")
            time.sleep(_POLL_INTERVAL)

    def _parse_result(
        self,
        payload: dict[str, Any],
        request: ExternalGenerationRequest,
        *,
        request_id: str | None,
    ) -> ExternalGenerationResult:
        image_items = payload.get("images")
        if not isinstance(image_items, list):
            data_items = payload.get("data")
            image_items = data_items if isinstance(data_items, list) else []

        seed_value = payload.get("seed")
        seed = seed_value if isinstance(seed_value, int) else request.seed
        images: list[ExternalGeneratedImage] = []
        for item in image_items:
            if not isinstance(item, dict):
                continue
            url = item.get("url") or item.get("image_url")
            if isinstance(url, str) and url:
                images.append(ExternalGeneratedImage(image=self._download_image(url), seed=seed))

        if not images:
            raise ExternalProviderRequestError("fal.ai response contained no images")

        return ExternalGenerationResult(
            images=images,
            seed_used=seed,
            provider_request_id=request_id,
            provider_metadata={"model": request.model.provider_model_id},
        )

    def _download_image(self, url: str) -> PILImageType:
        if url.startswith("data:image/"):
            try:
                encoded = url.split(",", 1)[1]
                return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
            except (IndexError, ValueError, OSError) as exc:
                raise ExternalProviderRequestError("fal.ai returned an invalid image data URI") from exc
        if not url.startswith("https://"):
            raise ExternalProviderRequestError("fal.ai returned a non-HTTPS image URL")

        response = self._request("GET", url, headers={}, timeout=_DOWNLOAD_TIMEOUT, stream=True)
        self._raise_for_response(response, "fal.ai image download")
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > _DOWNLOAD_MAX_BYTES:
                    raise ExternalProviderRequestError("fal.ai image exceeds the download size limit")
            except ValueError:
                pass

        buffer = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if chunk:
                buffer.extend(chunk)
                if len(buffer) > _DOWNLOAD_MAX_BYTES:
                    raise ExternalProviderRequestError("fal.ai image exceeds the download size limit")
        try:
            return Image.open(io.BytesIO(bytes(buffer))).convert("RGB")
        except OSError as exc:
            raise ExternalProviderRequestError("fal.ai returned invalid image data") from exc

    @staticmethod
    def _has_images(payload: dict[str, Any]) -> bool:
        images = payload.get("images")
        return isinstance(images, list) and bool(images)

    @staticmethod
    def _parse_json(response: requests.Response, label: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ExternalProviderRequestError(f"{label} was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ExternalProviderRequestError(f"{label} was not a JSON object")
        return payload

    @staticmethod
    def _request(method: str, url: str, **kwargs: Any) -> requests.Response:
        try:
            if method == "POST":
                return requests.post(url, **kwargs)
            if method == "PUT":
                return requests.put(url, **kwargs)
            return requests.get(url, **kwargs)
        except requests.RequestException as exc:
            raise ExternalProviderRequestError(f"fal.ai network request failed: {exc}") from exc

    @staticmethod
    def _raise_for_response(response: requests.Response, operation: str) -> None:
        if response.ok:
            return
        if response.status_code == 429:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            detail = f" Retry after {retry_after:.0f}s." if retry_after is not None else ""
            raise ExternalProviderRateLimitError(f"fal.ai rate limit exceeded.{detail}", retry_after=retry_after)
        if response.status_code in _RETRY_STATUS_CODES:
            raise ExternalProviderRequestError(f"{operation} failed with status {response.status_code}; retry later")
        raise ExternalProviderRequestError(f"{operation} failed with status {response.status_code}: {response.text}")


def _encode_png(image: PILImageType) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _select_aspect_ratio(width: int, height: int, allowed: list[str] | None) -> str:
    if width <= 0 or height <= 0:
        return "1:1"
    ratio = width / height
    candidates = allowed or list(_IMAGE_SIZE_BY_RATIO)
    parsed = [(value, _parse_ratio(value)) for value in candidates]
    valid = [(value, value_ratio) for value, value_ratio in parsed if value_ratio is not None]
    if not valid:
        return "1:1"
    return min(valid, key=lambda pair: abs(pair[1] - ratio))[0]


def _image_size_for_ratio(ratio: str) -> str:
    return _IMAGE_SIZE_BY_RATIO.get(ratio, "square_hd")


def _parse_ratio(value: str) -> float | None:
    try:
        left, right = value.split(":", 1)
        denominator = float(right)
        if denominator == 0:
            return None
        return float(left) / denominator
    except (ValueError, AttributeError):
        return None


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _retry_delay(response: requests.Response) -> float:
    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
    return min(retry_after if retry_after is not None else _POLL_INTERVAL, 60.0)


__all__ = ["FalProvider"]
