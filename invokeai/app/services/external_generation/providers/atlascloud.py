from __future__ import annotations

import io
import time
from typing import Any

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
        payload: dict[str, object] = {
            "model": request.model.provider_model_id,
            "prompt": request.prompt,
            "size": f"{request.width}*{request.height}",
            "num_images": request.num_images,
        }
        if request.seed is not None:
            payload["seed"] = request.seed

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
