from __future__ import annotations

import io
import time

import requests
from PIL import Image
from PIL.Image import Image as PILImageType

from invokeai.app.services.external_generation.errors import ExternalProviderRequestError
from invokeai.app.services.external_generation.external_generation_base import ExternalProvider
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGeneratedImage,
    ExternalGenerationRequest,
    ExternalGenerationResult,
)
from invokeai.app.services.external_generation.image_utils import decode_image_base64, encode_image_base64

# Models that support the synchronous multimodal-generation endpoint with messages format
_SYNC_MODELS = {
    "qwen-image-2.0-pro",
    "qwen-image-2.0",
    "qwen-image-max",
    "wan2.6-t2i",
    "qwen-image-edit-max",
}

# Models that use the async image-generation endpoint with flat prompt format.
# Currently no shipped starter model uses this path, but it is retained because
# users may install custom external models via `external://alibabacloud/<model_id>`.
_ASYNC_MODELS: set[str] = set()

_TASK_POLL_INTERVAL = 5  # seconds
_TASK_POLL_TIMEOUT = 300  # seconds
_DOWNLOAD_TIMEOUT = 60  # seconds
_DOWNLOAD_MAX_BYTES = 32 * 1024 * 1024  # 32 MiB safety cap on image downloads
_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 2  # total attempts = 1 + _MAX_RETRIES
_RETRY_BACKOFF_BASE = 2.0  # seconds


class AlibabaCloudProvider(ExternalProvider):
    provider_id = "alibabacloud"

    def is_configured(self) -> bool:
        return bool(self._app_config.external_alibabacloud_api_key)

    def generate(self, request: ExternalGenerationRequest) -> ExternalGenerationResult:
        api_key = self._app_config.external_alibabacloud_api_key
        if not api_key:
            raise ExternalProviderRequestError("Alibaba Cloud DashScope API key is not configured")

        base_url = (self._app_config.external_alibabacloud_base_url or "https://dashscope-intl.aliyuncs.com").rstrip(
            "/"
        )
        model_id = request.model.provider_model_id
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        size = f"{request.width}*{request.height}"

        if model_id in _SYNC_MODELS:
            return self._generate_sync(request, base_url, headers, model_id, size)
        if model_id in _ASYNC_MODELS:
            return self._generate_async(request, base_url, headers, model_id, size)
        raise ExternalProviderRequestError(
            f"Unknown DashScope model_id '{model_id}'. Add it to _SYNC_MODELS or _ASYNC_MODELS in alibabacloud.py."
        )

    def _generate_sync(
        self,
        request: ExternalGenerationRequest,
        base_url: str,
        headers: dict[str, str],
        model_id: str,
        size: str,
    ) -> ExternalGenerationResult:
        """Use the synchronous multimodal-generation endpoint (messages format)."""
        endpoint = f"{base_url}/api/v1/services/aigc/multimodal-generation/generation"

        content: list[dict[str, str]] = []

        # Reference images: DashScope multimodal accepts up to 3 input images for the
        # qwen-image-edit family; we let the API surface its own limit if exceeded.
        for ref in request.reference_images:
            content.append({"image": f"data:image/png;base64,{encode_image_base64(ref.image)}"})

        content.append({"text": request.prompt})

        parameters: dict[str, object] = {
            "size": size,
            "n": request.num_images,
            "prompt_extend": False,
            "watermark": False,
        }
        if request.seed is not None:
            parameters["seed"] = request.seed

        payload: dict[str, object] = {
            "model": model_id,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            },
            "parameters": parameters,
        }

        response = self._post_with_retry(endpoint, headers=headers, json=payload, timeout=120, label="DashScope sync")
        if not response.ok:
            raise ExternalProviderRequestError(
                f"DashScope request failed with status {response.status_code} for model '{model_id}': {response.text}"
            )

        data = response.json()
        request_id = data.get("request_id")
        return self._parse_sync_response(data, request, request_id)

    def _generate_async(
        self,
        request: ExternalGenerationRequest,
        base_url: str,
        headers: dict[str, str],
        model_id: str,
        size: str,
    ) -> ExternalGenerationResult:
        """Use the async image-generation endpoint (flat prompt format) with task polling."""
        endpoint = f"{base_url}/api/v1/services/aigc/image-generation/generation"
        async_headers = {**headers, "X-DashScope-Async": "enable"}

        parameters: dict[str, object] = {
            "size": size,
            "n": request.num_images,
            "prompt_extend": False,
            "watermark": False,
        }
        if request.seed is not None:
            parameters["seed"] = request.seed

        input_data: dict[str, object] = {"prompt": request.prompt}

        payload: dict[str, object] = {
            "model": model_id,
            "input": input_data,
            "parameters": parameters,
        }

        response = self._post_with_retry(
            endpoint, headers=async_headers, json=payload, timeout=60, label="DashScope async submit"
        )
        if not response.ok:
            raise ExternalProviderRequestError(
                f"DashScope async request failed with status {response.status_code} for model '{model_id}': {response.text}"
            )

        data = response.json()
        request_id = data.get("request_id")
        output = data.get("output", {})
        task_id = output.get("task_id")

        if not task_id:
            raise ExternalProviderRequestError(f"DashScope async response missing task_id: {data}")

        return self._poll_task(base_url, headers, task_id, request, request_id)

    def _poll_task(
        self,
        base_url: str,
        headers: dict[str, str],
        task_id: str,
        request: ExternalGenerationRequest,
        request_id: str | None,
    ) -> ExternalGenerationResult:
        """Poll an async task until completion."""
        task_url = f"{base_url}/api/v1/tasks/{task_id}"
        start_time = time.monotonic()
        poll_headers = {"Authorization": headers["Authorization"]}
        first_poll = True

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > _TASK_POLL_TIMEOUT:
                raise ExternalProviderRequestError(f"DashScope task {task_id} timed out after {_TASK_POLL_TIMEOUT}s")

            response = self._get_with_retry(task_url, headers=poll_headers, timeout=30, label="DashScope task poll")
            if not response.ok:
                raise ExternalProviderRequestError(
                    f"DashScope task poll failed with status {response.status_code}: {response.text}"
                )

            data = response.json()
            output = data.get("output", {})
            status = output.get("task_status")

            if first_poll:
                self._logger.info("DashScope task %s submitted (status=%s)", task_id, status)
                first_poll = False

            if status == "SUCCEEDED":
                return self._parse_async_response(output, request, request_id)
            if status in ("FAILED", "UNKNOWN"):
                message = output.get("message", "Unknown error")
                raise ExternalProviderRequestError(f"DashScope task {task_id} failed: {message}")

            self._logger.debug("DashScope task %s status: %s (%.0fs elapsed)", task_id, status, elapsed)
            time.sleep(_TASK_POLL_INTERVAL)

    def _parse_sync_response(
        self,
        data: dict[str, object],
        request: ExternalGenerationRequest,
        request_id: str | None,
    ) -> ExternalGenerationResult:
        """Parse the synchronous multimodal-generation response."""
        output = data.get("output")
        if not isinstance(output, dict):
            raise ExternalProviderRequestError(f"DashScope response missing output: {data}")

        choices = output.get("choices")
        if not isinstance(choices, list):
            raise ExternalProviderRequestError(f"DashScope response missing choices: {data}")

        images: list[ExternalGeneratedImage] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                image_url = part.get("image")
                if isinstance(image_url, str) and image_url:
                    pil_image = self._download_image(image_url)
                    images.append(ExternalGeneratedImage(image=pil_image, seed=request.seed))

        if not images:
            raise ExternalProviderRequestError(f"DashScope response contained no images: {data}")

        return ExternalGenerationResult(
            images=images,
            seed_used=request.seed,
            provider_request_id=request_id,
            provider_metadata={"model": request.model.provider_model_id},
        )

    def _parse_async_response(
        self,
        output: dict[str, object],
        request: ExternalGenerationRequest,
        request_id: str | None,
    ) -> ExternalGenerationResult:
        """Parse the async task completion response."""
        results = output.get("results")
        if not isinstance(results, list):
            raise ExternalProviderRequestError(f"DashScope async response missing results: {output}")

        images: list[ExternalGeneratedImage] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            url = result.get("url")
            if isinstance(url, str) and url:
                pil_image = self._download_image(url)
                images.append(ExternalGeneratedImage(image=pil_image, seed=request.seed))
                continue
            b64_image = result.get("b64_image")
            if isinstance(b64_image, str) and b64_image:
                pil_image = decode_image_base64(b64_image)
                images.append(ExternalGeneratedImage(image=pil_image, seed=request.seed))

        if not images:
            raise ExternalProviderRequestError(f"DashScope async response contained no images: {output}")

        return ExternalGenerationResult(
            images=images,
            seed_used=request.seed,
            provider_request_id=request_id,
            provider_metadata={"model": request.model.provider_model_id},
        )

    def _download_image(self, url: str) -> PILImageType:
        """Download an image from a URL and return it as a PIL Image, with a size cap."""
        try:
            response = requests.get(url, timeout=_DOWNLOAD_TIMEOUT, stream=True)
        except requests.RequestException as exc:
            raise ExternalProviderRequestError(f"Failed to download image from DashScope: {exc}") from exc

        with response:
            if not response.ok:
                raise ExternalProviderRequestError(
                    f"Failed to download image from DashScope (status {response.status_code})"
                )

            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    if int(content_length) > _DOWNLOAD_MAX_BYTES:
                        raise ExternalProviderRequestError(
                            f"DashScope image exceeds {_DOWNLOAD_MAX_BYTES} byte cap (Content-Length={content_length})"
                        )
                except ValueError:
                    pass

            buffer = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                buffer.extend(chunk)
                if len(buffer) > _DOWNLOAD_MAX_BYTES:
                    raise ExternalProviderRequestError(f"DashScope image exceeds {_DOWNLOAD_MAX_BYTES} byte cap")

        return Image.open(io.BytesIO(bytes(buffer))).convert("RGB")

    def _post_with_retry(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict,
        timeout: int,
        label: str,
    ) -> requests.Response:
        return self._request_with_retry("POST", url, headers=headers, json=json, timeout=timeout, label=label)

    def _get_with_retry(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: int,
        label: str,
    ) -> requests.Response:
        return self._request_with_retry("GET", url, headers=headers, timeout=timeout, label=label)

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        timeout: int,
        label: str,
        json: dict | None = None,
    ) -> requests.Response:
        """Issue a request with limited retries on transient failures (429/5xx, network errors).

        Honors `Retry-After` for 429 responses when present. Non-retryable errors
        (4xx other than 429, parse failures) are returned to the caller, which is
        responsible for raising a meaningful ExternalProviderRequestError.
        """
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                if method == "POST":
                    response = requests.post(url, headers=headers, json=json, timeout=timeout)
                else:
                    response = requests.get(url, headers=headers, timeout=timeout)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= _MAX_RETRIES:
                    raise ExternalProviderRequestError(f"{label} network error: {exc}") from exc
                delay = _RETRY_BACKOFF_BASE * (2**attempt)
                self._logger.warning(
                    "%s network error on attempt %d/%d: %s — retrying in %.1fs",
                    label,
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)
                continue

            if response.status_code in _RETRY_STATUS_CODES and attempt < _MAX_RETRIES:
                delay = self._retry_delay(response, attempt)
                self._logger.warning(
                    "%s got status %d on attempt %d/%d — retrying in %.1fs",
                    label,
                    response.status_code,
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    delay,
                )
                time.sleep(delay)
                continue

            return response

        # Unreachable: the loop either returns a response or raises.
        assert last_exc is not None
        raise ExternalProviderRequestError(f"{label} failed after retries: {last_exc}") from last_exc

    @staticmethod
    def _retry_delay(response: requests.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        return _RETRY_BACKOFF_BASE * (2**attempt)
