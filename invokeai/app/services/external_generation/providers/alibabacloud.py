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
    "wan2.6-image",
    "qwen-image-edit-max",
}

# Models that use the async image-generation endpoint with flat prompt format
_ASYNC_MODELS = {
    "qwen-image-plus",
    "qwen-image",
    "qwen-image-edit-plus",
    "qwen-image-edit",
    "wan2.5-t2i-preview",
    "wan2.2-t2i-flash",
    "wanx2.0-t2i-turbo",
}

# Models that support image editing (accept input images)
_EDIT_MODELS = {
    "wan2.6-image",
    "qwen-image-edit-max",
    "qwen-image-edit-plus",
    "qwen-image-edit",
}

_TASK_POLL_INTERVAL = 5  # seconds
_TASK_POLL_TIMEOUT = 300  # seconds


class AlibabaCloudProvider(ExternalProvider):
    provider_id = "alibabacloud"

    def is_configured(self) -> bool:
        return bool(self._app_config.external_alibabacloud_api_key)

    def generate(self, request: ExternalGenerationRequest) -> ExternalGenerationResult:
        api_key = self._app_config.external_alibabacloud_api_key
        if not api_key:
            raise ExternalProviderRequestError("Alibaba Cloud DashScope API key is not configured")

        base_url = (
            self._app_config.external_alibabacloud_base_url or "https://dashscope-intl.aliyuncs.com"
        ).rstrip("/")
        model_id = request.model.provider_model_id
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        size = f"{request.width}*{request.height}"

        if model_id in _SYNC_MODELS or model_id not in _ASYNC_MODELS:
            return self._generate_sync(request, base_url, headers, model_id, size)
        else:
            return self._generate_async(request, base_url, headers, model_id, size)

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

        # Add init image for editing
        if request.init_image is not None and model_id in _EDIT_MODELS:
            content.append({"image": f"data:image/png;base64,{encode_image_base64(request.init_image)}"})

        # Add reference images
        for ref in request.reference_images:
            content.append({"image": f"data:image/png;base64,{encode_image_base64(ref.image)}"})

        content.append({"text": request.prompt})

        parameters: dict[str, object] = {
            "size": size,
            "n": request.num_images,
            "prompt_extend": False,
            "watermark": False,
        }
        if request.negative_prompt:
            parameters["negative_prompt"] = request.negative_prompt
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

        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)

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
        if request.negative_prompt:
            parameters["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            parameters["seed"] = request.seed

        input_data: dict[str, object] = {"prompt": request.prompt}
        if request.negative_prompt:
            input_data["negative_prompt"] = request.negative_prompt

        payload: dict[str, object] = {
            "model": model_id,
            "input": input_data,
            "parameters": parameters,
        }

        response = requests.post(endpoint, headers=async_headers, json=payload, timeout=60)

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

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > _TASK_POLL_TIMEOUT:
                raise ExternalProviderRequestError(
                    f"DashScope task {task_id} timed out after {_TASK_POLL_TIMEOUT}s"
                )

            time.sleep(_TASK_POLL_INTERVAL)

            response = requests.get(task_url, headers={"Authorization": headers["Authorization"]}, timeout=30)
            if not response.ok:
                raise ExternalProviderRequestError(
                    f"DashScope task poll failed with status {response.status_code}: {response.text}"
                )

            data = response.json()
            output = data.get("output", {})
            status = output.get("task_status")

            if status == "SUCCEEDED":
                return self._parse_async_response(output, request, request_id)
            elif status in ("FAILED", "UNKNOWN"):
                message = output.get("message", "Unknown error")
                raise ExternalProviderRequestError(f"DashScope task {task_id} failed: {message}")

            self._logger.debug("DashScope task %s status: %s (%.0fs elapsed)", task_id, status, elapsed)

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
        """Download an image from a URL and return it as a PIL Image."""
        response = requests.get(url, timeout=60)
        if not response.ok:
            raise ExternalProviderRequestError(
                f"Failed to download image from DashScope (status {response.status_code})"
            )
        return Image.open(io.BytesIO(response.content)).convert("RGB")
