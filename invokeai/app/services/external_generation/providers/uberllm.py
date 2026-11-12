from __future__ import annotations

from typing import Any

import requests

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

_DEFAULT_BASE_URL = "https://api.uberllm.dev"
# Image requests are answered synchronously, so the timeout covers the whole generation.
_REQUEST_TIMEOUT = 300


class UberLLMProvider(ExternalProvider):
    provider_id = "uberllm"

    def is_configured(self) -> bool:
        return bool(self._app_config.external_uberllm_api_key)

    def generate(self, request: ExternalGenerationRequest) -> ExternalGenerationResult:
        api_key = self._app_config.external_uberllm_api_key
        if not api_key:
            raise ExternalProviderRequestError("UberLLM API key is not configured")

        base_url = (self._app_config.external_uberllm_base_url or _DEFAULT_BASE_URL).rstrip("/")
        model_id = request.model.provider_model_id
        # Every UberLLM image model takes an explicit "<width>x<height>" size; the starter models
        # declare those sizes as aspect-ratio buckets, so the service has already snapped to one.
        payload: dict[str, object] = {
            "model": model_id,
            "prompt": request.prompt,
            "n": request.num_images,
            "size": f"{request.width}x{request.height}",
            "response_format": "b64_json",
        }

        try:
            response = requests.post(
                f"{base_url}/v1/images/generations",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise ExternalProviderRequestError(f"UberLLM image request failed: {exc}") from exc

        if not response.ok:
            if response.status_code == 429:
                raise ExternalProviderRateLimitError(
                    "UberLLM rate limit exceeded", retry_after=_parse_retry_after(response.headers.get("Retry-After"))
                )
            raise ExternalProviderRequestError(
                f"UberLLM image request failed with status {response.status_code}: {_error_detail(response)}"
            )

        try:
            response_payload = response.json()
        except ValueError as exc:
            raise ExternalProviderRequestError("UberLLM image response was not valid JSON") from exc
        data_items = response_payload.get("data") if isinstance(response_payload, dict) else None
        if not isinstance(data_items, list):
            raise ExternalProviderRequestError("UberLLM image response missing image data")

        images: list[ExternalGeneratedImage] = []
        for item in data_items:
            encoded = item.get("b64_json") if isinstance(item, dict) else None
            if not isinstance(encoded, str) or not encoded:
                continue
            try:
                images.append(ExternalGeneratedImage(image=decode_image_base64(encoded), seed=request.seed))
            except Exception as exc:
                raise ExternalProviderRequestError("UberLLM output was not a valid image") from exc

        if not images:
            raise ExternalProviderRequestError("UberLLM response contained no images")

        usage = response_payload.get("usage")
        cost_usd = usage.get("cost_usd") if isinstance(usage, dict) else None
        return ExternalGenerationResult(
            images=images,
            seed_used=request.seed,
            provider_request_id=response.headers.get("x-request-id"),
            provider_metadata={"model": model_id, "cost_usd": cost_usd},
        )


def _error_detail(response: requests.Response) -> str:
    """UberLLM errors use the OpenAI shape, `{"error": {"message": ...}}`; fall back to the raw body."""
    try:
        payload: Any = response.json()
    except ValueError:
        return response.text
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return response.text


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
