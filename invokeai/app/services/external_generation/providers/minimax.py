from __future__ import annotations

import json

import requests
from PIL import Image
from PIL.Image import Image as PILImageType

from invokeai.app.services.external_generation.errors import (
    ExternalProviderCapabilityError,
    ExternalProviderRequestError,
)
from invokeai.app.services.external_generation.external_generation_base import ExternalProvider
from invokeai.app.services.external_generation.external_generation_common import (
    MINIMAX_MAX_INPUT_IMAGES,
    ExternalGeneratedImage,
    ExternalGenerationRequest,
    ExternalGenerationResult,
)
from invokeai.app.services.external_generation.image_utils import decode_image_base64, encode_image_base64

_DEFAULT_BASE_URL = "https://api.minimax.io"
_REQUEST_TIMEOUT = 600
_MAX_EDIT_BODY_BYTES = 64 * 1024 * 1024
_MAX_JPEG_RESPONSE_PIXELS = 80_000_000


class MiniMaxProvider(ExternalProvider):
    provider_id = "minimax"

    def is_configured(self) -> bool:
        return bool(self._app_config.external_minimax_api_key)

    def generate(self, request: ExternalGenerationRequest) -> ExternalGenerationResult:
        api_key = self._app_config.external_minimax_api_key
        base_url = (self._app_config.external_minimax_base_url or _DEFAULT_BASE_URL).rstrip("/")
        if not api_key:
            raise ExternalProviderRequestError("MiniMax API key is not configured")

        inputs = [reference.image for reference in request.reference_images]
        if request.init_image is not None:
            inputs.insert(0, request.init_image)
        self._validate_request(request, inputs)

        model_id = request.model.provider_model_id
        payload: dict[str, object] = {
            "model": model_id,
            "prompt": request.prompt,
            "n": request.num_images,
            "size": f"{request.width}x{request.height}",
            "output_format": "jpeg",
            "response_format": "b64_json",
        }
        if inputs:
            payload["images"] = self._encode_inputs(inputs)

        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if inputs and len(body) > _MAX_EDIT_BODY_BYTES:
            raise ExternalProviderCapabilityError("MiniMax image edits are limited to a 64 MiB request body")

        endpoint = "edits" if inputs else "generations"

        # The provider charges for successful images even after a client disconnects.
        try:
            response = requests.post(
                f"{base_url}/v1/images/{endpoint}",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                data=body,
                timeout=_REQUEST_TIMEOUT,
                allow_redirects=False,
            )
        except requests.Timeout:
            raise ExternalProviderRequestError("MiniMax image request timed out") from None
        except requests.RequestException:
            raise ExternalProviderRequestError("MiniMax image request could not be completed") from None

        if not 200 <= response.status_code < 300:
            # The shared service retries ExternalProviderRateLimitError. MiniMax generation is non-idempotent.
            raise ExternalProviderRequestError(f"MiniMax image request failed (HTTP {response.status_code})")

        try:
            result = response.json()
        except ValueError:
            raise ExternalProviderRequestError("MiniMax returned an invalid JSON response") from None
        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, list) or len(data) > request.num_images:
            raise ExternalProviderRequestError("MiniMax returned invalid image data")

        images: list[ExternalGeneratedImage] = []
        for item in data:
            encoded = item.get("b64_json") if isinstance(item, dict) and "error" not in item else None
            if not isinstance(encoded, str) or not encoded:
                continue
            try:
                images.append(ExternalGeneratedImage(image=decode_image_base64(encoded)))
            except (OSError, ValueError, Image.DecompressionBombError):
                raise ExternalProviderRequestError("MiniMax returned an invalid inline image") from None

        if not images:
            raise ExternalProviderRequestError("MiniMax response contained no inline images")

        request_id = result.get("request_id")
        return ExternalGenerationResult(
            images=images,
            provider_request_id=request_id if isinstance(request_id, str) else None,
            provider_metadata={"model": model_id},
        )

    @staticmethod
    def _validate_request(request: ExternalGenerationRequest, inputs: list[PILImageType]) -> None:
        if request.mode not in ("txt2img", "img2img") or request.mask_image is not None:
            raise ExternalProviderCapabilityError("MiniMax supports text-to-image and image edits without masks")
        if not 1 <= len(request.prompt.strip()) <= 32_000:
            raise ExternalProviderCapabilityError("MiniMax requires a prompt of 1 to 32,000 characters")
        if not 1 <= request.num_images <= 10:
            raise ExternalProviderCapabilityError("MiniMax supports 1 to 10 outputs per request")

        width, height = request.width, request.height
        if not (1 <= width <= 4096 and 1 <= height <= 4096):
            raise ExternalProviderCapabilityError("MiniMax output width and height must be between 1 and 4096 pixels")
        area = width * height
        if not 655_360 <= area <= 20_000_000 or max(width, height) > 3 * min(width, height):
            raise ExternalProviderCapabilityError(
                "MiniMax output size requires 655,360 to 20,000,000 pixels and an aspect ratio between 1:3 and 3:1"
            )
        if request.num_images * area > _MAX_JPEG_RESPONSE_PIXELS:
            raise ExternalProviderCapabilityError(
                "MiniMax inline JPEG output is limited to 80,000,000 pixels per request"
            )

        if request.mode == "img2img" and not inputs:
            raise ExternalProviderCapabilityError("MiniMax image edits require an input image")
        if len(inputs) > MINIMAX_MAX_INPUT_IMAGES:
            raise ExternalProviderCapabilityError(
                f"MiniMax image edits accept at most {MINIMAX_MAX_INPUT_IMAGES} inputs, including the init image"
            )
        for image in inputs:
            width, height = image.size
            if not (128 <= width <= 8192 and 128 <= height <= 8192) or max(width, height) > 4 * min(width, height):
                raise ExternalProviderCapabilityError(
                    "MiniMax edit inputs require dimensions between 128 and 8192 pixels and an aspect ratio "
                    "between 1:4 and 4:1"
                )

    @staticmethod
    def _encode_inputs(inputs: list[PILImageType]) -> list[dict[str, str]]:
        encoded_images: list[dict[str, str]] = []
        encoded_size = 0
        for image in inputs:
            image_url = f"data:image/png;base64,{encode_image_base64(image)}"
            encoded_size += len(image_url)
            if encoded_size > _MAX_EDIT_BODY_BYTES:
                raise ExternalProviderCapabilityError("MiniMax image edits are limited to a 64 MiB request body")
            encoded_images.append({"image_url": image_url})
        return encoded_images
