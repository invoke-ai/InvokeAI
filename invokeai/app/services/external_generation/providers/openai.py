from __future__ import annotations

import io

import requests
from PIL.Image import Image as PILImageType

from invokeai.app.services.external_generation.errors import ExternalProviderRequestError
from invokeai.app.services.external_generation.external_generation_base import ExternalProvider
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGeneratedImage,
    ExternalGenerationRequest,
    ExternalGenerationResult,
)
from invokeai.app.services.external_generation.image_utils import decode_image_base64


class OpenAIProvider(ExternalProvider):
    provider_id = "openai"

    def is_configured(self) -> bool:
        return bool(self._app_config.external_openai_api_key)

    def generate(self, request: ExternalGenerationRequest) -> ExternalGenerationResult:
        api_key = self._app_config.external_openai_api_key
        if not api_key:
            raise ExternalProviderRequestError("OpenAI API key is not configured")

        size = f"{request.width}x{request.height}"
        base_url = (self._app_config.external_openai_base_url or "https://api.openai.com").rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"}

        use_edits_endpoint = request.mode != "txt2img" or bool(request.reference_images)

        if not use_edits_endpoint:
            payload: dict[str, object] = {
                "prompt": request.prompt,
                "n": request.num_images,
                "size": size,
                "response_format": "b64_json",
            }
            if request.seed is not None:
                payload["seed"] = request.seed
            response = requests.post(
                f"{base_url}/v1/images/generations",
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
                raise ExternalProviderRequestError(
                    "OpenAI image edits require at least one image (init image or reference image)"
                )

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
                "prompt": request.prompt,
                "n": request.num_images,
                "size": size,
                "response_format": "b64_json",
            }
            response = requests.post(
                f"{base_url}/v1/images/edits",
                headers=headers,
                data=data,
                files=files,
                timeout=120,
            )

        if not response.ok:
            raise ExternalProviderRequestError(
                f"OpenAI request failed with status {response.status_code}: {response.text}"
            )

        payload = response.json()
        if not isinstance(payload, dict):
            raise ExternalProviderRequestError("OpenAI response payload was not a JSON object")
        images: list[ExternalGeneratedImage] = []
        data_items = payload.get("data")
        if not isinstance(data_items, list):
            raise ExternalProviderRequestError("OpenAI response payload missing image data")
        for item in data_items:
            if not isinstance(item, dict):
                continue
            encoded = item.get("b64_json")
            if not encoded:
                continue
            images.append(ExternalGeneratedImage(image=decode_image_base64(encoded), seed=request.seed))

        if not images:
            raise ExternalProviderRequestError("OpenAI response contained no images")

        return ExternalGenerationResult(
            images=images,
            seed_used=request.seed,
            provider_request_id=response.headers.get("x-request-id"),
            provider_metadata={"model": request.model.provider_model_id},
        )
