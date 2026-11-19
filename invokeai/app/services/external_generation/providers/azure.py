"""Azure AI Foundry image generation.

One Foundry resource serves two API families from the same endpoint
(`https://<resource>.services.ai.azure.com`), authenticated with the resource key:

- GPT Image deployments through the Azure OpenAI v1 API (`/openai/v1/images/generations` and
  `/openai/v1/images/edits`). These are OpenAI-compatible; the deployment name goes in `model`.
- Microsoft MAI image models through the MAI API (`/mai/v1/images/generations` and `/mai/v1/images/edits`).
  These take `width`/`height` instead of `size`, return one image per request, and their edits endpoint only
  accepts reference images in repeated `image` form fields (`image[]` is rejected).

The model ID is the deployment name. Starter models assume the deployment is named after its model, which is
the Foundry portal's default.
"""

from __future__ import annotations

import io

import requests
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

MAI_MODEL_PREFIX = "mai-image"
MAI_MAX_REFERENCE_IMAGES = 5


def is_mai_model(model_id: str) -> bool:
    return model_id.lower().startswith(MAI_MODEL_PREFIX)


class AzureAIFoundryProvider(ExternalProvider):
    provider_id = "azure"

    _TIMEOUT = 300

    def is_configured(self) -> bool:
        return bool(self._app_config.external_azure_api_key and self._app_config.external_azure_base_url)

    def generate(self, request: ExternalGenerationRequest) -> ExternalGenerationResult:
        api_key = self._app_config.external_azure_api_key
        base_url = self._app_config.external_azure_base_url
        if not api_key or not base_url:
            raise ExternalProviderRequestError(
                "Azure AI Foundry requires both an API key and the resource endpoint "
                "(https://<resource>.services.ai.azure.com) as the base URL"
            )
        base_url = base_url.rstrip("/")
        headers = {"api-key": api_key}
        model_id = request.model.provider_model_id

        images: list[PILImageType] = []
        if request.init_image is not None:
            images.append(request.init_image)
        images.extend(reference.image for reference in request.reference_images)

        if is_mai_model(model_id):
            responses = [
                self._post_mai(base_url, headers, request, model_id, images) for _ in range(max(1, request.num_images))
            ]
        else:
            responses = [self._post_gpt_image(base_url, headers, request, model_id, images)]

        generated: list[ExternalGeneratedImage] = []
        for response in responses:
            generated.extend(_parse_images(response, request.seed))
        if not generated:
            raise ExternalProviderRequestError("Azure AI Foundry response contained no images")

        last = responses[-1]
        return ExternalGenerationResult(
            images=generated,
            seed_used=request.seed,
            provider_request_id=last.headers.get("apim-request-id") or last.headers.get("x-request-id"),
            provider_metadata={"model": model_id},
        )

    def _post_gpt_image(
        self,
        base_url: str,
        headers: dict[str, str],
        request: ExternalGenerationRequest,
        model_id: str,
        images: list[PILImageType],
    ) -> requests.Response:
        opts = request.provider_options or {}
        fields: dict[str, object] = {
            "model": model_id,
            "prompt": request.prompt,
            "n": request.num_images,
            "size": f"{request.width}x{request.height}",
            "output_format": "png",
        }
        if opts.get("quality") and opts["quality"] != "auto":
            fields["quality"] = opts["quality"]
        if opts.get("background") and opts["background"] != "auto":
            fields["background"] = opts["background"]

        if not images:
            return _check(
                requests.post(
                    f"{base_url}/openai/v1/images/generations", headers=headers, json=fields, timeout=self._TIMEOUT
                )
            )

        field_name = "image" if len(images) == 1 else "image[]"
        files = [(field_name, _png_file(image, index)) for index, image in enumerate(images)]
        return _check(
            requests.post(
                f"{base_url}/openai/v1/images/edits", headers=headers, data=fields, files=files, timeout=self._TIMEOUT
            )
        )

    def _post_mai(
        self,
        base_url: str,
        headers: dict[str, str],
        request: ExternalGenerationRequest,
        model_id: str,
        images: list[PILImageType],
    ) -> requests.Response:
        opts = request.provider_options or {}
        fields: dict[str, object] = {
            "model": model_id,
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
        }
        if opts.get("background") and opts["background"] != "auto":
            fields["background"] = opts["background"]

        if not images:
            return _check(
                requests.post(
                    f"{base_url}/mai/v1/images/generations", headers=headers, json=fields, timeout=self._TIMEOUT
                )
            )

        if len(images) > MAI_MAX_REFERENCE_IMAGES:
            raise ExternalProviderRequestError(
                f"MAI image edits accept at most {MAI_MAX_REFERENCE_IMAGES} images, got {len(images)}"
            )
        # The MAI edits endpoint rejects `image[]`; every file must be a repeated `image` field.
        files = [("image", _png_file(image, index)) for index, image in enumerate(images)]
        return _check(
            requests.post(
                f"{base_url}/mai/v1/images/edits", headers=headers, data=fields, files=files, timeout=self._TIMEOUT
            )
        )


def _png_file(image: PILImageType, index: int) -> tuple[str, io.BytesIO, str]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return (f"image_{index}.png", buffer, "image/png")


def _check(response: requests.Response) -> requests.Response:
    if response.ok:
        return response
    if response.status_code == 429:
        retry_after = _parse_retry_after(response.headers.get("retry-after"))
        raise ExternalProviderRateLimitError(
            "Azure AI Foundry rate limit exceeded. "
            + (f"Retry after {retry_after:.0f}s." if retry_after else "Please try again later."),
            retry_after=retry_after,
        )
    raise ExternalProviderRequestError(
        f"Azure AI Foundry request failed with status {response.status_code}: {response.text}"
    )


def _parse_images(response: requests.Response, seed: int | None) -> list[ExternalGeneratedImage]:
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ExternalProviderRequestError("Azure AI Foundry response payload missing image data")
    images: list[ExternalGeneratedImage] = []
    for item in payload["data"]:
        if isinstance(item, dict) and item.get("b64_json"):
            images.append(ExternalGeneratedImage(image=decode_image_base64(item["b64_json"]), seed=seed))
    return images


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
