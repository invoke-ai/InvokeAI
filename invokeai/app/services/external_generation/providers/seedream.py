from __future__ import annotations

import json
import uuid

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
from invokeai.app.services.external_generation.image_utils import decode_image_base64, encode_image_base64

_SEEDREAM_BATCH_PREFIXES = (
    "seedream-5",
    "seedream-4.5",
    "seedream-4.0",
    "seedream-4-5",
    "seedream-4-0",
    "seedream-5-0",
)


class SeedreamProvider(ExternalProvider):
    provider_id = "seedream"

    def is_configured(self) -> bool:
        return bool(self._app_config.external_seedream_api_key)

    def generate(self, request: ExternalGenerationRequest) -> ExternalGenerationResult:
        api_key = self._app_config.external_seedream_api_key
        if not api_key:
            raise ExternalProviderRequestError("Seedream API key is not configured")

        base_url = (self._app_config.external_seedream_base_url or "https://ark.ap-southeast.bytepluses.com").rstrip(
            "/"
        )
        endpoint = f"{base_url}/api/v3/images/generations"
        headers = {"Authorization": f"Bearer {api_key}"}

        model_id = request.model.provider_model_id
        is_batch_model = any(model_id.startswith(prefix) for prefix in _SEEDREAM_BATCH_PREFIXES)

        opts = request.provider_options or {}

        payload: dict[str, object] = {
            "model": model_id,
            "prompt": request.prompt,
            "size": f"{request.width}x{request.height}",
            "response_format": "b64_json",
            "watermark": opts.get("watermark", False),
        }

        if opts.get("optimize_prompt"):
            payload["optimize_prompt_options"] = {"optimize_prompt": True}

        # Seed and guidance_scale are only supported on 3.0 models
        if not is_batch_model and request.seed is not None and request.seed >= 0:
            payload["seed"] = request.seed
        if not is_batch_model and opts.get("guidance_scale") is not None:
            payload["guidance_scale"] = opts["guidance_scale"]

        # Batch generation for 4.x/5.x models
        if is_batch_model:
            if request.num_images > 1:
                payload["sequential_image_generation"] = "auto"
                payload["sequential_image_generation_options"] = {"max_images": request.num_images}
            else:
                payload["sequential_image_generation"] = "disabled"

        # Image input: init_image for img2img, reference images for 4.x
        images_b64: list[str] = []
        if request.init_image is not None:
            images_b64.append(f"data:image/png;base64,{encode_image_base64(request.init_image)}")
        for reference in request.reference_images:
            images_b64.append(f"data:image/png;base64,{encode_image_base64(reference.image)}")

        if images_b64:
            payload["image"] = images_b64 if len(images_b64) > 1 else images_b64[0]

        self._dump_debug_payload("request", payload)

        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)

        if not response.ok:
            if response.status_code == 429:
                retry_after = _parse_retry_after(response.headers.get("retry-after"))
                raise ExternalProviderRateLimitError(
                    f"Seedream rate limit exceeded. {f'Retry after {retry_after:.0f}s.' if retry_after else 'Please try again later.'}",
                    retry_after=retry_after,
                )
            raise ExternalProviderRequestError(
                f"Seedream request failed with status {response.status_code}: {response.text}"
            )

        body = response.json()
        self._dump_debug_payload("response", body)
        if not isinstance(body, dict):
            raise ExternalProviderRequestError("Seedream response payload was not a JSON object")

        generated_images: list[ExternalGeneratedImage] = []
        data_items = body.get("data")
        if not isinstance(data_items, list):
            raise ExternalProviderRequestError("Seedream response payload missing image data")

        for item in data_items:
            if not isinstance(item, dict):
                continue
            # Items may be error objects for failed images in batch
            if "error" in item:
                continue
            encoded = item.get("b64_json")
            if not encoded:
                continue
            image = decode_image_base64(encoded)
            self._dump_debug_image(image)
            generated_images.append(ExternalGeneratedImage(image=image, seed=request.seed))

        if not generated_images:
            raise ExternalProviderRequestError("Seedream response contained no images")

        return ExternalGenerationResult(
            images=generated_images,
            seed_used=request.seed,
            provider_metadata={"model": model_id},
        )

    def _dump_debug_payload(self, label: str, payload: object) -> None:
        """TODO: remove debug payload dump once Seedream is stable."""
        try:
            outputs_path = self._app_config.outputs_path
            if outputs_path is None:
                return
            debug_dir = outputs_path / "external_debug" / "seedream"
            debug_dir.mkdir(parents=True, exist_ok=True)
            path = debug_dir / f"{label}_{uuid.uuid4().hex}.json"
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            self._logger.debug("Failed to write Seedream debug payload: %s", exc)

    def _dump_debug_image(self, image: PILImageType) -> None:
        """TODO: remove debug image dump once Seedream is stable."""
        try:
            outputs_path = self._app_config.outputs_path
            if outputs_path is None:
                return
            debug_dir = outputs_path / "external_debug" / "seedream"
            debug_dir.mkdir(parents=True, exist_ok=True)
            path = debug_dir / f"decoded_{uuid.uuid4().hex}.png"
            image.save(path, format="PNG")
        except Exception as exc:
            self._logger.debug("Failed to write Seedream debug image: %s", exc)


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
