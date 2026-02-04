from __future__ import annotations

import json
import uuid

import requests
from PIL.Image import Image as PILImageType

from invokeai.app.services.external_generation.errors import ExternalProviderRequestError
from invokeai.app.services.external_generation.external_generation_base import ExternalProvider
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGeneratedImage,
    ExternalGenerationRequest,
    ExternalGenerationResult,
)
from invokeai.app.services.external_generation.image_utils import decode_image_base64, encode_image_base64


class GeminiProvider(ExternalProvider):
    provider_id = "gemini"
    _SYSTEM_INSTRUCTION = (
        "You are an image generation model. Always respond with an image based on the user's prompt. "
        "Do not return text-only responses. If the user input is not an edit instruction, "
        "interpret it as a request to create a new image."
    )

    def is_configured(self) -> bool:
        return bool(self._app_config.external_gemini_api_key)

    def generate(self, request: ExternalGenerationRequest) -> ExternalGenerationResult:
        api_key = self._app_config.external_gemini_api_key
        if not api_key:
            raise ExternalProviderRequestError("Gemini API key is not configured")

        base_url = (self._app_config.external_gemini_base_url or "https://generativelanguage.googleapis.com").rstrip(
            "/"
        )
        if not base_url.endswith("/v1") and not base_url.endswith("/v1beta"):
            base_url = f"{base_url}/v1beta"
        model_id = request.model.provider_model_id.removeprefix("models/")
        endpoint = f"{base_url}/models/{model_id}:generateContent"

        request_parts: list[dict[str, object]] = []

        if request.init_image is not None:
            request_parts.append(
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": encode_image_base64(request.init_image),
                    }
                }
            )

        request_parts.append({"text": request.prompt})

        for reference in request.reference_images:
            request_parts.append(
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": encode_image_base64(reference.image),
                    }
                }
            )

        generation_config: dict[str, object] = {
            "candidateCount": request.num_images,
            "responseModalities": ["IMAGE"],
        }
        aspect_ratio = _select_aspect_ratio(
            request.width,
            request.height,
            request.model.capabilities.allowed_aspect_ratios,
        )
        system_instruction = self._SYSTEM_INSTRUCTION
        if request.init_image is not None:
            system_instruction = (
                f"{system_instruction} An input image is provided. "
                "Treat the prompt as an edit instruction and modify the image accordingly. "
                "Do not return the original image unchanged."
            )
        if aspect_ratio is not None:
            system_instruction = f"{system_instruction} Use an aspect ratio of {aspect_ratio}."

        payload: dict[str, object] = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": request_parts}],
            "generationConfig": generation_config,
        }

        self._dump_debug_payload("request", payload)

        response = requests.post(
            endpoint,
            params={"key": api_key},
            json=payload,
            timeout=120,
        )

        if not response.ok:
            raise ExternalProviderRequestError(
                f"Gemini request failed with status {response.status_code} for model '{model_id}': {response.text}"
            )

        data = response.json()
        self._dump_debug_payload("response", data)
        if not isinstance(data, dict):
            raise ExternalProviderRequestError("Gemini response payload was not a JSON object")
        images: list[ExternalGeneratedImage] = []
        text_parts: list[str] = []
        finish_messages: list[str] = []
        candidates = data.get("candidates")
        if not isinstance(candidates, list):
            raise ExternalProviderRequestError("Gemini response payload missing candidates")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            finish_message = candidate.get("finishMessage")
            finish_reason = candidate.get("finishReason")
            if isinstance(finish_message, str):
                finish_messages.append(finish_message)
            elif isinstance(finish_reason, str):
                finish_messages.append(f"Finish reason: {finish_reason}")
            for part in _iter_response_parts(candidate):
                inline_data = part.get("inline_data") or part.get("inlineData")
                if isinstance(inline_data, dict):
                    encoded = inline_data.get("data")
                    if encoded:
                        image = decode_image_base64(encoded)
                        images.append(ExternalGeneratedImage(image=image, seed=request.seed))
                        self._dump_debug_image(image)
                        continue
                file_data = part.get("fileData") or part.get("file_data")
                if isinstance(file_data, dict):
                    file_uri = file_data.get("fileUri") or file_data.get("file_uri")
                    if isinstance(file_uri, str) and file_uri:
                        raise ExternalProviderRequestError(
                            f"Gemini returned fileUri instead of inline image data: {file_uri}"
                        )
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)

        if not images:
            self._logger.error("Gemini response contained no images: %s", data)
            detail = ""
            if finish_messages:
                combined = " ".join(message.strip() for message in finish_messages if message.strip())
                if combined:
                    detail = f" Response status: {combined[:500]}"
            elif text_parts:
                combined = " ".join(text_parts).strip()
                if combined:
                    detail = f" Response text: {combined[:500]}"
            raise ExternalProviderRequestError(f"Gemini response contained no images.{detail}")

        return ExternalGenerationResult(
            images=images,
            seed_used=request.seed,
            provider_metadata={"model": request.model.provider_model_id},
        )

    def _dump_debug_payload(self, label: str, payload: object) -> None:
        """TODO: remove debug payload dump once Gemini is stable."""
        try:
            outputs_path = self._app_config.outputs_path
            if outputs_path is None:
                return
            debug_dir = outputs_path / "external_debug" / "gemini"
            debug_dir.mkdir(parents=True, exist_ok=True)
            path = debug_dir / f"{label}_{uuid.uuid4().hex}.json"
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            self._logger.debug("Failed to write Gemini debug payload: %s", exc)

    def _dump_debug_image(self, image: "PILImageType") -> None:
        """TODO: remove debug image dump once Gemini is stable."""
        try:
            outputs_path = self._app_config.outputs_path
            if outputs_path is None:
                return
            debug_dir = outputs_path / "external_debug" / "gemini"
            debug_dir.mkdir(parents=True, exist_ok=True)
            path = debug_dir / f"decoded_{uuid.uuid4().hex}.png"
            image.save(path, format="PNG")
        except Exception as exc:
            self._logger.debug("Failed to write Gemini debug image: %s", exc)


def _iter_response_parts(candidate: dict[str, object]) -> list[dict[str, object]]:
    content = candidate.get("content")
    if isinstance(content, dict):
        content_parts = content.get("parts")
        if isinstance(content_parts, list):
            return [part for part in content_parts if isinstance(part, dict)]
    contents = candidate.get("contents")
    if isinstance(contents, list):
        parts: list[dict[str, object]] = []
        for item in contents:
            if not isinstance(item, dict):
                continue
            item_parts = item.get("parts")
            if isinstance(item_parts, list):
                parts.extend([part for part in item_parts if isinstance(part, dict)])
        if parts:
            return parts
    return []


def _select_aspect_ratio(width: int, height: int, allowed: list[str] | None) -> str | None:
    if width <= 0 or height <= 0:
        return None
    ratio = width / height
    default_ratio = _format_aspect_ratio(width, height)
    if not allowed:
        return default_ratio
    parsed = [(value, _parse_ratio(value)) for value in allowed]
    filtered = [(value, parsed_ratio) for value, parsed_ratio in parsed if parsed_ratio is not None]
    if not filtered:
        return default_ratio
    return min(filtered, key=lambda item: abs(item[1] - ratio))[0]


def _format_aspect_ratio(width: int, height: int) -> str | None:
    if width <= 0 or height <= 0:
        return None
    divisor = _gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


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


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a
