from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import requests

from invokeai.app.services.external_generation.errors import ExternalProviderRateLimitError, ExternalProviderRequestError

_DEFAULT_CATALOG_URL = "https://api.fal.ai"
_DEFAULT_SCHEMA_URL = "https://fal.ai/api/openapi/queue/openapi.json"
_CATALOG_TIMEOUT = 30
_MAX_PAGE_SIZE = 100


class FalEndpointKind(StrEnum):
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_IMAGE = "image-to-image"
    INPAINT = "inpaint"
    UPSCALE = "upscale"
    TEXT_TO_VIDEO = "text-to-video"
    IMAGE_TO_VIDEO = "image-to-video"
    VIDEO_TO_VIDEO = "video-to-video"
    AUDIO = "audio"
    GENERIC = "generic"


@dataclass(frozen=True)
class FalCatalogModel:
    endpoint_id: str
    display_name: str
    description: str
    category: str
    model_url: str | None
    thumbnail_url: str | None
    tags: tuple[str, ...]


@dataclass(frozen=True)
class FalCatalogPage:
    models: list[FalCatalogModel]
    next_cursor: str | None
    has_more: bool


@dataclass(frozen=True)
class FalEndpointSchema:
    endpoint_id: str
    kind: FalEndpointKind
    output_kind: FalEndpointKind
    category: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    common_fields: dict[str, str]
    public_properties: tuple[str, ...]


class FalCatalogClient:
    """Read fal.ai's model catalog and per-endpoint OpenAPI schemas."""

    def __init__(
        self,
        api_key: str,
        *,
        catalog_url: str = _DEFAULT_CATALOG_URL,
        schema_url: str = _DEFAULT_SCHEMA_URL,
    ) -> None:
        self._api_key = api_key
        self._catalog_url = catalog_url.rstrip("/")
        self._schema_url = schema_url

    def list_models(
        self,
        *,
        limit: int = 50,
        cursor: str | None = None,
        search: str | None = None,
    ) -> FalCatalogPage:
        params: dict[str, Any] = {"limit": min(max(limit, 1), _MAX_PAGE_SIZE)}
        if cursor:
            params["cursor"] = cursor
        if search:
            params["search"] = search

        response = self._get(f"{self._catalog_url}/v1/models", params=params)
        payload = self._parse_object(response, "fal.ai catalog response")
        raw_models = payload.get("models")
        if not isinstance(raw_models, list):
            raise ExternalProviderRequestError("fal.ai catalog response missing models")

        models: list[FalCatalogModel] = []
        for raw_model in raw_models:
            if not isinstance(raw_model, dict):
                continue
            endpoint_id = raw_model.get("endpoint_id")
            metadata = raw_model.get("metadata")
            if not isinstance(endpoint_id, str) or not endpoint_id or not isinstance(metadata, dict):
                continue
            models.append(
                FalCatalogModel(
                    endpoint_id=endpoint_id,
                    display_name=_string_or_default(metadata.get("display_name"), endpoint_id.rsplit("/", 1)[-1]),
                    description=_string_or_default(metadata.get("description"), ""),
                    category=_string_or_default(metadata.get("category"), "generic"),
                    model_url=_optional_string(metadata.get("model_url")),
                    thumbnail_url=_optional_string(metadata.get("thumbnail_url")),
                    tags=tuple(value for value in metadata.get("tags", []) if isinstance(value, str)),
                )
            )

        next_cursor = payload.get("next_cursor")
        return FalCatalogPage(
            models=models,
            next_cursor=next_cursor if isinstance(next_cursor, str) and next_cursor else None,
            has_more=bool(payload.get("has_more")),
        )

    def get_schema(self, endpoint_id: str) -> FalEndpointSchema:
        if not endpoint_id or endpoint_id.startswith("/") or ".." in endpoint_id.split("/"):
            raise ExternalProviderRequestError("fal.ai catalog endpoint ID is invalid")
        response = self._get(self._schema_url, params={"endpoint_id": endpoint_id})
        payload = self._parse_object(response, "fal.ai catalog schema response")
        return normalize_openapi_schema(endpoint_id, payload)

    def _get(self, url: str, *, params: dict[str, Any]) -> requests.Response:
        try:
            response = requests.get(
                url,
                headers={"Authorization": f"Key {self._api_key}"},
                params=params,
                timeout=_CATALOG_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise ExternalProviderRequestError(f"fal.ai catalog network request failed: {exc}") from exc
        if response.status_code == 429:
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            raise ExternalProviderRateLimitError("fal.ai catalog rate limit exceeded", retry_after=retry_after)
        if not response.ok:
            raise ExternalProviderRequestError(
                f"fal.ai catalog request failed with status {response.status_code}: {response.text}"
            )
        return response

    @staticmethod
    def _parse_object(response: requests.Response, label: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ExternalProviderRequestError(f"{label} was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ExternalProviderRequestError(f"{label} was not a JSON object")
        return payload


def normalize_openapi_schema(endpoint_id: str, document: dict[str, Any]) -> FalEndpointSchema:
    """Extract safe request/output metadata from one fal.ai queue OpenAPI document."""
    components = document.get("components", {}).get("schemas", {})
    if not isinstance(components, dict):
        components = {}

    metadata = document.get("info", {}).get("x-fal-metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    category = _string_or_default(metadata.get("category"), "generic")

    paths = document.get("paths")
    if not isinstance(paths, dict):
        raise ExternalProviderRequestError("fal.ai catalog schema response missing paths")
    operation = next(
        (
            item.get("post")
            for path, item in paths.items()
            if isinstance(path, str) and "/requests/" not in path and isinstance(item, dict) and isinstance(item.get("post"), dict)
        ),
        None,
    )
    if not isinstance(operation, dict):
        raise ExternalProviderRequestError("fal.ai catalog schema response missing queue POST operation")

    request_schema = _resolve_schema(
        operation.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema"), components
    )
    if request_schema.get("type") != "object" or not isinstance(request_schema.get("properties"), dict):
        raise ExternalProviderRequestError("fal.ai catalog endpoint has no object input schema")

    output_schema = _find_output_schema(components)
    kind = classify_endpoint(category, request_schema, endpoint_id=endpoint_id)
    output_kind = _classify_output_schema(output_schema, fallback=kind)
    properties = request_schema["properties"]
    public_properties = tuple(
        name for name, value in properties.items() if isinstance(name, str) and isinstance(value, dict) and not value.get("writeOnly")
    )

    return FalEndpointSchema(
        endpoint_id=endpoint_id,
        kind=kind,
        output_kind=output_kind,
        category=category,
        input_schema=request_schema,
        output_schema=output_schema,
        common_fields=_find_common_fields(properties),
        public_properties=public_properties,
    )


def classify_endpoint(category: str, schema: dict[str, Any], *, endpoint_id: str = "") -> FalEndpointKind:
    normalized = category.strip().lower().replace("_", "-")
    endpoint = endpoint_id.lower()
    if "upscal" in normalized or "upscal" in endpoint:
        return FalEndpointKind.UPSCALE
    if normalized in {"text-to-image", "text2image", "text-to-img"}:
        return FalEndpointKind.TEXT_TO_IMAGE
    if normalized in {"image-to-image", "image-editing", "image-edit", "inpainting", "inpaint"}:
        return FalEndpointKind.INPAINT if "inpaint" in normalized or "fill" in endpoint else FalEndpointKind.IMAGE_TO_IMAGE
    if normalized in {"text-to-video", "text2video"}:
        return FalEndpointKind.TEXT_TO_VIDEO
    if normalized in {"image-to-video", "image2video"}:
        return FalEndpointKind.IMAGE_TO_VIDEO
    if normalized in {"video-to-video", "video2video", "video-editing", "video-edit"}:
        return FalEndpointKind.VIDEO_TO_VIDEO
    if normalized.startswith("audio") or normalized.endswith("-to-audio"):
        return FalEndpointKind.AUDIO
    properties = schema.get("properties", {})
    if isinstance(properties, dict) and any("video" in str(name).lower() for name in properties):
        return FalEndpointKind.IMAGE_TO_VIDEO
    return FalEndpointKind.GENERIC


def _find_output_schema(components: dict[str, Any]) -> dict[str, Any]:
    for name, schema in components.items():
        if isinstance(name, str) and name.lower().endswith("output") and isinstance(schema, dict):
            resolved = _resolve_schema(schema, components)
            if resolved.get("type") == "object":
                return resolved
    return {}


def _classify_output_schema(schema: dict[str, Any], *, fallback: FalEndpointKind) -> FalEndpointKind:
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return fallback
    names = " ".join(str(name).lower() for name in properties)
    if "video" in names:
        return FalEndpointKind.IMAGE_TO_VIDEO
    if "audio" in names or "speech" in names:
        return FalEndpointKind.AUDIO
    if "image" in names:
        return FalEndpointKind.IMAGE_TO_IMAGE
    return fallback


def _find_common_fields(properties: dict[str, Any]) -> dict[str, str]:
    aliases = {
        "prompt": ("prompt", "text", "input_text"),
        "negative_prompt": ("negative_prompt", "negative_prompt_text"),
        "init_image": (
            "image_url",
            "image_urls",
            "input_image_url",
            "input_image",
            "start_image_url",
            "first_frame_url",
        ),
        "mask_image": ("mask_url", "mask_image_url", "mask"),
        "init_video": ("video_url", "video_urls", "input_video_url"),
        "seed": ("seed",),
        "num_images": ("num_images", "num_outputs", "num_inference_images"),
        "width": ("width", "output_width"),
        "height": ("height", "output_height"),
        "aspect_ratio": ("aspect_ratio",),
        "image_size": ("image_size", "resolution", "output_size"),
        "duration": ("duration", "video_length", "num_frames"),
        "fps": ("fps", "frame_rate"),
    }
    return {
        common_name: next((candidate for candidate in candidates if candidate in properties), "")
        for common_name, candidates in aliases.items()
        if any(candidate in properties for candidate in candidates)
    }


def _resolve_schema(schema: Any, components: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    resolved = copy.deepcopy(schema)
    reference = resolved.pop("$ref", None)
    if isinstance(reference, str) and reference.startswith("#/components/schemas/"):
        target = components.get(reference.removeprefix("#/components/schemas/"))
        if isinstance(target, dict):
            resolved = _resolve_schema(target, components)
            resolved.update({key: value for key, value in schema.items() if key != "$ref"})
    return resolved


def _string_or_default(value: Any, default: str) -> str:
    return value if isinstance(value, str) and value else default


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


__all__ = [
    "FalCatalogClient",
    "FalCatalogModel",
    "FalCatalogPage",
    "FalEndpointKind",
    "FalEndpointSchema",
    "classify_endpoint",
    "normalize_openapi_schema",
]
