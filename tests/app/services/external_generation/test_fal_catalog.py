from typing import Any

import pytest

from invokeai.app.services.external_generation.errors import ExternalProviderRequestError
from invokeai.app.services.external_generation.providers.fal_catalog import (
    FalCatalogClient,
    FalEndpointKind,
    classify_endpoint,
    normalize_openapi_schema,
)


class DummyResponse:
    def __init__(self, *, json_data: dict[str, Any], status_code: int = 200, text: str = "") -> None:
        self.status_code = status_code
        self.ok = status_code < 400
        self._json_data = json_data
        self.text = text
        self.headers: dict[str, str] = {}

    def json(self) -> dict[str, Any]:
        return self._json_data


def _schema() -> dict[str, Any]:
    return {
        "openapi": "3.0.4",
        "info": {
            "title": "Queue OpenAPI for fal-ai/test",
            "x-fal-metadata": {
                "endpointId": "fal-ai/test",
                "category": "image-to-video",
                "about": "test endpoint",
            },
        },
        "paths": {
            "/fal-ai/test": {
                "post": {
                    "requestBody": {
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/TestInput"}}}
                    },
                    "responses": {
                        "200": {
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/TestOutput"}}}
                        }
                    },
                }
            }
        },
        "components": {
            "schemas": {
                "TestInput": {
                    "type": "object",
                    "required": ["prompt", "start_image_url"],
                    "properties": {
                        "prompt": {"type": "string", "description": "Motion prompt"},
                        "start_image_url": {"type": "string", "format": "uri"},
                        "duration": {"type": "string", "enum": ["5", "10"], "default": "5"},
                        "secret": {"type": "string", "writeOnly": True},
                    },
                },
                "TestOutput": {
                    "type": "object",
                    "properties": {"video": {"type": "object", "properties": {"url": {"type": "string"}}}},
                },
            }
        },
    }


def test_normalize_openapi_schema_resolves_refs_and_common_aliases() -> None:
    normalized = normalize_openapi_schema("fal-ai/test", _schema())

    assert normalized.endpoint_id == "fal-ai/test"
    assert normalized.kind is FalEndpointKind.IMAGE_TO_VIDEO
    assert normalized.input_schema["required"] == ["prompt", "start_image_url"]
    assert normalized.input_schema["properties"]["duration"]["enum"] == ["5", "10"]
    assert normalized.common_fields == {"prompt": "prompt", "init_image": "start_image_url", "duration": "duration"}
    assert normalized.output_kind is FalEndpointKind.IMAGE_TO_VIDEO
    assert "secret" not in normalized.public_properties


def test_classify_endpoint_keeps_unknown_categories_generic() -> None:
    assert classify_endpoint("speech-to-text", {}) is FalEndpointKind.GENERIC
    assert classify_endpoint("text-to-image", {}) is FalEndpointKind.TEXT_TO_IMAGE
    assert classify_endpoint("upscaling", {}) is FalEndpointKind.UPSCALE
    assert (
        classify_endpoint("image-to-image", {"properties": {"image_url": {}, "mask_url": {}}})
        is FalEndpointKind.INPAINT
    )


def test_classify_endpoint_keeps_segmentation_out_of_inpaint_mode() -> None:
    assert (
        classify_endpoint(
            "image-to-image",
            {"properties": {"image_url": {}, "min_mask_region_area": {}}},
            endpoint_id="fal-ai/sam2/auto-segment",
        )
        is FalEndpointKind.IMAGE_TO_IMAGE
    )


def test_catalog_client_lists_pages_and_fetches_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_get(url: str, **kwargs: Any) -> DummyResponse:
        calls.append((url, kwargs))
        if url.endswith("/v1/models"):
            return DummyResponse(
                json_data={
                    "models": [
                        {
                            "endpoint_id": "fal-ai/test",
                            "metadata": {
                                "display_name": "Test model",
                                "category": "image-to-video",
                                "description": "A test model",
                                "tags": ["video"],
                                "thumbnail_url": "https://cdn.test/thumb.jpg",
                            },
                        }
                    ],
                    "next_cursor": "next",
                    "has_more": True,
                }
            )
        return DummyResponse(json_data=_schema())

    monkeypatch.setattr("requests.get", fake_get)
    client = FalCatalogClient("test-key")

    page = client.list_models(limit=10, cursor="old")
    schema = client.get_schema("fal-ai/test")

    assert page.next_cursor == "next"
    assert page.has_more is True
    assert page.models[0].endpoint_id == "fal-ai/test"
    assert page.models[0].display_name == "Test model"
    assert schema.kind is FalEndpointKind.IMAGE_TO_VIDEO
    assert calls[0][0] == "https://api.fal.ai/v1/models"
    assert calls[0][1]["params"] == {"limit": 10, "cursor": "old"}
    assert calls[1][0] == "https://fal.ai/api/openapi/queue/openapi.json"
    assert calls[1][1]["params"] == {"endpoint_id": "fal-ai/test"}


def test_catalog_client_filters_search_across_api_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_get(url: str, **kwargs: Any) -> DummyResponse:
        del url
        calls.append(kwargs["params"])
        if len(calls) == 1:
            return DummyResponse(
                json_data={
                    "models": [
                        {"endpoint_id": "fal-ai/other", "metadata": {"display_name": "Other"}},
                    ],
                    "next_cursor": "next",
                    "has_more": True,
                }
            )
        return DummyResponse(
            json_data={
                "models": [
                    {"endpoint_id": "fal-ai/flux/schnell", "metadata": {"display_name": "Flux Schnell"}},
                ],
                "next_cursor": None,
                "has_more": False,
            }
        )

    monkeypatch.setattr("requests.get", fake_get)
    page = FalCatalogClient("test-key").list_models(limit=1, search="flux")

    assert [model.endpoint_id for model in page.models] == ["fal-ai/flux/schnell"]
    assert calls == [{"limit": 100}, {"limit": 100, "cursor": "next"}]
    assert page.has_more is False


def test_catalog_client_rejects_non_object_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: DummyResponse(json_data={"models": []}))
    client = FalCatalogClient("test-key")

    with pytest.raises(ExternalProviderRequestError, match="catalog"):
        client.get_schema("fal-ai/bad")
