import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.backend.model_manager.configs.external_api import (
    ExternalApiModelConfig,
    ExternalModelCapabilities,
    ExternalModelPanelSchema,
)
from invokeai.backend.model_manager.taxonomy import ModelType


@pytest.fixture(autouse=True, scope="module")
def client(invokeai_root_dir: Path) -> TestClient:
    os.environ["INVOKEAI_ROOT"] = invokeai_root_dir.as_posix()
    return TestClient(app)


class DummyModelImages:
    def get_url(self, key: str) -> str:
        return f"https://example.com/models/{key}.png"


class DummyInvoker:
    def __init__(self, services: Any) -> None:
        self.services = services


class MockApiDependencies(ApiDependencies):
    invoker: DummyInvoker

    def __init__(self, invoker: DummyInvoker) -> None:
        self.invoker = invoker


def test_model_manager_external_config_round_trip(
    monkeypatch: Any, client: TestClient, mm2_model_manager: Any, mm2_app_config: Any
) -> None:
    config = ExternalApiModelConfig(
        key="external_test",
        name="External Test",
        provider_id="openai",
        provider_model_id="gpt-image-1",
        capabilities=ExternalModelCapabilities(modes=["txt2img"]),
    )
    mm2_model_manager.store.add_model(config)

    services = type("Services", (), {})()
    services.model_manager = mm2_model_manager
    services.model_images = DummyModelImages()
    services.configuration = mm2_app_config

    invoker = DummyInvoker(services)
    monkeypatch.setattr("invokeai.app.api.routers.model_manager.ApiDependencies", MockApiDependencies(invoker))

    response = client.get("/api/v2/models/", params={"model_type": ModelType.ExternalImageGenerator.value})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["models"]) == 1
    assert payload["models"][0]["key"] == "external_test"
    assert payload["models"][0]["provider_id"] == "openai"
    assert payload["models"][0]["cover_image"] == "https://example.com/models/external_test.png"

    get_response = client.get("/api/v2/models/i/external_test")

    assert get_response.status_code == 200
    model_payload = get_response.json()
    assert model_payload["provider_model_id"] == "gpt-image-1"
    assert model_payload["cover_image"] == "https://example.com/models/external_test.png"


def test_model_manager_external_config_preserves_custom_panel_schema(
    monkeypatch: Any, client: TestClient, mm2_model_manager: Any, mm2_app_config: Any
) -> None:
    config = ExternalApiModelConfig(
        key="external_custom_schema",
        name="External Custom Schema",
        provider_id="custom",
        provider_model_id="custom-model",
        capabilities=ExternalModelCapabilities(modes=["txt2img"]),
        panel_schema=ExternalModelPanelSchema(
            prompts=[{"name": "reference_images"}],
            image=[{"name": "dimensions"}],
        ),
        source="external://custom/custom-model",
    )
    mm2_model_manager.store.add_model(config)

    services = type("Services", (), {})()
    services.model_manager = mm2_model_manager
    services.model_images = DummyModelImages()
    services.configuration = mm2_app_config

    invoker = DummyInvoker(services)
    monkeypatch.setattr("invokeai.app.api.routers.model_manager.ApiDependencies", MockApiDependencies(invoker))

    response = client.get("/api/v2/models/i/external_custom_schema")

    assert response.status_code == 200
    payload = response.json()
    assert [control["name"] for control in payload["panel_schema"]["prompts"]] == ["reference_images"]
    assert [control["name"] for control in payload["panel_schema"]["image"]] == ["dimensions"]


def test_model_manager_external_starter_model_applies_panel_schema_overrides(
    monkeypatch: Any, client: TestClient, mm2_model_manager: Any, mm2_app_config: Any
) -> None:
    config = ExternalApiModelConfig(
        key="external_starter_schema",
        name="Starter Schema Test",
        provider_id="openai",
        provider_model_id="gpt-image-1",
        capabilities=ExternalModelCapabilities(
            modes=["txt2img"],
            supports_reference_images=False,
        ),
    )
    mm2_model_manager.store.add_model(config)

    services = type("Services", (), {})()
    services.model_manager = mm2_model_manager
    services.model_images = DummyModelImages()
    services.configuration = mm2_app_config

    invoker = DummyInvoker(services)
    monkeypatch.setattr("invokeai.app.api.routers.model_manager.ApiDependencies", MockApiDependencies(invoker))

    response = client.get("/api/v2/models/i/external_starter_schema")

    assert response.status_code == 200
    payload = response.json()
    assert [control["name"] for control in payload["panel_schema"]["prompts"]] == ["reference_images"]
    assert [control["name"] for control in payload["panel_schema"]["image"]] == ["dimensions"]
    assert payload["panel_schema"]["generation"] == []


def test_model_manager_gemini_starter_model_applies_reference_and_resolution_overrides(
    monkeypatch: Any, client: TestClient, mm2_model_manager: Any, mm2_app_config: Any
) -> None:
    config = ExternalApiModelConfig(
        key="external_gemini_schema",
        name="Gemini Starter Schema Test",
        provider_id="gemini",
        provider_model_id="gemini-3.1-flash-image-preview",
        capabilities=ExternalModelCapabilities(modes=["txt2img"]),
        source="external://gemini/gemini-3.1-flash-image-preview",
    )
    mm2_model_manager.store.add_model(config)

    services = type("Services", (), {})()
    services.model_manager = mm2_model_manager
    services.model_images = DummyModelImages()
    services.configuration = mm2_app_config

    invoker = DummyInvoker(services)
    monkeypatch.setattr("invokeai.app.api.routers.model_manager.ApiDependencies", MockApiDependencies(invoker))

    response = client.get("/api/v2/models/i/external_gemini_schema")

    assert response.status_code == 200
    payload = response.json()
    assert payload["capabilities"]["max_reference_images"] == 14
    assert payload["capabilities"]["max_image_size"] == {"width": 4096, "height": 4096}
    assert payload["capabilities"]["allowed_aspect_ratios"] == [
        "1:1",
        "1:4",
        "1:8",
        "2:3",
        "3:2",
        "3:4",
        "4:1",
        "4:3",
        "4:5",
        "5:4",
        "8:1",
        "9:16",
        "16:9",
        "21:9",
    ]


def _make_stats_services(ram_caches: dict) -> Any:
    class _Load:
        pass

    load = _Load()
    load.ram_caches = ram_caches

    class _ModelManager:
        pass

    mm = _ModelManager()
    mm.load = load
    services = type("Services", (), {})()
    services.model_manager = mm
    return services


def test_get_stats_aggregates_per_device_caches(monkeypatch: Any, client: TestClient) -> None:
    """In multi-GPU mode there is one cache per device; the stats endpoint must not report only
    the API thread's default cache."""
    from invokeai.backend.model_manager.load.model_cache.cache_stats import CacheStats

    class _Cache:
        def __init__(self, stats: CacheStats | None) -> None:
            self.stats = stats

    stats_0 = CacheStats(hits=3, misses=1, in_cache=2, cleared=1, cache_size=100, high_watermark=80)
    stats_0.loaded_model_sizes = {"m1": 50}
    stats_1 = CacheStats(hits=5, misses=2, in_cache=1, cleared=0, cache_size=200, high_watermark=120)
    stats_1.loaded_model_sizes = {"m2": 70}

    services = _make_stats_services({"cuda:0": _Cache(stats_0), "cuda:1": _Cache(stats_1)})
    invoker = DummyInvoker(services)
    monkeypatch.setattr("invokeai.app.api.routers.model_manager.ApiDependencies", MockApiDependencies(invoker))

    response = client.get("/api/v2/models/stats")

    assert response.status_code == 200
    payload = response.json()
    assert payload["hits"] == 8
    assert payload["misses"] == 3
    assert payload["in_cache"] == 3
    assert payload["cleared"] == 1
    assert payload["cache_size"] == 300
    assert payload["high_watermark"] == 200
    assert payload["loaded_model_sizes"] == {"m1": 50, "m2": 70}


def test_get_stats_counts_duplicate_cache_objects_once(monkeypatch: Any, client: TestClient) -> None:
    """ram_caches can map several device keys to the same cache object (the default cache is
    always included under its own device key); its stats must not be double-counted."""
    from invokeai.backend.model_manager.load.model_cache.cache_stats import CacheStats

    class _Cache:
        def __init__(self, stats: CacheStats | None) -> None:
            self.stats = stats

    shared = _Cache(CacheStats(hits=4, misses=2, in_cache=1, cache_size=100, high_watermark=60))
    services = _make_stats_services({"cuda:0": shared, "cpu": shared})
    invoker = DummyInvoker(services)
    monkeypatch.setattr("invokeai.app.api.routers.model_manager.ApiDependencies", MockApiDependencies(invoker))

    response = client.get("/api/v2/models/stats")

    assert response.status_code == 200
    assert response.json()["hits"] == 4


def test_get_stats_returns_null_when_no_stats(monkeypatch: Any, client: TestClient) -> None:
    class _Cache:
        def __init__(self) -> None:
            self.stats = None

    services = _make_stats_services({"cuda:0": _Cache(), "cuda:1": _Cache()})
    invoker = DummyInvoker(services)
    monkeypatch.setattr("invokeai.app.api.routers.model_manager.ApiDependencies", MockApiDependencies(invoker))

    response = client.get("/api/v2/models/stats")

    assert response.status_code == 200
    assert response.json() is None
