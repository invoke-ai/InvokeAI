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
