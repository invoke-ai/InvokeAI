import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig, ExternalModelCapabilities
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
