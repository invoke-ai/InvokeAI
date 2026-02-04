import os
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.config.config_default import get_config, load_and_migrate_config
from invokeai.app.services.external_generation.external_generation_common import ExternalProviderStatus
from invokeai.app.services.invoker import Invoker


@pytest.fixture(autouse=True, scope="module")
def client(invokeai_root_dir: Path) -> TestClient:
    os.environ["INVOKEAI_ROOT"] = invokeai_root_dir.as_posix()
    return TestClient(app)


class MockApiDependencies(ApiDependencies):
    invoker: Invoker

    def __init__(self, invoker: Invoker) -> None:
        self.invoker = invoker


def test_get_external_provider_statuses(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    statuses = {
        "gemini": ExternalProviderStatus(provider_id="gemini", configured=True, message=None),
        "openai": ExternalProviderStatus(provider_id="openai", configured=False, message="Missing key"),
    }

    monkeypatch.setattr("invokeai.app.api.routers.app_info.ApiDependencies", MockApiDependencies(mock_invoker))
    monkeypatch.setattr(mock_invoker.services.external_generation, "get_provider_statuses", lambda: statuses)

    response = client.get("/api/v1/app/external_providers/status")

    assert response.status_code == 200
    payload = sorted(response.json(), key=lambda item: item["provider_id"])
    assert payload == [
        {"provider_id": "gemini", "configured": True, "message": None},
        {"provider_id": "openai", "configured": False, "message": "Missing key"},
    ]


def test_external_provider_config_update_and_reset(client: TestClient) -> None:
    for provider_id in ("gemini", "openai"):
        response = client.delete(f"/api/v1/app/external_providers/config/{provider_id}")
        assert response.status_code == 200

    response = client.get("/api/v1/app/external_providers/config")
    assert response.status_code == 200
    payload = response.json()
    openai_config = _get_provider_config(payload, "openai")
    assert openai_config["api_key_configured"] is False
    assert openai_config["base_url"] is None

    response = client.post(
        "/api/v1/app/external_providers/config/openai",
        json={"api_key": "openai-key", "base_url": "https://api.openai.test"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["api_key_configured"] is True
    assert payload["base_url"] == "https://api.openai.test"

    response = client.get("/api/v1/app/external_providers/config")
    assert response.status_code == 200
    payload = response.json()
    openai_config = _get_provider_config(payload, "openai")
    assert openai_config["api_key_configured"] is True
    assert openai_config["base_url"] == "https://api.openai.test"

    config_path = get_config().config_file_path
    file_config = load_and_migrate_config(config_path)
    assert file_config.external_openai_api_key == "openai-key"
    assert file_config.external_openai_base_url == "https://api.openai.test"

    response = client.delete("/api/v1/app/external_providers/config/openai")
    assert response.status_code == 200
    payload = response.json()
    assert payload["api_key_configured"] is False
    assert payload["base_url"] is None

    file_config = load_and_migrate_config(config_path)
    assert file_config.external_openai_api_key is None
    assert file_config.external_openai_base_url is None


def _get_provider_config(payload: list[dict[str, Any]], provider_id: str) -> dict[str, Any]:
    return next(item for item in payload if item["provider_id"] == provider_id)
