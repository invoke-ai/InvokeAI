import os
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers import app_info
from invokeai.app.api_app import app
from invokeai.app.services.auth.token_service import TokenData
from invokeai.app.services.config.config_default import get_config, load_and_migrate_config, load_external_api_keys
from invokeai.app.services.external_generation.external_generation_common import ExternalProviderStatus
from invokeai.app.services.image_files.image_subfolder_strategy import DateStrategy, create_subfolder_strategy
from invokeai.app.services.invoker import Invoker
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig, ExternalModelCapabilities
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


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


def test_external_provider_config_update_and_reset(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    mock_store = Mock()
    mock_store.search_by_attr.return_value = []
    mock_install = Mock()
    mock_model_manager = Mock()
    mock_model_manager.store = mock_store
    mock_model_manager.install = mock_install
    mock_invoker.services.model_manager = mock_model_manager
    monkeypatch.setattr("invokeai.app.api.routers.app_info.ApiDependencies", MockApiDependencies(mock_invoker))

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
    api_keys_path = get_config().api_keys_file_path
    file_config = load_and_migrate_config(config_path)
    assert file_config.external_openai_api_key is None
    assert file_config.external_openai_base_url is None
    assert "external_openai_api_key" not in config_path.read_text()
    assert "external_openai_base_url" not in config_path.read_text()
    api_keys = load_external_api_keys(api_keys_path)
    assert api_keys["external_openai_api_key"] == "openai-key"
    assert api_keys["external_openai_base_url"] == "https://api.openai.test"

    response = client.delete("/api/v1/app/external_providers/config/openai")
    assert response.status_code == 200
    payload = response.json()
    assert payload["api_key_configured"] is False
    assert payload["base_url"] is None

    file_config = load_and_migrate_config(config_path)
    api_keys = load_external_api_keys(api_keys_path)
    assert file_config.external_openai_api_key is None
    assert file_config.external_openai_base_url is None
    assert "external_openai_api_key" not in config_path.read_text()
    assert "external_openai_api_key" not in api_keys
    assert "external_openai_base_url" not in api_keys


def test_reset_external_provider_config_removes_provider_models(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    openai_model = ExternalApiModelConfig(
        key="openai_model",
        name="OpenAI Model",
        provider_id="openai",
        provider_model_id="gpt-image-1",
        capabilities=ExternalModelCapabilities(modes=["txt2img"]),
    )
    gemini_model = ExternalApiModelConfig(
        key="gemini_model",
        name="Gemini Model",
        provider_id="gemini",
        provider_model_id="gemini-2.5-flash-image",
        capabilities=ExternalModelCapabilities(modes=["txt2img"]),
    )
    mock_store = Mock()
    mock_store.search_by_attr.return_value = [openai_model, gemini_model]
    mock_install = Mock()
    mock_model_manager = Mock()
    mock_model_manager.store = mock_store
    mock_model_manager.install = mock_install
    mock_invoker.services.model_manager = mock_model_manager

    monkeypatch.setattr("invokeai.app.api.routers.app_info.ApiDependencies", MockApiDependencies(mock_invoker))

    response = client.delete("/api/v1/app/external_providers/config/openai")

    assert response.status_code == 200
    mock_store.search_by_attr.assert_called_once_with(
        base_model=BaseModelType.External,
        model_type=ModelType.ExternalImageGenerator,
    )
    mock_install.delete.assert_called_once_with("openai_model")


def test_set_external_provider_config_clears_provider_models_when_api_key_removed(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    openai_model = ExternalApiModelConfig(
        key="openai_model",
        name="OpenAI Model",
        provider_id="openai",
        provider_model_id="gpt-image-1",
        capabilities=ExternalModelCapabilities(modes=["txt2img"]),
    )
    mock_store = Mock()
    mock_store.search_by_attr.return_value = [openai_model]
    mock_install = Mock()
    mock_model_manager = Mock()
    mock_model_manager.store = mock_store
    mock_model_manager.install = mock_install
    mock_invoker.services.model_manager = mock_model_manager

    monkeypatch.setattr("invokeai.app.api.routers.app_info.ApiDependencies", MockApiDependencies(mock_invoker))

    response = client.post("/api/v1/app/external_providers/config/openai", json={"api_key": " "})

    assert response.status_code == 200
    mock_store.search_by_attr.assert_called_once_with(
        base_model=BaseModelType.External,
        model_type=ModelType.ExternalImageGenerator,
    )
    mock_install.delete.assert_called_once_with("openai_model")


def test_update_runtime_config_persists_image_subfolder_strategy(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", MockApiDependencies(mock_invoker))

    response = client.patch("/api/v1/app/runtime_config", json={"image_subfolder_strategy": "date"})

    assert response.status_code == 200
    assert response.json()["config"]["image_subfolder_strategy"] == "date"

    config_path = get_config().config_file_path
    file_config = load_and_migrate_config(config_path)
    assert file_config.image_subfolder_strategy == "date"
    assert "image_subfolder_strategy: date" in config_path.read_text()
    assert get_config().image_subfolder_strategy == "date"
    assert isinstance(create_subfolder_strategy(get_config().image_subfolder_strategy), DateStrategy)


def test_update_runtime_config_rejects_null_image_subfolder_strategy(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", MockApiDependencies(mock_invoker))

    response = client.patch("/api/v1/app/runtime_config", json={"image_subfolder_strategy": None})

    assert response.status_code == 422


def test_update_runtime_config_image_subfolder_strategy_schema() -> None:
    app.openapi_schema = None
    property_schema = app.openapi()["components"]["schemas"]["UpdateAppGenerationSettingsRequest"]["properties"][
        "image_subfolder_strategy"
    ]

    assert property_schema == {
        "description": "Strategy for organizing images into subfolders.",
        "enum": ["flat", "date", "type", "hash"],
        "title": "Image Subfolder Strategy",
        "type": "string",
    }


def test_update_runtime_config_reads_and_writes_yaml_under_config_lock(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", MockApiDependencies(mock_invoker))

    class TrackingLock:
        is_locked = False
        load_seen = False
        write_seen = False

        def __enter__(self) -> None:
            self.is_locked = True

        def __exit__(self, *_: Any) -> None:
            self.is_locked = False

    tracking_lock = TrackingLock()
    original_load_and_migrate_config = app_info.load_and_migrate_config
    original_write_file = app_info.InvokeAIAppConfig.write_file

    def load_and_migrate_config_with_lock_assertion(config_path: Path) -> Any:
        assert tracking_lock.is_locked
        tracking_lock.load_seen = True
        return original_load_and_migrate_config(config_path)

    def write_file_with_lock_assertion(
        config: app_info.InvokeAIAppConfig, dest_path: Path, as_example: bool = False
    ) -> None:
        assert tracking_lock.is_locked
        tracking_lock.write_seen = True
        return original_write_file(config, dest_path, as_example)

    monkeypatch.setattr(app_info, "_EXTERNAL_PROVIDER_CONFIG_LOCK", tracking_lock)
    monkeypatch.setattr(app_info, "load_and_migrate_config", load_and_migrate_config_with_lock_assertion)
    monkeypatch.setattr(app_info.InvokeAIAppConfig, "write_file", write_file_with_lock_assertion)

    response = client.patch("/api/v1/app/runtime_config", json={"max_queue_history": 10})

    assert response.status_code == 200
    assert tracking_lock.load_seen
    assert tracking_lock.write_seen


def test_update_runtime_config_rejects_non_admin_users(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", MockApiDependencies(mock_invoker))
    monkeypatch.setattr(mock_invoker.services.configuration, "multiuser", True)
    monkeypatch.setattr(
        "invokeai.app.api.auth_dependencies.verify_token",
        lambda _: TokenData(user_id="user-1", email="user@example.com", is_admin=False),
    )
    monkeypatch.setattr(mock_invoker.services.users, "get", Mock(return_value=Mock(is_active=True)))

    response = client.patch(
        "/api/v1/app/runtime_config",
        json={"image_subfolder_strategy": "date"},
        headers={"Authorization": "Bearer non-admin-token"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Admin privileges required"


def _get_provider_config(payload: list[dict[str, Any]], provider_id: str) -> dict[str, Any]:
    return next(item for item in payload if item["provider_id"] == provider_id)
