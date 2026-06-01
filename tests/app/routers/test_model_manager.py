import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.model_records import ModelRecordChanges
from invokeai.backend.model_manager.configs.external_api import (
    ExternalApiModelConfig,
    ExternalModelCapabilities,
    ExternalModelPanelSchema,
)
from invokeai.backend.model_manager.metadata import CivitaiMetadata, UnknownMetadataException
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


def _patch_model_manager_dependencies(monkeypatch: Any, mm2_model_manager: Any, mm2_app_config: Any) -> None:
    services = type("Services", (), {})()
    services.model_manager = mm2_model_manager
    services.model_images = DummyModelImages()
    services.configuration = mm2_app_config

    invoker = DummyInvoker(services)
    monkeypatch.setattr("invokeai.app.api.routers.model_manager.ApiDependencies", MockApiDependencies(invoker))
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", MockApiDependencies(invoker))


def _civitai_metadata(
    trained_words: list[str], source_url: str | None = "https://civitai.com/models/111?modelVersionId=222"
) -> CivitaiMetadata:
    return CivitaiMetadata(
        name="Test CivitAI LoRA",
        model_id=111,
        model_version_id=222,
        trained_words=trained_words,
        api_response=json.dumps({"trainedWords": trained_words}),
        source_url=source_url,
    )


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


def test_refresh_trigger_phrases_restores_deleted_civitai_words_from_source_url(
    monkeypatch: Any, client: TestClient, mm2_model_manager: Any, mm2_app_config: Any
) -> None:
    _patch_model_manager_dependencies(monkeypatch, mm2_model_manager, mm2_app_config)
    mm2_model_manager.store.update_model(
        "test_config_4",
        ModelRecordChanges(
            source_url="https://civitai.com/models/111/test-lora?modelVersionId=222",
            trigger_phrases={"custom"},
        ),
    )

    class FakeCivitaiMetadataFetch:
        def from_url(self, url: str) -> CivitaiMetadata:
            assert url == "https://civitai.com/models/111/test-lora?modelVersionId=222"
            return _civitai_metadata(["alpha", "custom"])

        def from_hash(self, hash_value: str) -> CivitaiMetadata:
            raise AssertionError("hash lookup should not be used when source URL resolves")

    monkeypatch.setattr(
        "invokeai.app.api.routers.model_manager.CivitaiMetadataFetch", lambda: FakeCivitaiMetadataFetch()
    )

    response = client.post("/api/v2/models/i/test_config_4/refresh_trigger_phrases")

    assert response.status_code == 200
    assert set(response.json()["trigger_phrases"]) == {"alpha", "custom"}
    updated_config = mm2_model_manager.store.get_model("test_config_4")
    assert set(updated_config.trigger_phrases or []) == {"alpha", "custom"}
    assert updated_config.source_api_response == '{"trainedWords": ["alpha", "custom"]}'
    assert updated_config.source_url == "https://civitai.com/models/111/test-lora?modelVersionId=222"


def test_refresh_trigger_phrases_uses_hash_lookup_when_no_civitai_source(
    monkeypatch: Any, client: TestClient, mm2_model_manager: Any, mm2_app_config: Any
) -> None:
    _patch_model_manager_dependencies(monkeypatch, mm2_model_manager, mm2_app_config)

    class FakeCivitaiMetadataFetch:
        def from_url(self, url: str) -> CivitaiMetadata:
            raise AssertionError("source URL lookup should not be used for non-CivitAI sources")

        def from_hash(self, hash_value: str) -> CivitaiMetadata:
            assert hash_value == "111222333444"
            return _civitai_metadata(["hash word"])

    monkeypatch.setattr(
        "invokeai.app.api.routers.model_manager.CivitaiMetadataFetch", lambda: FakeCivitaiMetadataFetch()
    )

    response = client.post("/api/v2/models/i/test_config_4/refresh_trigger_phrases")

    assert response.status_code == 200
    assert set(response.json()["trigger_phrases"]) == {"hash word"}
    updated_config = mm2_model_manager.store.get_model("test_config_4")
    assert updated_config.source_url == "https://civitai.com/models/111?modelVersionId=222"


def test_refresh_trigger_phrases_uses_hash_lookup_for_generic_civitai_source(
    monkeypatch: Any, client: TestClient, mm2_model_manager: Any, mm2_app_config: Any
) -> None:
    _patch_model_manager_dependencies(monkeypatch, mm2_model_manager, mm2_app_config)
    mm2_model_manager.store.update_model(
        "test_config_4",
        ModelRecordChanges(source_url="https://civitai.com/models/111/test-lora"),
    )

    class FakeCivitaiMetadataFetch:
        def from_url(self, url: str) -> CivitaiMetadata:
            raise AssertionError("generic CivitAI model pages should not be used for URL lookup")

        def from_hash(self, hash_value: str) -> CivitaiMetadata:
            assert hash_value == "111222333444"
            return _civitai_metadata(["hash word"])

    monkeypatch.setattr(
        "invokeai.app.api.routers.model_manager.CivitaiMetadataFetch", lambda: FakeCivitaiMetadataFetch()
    )

    response = client.post("/api/v2/models/i/test_config_4/refresh_trigger_phrases")

    assert response.status_code == 200
    assert set(response.json()["trigger_phrases"]) == {"hash word"}
    updated_config = mm2_model_manager.store.get_model("test_config_4")
    assert updated_config.source_url == "https://civitai.com/models/111/test-lora?modelVersionId=222"


def test_refresh_trigger_phrases_rejects_non_lora_model(
    monkeypatch: Any, client: TestClient, mm2_model_manager: Any, mm2_app_config: Any
) -> None:
    _patch_model_manager_dependencies(monkeypatch, mm2_model_manager, mm2_app_config)

    response = client.post("/api/v2/models/i/test_config_2/refresh_trigger_phrases")

    assert response.status_code == 400
    assert "LoRA" in response.json()["detail"]


def test_refresh_trigger_phrases_saves_source_metadata_when_civitai_has_no_words(
    monkeypatch: Any, client: TestClient, mm2_model_manager: Any, mm2_app_config: Any
) -> None:
    _patch_model_manager_dependencies(monkeypatch, mm2_model_manager, mm2_app_config)
    mm2_model_manager.store.update_model(
        "test_config_4",
        ModelRecordChanges(trigger_phrases={"custom"}),
    )

    class FakeCivitaiMetadataFetch:
        def from_hash(self, hash_value: str) -> CivitaiMetadata:
            return _civitai_metadata([])

    monkeypatch.setattr(
        "invokeai.app.api.routers.model_manager.CivitaiMetadataFetch", lambda: FakeCivitaiMetadataFetch()
    )

    response = client.post("/api/v2/models/i/test_config_4/refresh_trigger_phrases")

    assert response.status_code == 200
    assert set(response.json()["trigger_phrases"]) == {"custom"}
    updated_config = mm2_model_manager.store.get_model("test_config_4")
    assert set(updated_config.trigger_phrases or []) == {"custom"}
    assert updated_config.source_api_response == '{"trainedWords": []}'
    assert updated_config.source_url == "https://civitai.com/models/111?modelVersionId=222"


def test_refresh_trigger_phrases_falls_back_to_stored_civitai_response(
    monkeypatch: Any, client: TestClient, mm2_model_manager: Any, mm2_app_config: Any
) -> None:
    _patch_model_manager_dependencies(monkeypatch, mm2_model_manager, mm2_app_config)
    mm2_model_manager.store.update_model(
        "test_config_4",
        ModelRecordChanges(
            source_url="https://civitai.com/models/111/test-lora",
            source_api_response='{"trainedWords": ["cached"]}',
            trigger_phrases={"custom"},
        ),
    )

    class FakeCivitaiMetadataFetch:
        def from_url(self, url: str) -> CivitaiMetadata:
            raise AssertionError("generic CivitAI model pages should not be used for URL lookup")

        def from_hash(self, hash_value: str) -> CivitaiMetadata:
            raise UnknownMetadataException("live lookup failed")

        def from_api_response(self, json_str: str) -> CivitaiMetadata:
            assert json_str == '{"trainedWords": ["cached"]}'
            return _civitai_metadata(["cached"])

    monkeypatch.setattr(
        "invokeai.app.api.routers.model_manager.CivitaiMetadataFetch", lambda: FakeCivitaiMetadataFetch()
    )

    response = client.post("/api/v2/models/i/test_config_4/refresh_trigger_phrases")

    assert response.status_code == 200
    assert set(response.json()["trigger_phrases"]) == {"cached", "custom"}
    updated_config = mm2_model_manager.store.get_model("test_config_4")
    assert updated_config.source_url == "https://civitai.com/models/111/test-lora?modelVersionId=222"


def test_refresh_trigger_phrases_preserves_source_url_when_cached_metadata_has_no_source_url(
    monkeypatch: Any, client: TestClient, mm2_model_manager: Any, mm2_app_config: Any
) -> None:
    _patch_model_manager_dependencies(monkeypatch, mm2_model_manager, mm2_app_config)
    mm2_model_manager.store.update_model(
        "test_config_4",
        ModelRecordChanges(
            source_url="https://civitai.com/models/111/test-lora",
            source_api_response='{"trainedWords": ["cached"]}',
            trigger_phrases={"custom"},
        ),
    )

    class FakeCivitaiMetadataFetch:
        def from_hash(self, hash_value: str) -> CivitaiMetadata:
            raise UnknownMetadataException("live lookup failed")

        def from_api_response(self, json_str: str) -> CivitaiMetadata:
            assert json_str == '{"trainedWords": ["cached"]}'
            return _civitai_metadata(["cached"], source_url=None)

    monkeypatch.setattr(
        "invokeai.app.api.routers.model_manager.CivitaiMetadataFetch", lambda: FakeCivitaiMetadataFetch()
    )

    response = client.post("/api/v2/models/i/test_config_4/refresh_trigger_phrases")

    assert response.status_code == 200
    updated_config = mm2_model_manager.store.get_model("test_config_4")
    assert updated_config.source_url == "https://civitai.com/models/111/test-lora"


def test_refresh_trigger_phrases_errors_when_metadata_cannot_be_resolved(
    monkeypatch: Any, client: TestClient, mm2_model_manager: Any, mm2_app_config: Any
) -> None:
    _patch_model_manager_dependencies(monkeypatch, mm2_model_manager, mm2_app_config)

    class FakeCivitaiMetadataFetch:
        def from_hash(self, hash_value: str) -> CivitaiMetadata:
            raise UnknownMetadataException("not found")

    monkeypatch.setattr(
        "invokeai.app.api.routers.model_manager.CivitaiMetadataFetch", lambda: FakeCivitaiMetadataFetch()
    )

    response = client.post("/api/v2/models/i/test_config_4/refresh_trigger_phrases")

    assert response.status_code == 404
    assert "No version-specific CivitAI URL, matching CivitAI hash, or cached CivitAI response" in response.json()[
        "detail"
    ]
