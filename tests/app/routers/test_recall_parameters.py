"""Tests for the recall parameters router.

Focused on the ``reference_images`` field added for model-free reference
images (FLUX.2 Klein, FLUX Kontext, Qwen Image Edit). The existing
``loras`` / ``control_layers`` / ``ip_adapters`` paths are exercised via
integration tests elsewhere; this file pins down the new field's
request-validation, resolver behavior, and event payload.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers import recall_parameters as recall_module
from invokeai.app.api_app import app
from invokeai.app.services.invoker import Invoker


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class MockApiDependencies(ApiDependencies):
    """Minimal ApiDependencies stand-in that only wires up an invoker."""

    invoker: Invoker

    def __init__(self, invoker: Invoker) -> None:
        self.invoker = invoker


@pytest.fixture
def patched_dependencies(monkeypatch: Any, mock_invoker: Invoker) -> MockApiDependencies:
    """Install a mock ApiDependencies for the recall_parameters router.

    The router persists each parameter via ``client_state_persistence.set_by_key``,
    whose ``user_id`` column has a FOREIGN KEY constraint back to the users
    table. The mock invoker uses an in-memory SQLite database that is not
    pre-populated with any users, so persistence would fail with "FOREIGN
    KEY constraint failed" — that's an orthogonal concern to the reference-
    images resolver under test, so we stub it out.
    """
    dependencies = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.recall_parameters.ApiDependencies", dependencies)
    monkeypatch.setattr(
        mock_invoker.services.client_state_persistence,
        "set_by_key",
        lambda user_id, key, value: value,
    )
    return dependencies


class TestReferenceImagesRecall:
    def test_reference_images_forwarded_when_image_exists(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """Reference images whose files exist should flow through to the event payload."""

        # Stub load_image_file so we don't need a real outputs/images directory.
        def fake_load_image_file(image_name: str) -> dict[str, Any] | None:
            return {"image_name": image_name, "width": 1024, "height": 768}

        monkeypatch.setattr(recall_module, "load_image_file", fake_load_image_file)

        response = client.post(
            "/api/v1/recall/default",
            json={
                "reference_images": [
                    {"image_name": "cat.png"},
                    {"image_name": "dog.png"},
                ]
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["queue_id"] == "default"
        # Both references came through, in order.
        resolved = body["parameters"]["reference_images"]
        assert len(resolved) == 2
        assert resolved[0]["image"]["image_name"] == "cat.png"
        assert resolved[1]["image"]["image_name"] == "dog.png"
        assert resolved[0]["image"]["width"] == 1024

    def test_missing_reference_images_are_dropped_without_failing(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """An image that can't be loaded should be skipped — never 500."""

        def fake_load_image_file(image_name: str) -> dict[str, Any] | None:
            if image_name == "present.png":
                return {"image_name": image_name, "width": 512, "height": 512}
            return None

        monkeypatch.setattr(recall_module, "load_image_file", fake_load_image_file)

        response = client.post(
            "/api/v1/recall/default",
            json={
                "reference_images": [
                    {"image_name": "missing.png"},
                    {"image_name": "present.png"},
                ]
            },
        )
        assert response.status_code == 200
        resolved = response.json()["parameters"]["reference_images"]
        assert len(resolved) == 1
        assert resolved[0]["image"]["image_name"] == "present.png"

    def test_reference_images_do_not_require_model_name(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """The schema must accept a reference image entry with only ``image_name``.

        This pins down the "model-free" contract: unlike ``ip_adapters``,
        these entries are for FLUX.2 Klein / FLUX Kontext / Qwen Image Edit,
        where the reference image feeds the main model directly and there is
        no adapter model to name. Callers should be able to omit every
        field except ``image_name``.
        """
        monkeypatch.setattr(
            recall_module,
            "load_image_file",
            lambda image_name: {"image_name": image_name, "width": 64, "height": 64},
        )

        response = client.post(
            "/api/v1/recall/default",
            json={"reference_images": [{"image_name": "ok.png"}]},
        )
        assert response.status_code == 200
        resolved = response.json()["parameters"]["reference_images"]
        assert resolved == [{"image": {"image_name": "ok.png", "width": 64, "height": 64}}]

    def test_empty_reference_images_is_noop_for_other_fields(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """Sending an empty reference_images list should not break other fields."""
        monkeypatch.setattr(
            recall_module,
            "load_image_file",
            lambda image_name: {"image_name": image_name, "width": 1, "height": 1},
        )

        response = client.post(
            "/api/v1/recall/default",
            json={
                "positive_prompt": "hello",
                "reference_images": [],
            },
        )
        assert response.status_code == 200
        params = response.json()["parameters"]
        assert params["positive_prompt"] == "hello"
        assert params["reference_images"] == []
