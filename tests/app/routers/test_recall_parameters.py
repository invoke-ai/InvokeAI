"""Tests for the recall parameters router.

These tests monkey-patch the heavy-weight lookup helpers
(``resolve_model_name_to_key``, ``load_image_file``,
``process_controlnet_image``) rather than wiring up a real model manager
or image-files service. This keeps each test focused on the router's
request-validation, resolver sequencing, and broadcast payload shape.
"""

from collections.abc import Callable
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers import recall_parameters as recall_module
from invokeai.app.api_app import app
from invokeai.app.services.invoker import Invoker
from invokeai.backend.model_manager.taxonomy import ModelType


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
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", dependencies)
    monkeypatch.setattr(
        mock_invoker.services.client_state_persistence,
        "set_by_key",
        lambda user_id, key, value: value,
    )
    return dependencies


def make_name_to_key_stub(
    mapping: dict[tuple[str, ModelType], str],
) -> Callable[[str, ModelType], Optional[str]]:
    """Build a ``resolve_model_name_to_key`` stand-in from a (name, type) dict.

    Any lookup that is not present in ``mapping`` returns ``None``, mirroring
    what the real resolver does when the model manager cannot find a match.
    """

    def _lookup(model_name: str, model_type: ModelType = ModelType.Main) -> Optional[str]:
        return mapping.get((model_name, model_type))

    return _lookup


def make_load_image_file_stub(
    known_images: dict[str, tuple[int, int]],
) -> Callable[[str], Optional[dict[str, Any]]]:
    """Build a ``load_image_file`` stand-in from a name → (width, height) dict."""

    def _load(image_name: str) -> Optional[dict[str, Any]]:
        dims = known_images.get(image_name)
        if dims is None:
            return None
        width, height = dims
        return {"image_name": image_name, "width": width, "height": height}

    return _load


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


class TestLorasRecall:
    def test_multiple_loras_resolved_with_weights_and_is_enabled(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """Each LoRA's model name is resolved to a key and weight/is_enabled pass through."""
        monkeypatch.setattr(
            recall_module,
            "resolve_model_name_to_key",
            make_name_to_key_stub(
                {
                    ("detail-lora", ModelType.LoRA): "key-detail",
                    ("style-lora", ModelType.LoRA): "key-style",
                }
            ),
        )

        response = client.post(
            "/api/v1/recall/default",
            json={
                "loras": [
                    {"model_name": "detail-lora", "weight": 0.8, "is_enabled": True},
                    {"model_name": "style-lora", "weight": 0.5, "is_enabled": False},
                ]
            },
        )
        assert response.status_code == 200
        loras = response.json()["parameters"]["loras"]
        assert loras == [
            {"model_key": "key-detail", "weight": 0.8, "is_enabled": True},
            {"model_key": "key-style", "weight": 0.5, "is_enabled": False},
        ]

    def test_unresolvable_loras_are_dropped(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """LoRAs whose names do not resolve are silently skipped — not an error."""
        monkeypatch.setattr(
            recall_module,
            "resolve_model_name_to_key",
            make_name_to_key_stub({("keeper", ModelType.LoRA): "key-keeper"}),
        )

        response = client.post(
            "/api/v1/recall/default",
            json={
                "loras": [
                    {"model_name": "keeper", "weight": 0.7},
                    {"model_name": "ghost-lora"},
                ]
            },
        )
        assert response.status_code == 200
        loras = response.json()["parameters"]["loras"]
        assert len(loras) == 1
        assert loras[0]["model_key"] == "key-keeper"

    def test_is_enabled_defaults_to_true(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """Omitting is_enabled should default to True per the pydantic schema."""
        monkeypatch.setattr(
            recall_module,
            "resolve_model_name_to_key",
            make_name_to_key_stub({("x", ModelType.LoRA): "key-x"}),
        )

        response = client.post(
            "/api/v1/recall/default",
            json={"loras": [{"model_name": "x"}]},
        )
        assert response.status_code == 200
        assert response.json()["parameters"]["loras"][0]["is_enabled"] is True


class TestControlLayersRecall:
    def test_controlnet_resolution_takes_precedence(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """A name that matches a ControlNet model should resolve to it directly."""
        monkeypatch.setattr(
            recall_module,
            "resolve_model_name_to_key",
            make_name_to_key_stub({("canny", ModelType.ControlNet): "key-canny"}),
        )
        monkeypatch.setattr(
            recall_module,
            "load_image_file",
            make_load_image_file_stub({"ctl.png": (512, 512)}),
        )
        monkeypatch.setattr(recall_module, "process_controlnet_image", lambda *a, **kw: None)

        response = client.post(
            "/api/v1/recall/default",
            json={
                "control_layers": [
                    {
                        "model_name": "canny",
                        "image_name": "ctl.png",
                        "weight": 0.75,
                        "begin_step_percent": 0.1,
                        "end_step_percent": 0.9,
                        "control_mode": "balanced",
                    }
                ]
            },
        )
        assert response.status_code == 200
        layer = response.json()["parameters"]["control_layers"][0]
        assert layer["model_key"] == "key-canny"
        assert layer["weight"] == 0.75
        assert layer["begin_step_percent"] == 0.1
        assert layer["end_step_percent"] == 0.9
        assert layer["control_mode"] == "balanced"
        assert layer["image"] == {"image_name": "ctl.png", "width": 512, "height": 512}
        # processor returned None → no processed_image field
        assert "processed_image" not in layer

    def test_falls_back_to_t2i_adapter(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """When no ControlNet match exists, T2I Adapter is tried next."""
        monkeypatch.setattr(
            recall_module,
            "resolve_model_name_to_key",
            make_name_to_key_stub({("sketchy", ModelType.T2IAdapter): "key-t2i"}),
        )
        monkeypatch.setattr(recall_module, "load_image_file", make_load_image_file_stub({}))
        monkeypatch.setattr(recall_module, "process_controlnet_image", lambda *a, **kw: None)

        response = client.post(
            "/api/v1/recall/default",
            json={"control_layers": [{"model_name": "sketchy", "weight": 1.0}]},
        )
        assert response.status_code == 200
        assert response.json()["parameters"]["control_layers"][0]["model_key"] == "key-t2i"

    def test_falls_back_to_control_lora(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """When neither ControlNet nor T2I Adapter matches, Control LoRA is tried last."""
        monkeypatch.setattr(
            recall_module,
            "resolve_model_name_to_key",
            make_name_to_key_stub({("clora", ModelType.LoRA): "key-clora"}),
        )
        monkeypatch.setattr(recall_module, "load_image_file", make_load_image_file_stub({}))
        monkeypatch.setattr(recall_module, "process_controlnet_image", lambda *a, **kw: None)

        response = client.post(
            "/api/v1/recall/default",
            json={"control_layers": [{"model_name": "clora", "weight": 1.0}]},
        )
        assert response.status_code == 200
        assert response.json()["parameters"]["control_layers"][0]["model_key"] == "key-clora"

    def test_missing_image_still_resolves_config(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """A missing control image is warned about but does not block the rest of the config."""
        monkeypatch.setattr(
            recall_module,
            "resolve_model_name_to_key",
            make_name_to_key_stub({("canny", ModelType.ControlNet): "key-canny"}),
        )
        monkeypatch.setattr(recall_module, "load_image_file", make_load_image_file_stub({}))
        monkeypatch.setattr(recall_module, "process_controlnet_image", lambda *a, **kw: None)

        response = client.post(
            "/api/v1/recall/default",
            json={
                "control_layers": [
                    {
                        "model_name": "canny",
                        "image_name": "missing.png",
                        "weight": 0.75,
                    }
                ]
            },
        )
        assert response.status_code == 200
        layer = response.json()["parameters"]["control_layers"][0]
        assert layer["model_key"] == "key-canny"
        assert layer["weight"] == 0.75
        assert "image" not in layer
        assert "processed_image" not in layer

    def test_processed_image_included_when_processor_returns_data(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """When the processor produces a derived image, it is attached to the resolved layer."""
        monkeypatch.setattr(
            recall_module,
            "resolve_model_name_to_key",
            make_name_to_key_stub({("canny", ModelType.ControlNet): "key-canny"}),
        )
        monkeypatch.setattr(
            recall_module,
            "load_image_file",
            make_load_image_file_stub({"ctl.png": (768, 768)}),
        )
        monkeypatch.setattr(
            recall_module,
            "process_controlnet_image",
            lambda image_name, model_key, services: {
                "image_name": f"processed-{image_name}",
                "width": 768,
                "height": 768,
            },
        )

        response = client.post(
            "/api/v1/recall/default",
            json={"control_layers": [{"model_name": "canny", "image_name": "ctl.png", "weight": 1.0}]},
        )
        assert response.status_code == 200
        layer = response.json()["parameters"]["control_layers"][0]
        assert layer["processed_image"]["image_name"] == "processed-ctl.png"

    def test_unresolvable_control_layers_are_dropped(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """Control entries whose model doesn't resolve by any type are skipped."""
        monkeypatch.setattr(
            recall_module,
            "resolve_model_name_to_key",
            make_name_to_key_stub({}),
        )
        monkeypatch.setattr(recall_module, "load_image_file", make_load_image_file_stub({}))
        monkeypatch.setattr(recall_module, "process_controlnet_image", lambda *a, **kw: None)

        response = client.post(
            "/api/v1/recall/default",
            json={"control_layers": [{"model_name": "unknown", "weight": 1.0}]},
        )
        assert response.status_code == 200
        assert response.json()["parameters"]["control_layers"] == []


class TestIPAdaptersRecall:
    def test_ip_adapter_resolved_with_image_and_method(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """IPAdapter lookup is tried first and all config fields pass through."""
        monkeypatch.setattr(
            recall_module,
            "resolve_model_name_to_key",
            make_name_to_key_stub({("ipa-face", ModelType.IPAdapter): "key-ipa"}),
        )
        monkeypatch.setattr(
            recall_module,
            "load_image_file",
            make_load_image_file_stub({"ref.png": (1024, 1024)}),
        )

        response = client.post(
            "/api/v1/recall/default",
            json={
                "ip_adapters": [
                    {
                        "model_name": "ipa-face",
                        "image_name": "ref.png",
                        "weight": 0.7,
                        "begin_step_percent": 0.0,
                        "end_step_percent": 0.8,
                        "method": "style",
                    }
                ]
            },
        )
        assert response.status_code == 200
        adapter = response.json()["parameters"]["ip_adapters"][0]
        assert adapter["model_key"] == "key-ipa"
        assert adapter["weight"] == 0.7
        assert adapter["begin_step_percent"] == 0.0
        assert adapter["end_step_percent"] == 0.8
        assert adapter["method"] == "style"
        assert adapter["image"] == {"image_name": "ref.png", "width": 1024, "height": 1024}
        # image_influence was not sent, so it must not appear in the resolved config
        assert "image_influence" not in adapter

    def test_falls_back_to_flux_redux(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """When the name doesn't match an IPAdapter, FluxRedux is tried next."""
        monkeypatch.setattr(
            recall_module,
            "resolve_model_name_to_key",
            make_name_to_key_stub({("redux-1", ModelType.FluxRedux): "key-redux"}),
        )
        monkeypatch.setattr(
            recall_module,
            "load_image_file",
            make_load_image_file_stub({"ref.png": (512, 512)}),
        )

        response = client.post(
            "/api/v1/recall/default",
            json={
                "ip_adapters": [
                    {
                        "model_name": "redux-1",
                        "image_name": "ref.png",
                        "weight": 1.0,
                        "image_influence": "high",
                    }
                ]
            },
        )
        assert response.status_code == 200
        adapter = response.json()["parameters"]["ip_adapters"][0]
        assert adapter["model_key"] == "key-redux"
        assert adapter["image_influence"] == "high"

    def test_missing_image_still_resolves_config(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """A missing reference image is warned about but the adapter still lands."""
        monkeypatch.setattr(
            recall_module,
            "resolve_model_name_to_key",
            make_name_to_key_stub({("ipa", ModelType.IPAdapter): "key-ipa"}),
        )
        monkeypatch.setattr(recall_module, "load_image_file", make_load_image_file_stub({}))

        response = client.post(
            "/api/v1/recall/default",
            json={"ip_adapters": [{"model_name": "ipa", "image_name": "missing.png", "weight": 0.5}]},
        )
        assert response.status_code == 200
        adapter = response.json()["parameters"]["ip_adapters"][0]
        assert adapter["model_key"] == "key-ipa"
        assert adapter["weight"] == 0.5
        assert "image" not in adapter

    def test_unresolvable_ip_adapters_are_dropped(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """Adapters whose model can't be resolved (neither IPAdapter nor FluxRedux) are skipped."""
        monkeypatch.setattr(
            recall_module,
            "resolve_model_name_to_key",
            make_name_to_key_stub({}),
        )
        monkeypatch.setattr(recall_module, "load_image_file", make_load_image_file_stub({}))

        response = client.post(
            "/api/v1/recall/default",
            json={"ip_adapters": [{"model_name": "unknown", "weight": 1.0}]},
        )
        assert response.status_code == 200
        assert response.json()["parameters"]["ip_adapters"] == []


class TestCombinedRecall:
    def test_all_collection_fields_together(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """Exercise the full happy path: prompts, model, loras, control_layers, ip_adapters, reference_images."""
        monkeypatch.setattr(
            recall_module,
            "resolve_model_name_to_key",
            make_name_to_key_stub(
                {
                    ("my-model", ModelType.Main): "key-main",
                    ("detail-lora", ModelType.LoRA): "key-lora",
                    ("canny", ModelType.ControlNet): "key-canny",
                    ("ipa-face", ModelType.IPAdapter): "key-ipa",
                }
            ),
        )
        monkeypatch.setattr(
            recall_module,
            "load_image_file",
            make_load_image_file_stub(
                {
                    "ctl.png": (512, 512),
                    "face.png": (768, 768),
                    "ref.png": (1024, 1024),
                }
            ),
        )
        monkeypatch.setattr(recall_module, "process_controlnet_image", lambda *a, **kw: None)

        response = client.post(
            "/api/v1/recall/default",
            json={
                "positive_prompt": "a cat",
                "negative_prompt": "blurry",
                "model": "my-model",
                "steps": 30,
                "cfg_scale": 7.5,
                "width": 512,
                "height": 512,
                "seed": 42,
                "loras": [{"model_name": "detail-lora", "weight": 0.6}],
                "control_layers": [{"model_name": "canny", "image_name": "ctl.png", "weight": 0.75}],
                "ip_adapters": [
                    {"model_name": "ipa-face", "image_name": "face.png", "weight": 0.5, "method": "composition"}
                ],
                "reference_images": [{"image_name": "ref.png"}],
            },
        )
        assert response.status_code == 200
        params = response.json()["parameters"]

        # Core fields
        assert params["positive_prompt"] == "a cat"
        assert params["negative_prompt"] == "blurry"
        assert params["model"] == "key-main"
        assert params["steps"] == 30
        assert params["seed"] == 42

        # Collections
        assert params["loras"] == [{"model_key": "key-lora", "weight": 0.6, "is_enabled": True}]
        assert params["control_layers"][0]["model_key"] == "key-canny"
        assert params["control_layers"][0]["image"]["image_name"] == "ctl.png"
        assert params["ip_adapters"][0]["model_key"] == "key-ipa"
        assert params["ip_adapters"][0]["method"] == "composition"
        assert params["reference_images"] == [{"image": {"image_name": "ref.png", "width": 1024, "height": 1024}}]

    def test_unresolvable_main_model_drops_from_payload(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """A model name that doesn't resolve should be scrubbed from the broadcast payload."""
        monkeypatch.setattr(
            recall_module,
            "resolve_model_name_to_key",
            make_name_to_key_stub({}),
        )

        response = client.post(
            "/api/v1/recall/default",
            json={"positive_prompt": "x", "model": "ghost-model"},
        )
        assert response.status_code == 200
        params = response.json()["parameters"]
        assert params["positive_prompt"] == "x"
        assert "model" not in params


class TestStrictMode:
    """Regression tests for the ``strict`` query parameter.

    When ``strict=True``, parameters not included in the request body must
    be reset — list-typed fields to ``[]`` and scalar fields to ``None``.
    """

    def test_strict_clears_list_fields(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """List fields (loras, control_layers, ip_adapters, reference_images) are
        sent as empty lists when omitted in strict mode."""
        monkeypatch.setattr(recall_module, "resolve_model_name_to_key", make_name_to_key_stub({}))

        response = client.post(
            "/api/v1/recall/default?strict=true",
            json={"positive_prompt": "hello"},
        )
        assert response.status_code == 200
        params = response.json()["parameters"]
        assert params["positive_prompt"] == "hello"
        assert params["loras"] == []
        assert params["control_layers"] == []
        assert params["ip_adapters"] == []
        assert params["reference_images"] == []

    def test_strict_clears_scalar_fields(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """Scalar fields not in the request are sent as None in strict mode."""
        monkeypatch.setattr(recall_module, "resolve_model_name_to_key", make_name_to_key_stub({}))

        response = client.post(
            "/api/v1/recall/default?strict=true",
            json={"steps": 20},
        )
        assert response.status_code == 200
        params = response.json()["parameters"]
        assert params["steps"] == 20
        assert params["positive_prompt"] is None
        assert params["seed"] is None
        assert params["loras"] == []

    def test_non_strict_omits_unset_fields(
        self, monkeypatch: Any, patched_dependencies: MockApiDependencies, client: TestClient
    ) -> None:
        """Default (non-strict) behaviour: unset fields are absent from the response."""
        monkeypatch.setattr(recall_module, "resolve_model_name_to_key", make_name_to_key_stub({}))

        response = client.post(
            "/api/v1/recall/default",
            json={"positive_prompt": "hello"},
        )
        assert response.status_code == 200
        params = response.json()["parameters"]
        assert params["positive_prompt"] == "hello"
        assert "loras" not in params
        assert "reference_images" not in params
        assert "seed" not in params
