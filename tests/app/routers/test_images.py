import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from invokeai.app.api.auth_dependencies import get_current_user_or_default
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.auth.token_service import TokenData
from invokeai.app.services.board_records.board_records_common import BoardRecord
from invokeai.app.services.invoker import Invoker


@pytest.fixture(autouse=True, scope="module")
def client(invokeai_root_dir: Path) -> TestClient:
    os.environ["INVOKEAI_ROOT"] = invokeai_root_dir.as_posix()
    return TestClient(app)


class MockApiDependencies(ApiDependencies):
    invoker: Invoker

    def __init__(self, invoker) -> None:
        self.invoker = invoker


def test_download_images_from_list(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    prepare_download_images_test(monkeypatch, mock_invoker)

    response = client.post("/api/v1/images/download", json={"image_names": ["test.png"]})
    json_response = response.json()
    assert response.status_code == 202
    assert json_response["bulk_download_item_name"] == "test.zip"


def test_download_images_from_board_id_empty_image_name_list(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    expected_board_name = "test"

    def mock_get(*args, **kwargs):
        return BoardRecord(board_id="12345", board_name=expected_board_name, created_at="None", updated_at="None")

    monkeypatch.setattr(mock_invoker.services.board_records, "get", mock_get)
    prepare_download_images_test(monkeypatch, mock_invoker)

    response = client.post("/api/v1/images/download", json={"board_id": "test"})
    json_response = response.json()
    assert response.status_code == 202
    assert json_response["bulk_download_item_name"] == "test.zip"


def prepare_download_images_test(monkeypatch: Any, mock_invoker: Invoker) -> None:
    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.images.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)
    monkeypatch.setattr(
        "invokeai.app.api.routers.images.ApiDependencies.invoker.services.bulk_download.generate_item_id",
        lambda arg: "test",
    )

    def mock_add_task(*args, **kwargs):
        return None

    monkeypatch.setattr(BackgroundTasks, "add_task", mock_add_task)


def prepare_image_maintenance_test(monkeypatch: Any, mock_invoker: Invoker) -> None:
    mock_deps = MockApiDependencies(mock_invoker)
    mock_invoker.services.image_moves = MagicMock()
    mock_invoker.services.image_moves.is_maintenance_active.return_value = True
    monkeypatch.setattr(mock_invoker.services.image_records, "get_user_id", MagicMock(return_value="system"))
    monkeypatch.setattr(mock_invoker.services.board_image_records, "get_board_for_image", MagicMock(return_value=None))
    monkeypatch.setattr("invokeai.app.api.routers.images.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers._access.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.image_move_maintenance.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("get", "/api/v1/images/i/test.png/full", None),
        ("head", "/api/v1/images/i/test.png/full", None),
        ("get", "/api/v1/images/i/test.png/thumbnail", None),
        ("get", "/api/v1/images/i/test.png/workflow", None),
        ("delete", "/api/v1/images/i/test.png", None),
        ("delete", "/api/v1/images/intermediates", None),
        ("delete", "/api/v1/images/uncategorized", None),
        ("patch", "/api/v1/images/i/test.png", {"starred": True}),
        ("post", "/api/v1/images/delete", {"image_names": ["test.png"]}),
        ("post", "/api/v1/images/star", {"image_names": ["test.png"]}),
        ("post", "/api/v1/images/unstar", {"image_names": ["test.png"]}),
        ("post", "/api/v1/images/download", {"image_names": ["test.png"]}),
    ],
)
def test_image_operations_are_blocked_during_image_move_maintenance(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient, method: str, path: str, json_body: dict | None
) -> None:
    prepare_image_maintenance_test(monkeypatch, mock_invoker)

    if json_body is not None:
        response = getattr(client, method)(path, json=json_body)
    else:
        response = getattr(client, method)(path)

    assert response.status_code == 409
    if method != "head":
        assert response.json()["detail"] == "Image storage maintenance is active"


def test_image_mutation_checks_access_before_image_move_maintenance(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    prepare_image_maintenance_test(monkeypatch, mock_invoker)
    monkeypatch.setattr(mock_invoker.services.image_records, "get_user_id", MagicMock(return_value="other-user"))

    async def current_user_override() -> TokenData:
        return TokenData(user_id="request-user", email="request-user@example.com", is_admin=False)

    app.dependency_overrides[get_current_user_or_default] = current_user_override
    try:
        response = client.delete("/api/v1/images/i/test.png")

        assert response.status_code == 403
        mock_invoker.services.image_moves.is_maintenance_active.assert_not_called()
    finally:
        app.dependency_overrides.pop(get_current_user_or_default, None)


def test_image_upload_is_blocked_during_image_move_maintenance(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    prepare_image_maintenance_test(monkeypatch, mock_invoker)

    response = client.post(
        "/api/v1/images/upload",
        params={"image_category": "general", "is_intermediate": False},
        files={"file": ("test.png", b"not-read-during-maintenance", "image/png")},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Image storage maintenance is active"


def test_image_to_prompt_is_blocked_during_image_move_maintenance(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    prepare_image_maintenance_test(monkeypatch, mock_invoker)

    response = client.post(
        "/api/v1/utilities/image-to-prompt",
        json={"image_name": "test.png", "model_key": "model-key", "instruction": "describe"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Image storage maintenance is active"


def test_download_images_with_empty_image_list_and_no_board_id(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    prepare_download_images_test(monkeypatch, mock_invoker)

    response = client.post("/api/v1/images/download", json={"image_names": []})

    assert response.status_code == 400


def test_get_bulk_download_image(tmp_path: Path, monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    mock_file: Path = tmp_path / "test.zip"
    mock_file.write_text("contents")

    monkeypatch.setattr(mock_invoker.services.bulk_download, "get_path", lambda x: str(mock_file))
    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.images.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)

    def mock_add_task(*args, **kwargs):
        return None

    monkeypatch.setattr(BackgroundTasks, "add_task", mock_add_task)

    response = client.get("/api/v1/images/download/test.zip")

    assert response.status_code == 200
    assert response.content == b"contents"


def test_get_bulk_download_image_not_found(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.images.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)

    def mock_add_task(*args, **kwargs):
        return None

    monkeypatch.setattr(BackgroundTasks, "add_task", mock_add_task)

    response = client.get("/api/v1/images/download/test.zip")

    assert response.status_code == 404


def test_get_bulk_download_image_image_deleted_after_response(
    monkeypatch: Any, mock_invoker: Invoker, tmp_path: Path, client: TestClient
) -> None:
    mock_file: Path = tmp_path / "test.zip"
    mock_file.write_text("contents")

    monkeypatch.setattr(mock_invoker.services.bulk_download, "get_path", lambda x: str(mock_file))
    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.images.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)

    client.get("/api/v1/images/download/test.zip")

    assert not (tmp_path / "test.zip").exists()


def test_search_image_names_endpoint(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    monkeypatch.setattr("invokeai.app.api.routers.images.ApiDependencies", MockApiDependencies(mock_invoker))

    captured: dict[str, Any] = {}

    def mock_get_image_names(**kwargs):
        captured.update(kwargs)
        return {"image_names": ["test.png"], "starred_count": 0, "total_count": 1}

    monkeypatch.setattr(mock_invoker.services.images, "get_image_names", mock_get_image_names)

    response = client.post(
        "/api/v1/images/search/names",
        json={
            "file_name_term": "test",
            "metadata_term": "prompt",
            "width_min": 512,
            "height_exact": 768,
            "board_ids": ["none"],
            "starred_mode": "only",
        },
    )

    assert response.status_code == 200
    assert response.json()["image_names"] == ["test.png"]
    assert captured["file_name_term"] == "test"
    assert captured["metadata_term"] == "prompt"
    assert captured["width_min"] == 512
    assert captured["height_exact"] == 768
    assert captured["board_ids"] == ["none"]
    assert captured["starred_mode"] == "only"
