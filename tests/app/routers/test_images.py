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


# ── Transactional single-image deletion (DELETE /api/v1/images/i/{image_name}) ──


def prepare_delete_image_test(monkeypatch: Any, mock_invoker: Invoker, tmp_path: Path):
    """Wire the delete route to a real ImageService + real DiskImageFileStorage + real SQLite records."""
    from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage

    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.images.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers._access.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.image_move_maintenance.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)

    mock_invoker.services.urls = MagicMock()
    mock_invoker.services.urls.get_image_url.return_value = "http://localhost/img.png"

    storage = DiskImageFileStorage(tmp_path / "outputs")
    mock_invoker.services.image_files = storage
    storage.start(mock_invoker)
    mock_invoker.services.images.start(mock_invoker)
    return storage


def _save_deletable_image(mock_invoker: Invoker, storage, image_name: str) -> None:
    from PIL import Image

    from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin

    mock_invoker.services.image_records.save(
        image_name=image_name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=64,
        height=64,
        has_workflow=False,
    )
    storage.save(image=Image.new("RGB", (64, 64)), image_name=image_name)


def test_delete_image_success_deletes_files_and_record(
    monkeypatch: Any, mock_invoker: Invoker, tmp_path: Path, client: TestClient
) -> None:
    from invokeai.app.services.image_records.image_records_common import ImageRecordNotFoundException

    storage = prepare_delete_image_test(monkeypatch, mock_invoker, tmp_path)
    _save_deletable_image(mock_invoker, storage, "del.png")

    response = client.delete("/api/v1/images/i/del.png")

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["deleted_images"] == ["del.png"]
    assert json_response["affected_boards"] == ["none"]
    assert not storage.get_path("del.png").exists()
    assert not storage.get_path("del.png", thumbnail=True).exists()
    with pytest.raises(ImageRecordNotFoundException):
        mock_invoker.services.image_records.get("del.png")
    assert list(storage.image_root.glob(".delete_*")) == []


def test_delete_image_not_found_returns_404(
    monkeypatch: Any, mock_invoker: Invoker, tmp_path: Path, client: TestClient
) -> None:
    prepare_delete_image_test(monkeypatch, mock_invoker, tmp_path)

    response = client.delete("/api/v1/images/i/does-not-exist.png")

    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found"


def test_delete_image_db_failure_returns_500_and_restores_files(
    monkeypatch: Any, mock_invoker: Invoker, tmp_path: Path, client: TestClient
) -> None:
    from invokeai.app.services.image_records.image_records_common import ImageRecordDeleteException

    storage = prepare_delete_image_test(monkeypatch, mock_invoker, tmp_path)
    _save_deletable_image(mock_invoker, storage, "del.png")

    def failing_delete(image_name: str) -> None:
        raise ImageRecordDeleteException()

    monkeypatch.setattr(mock_invoker.services.image_records, "delete", failing_delete)

    response = client.delete("/api/v1/images/i/del.png")

    # The route must report the failure, not a success-shaped empty payload.
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to delete image"
    # The staged files must be rolled back: image and thumbnail restored, record intact.
    assert storage.get_path("del.png").exists()
    assert storage.get_path("del.png", thumbnail=True).exists()
    assert mock_invoker.services.image_records.get("del.png").image_name == "del.png"
    assert list(storage.image_root.glob(".delete_*")) == []
