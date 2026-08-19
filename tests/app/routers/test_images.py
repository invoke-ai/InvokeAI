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


def prepare_starred_delete_test(
    monkeypatch: Any,
    mock_invoker: Invoker,
    starred_names: set[str],
    board_ids: dict[str, str | None],
) -> None:
    mock_deps = MockApiDependencies(mock_invoker)
    mock_invoker.services.image_moves = MagicMock()
    mock_invoker.services.image_moves.is_maintenance_active.return_value = False
    mock_invoker.services.board_image_records = MagicMock()
    mock_invoker.services.board_image_records.get_board_for_image.side_effect = board_ids.__getitem__
    mock_invoker.services.board_images = MagicMock()
    mock_invoker.services.images = MagicMock()
    mock_invoker.services.images.delete.side_effect = (
        lambda image_name, delete_starred=True: delete_starred or image_name not in starred_names
    )
    monkeypatch.setattr("invokeai.app.api.routers.images.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers._access.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.image_move_maintenance.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)


def test_delete_starred_image_is_skipped_when_protected(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    image_name = "starred.png"
    prepare_starred_delete_test(
        monkeypatch,
        mock_invoker,
        starred_names={image_name},
        board_ids={image_name: "board-id"},
    )

    response = client.delete(f"/api/v1/images/i/{image_name}", params={"delete_starred": False})

    assert response.status_code == 200
    assert response.json() == {
        "deleted_images": [],
        "failed_images": [],
        "affected_boards": ["board-id"],
        "starred_skipped": [image_name],
    }
    mock_invoker.services.images.delete.assert_called_once_with(image_name, delete_starred=False)


def test_bulk_delete_only_deletes_unstarred_images_when_protected(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    starred_name = "starred.png"
    unstarred_name = "unstarred.png"
    prepare_starred_delete_test(
        monkeypatch,
        mock_invoker,
        starred_names={starred_name},
        board_ids={starred_name: "board-id", unstarred_name: "board-id"},
    )

    response = client.post(
        "/api/v1/images/delete",
        json={"image_names": [starred_name, unstarred_name], "delete_starred": False},
    )

    assert response.status_code == 200
    assert response.json() == {
        "deleted_images": [unstarred_name],
        "failed_images": [],
        "affected_boards": ["board-id"],
        "starred_skipped": [starred_name],
    }
    assert mock_invoker.services.images.delete.call_count == 2
    mock_invoker.services.images.delete.assert_any_call(starred_name, delete_starred=False)
    mock_invoker.services.images.delete.assert_any_call(unstarred_name, delete_starred=False)


def test_bulk_delete_deduplicates_image_names(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    prepare_starred_delete_test(
        monkeypatch,
        mock_invoker,
        starred_names=set(),
        board_ids={"duplicate.png": "board-id", "other.png": "board-id"},
    )

    response = client.post(
        "/api/v1/images/delete",
        json={"image_names": ["duplicate.png", "duplicate.png", "other.png"]},
    )

    assert response.status_code == 200
    assert sorted(response.json()["deleted_images"]) == ["duplicate.png", "other.png"]
    assert response.json()["failed_images"] == []
    assert [call.args[0] for call in mock_invoker.services.images.delete.call_args_list] == [
        "duplicate.png",
        "other.png",
    ]


def test_delete_uncategorized_only_deletes_unstarred_images_when_protected(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    starred_name = "starred.png"
    unstarred_name = "unstarred.png"
    prepare_starred_delete_test(
        monkeypatch,
        mock_invoker,
        starred_names={starred_name},
        board_ids={starred_name: None, unstarred_name: None},
    )
    mock_invoker.services.board_images.get_all_board_image_names_for_board.return_value = [
        starred_name,
        unstarred_name,
    ]

    response = client.delete("/api/v1/images/uncategorized", params={"delete_starred": False})

    assert response.status_code == 200
    assert response.json() == {
        "deleted_images": [unstarred_name],
        "failed_images": [],
        "affected_boards": ["none"],
        "starred_skipped": [starred_name],
    }
    assert mock_invoker.services.images.delete.call_count == 2


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
