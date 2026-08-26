import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from invokeai.app.api.auth_dependencies import get_current_user_or_default
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers.images import MAX_IMAGE_BATCH_SIZE
from invokeai.app.api_app import app
from invokeai.app.services.auth.token_service import TokenData
from invokeai.app.services.board_records.board_records_common import BoardRecord
from invokeai.app.services.image_records.image_records_common import ImageNamesResult, ImageRecordNotFoundException
from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import MAX_PAGE_SIZE, OffsetPaginatedResults


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

    mock_get = MagicMock(
        return_value=BoardRecord(board_id="12345", board_name=expected_board_name, created_at="None", updated_at="None")
    )
    monkeypatch.setattr(mock_invoker.services.board_records, "get", mock_get)
    prepare_download_images_test(monkeypatch, mock_invoker)

    response = client.post("/api/v1/images/download", json={"board_id": "test"})
    json_response = response.json()
    assert response.status_code == 202
    assert json_response["bulk_download_item_name"] == "test.zip"
    mock_get.assert_called_once_with("test")


def prepare_download_images_test(monkeypatch: Any, mock_invoker: Invoker) -> None:
    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.images.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers._access.ApiDependencies", mock_deps)
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
    monkeypatch.setattr(mock_invoker.services.image_records, "exists", MagicMock(return_value=True))
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


def prepare_created_range_test(monkeypatch: Any, mock_invoker: Invoker) -> tuple[MagicMock, MagicMock]:
    """Patches list endpoints' service calls with capturing mocks; returns (get_many, get_image_names)."""
    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.images.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)

    mock_get_many = MagicMock(return_value=OffsetPaginatedResults(items=[], offset=0, limit=10, total=0))
    mock_get_image_names = MagicMock(return_value=ImageNamesResult(image_names=[], starred_count=0, total_count=0))
    monkeypatch.setattr(mock_invoker.services.images, "get_many", mock_get_many)
    monkeypatch.setattr(mock_invoker.services.images, "get_image_names", mock_get_image_names)
    return mock_get_many, mock_get_image_names


@pytest.mark.parametrize("path", ["/api/v1/images/", "/api/v1/images/names"])
@pytest.mark.parametrize("bad_value", ["next-tuesday", "2026-02-31"])
@pytest.mark.parametrize("param", ["created_from", "created_to"])
def test_list_endpoints_reject_invalid_created_range_dates(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient, path: str, bad_value: str, param: str
) -> None:
    """Malformed shapes and impossible calendar dates are both rejected with 422."""
    prepare_created_range_test(monkeypatch, mock_invoker)

    response = client.get(path, params={param: bad_value})

    assert response.status_code == 422


def test_list_image_dtos_forwards_created_range(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    mock_get_many, _ = prepare_created_range_test(monkeypatch, mock_invoker)

    response = client.get("/api/v1/images/", params={"created_from": "2026-07-01", "created_to": "2026-07-15"})

    assert response.status_code == 200
    kwargs = mock_get_many.call_args.kwargs
    assert kwargs["created_from"] == "2026-07-01"
    assert kwargs["created_to"] == "2026-07-15"


def test_list_image_dtos_omits_created_range_by_default(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    mock_get_many, _ = prepare_created_range_test(monkeypatch, mock_invoker)

    response = client.get("/api/v1/images/")

    assert response.status_code == 200
    kwargs = mock_get_many.call_args.kwargs
    assert kwargs["created_from"] is None
    assert kwargs["created_to"] is None


def test_get_image_names_forwards_created_range(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    _, mock_get_image_names = prepare_created_range_test(monkeypatch, mock_invoker)

    response = client.get("/api/v1/images/names", params={"created_from": "2026-07-01", "created_to": "2026-07-15"})

    assert response.status_code == 200
    kwargs = mock_get_image_names.call_args.kwargs
    assert kwargs["created_from"] == "2026-07-01"
    assert kwargs["created_to"] == "2026-07-15"


@pytest.mark.parametrize("path", ["/api/v1/images/", "/api/v1/images/names"])
@pytest.mark.parametrize("board_id", ["all", "none"])
def test_list_image_sentinel_scopes_skip_concrete_board_access_check(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient, path: str, board_id: str
) -> None:
    prepare_created_range_test(monkeypatch, mock_invoker)
    access_check = MagicMock()
    monkeypatch.setattr("invokeai.app.api.routers.images._assert_board_read_access", access_check)

    response = client.get(path, params={"board_id": board_id})

    assert response.status_code == 200
    access_check.assert_not_called()


@pytest.mark.parametrize("path", ["/api/v1/images/", "/api/v1/images/names"])
def test_list_image_concrete_board_requires_read_access(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient, path: str
) -> None:
    prepare_created_range_test(monkeypatch, mock_invoker)
    access_check = MagicMock()
    monkeypatch.setattr("invokeai.app.api.routers.images._assert_board_read_access", access_check)

    response = client.get(path, params={"board_id": "board-123"})

    assert response.status_code == 200
    access_check.assert_called_once()
    assert access_check.call_args.args[0] == "board-123"


@pytest.mark.parametrize("path", ["/api/v1/images/", "/api/v1/images/names"])
def test_list_image_all_scope_combines_with_created_range(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient, path: str
) -> None:
    mock_get_many, mock_get_image_names = prepare_created_range_test(monkeypatch, mock_invoker)

    response = client.get(
        path,
        params={"board_id": "all", "created_from": "2026-07-01", "created_to": "2026-07-15"},
    )

    assert response.status_code == 200
    service_call = mock_get_image_names if path.endswith("/names") else mock_get_many
    assert service_call.call_args.kwargs["board_id"] == "all"
    assert service_call.call_args.kwargs["created_from"] == "2026-07-01"
    assert service_call.call_args.kwargs["created_to"] == "2026-07-15"


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


def prepare_image_batch_test(monkeypatch: Any, mock_invoker: Invoker) -> MagicMock:
    """Wires the image router to a MagicMock image service with maintenance inactive.

    Returns the mock service so tests can script per-name update outcomes.
    """
    images_service = MagicMock()
    monkeypatch.setattr(mock_invoker.services, "images", images_service)
    mock_invoker.services.image_moves = MagicMock()
    mock_invoker.services.image_moves.is_maintenance_active.return_value = False
    monkeypatch.setattr(mock_invoker.services.image_records, "exists", MagicMock(return_value=True))
    monkeypatch.setattr(mock_invoker.services.board_image_records, "get_board_for_image", MagicMock(return_value=None))

    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.images.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers._access.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.image_move_maintenance.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)
    return images_service


@pytest.fixture
def non_admin_user():
    """Makes ownership decisions depend on image_records.get_user_id rather than admin bypass."""

    async def current_user_override() -> TokenData:
        return TokenData(user_id="request-user", email="request-user@example.com", is_admin=False)

    app.dependency_overrides[get_current_user_or_default] = current_user_override
    yield
    app.dependency_overrides.pop(get_current_user_or_default, None)


@pytest.mark.parametrize(
    ("route", "updated_key"),
    [("star", "starred_images"), ("unstar", "unstarred_images")],
)
def test_star_unstar_reports_failures_and_keeps_partial_successes(
    monkeypatch: Any,
    mock_invoker: Invoker,
    client: TestClient,
    non_admin_user: None,
    route: str,
    updated_key: str,
) -> None:
    """A foreign name is skipped, a storage failure is reported, and the rest still apply.

    Both used to abort the whole batch on the first foreign name (discarding the images
    that HAD been updated) and to silently swallow storage failures, so the client cached
    a star that never reached the DB.
    """
    images_service = prepare_image_batch_test(monkeypatch, mock_invoker)
    owners = {"ok.png": "request-user", "broken.png": "request-user", "foreign.png": "someone-else"}
    monkeypatch.setattr(
        mock_invoker.services.image_records, "get_user_id", MagicMock(side_effect=lambda name: owners.get(name))
    )

    def update(image_name: str, changes: Any) -> MagicMock:
        del changes
        if image_name == "broken.png":
            raise RuntimeError("storage is on fire")
        dto = MagicMock()
        dto.board_id = "board-1"
        return dto

    images_service.update.side_effect = update

    response = client.post(f"/api/v1/images/{route}", json={"image_names": ["ok.png", "broken.png", "foreign.png"]})

    assert response.status_code == 200
    body = response.json()
    assert body[updated_key] == ["ok.png"]
    # The genuine failure is reported; the foreign name is an intentional skip and must
    # not be toasted as a failure.
    assert body["failed_images"] == ["broken.png"]
    assert body["affected_boards"] == ["board-1"]


@pytest.mark.parametrize(
    ("route", "updated_key"),
    [("star", "starred_images"), ("unstar", "unstarred_images")],
)
def test_star_unstar_skips_names_deleted_mid_batch(
    monkeypatch: Any,
    mock_invoker: Invoker,
    client: TestClient,
    route: str,
    updated_key: str,
) -> None:
    """A name deleted by a concurrent session is a skip, not a storage failure.

    Reported as an admin, because that is the only caller for which this is reachable: for
    anyone else `get_user_id` returns None for a missing record and the ownership check
    already answers 403. It is also the default single-user path, so this WAS the common
    case -- the name landed in failed_images and toasted "1 image could not be updated"
    for an image the user no longer had.
    """
    images_service = prepare_image_batch_test(monkeypatch, mock_invoker)

    def update(image_name: str, changes: Any) -> MagicMock:
        del changes
        if image_name == "vanished.png":
            raise ImageRecordNotFoundException
        dto = MagicMock()
        dto.board_id = "board-1"
        return dto

    images_service.update.side_effect = update

    response = client.post(f"/api/v1/images/{route}", json={"image_names": ["ok.png", "vanished.png"]})

    assert response.status_code == 200
    body = response.json()
    assert body[updated_key] == ["ok.png"]
    assert body["failed_images"] == []


@pytest.mark.parametrize("raise_from", ["get_dto", "delete"])
def test_delete_skips_names_deleted_mid_batch(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient, raise_from: str
) -> None:
    """Same for /delete: the caller asked for the image to be gone, and it is.

    Parametrized over both raise sites because the race window spans them: the loop reads the
    DTO for its board id and only then deletes, and ImageService.delete re-reads the record, so
    a name can vanish after the first read succeeds. The two sites answer differently on
    purpose, and the difference is what this asserts — see the branch below.
    """
    images_service = prepare_image_batch_test(monkeypatch, mock_invoker)

    def get_dto(image_name: str) -> MagicMock:
        if image_name == "vanished.png" and raise_from == "get_dto":
            raise ImageRecordNotFoundException
        dto = MagicMock()
        # Distinct boards on purpose: sharing one would let the surviving image supply the
        # vanished one's board, and the assertion below could not tell whether it was reported.
        dto.board_id = "board-2" if image_name == "vanished.png" else "board-1"
        return dto

    def delete(image_name: str) -> None:
        if image_name == "vanished.png" and raise_from == "delete":
            raise ImageRecordNotFoundException

    images_service.get_dto.side_effect = get_dto
    images_service.delete.side_effect = delete

    response = client.post("/api/v1/images/delete", json={"image_names": ["ok.png", "vanished.png"]})

    assert response.status_code == 200
    body = response.json()
    assert body["failed_images"] == []
    if raise_from == "delete":
        # Read a line earlier, so the record verifiably existed and a concurrent session removed
        # it: the requested postcondition holds and must reach the client cleanup path as a
        # confirmed deletion. Order is intentionally unspecified — the route accumulates a set.
        assert set(body["deleted_images"]) == {"ok.png", "vanished.png"}
        # Reported with its board. Every board-scoped tag getDeleteImagesTags publishes comes
        # from affected_boards, and it ignores deleted_images by design, so dropping the board
        # here leaves its counts stale while the name is reported gone.
        assert set(body["affected_boards"]) == {"board-1", "board-2"}
    else:
        # The read itself failed, so nothing established the record ever existed. That matters
        # because assert_image_owner returns immediately for an admin — the default single-user
        # identity — without touching storage, so a name that never existed reaches this path.
        # Reporting it deleted would answer for something the caller never had.
        assert body["deleted_images"] == ["ok.png"]
        assert set(body["affected_boards"]) == {"board-1"}


def test_delete_does_not_report_a_name_that_never_existed(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    """The admin path specifically: the ownership check is a no-op that proves nothing."""
    images_service = prepare_image_batch_test(monkeypatch, mock_invoker)

    def get_dto(image_name: str) -> MagicMock:
        raise ImageRecordNotFoundException

    images_service.get_dto.side_effect = get_dto

    response = client.post("/api/v1/images/delete", json={"image_names": ["never-existed.png"]})

    assert response.status_code == 200
    body = response.json()
    assert body["deleted_images"] == []
    assert body["failed_images"] == []


def test_image_records_get_does_not_disguise_a_storage_error_as_not_found(monkeypatch: Any) -> None:
    """The narrowing itself: a sqlite3.Error out of the SELECT must stay a sqlite3.Error."""
    from invokeai.app.services.image_records.image_records_sqlite import SqliteImageRecordStorage

    storage = SqliteImageRecordStorage.__new__(SqliteImageRecordStorage)

    class _Cursor:
        def execute(self, *args: Any, **kwargs: Any) -> None:
            raise sqlite3.OperationalError("database disk image is malformed")

    class _Db:
        @contextmanager
        def transaction(self):
            yield _Cursor()

    storage._db = _Db()  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(sqlite3.OperationalError):
        storage.get("a.png")


def test_delete_still_reports_a_genuine_storage_failure(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    """The not-found skip must stay narrow -- a real deletion failure is still reported.

    This is the half of the storage-error story the route owns. The other half is at the
    storage layer: image_records.get() used to re-raise every sqlite3.Error as
    ImageRecordNotFoundException, so a locked or corrupt database was indistinguishable from a
    concurrent delete and the skip would have answered 200 with two empty lists and no toast at
    all. See test_image_records_get_does_not_disguise_a_storage_error_as_not_found.
    """
    images_service = prepare_image_batch_test(monkeypatch, mock_invoker)
    dto = MagicMock()
    dto.board_id = "board-1"
    images_service.get_dto.return_value = dto

    def delete(image_name: str) -> None:
        if image_name == "broken.png":
            raise RuntimeError("storage is on fire")

    images_service.delete.side_effect = delete

    response = client.post("/api/v1/images/delete", json={"image_names": ["ok.png", "broken.png"]})

    assert response.status_code == 200
    body = response.json()
    assert body["deleted_images"] == ["ok.png"]
    assert body["failed_images"] == ["broken.png"]


@pytest.mark.parametrize("route", ["star", "unstar"])
def test_star_unstar_dedupes_repeated_names(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient, non_admin_user: None, route: str
) -> None:
    images_service = prepare_image_batch_test(monkeypatch, mock_invoker)
    monkeypatch.setattr(mock_invoker.services.image_records, "get_user_id", MagicMock(return_value="request-user"))
    dto = MagicMock()
    dto.board_id = "board-1"
    images_service.update.return_value = dto

    response = client.post(f"/api/v1/images/{route}", json={"image_names": ["a.png", "a.png", "a.png"]})

    assert response.status_code == 200
    assert images_service.update.call_count == 1


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/images/delete",
        "/api/v1/images/star",
        "/api/v1/images/unstar",
        "/api/v1/images/images_by_names",
        "/api/v1/images/download",
        "/api/v1/images/copy",
        "/api/v1/board_images/batch",
        "/api/v1/board_images/batch/delete",
    ],
)
def test_image_name_batches_are_bounded(monkeypatch: Any, mock_invoker: Invoker, client: TestClient, path: str) -> None:
    """An unbounded name list is a free amplification: each name costs at least one DB lookup,
    and up to six when the caller is reading someone else's shared board."""
    images_service = prepare_image_batch_test(monkeypatch, mock_invoker)
    # /download would otherwise authorize every name and schedule a background task; if the
    # bound ever regresses we want the assertion below to fail, not the service call to blow up.
    bulk_download = MagicMock()
    bulk_download.generate_item_id.return_value = "test"
    monkeypatch.setattr(mock_invoker.services, "bulk_download", bulk_download)

    body: dict[str, Any] = {"image_names": [f"image-{index}.png" for index in range(MAX_IMAGE_BATCH_SIZE + 1)]}
    if path == "/api/v1/board_images/batch":
        body["board_id"] = "board-1"
    response = client.post(path, json=body)
    assert response.status_code == 422

    response = client.post(path, json={**body, "image_names": ["x" * 256]})
    assert response.status_code == 422

    # Rejection is FastAPI request validation, so it lands before the route body runs at all.
    # These two only bite on the /v1/images routes -- the board_images router reads its own
    # module-level ApiDependencies, which prepare_image_batch_test does not patch -- but they
    # are what catches a bound that gets "enforced" inside the handler instead of on the body.
    assert images_service.mock_calls == []
    assert bulk_download.mock_calls == []


def test_every_image_names_body_is_bounded(client: TestClient) -> None:
    """Drift guard: a new explicit-name batch route must not ship without a bound.

    /download shipped unbounded because the limits were applied route-by-route rather
    than to the shape. Rather than restate the limit on every route, assert the published
    contract: every request body that takes an `image_names` array declares both a list
    bound and a per-name length bound.
    """
    schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]

    unbounded: list[str] = []
    checked: list[str] = []
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            ref = (
                operation.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema", {})
                .get("$ref")
            )
            if ref is None:
                continue
            body = components[ref.rsplit("/", 1)[-1]]
            image_names = body.get("properties", {}).get("image_names")
            if image_names is None:
                continue
            # Optional fields are wrapped in anyOf: [{array}, {null}].
            variants = image_names.get("anyOf", [image_names])
            array = next((variant for variant in variants if variant.get("type") == "array"), None)
            if array is None:
                continue
            checked.append(path)
            if array.get("maxItems") is None or array.get("items", {}).get("maxLength") is None:
                unbounded.append(f"{method.upper()} {path}")

    # Pin the exact route set rather than a floor. A floor cannot see the failure this test
    # exists to catch: a route the walk *skips* (a body that nests image_names inside a model
    # rather than declaring it flat with Body(embed=True)) leaves the count unchanged and the
    # guard green. Adding a route here is deliberate — bound it, then add it to this list.
    assert sorted(checked) == [
        "/api/v1/board_images/batch",
        "/api/v1/board_images/batch/delete",
        "/api/v1/images/copy",
        "/api/v1/images/delete",
        "/api/v1/images/download",
        "/api/v1/images/images_by_names",
        "/api/v1/images/star",
        "/api/v1/images/unstar",
    ]
    assert unbounded == [], f"unbounded image_names batch bodies: {unbounded}"


@pytest.mark.parametrize(
    "params",
    [
        # A negative LIMIT means *unlimited* in SQLite — every image row, materialized.
        {"limit": -1},
        {"limit": MAX_PAGE_SIZE + 1},
        {"offset": -1},
    ],
)
def test_list_image_dtos_rejects_out_of_range_pagination(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient, params: dict[str, int]
) -> None:
    prepare_image_batch_test(monkeypatch, mock_invoker)
    assert client.get("/api/v1/images/", params=params).status_code == 422


def test_list_image_dtos_allows_count_only_query(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    """The frontend issues limit=0 to read `total` without fetching rows."""
    images_service = prepare_image_batch_test(monkeypatch, mock_invoker)
    images_service.get_many.return_value = OffsetPaginatedResults[ImageDTO](items=[], offset=0, limit=0, total=7)

    response = client.get("/api/v1/images/", params={"limit": 0})

    assert response.status_code == 200
    assert response.json()["total"] == 7
