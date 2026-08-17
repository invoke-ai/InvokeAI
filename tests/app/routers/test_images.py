import os
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


def prepare_image_batch_test(monkeypatch: Any, mock_invoker: Invoker) -> MagicMock:
    """Wires the image router to a MagicMock image service with maintenance inactive.

    Returns the mock service so tests can script per-name update outcomes.
    """
    images_service = MagicMock()
    monkeypatch.setattr(mock_invoker.services, "images", images_service)
    mock_invoker.services.image_moves = MagicMock()
    mock_invoker.services.image_moves.is_maintenance_active.return_value = False
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
    ],
)
def test_image_name_batches_are_bounded(monkeypatch: Any, mock_invoker: Invoker, client: TestClient, path: str) -> None:
    """An unbounded name list is a free amplification: each name costs a DB lookup."""
    images_service = prepare_image_batch_test(monkeypatch, mock_invoker)
    # /download would otherwise authorize every name and schedule a background task; if the
    # bound ever regresses we want the assertion below to fail, not the service call to blow up.
    bulk_download = MagicMock()
    bulk_download.generate_item_id.return_value = "test"
    monkeypatch.setattr(mock_invoker.services, "bulk_download", bulk_download)

    response = client.post(
        path, json={"image_names": [f"image-{index}.png" for index in range(MAX_IMAGE_BATCH_SIZE + 1)]}
    )
    assert response.status_code == 422

    response = client.post(path, json={"image_names": ["x" * 256]})
    assert response.status_code == 422

    # Rejection is FastAPI request validation, so it happens before the route body runs:
    # no per-name authorization lookups and no background task were scheduled.
    assert images_service.get_dto.call_count == 0
    assert bulk_download.generate_item_id.call_count == 0


def test_every_image_names_body_is_bounded(client: TestClient) -> None:
    """Drift guard: a new explicit-name batch route must not ship without a bound.

    /download shipped unbounded because the limits were applied route-by-route rather
    than to the shape. Rather than restate the limit on every route, assert the published
    contract: every images-router request body that takes an `image_names` array declares
    both a list bound and a per-name length bound.

    Scoped to /v1/images deliberately. The two /v1/board_images batch routes are unbounded
    too, but bounding them would reject a >MAX_IMAGE_BATCH_SIZE change-board request that
    the UI can produce today, so they are left for a follow-up that pairs the bound with
    client-side chunking.
    """
    schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]

    unbounded: list[str] = []
    checked = 0
    for path, operations in schema["paths"].items():
        if not path.startswith("/api/v1/images/"):
            continue
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
            checked += 1
            if array.get("maxItems") is None or array.get("items", {}).get("maxLength") is None:
                unbounded.append(f"{method.upper()} {path}")

    # Floor guards against the walk silently matching nothing if the schema shape changes.
    assert checked >= 5
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
