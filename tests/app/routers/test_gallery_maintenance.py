import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers import gallery_maintenance, image_map
from invokeai.app.api_app import app
from invokeai.app.services.auth.token_service import TokenData
from invokeai.app.services.gallery_maintenance.gallery_maintenance_common import (
    GalleryMaintenanceError,
    GalleryMaintenancePreviewChanged,
)
from invokeai.app.services.image_moves.image_moves_default import ImageMoveQueueActive, ImageMoveService
from invokeai.app.services.invoker import Invoker


def test_gallery_maintenance_openapi_contract() -> None:
    paths = app.openapi()["paths"]

    expected_operations = {
        "/api/v1/app/gallery/maintenance/preview": "preview_gallery_maintenance",
        "/api/v1/app/gallery/maintenance/remove-missing": "remove_missing_images",
        "/api/v1/app/gallery/maintenance/archive-untracked": "archive_untracked_images",
        "/api/v1/app/gallery/maintenance/regenerate-thumbnails": "regenerate_missing_thumbnails",
    }

    for path, operation_id in expected_operations.items():
        operation = paths[path]["post"]
        assert operation["operationId"] == operation_id
        assert operation["security"] == [{"HTTPBearer": []}]
        assert {"401", "403", "409", "500", "503"}.issubset(operation["responses"])


@pytest.fixture(scope="module")
def client(invokeai_root_dir: Path) -> TestClient:
    os.environ["INVOKEAI_ROOT"] = invokeai_root_dir.as_posix()
    return TestClient(app)


class MockApiDependencies(ApiDependencies):
    invoker: Invoker

    def __init__(self, invoker: Invoker) -> None:
        self.invoker = invoker


def _patch_dependencies(monkeypatch: Any, mock_invoker: Invoker) -> Mock:
    maintenance_service = Mock()
    mock_invoker.services.gallery_maintenance = maintenance_service
    dependencies = MockApiDependencies(mock_invoker)
    monkeypatch.setattr(gallery_maintenance, "ApiDependencies", dependencies)
    monkeypatch.setattr(image_map, "ApiDependencies", dependencies)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", dependencies)
    monkeypatch.setattr(mock_invoker.services.configuration, "multiuser", False)
    return maintenance_service


def test_preview_and_execute_wait_for_service_results(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    service = _patch_dependencies(monkeypatch, mock_invoker)
    preview = {
        "operation": "remove_missing",
        "fingerprint": "a" * 64,
        "examined_count": 8,
        "affected_count": 2,
        "skipped_count": 0,
        "error_count": 0,
        "errors": [],
        "archive_path": "/outputs/images/archive/run-1",
    }
    result = {
        "operation": "remove_missing",
        "status": "partial",
        "examined_count": 8,
        "skipped_count": 0,
        "failed_count": 1,
        "records_removed": 2,
        "images_archived": 0,
        "thumbnails_archived": 2,
        "thumbnails_regenerated": 0,
        "archive_path": "/outputs/images/archive/run-1",
        "backup_path": "/outputs/images/archive/run-1/database.sqlite3",
        "errors": ["Some image records were removed, but thumbnail archival failed."],
    }
    service.preview.return_value = preview
    service.execute.return_value = result

    preview_response = client.post("/api/v1/app/gallery/maintenance/preview", json={"operation": "remove_missing"})
    assert preview_response.status_code == 200
    assert preview_response.json() == preview
    service.execute.assert_not_called()

    execute_response = client.post("/api/v1/app/gallery/maintenance/remove-missing", json={"fingerprint": "a" * 64})

    assert execute_response.status_code == 200
    assert execute_response.json() == result
    service.preview.assert_called_once_with("remove_missing")
    service.execute.assert_called_once_with("remove_missing", "a" * 64)


def test_multiuser_admin_can_preview(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    service = _patch_dependencies(monkeypatch, mock_invoker)
    monkeypatch.setattr(mock_invoker.services.configuration, "multiuser", True)
    monkeypatch.setattr(
        "invokeai.app.api.auth_dependencies.verify_token",
        lambda _: TokenData(user_id="admin-1", email="admin@example.com", is_admin=True),
    )
    monkeypatch.setattr(
        mock_invoker.services.users,
        "get",
        Mock(
            return_value=Mock(
                user_id="admin-1", email="admin@example.com", is_admin=True, is_active=True, token_epoch=0
            )
        ),
    )
    service.preview.return_value = {
        "operation": "remove_missing",
        "fingerprint": "d" * 64,
        "examined_count": 1,
        "affected_count": 0,
        "skipped_count": 0,
        "error_count": 0,
        "errors": [],
        "archive_path": None,
    }
    result = {
        "operation": "remove_missing",
        "status": "no_op",
        "examined_count": 1,
        "skipped_count": 0,
        "failed_count": 0,
        "records_removed": 0,
        "images_archived": 0,
        "thumbnails_archived": 0,
        "thumbnails_regenerated": 0,
        "archive_path": None,
        "backup_path": None,
        "errors": [],
    }
    service.execute.return_value = result

    response = client.post(
        "/api/v1/app/gallery/maintenance/preview",
        json={"operation": "remove_missing"},
        headers={"Authorization": "Bearer admin-token"},
    )
    execute_response = client.post(
        "/api/v1/app/gallery/maintenance/remove-missing",
        json={"fingerprint": "d" * 64},
        headers={"Authorization": "Bearer admin-token"},
    )

    assert response.status_code == 200
    assert response.json()["fingerprint"] == "d" * 64
    assert execute_response.status_code == 200
    assert execute_response.json() == result
    service.preview.assert_called_once_with("remove_missing")
    service.execute.assert_called_once_with("remove_missing", "d" * 64)


def test_blocking_execute_returns_only_after_completion_while_async_route_responds(
    monkeypatch: Any, mock_invoker: Invoker
) -> None:
    service = _patch_dependencies(monkeypatch, mock_invoker)
    mock_invoker.services.configuration.image_index_enabled = False
    started = Event()
    finish = Event()
    result = {
        "operation": "regenerate_thumbnails",
        "status": "completed",
        "examined_count": 1,
        "skipped_count": 0,
        "failed_count": 0,
        "records_removed": 0,
        "images_archived": 0,
        "thumbnails_archived": 0,
        "thumbnails_regenerated": 1,
        "archive_path": None,
        "backup_path": None,
        "errors": [],
    }

    def wait_for_completion(operation: str, fingerprint: str) -> dict[str, Any]:
        started.set()
        if not finish.wait(timeout=10):
            raise TimeoutError("test did not release gallery maintenance")
        return result

    service.execute.side_effect = wait_for_completion

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespanless_app(_app):
        yield

    monkeypatch.setattr(app.router, "lifespan_context", lifespanless_app)
    with TestClient(app) as concurrent_client:
        with ThreadPoolExecutor(max_workers=1) as executor:
            execute_future = executor.submit(
                concurrent_client.post,
                "/api/v1/app/gallery/maintenance/regenerate-thumbnails",
                json={"fingerprint": "e" * 64},
            )
            assert started.wait(timeout=5)

            async_response = concurrent_client.get("/api/v1/image_map/points")

            assert async_response.status_code == 200
            assert async_response.json()["state"] == "disabled"
            assert not execute_future.done()
            finish.set()
            execute_response = execute_future.result(timeout=5)

    assert execute_response.status_code == 200
    assert execute_response.json() == result


def test_disconnected_execute_keeps_reservation_until_completion_without_retry(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    """An abandoned HTTP response must not release the operation's image-storage reservation."""
    service = _patch_dependencies(monkeypatch, mock_invoker)
    image_moves = ImageMoveService(db=Mock(), image_files=Mock(), config=Mock(), logger=Mock())
    image_moves._invoker = mock_invoker
    monkeypatch.setattr(image_moves, "_get_active_job_id", lambda: None)
    monkeypatch.setattr(image_moves, "_assert_no_active_queue_work", lambda: None)
    mock_invoker.services.image_moves = image_moves
    started = Event()
    finish = Event()
    started_count = 0
    started_lock = Lock()
    result = {
        "operation": "archive_untracked",
        "status": "completed",
        "examined_count": 1,
        "skipped_count": 0,
        "failed_count": 0,
        "records_removed": 0,
        "images_archived": 1,
        "thumbnails_archived": 0,
        "thumbnails_regenerated": 0,
        "archive_path": "/outputs/images/archive/run-1",
        "backup_path": "/outputs/images/archive/run-1/database.sqlite3",
        "errors": [],
    }

    def execute_with_reservation(operation: str, fingerprint: str) -> dict[str, Any]:
        nonlocal started_count
        with image_moves.reserve_gallery_maintenance():
            with started_lock:
                started_count += 1
                is_first_execution = started_count == 1
            if is_first_execution:
                started.set()
                if not finish.wait(timeout=10):
                    raise TimeoutError("test did not release gallery maintenance")
            return result

    service.execute.side_effect = execute_with_reservation

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespanless_app(_app):
        yield

    monkeypatch.setattr(app.router, "lifespan_context", lifespanless_app)

    async def request(disconnected: bool) -> list[dict[str, Any]]:
        request_sent = False
        messages: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {
                    "type": "http.request",
                    "body": b'{"fingerprint":"' + b"f" * 64 + b'"}',
                    "more_body": False,
                }
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            if disconnected:
                raise OSError("client disconnected before receiving the response")
            messages.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v1/app/gallery/maintenance/archive-untracked",
            "raw_path": b"/api/v1/app/gallery/maintenance/archive-untracked",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", b"testserver"), (b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }
        await app(scope, receive, send)
        return messages

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            abandoned_future = executor.submit(asyncio.run, request(True))
            assert started.wait(timeout=5)

            excluded_response = client.post(
                "/api/v1/app/gallery/maintenance/archive-untracked", json={"fingerprint": "f" * 64}
            )

            assert excluded_response.status_code == 409
            assert started_count == 1
            assert not abandoned_future.done()

            finish.set()
            with pytest.raises(OSError, match="client disconnected"):
                abandoned_future.result(timeout=5)

        assert started_count == 1
    finally:
        finish.set()
        image_moves.stop()


def test_non_admin_cannot_preview_or_execute(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    service = _patch_dependencies(monkeypatch, mock_invoker)
    monkeypatch.setattr(mock_invoker.services.configuration, "multiuser", True)
    monkeypatch.setattr(
        "invokeai.app.api.auth_dependencies.verify_token",
        lambda _: TokenData(user_id="user-1", email="user@example.com", is_admin=False),
    )
    monkeypatch.setattr(
        mock_invoker.services.users,
        "get",
        Mock(
            return_value=Mock(user_id="user-1", email="user@example.com", is_admin=False, is_active=True, token_epoch=0)
        ),
    )

    headers = {"Authorization": "Bearer regular-user-token"}
    preview_response = client.post(
        "/api/v1/app/gallery/maintenance/preview",
        json={"operation": "remove_missing"},
        headers=headers,
    )
    execute_response = client.post(
        "/api/v1/app/gallery/maintenance/remove-missing", json={"fingerprint": "b" * 64}, headers=headers
    )

    assert preview_response.status_code == 403
    assert execute_response.status_code == 403
    service.preview.assert_not_called()
    service.execute.assert_not_called()


def test_multiuser_unauthenticated_preview_is_rejected(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    service = _patch_dependencies(monkeypatch, mock_invoker)
    monkeypatch.setattr(mock_invoker.services.configuration, "multiuser", True)

    response = client.post("/api/v1/app/gallery/maintenance/preview", json={"operation": "remove_missing"})

    assert response.status_code == 401
    service.preview.assert_not_called()


def test_unavailable_service_returns_503(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    _patch_dependencies(monkeypatch, mock_invoker)
    mock_invoker.services.gallery_maintenance = None

    response = client.post("/api/v1/app/gallery/maintenance/preview", json={"operation": "remove_missing"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Gallery maintenance unavailable"}


def test_invalid_request_is_rejected_before_service_call(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    service = _patch_dependencies(monkeypatch, mock_invoker)

    preview_response = client.post("/api/v1/app/gallery/maintenance/preview", json={"operation": "rebuild_everything"})
    execute_response = client.post(
        "/api/v1/app/gallery/maintenance/regenerate-thumbnails", json={"fingerprint": "not-a-fingerprint"}
    )

    assert preview_response.status_code == 422
    assert execute_response.status_code == 422
    service.preview.assert_not_called()
    service.execute.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        ImageMoveQueueActive("internal queue detail"),
        GalleryMaintenancePreviewChanged("internal fingerprint detail"),
    ],
)
def test_execute_conflicts_are_sanitized(
    failure: Exception, monkeypatch: Any, mock_invoker: Invoker, client: TestClient
) -> None:
    service = _patch_dependencies(monkeypatch, mock_invoker)
    service.execute.side_effect = failure

    response = client.post("/api/v1/app/gallery/maintenance/remove-missing", json={"fingerprint": "c" * 64})

    assert response.status_code == 409
    assert "internal" not in response.text
    service.execute.assert_called_once_with("remove_missing", "c" * 64)


def test_preview_busy_conflict_is_sanitized(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    service = _patch_dependencies(monkeypatch, mock_invoker)
    service.preview.side_effect = ImageMoveQueueActive("private queue state")

    response = client.post("/api/v1/app/gallery/maintenance/preview", json={"operation": "remove_missing"})

    assert response.status_code == 409
    assert "private queue state" not in response.text
    service.preview.assert_called_once_with("remove_missing")


def test_service_failure_returns_safe_error(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    service = _patch_dependencies(monkeypatch, mock_invoker)
    service.preview.side_effect = GalleryMaintenanceError("private filesystem path")

    response = client.post("/api/v1/app/gallery/maintenance/preview", json={"operation": "remove_missing"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Gallery maintenance failed"}
    assert "private filesystem path" not in response.text
    service.preview.assert_called_once_with("remove_missing")
