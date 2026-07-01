import logging
from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.auth.token_service import set_jwt_secret
from invokeai.app.services.board_image_records.board_image_records_sqlite import SqliteBoardImageRecordStorage
from invokeai.app.services.board_records.board_records_sqlite import SqliteBoardRecordStorage
from invokeai.app.services.board_video_records.board_video_records_sqlite import SqliteBoardVideoRecordStorage
from invokeai.app.services.boards.boards_default import BoardService
from invokeai.app.services.bulk_download.bulk_download_default import BulkDownloadService
from invokeai.app.services.client_state_persistence.client_state_persistence_sqlite import ClientStatePersistenceSqlite
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_moves.image_moves_default import ImageMoveJobAlreadyRunning, ImageMoveQueueActive
from invokeai.app.services.image_records.image_records_sqlite import SqliteImageRecordStorage
from invokeai.app.services.images.images_default import ImageService
from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.invocation_stats.invocation_stats_default import InvocationStatsService
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.users.users_default import UserService
from invokeai.app.services.video_records.video_records_sqlite import SqliteVideoRecordStorage
from invokeai.app.services.workflow_records.workflow_records_sqlite import SqliteWorkflowRecordsStorage
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database
from tests.test_nodes import TestEventService


class MockApiDependencies(ApiDependencies):
    invoker: Invoker

    def __init__(self, invoker: Invoker) -> None:
        self.invoker = invoker


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def mock_services() -> InvocationServices:
    configuration = InvokeAIAppConfig(use_memory_db=True, node_cache_size=0)
    logger = InvokeAILogger.get_logger()
    db = create_mock_sqlite_database(configuration, logger)
    image_moves = MagicMock()
    return InvocationServices(
        board_image_records=SqliteBoardImageRecordStorage(db=db),
        board_images=None,  # type: ignore
        board_records=SqliteBoardRecordStorage(db=db),
        boards=BoardService(),
        bulk_download=BulkDownloadService(),
        configuration=configuration,
        events=TestEventService(),
        image_files=None,  # type: ignore
        image_records=SqliteImageRecordStorage(db=db),
        images=ImageService(),
        invocation_cache=MemoryInvocationCache(max_cache_size=0),
        logger=logging,  # type: ignore
        model_images=None,  # type: ignore
        model_manager=None,  # type: ignore
        download_queue=None,  # type: ignore
        external_generation=None,  # type: ignore
        names=None,  # type: ignore
        performance_statistics=InvocationStatsService(),
        session_processor=None,  # type: ignore
        session_queue=None,  # type: ignore
        urls=None,  # type: ignore
        workflow_records=SqliteWorkflowRecordsStorage(db=db),
        tensors=None,  # type: ignore
        conditioning=None,  # type: ignore
        style_preset_records=None,  # type: ignore
        style_preset_image_files=None,  # type: ignore
        workflow_thumbnails=None,  # type: ignore
        model_relationship_records=None,  # type: ignore
        model_relationships=None,  # type: ignore
        client_state_persistence=ClientStatePersistenceSqlite(db=db),
        users=UserService(db),
        videos=None,  # type: ignore
        video_files=None,  # type: ignore
        video_records=SqliteVideoRecordStorage(db=db),
        board_video_records=SqliteBoardVideoRecordStorage(db=db),
        gallery=None,  # type: ignore
        image_moves=image_moves,
    )


@pytest.fixture
def mock_invoker(mock_services: InvocationServices, monkeypatch: pytest.MonkeyPatch) -> Invoker:
    invoker = Invoker(services=mock_services)
    mock_deps = MockApiDependencies(invoker)
    monkeypatch.setattr("invokeai.app.api.routers.image_moves.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", mock_deps)
    return invoker


def _status_payload(is_running: bool = True, operation: str = "move_all") -> dict:
    return {
        "is_running": is_running,
        "operation": operation,
        "active_job_id": None,
        "latest_job": None,
        "last_error": None,
        "needs_move_count": 0,
    }


def _create_user(invoker: Invoker, email: str, is_admin: bool) -> None:
    invoker.services.users.create(
        UserCreateRequest(
            email=email,
            display_name=email,
            password="TestPass123",
            is_admin=is_admin,
        )
    )


def _login(client: TestClient, email: str) -> str:
    response = client.post("/api/v1/auth/login", json={"email": email, "password": "TestPass123"})
    assert response.status_code == status.HTTP_200_OK
    return response.json()["token"]


def test_start_image_move_returns_accepted_without_running_job_inline(
    client: TestClient, mock_invoker: Invoker
) -> None:
    image_moves = mock_invoker.services.image_moves
    image_moves.start_background_move_all.return_value = _status_payload()

    response = client.post("/api/v1/image_moves/start")

    assert response.status_code == status.HTTP_202_ACCEPTED
    assert response.json()["is_running"] is True
    image_moves.start_background_move_all.assert_called_once_with()
    image_moves.move_all_images.assert_not_called()


def test_start_image_move_rejects_overlapping_background_job(client: TestClient, mock_invoker: Invoker) -> None:
    image_moves = mock_invoker.services.image_moves
    image_moves.start_background_move_all.side_effect = ImageMoveJobAlreadyRunning("already running")

    response = client.post("/api/v1/image_moves/start")

    assert response.status_code == status.HTTP_409_CONFLICT
    assert response.json()["detail"] == "already running"


def test_start_image_move_rejects_active_queue_work(client: TestClient, mock_invoker: Invoker) -> None:
    image_moves = mock_invoker.services.image_moves
    image_moves.start_background_move_all.side_effect = ImageMoveQueueActive("queue work is active")

    response = client.post("/api/v1/image_moves/start")

    assert response.status_code == status.HTTP_409_CONFLICT
    assert response.json()["detail"] == "queue work is active"


def test_force_recovery_returns_accepted(client: TestClient, mock_invoker: Invoker) -> None:
    image_moves = mock_invoker.services.image_moves
    image_moves.start_background_recovery.return_value = _status_payload(operation="recovery")

    response = client.post("/api/v1/image_moves/recover")

    assert response.status_code == status.HTTP_202_ACCEPTED
    assert response.json()["operation"] == "recovery"
    image_moves.start_background_recovery.assert_called_once_with()


def test_image_move_status_uses_service_status(client: TestClient, mock_invoker: Invoker) -> None:
    image_moves = mock_invoker.services.image_moves
    image_moves.get_background_status.return_value = _status_payload(is_running=False)

    response = client.get("/api/v1/image_moves/status")

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["is_running"] is False
    assert response.json()["needs_move_count"] == 0
    image_moves.get_background_status.assert_called_once_with()


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/v1/image_moves/start"),
        ("post", "/api/v1/image_moves/recover"),
        ("get", "/api/v1/image_moves/status"),
    ],
)
def test_image_move_endpoints_require_admin_in_multiuser_mode(
    client: TestClient, mock_invoker: Invoker, method: str, path: str
) -> None:
    set_jwt_secret("test-secret")
    mock_invoker.services.configuration.multiuser = True
    _create_user(mock_invoker, "user@test.com", is_admin=False)
    token = _login(client, "user@test.com")

    response = getattr(client, method)(path, headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == status.HTTP_403_FORBIDDEN
