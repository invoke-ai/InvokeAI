"""Tests for session queue API behavior with workflow-call queue items."""

import logging
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_processor.session_processor_common import SessionProcessorStatus
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.workflow_records.workflow_records_sqlite import SqliteWorkflowRecordsStorage
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database


class MockApiDependencies(ApiDependencies):
    invoker: Invoker

    def __init__(self, invoker: Invoker) -> None:
        self.invoker = invoker


@pytest.fixture
def setup_jwt_secret():
    from invokeai.app.services.auth.token_service import set_jwt_secret

    set_jwt_secret("test-secret-key-for-unit-tests-only-do-not-use-in-production")


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_services() -> InvocationServices:
    from invokeai.app.services.board_image_records.board_image_records_sqlite import SqliteBoardImageRecordStorage
    from invokeai.app.services.board_records.board_records_sqlite import SqliteBoardRecordStorage
    from invokeai.app.services.boards.boards_default import BoardService
    from invokeai.app.services.bulk_download.bulk_download_default import BulkDownloadService
    from invokeai.app.services.client_state_persistence.client_state_persistence_sqlite import (
        ClientStatePersistenceSqlite,
    )
    from invokeai.app.services.image_records.image_records_sqlite import SqliteImageRecordStorage
    from invokeai.app.services.images.images_default import ImageService
    from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
    from invokeai.app.services.invocation_stats.invocation_stats_default import InvocationStatsService
    from invokeai.app.services.users.users_default import UserService
    from tests.test_nodes import TestEventService

    configuration = InvokeAIAppConfig(use_memory_db=True, node_cache_size=0)
    logger = InvokeAILogger.get_logger()
    db = create_mock_sqlite_database(configuration, logger)

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
        external_generation=None,  # type: ignore
    )


@pytest.fixture
def mock_invoker(mock_services: InvocationServices) -> Invoker:
    invoker = Invoker(services=mock_services)
    queue = SqliteSessionQueue(db=mock_services.board_records._db)
    mock_services.session_queue = queue
    mock_services.session_processor = MagicMock()
    mock_services.session_processor.get_status.return_value = SessionProcessorStatus(
        is_started=True, is_processing=False
    )
    queue.start(invoker)
    return invoker


def _create_user(mock_invoker: Invoker, email: str, display_name: str, is_admin: bool = False) -> str:
    user = mock_invoker.services.users.create(
        UserCreateRequest(email=email, display_name=display_name, password="TestPass123", is_admin=is_admin)
    )
    return user.user_id


def _login(client: TestClient, email: str) -> str:
    response = client.post("/api/v1/auth/login", json={"email": email, "password": "TestPass123", "remember_me": False})
    assert response.status_code == 200
    return response.json()["token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def enable_multiuser(monkeypatch: Any, mock_invoker: Invoker):
    mock_invoker.services.configuration.multiuser = True
    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.session_queue.ApiDependencies", mock_deps)
    yield


@pytest.fixture
def admin_token(setup_jwt_secret: None, enable_multiuser: Any, mock_invoker: Invoker, client: TestClient):
    _create_user(mock_invoker, "admin@test.com", "Admin", is_admin=True)
    return _login(client, "admin@test.com")


@pytest.fixture
def user1_token(enable_multiuser: Any, mock_invoker: Invoker, client: TestClient, admin_token: str):
    _create_user(mock_invoker, "user1@test.com", "User One")
    return _login(client, "user1@test.com")


@pytest.fixture
def user2_token(enable_multiuser: Any, mock_invoker: Invoker, client: TestClient, admin_token: str):
    _create_user(mock_invoker, "user2@test.com", "User Two")
    return _login(client, "user2@test.com")


def _insert_queue_item(
    session_queue: SqliteSessionQueue,
    *,
    user_id: str,
    status: str,
    session: GraphExecutionState | None = None,
    workflow_call_id: str | None = None,
    parent_item_id: int | None = None,
    parent_session_id: str | None = None,
    root_item_id: int | None = None,
    workflow_call_depth: int | None = None,
) -> int:
    session = session or GraphExecutionState(graph=Graph())
    with session_queue._db.transaction() as cursor:
        cursor.execute(
            """--sql
            INSERT INTO session_queue (
                queue_id, session, session_id, batch_id, field_values, priority, workflow, origin, destination,
                retried_from_item_id, user_id, status, workflow_call_id, parent_item_id, parent_session_id,
                root_item_id, workflow_call_depth
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default",
                session.model_dump_json(warnings=False),
                session.id,
                str(uuid.uuid4()),
                None,
                0,
                None,
                None,
                None,
                None,
                user_id,
                status,
                workflow_call_id,
                parent_item_id,
                parent_session_id,
                root_item_id,
                workflow_call_depth,
            ),
        )
        return cursor.lastrowid


def test_get_queue_status_reports_waiting_for_owner(
    client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
) -> None:
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    user2 = mock_invoker.services.users.get_by_email("user2@test.com")
    assert user1 is not None and user2 is not None

    _insert_queue_item(mock_invoker.services.session_queue, user_id=user1.user_id, status="waiting")
    _insert_queue_item(mock_invoker.services.session_queue, user_id=user2.user_id, status="pending")

    response = client.get("/api/v1/queue/default/status", headers=_auth(user1_token))

    assert response.status_code == 200
    payload = response.json()
    assert payload["queue"]["waiting"] == 1
    assert payload["queue"]["pending"] == 0
    assert payload["queue"]["in_progress"] == 0
    assert payload["queue"]["total"] == 1
    assert payload["queue"]["item_id"] is None


def test_get_queue_item_sanitizes_workflow_call_metadata_for_non_owner(
    client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
) -> None:
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None

    item_id = _insert_queue_item(
        mock_invoker.services.session_queue,
        user_id=user1.user_id,
        status="waiting",
        workflow_call_id="workflow-call-1",
        parent_item_id=41,
        parent_session_id="parent-session-1",
        root_item_id=17,
        workflow_call_depth=2,
    )

    response = client.get(f"/api/v1/queue/default/i/{item_id}", headers=_auth(user2_token))

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "waiting"
    assert payload["item_id"] == item_id
    assert payload["user_id"] == "redacted"
    assert payload["batch_id"] == "redacted"
    assert payload["session_id"] == "redacted"
    assert payload.get("workflow_call_id") is None
    assert payload.get("parent_item_id") is None
    assert payload.get("parent_session_id") is None
    assert payload.get("root_item_id") is None
    assert payload.get("workflow_call_depth") is None


def test_retry_items_by_id_normalizes_child_to_root_at_router(
    client: TestClient, mock_invoker: Invoker, user1_token: str
) -> None:
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None

    root_item_id = _insert_queue_item(mock_invoker.services.session_queue, user_id=user1.user_id, status="failed")
    child_item_id = _insert_queue_item(
        mock_invoker.services.session_queue,
        user_id=user1.user_id,
        status="failed",
        workflow_call_id="workflow-call-1",
        parent_item_id=root_item_id,
        parent_session_id="parent-session-1",
        root_item_id=root_item_id,
        workflow_call_depth=1,
    )

    response = client.put("/api/v1/queue/default/retry_items_by_id", headers=_auth(user1_token), json=[child_item_id])

    assert response.status_code == 200
    assert response.json()["retried_item_ids"] == [root_item_id]


def test_cancel_queue_item_cascades_to_waiting_parent_via_router(
    client: TestClient, mock_invoker: Invoker, user1_token: str
) -> None:
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None

    parent_item_id = _insert_queue_item(mock_invoker.services.session_queue, user_id=user1.user_id, status="waiting")
    child_item_id = _insert_queue_item(
        mock_invoker.services.session_queue,
        user_id=user1.user_id,
        status="pending",
        workflow_call_id="workflow-call-1",
        parent_item_id=parent_item_id,
        parent_session_id="parent-session-1",
        root_item_id=parent_item_id,
        workflow_call_depth=1,
    )

    response = client.put(f"/api/v1/queue/default/i/{child_item_id}/cancel", headers=_auth(user1_token))

    assert response.status_code == 200
    assert response.json()["status"] == "canceled"
    assert mock_invoker.services.session_queue.get_queue_item(parent_item_id).status == "canceled"
    assert mock_invoker.services.session_queue.get_queue_item(child_item_id).status == "canceled"
