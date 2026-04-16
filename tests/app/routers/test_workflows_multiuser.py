"""Tests for multiuser workflow library functionality."""

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.workflow_records.workflow_records_sqlite import SqliteWorkflowRecordsStorage
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database


class MockApiDependencies(ApiDependencies):
    invoker: Invoker

    def __init__(self, invoker: Invoker) -> None:
        self.invoker = invoker


WORKFLOW_BODY = {
    "name": "Test Workflow",
    "author": "",
    "description": "A test workflow",
    "version": "1.0.0",
    "contact": "",
    "tags": "",
    "notes": "",
    "nodes": [],
    "edges": [],
    "exposedFields": [],
    "meta": {"version": "3.0.0", "category": "user"},
    "id": None,
    "form_fields": [],
}


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
    )


def create_test_user(mock_invoker: Invoker, email: str, display_name: str, is_admin: bool = False) -> str:
    user_service = mock_invoker.services.users
    user_data = UserCreateRequest(email=email, display_name=display_name, password="TestPass123", is_admin=is_admin)
    user = user_service.create(user_data)
    return user.user_id


def get_user_token(client: TestClient, email: str) -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "TestPass123", "remember_me": False},
    )
    assert response.status_code == 200
    return response.json()["token"]


@pytest.fixture
def enable_multiuser(monkeypatch: Any, mock_invoker: Invoker):
    mock_invoker.services.configuration.multiuser = True
    mock_workflow_thumbnails = MagicMock()
    mock_workflow_thumbnails.get_url.return_value = None
    mock_invoker.services.workflow_thumbnails = mock_workflow_thumbnails

    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.workflows.ApiDependencies", mock_deps)
    yield


@pytest.fixture
def admin_token(setup_jwt_secret: None, enable_multiuser: Any, mock_invoker: Invoker, client: TestClient):
    create_test_user(mock_invoker, "admin@test.com", "Admin", is_admin=True)
    return get_user_token(client, "admin@test.com")


@pytest.fixture
def user1_token(enable_multiuser: Any, mock_invoker: Invoker, client: TestClient, admin_token: str):
    create_test_user(mock_invoker, "user1@test.com", "User One", is_admin=False)
    return get_user_token(client, "user1@test.com")


@pytest.fixture
def user2_token(enable_multiuser: Any, mock_invoker: Invoker, client: TestClient, admin_token: str):
    create_test_user(mock_invoker, "user2@test.com", "User Two", is_admin=False)
    return get_user_token(client, "user2@test.com")


def create_workflow(client: TestClient, token: str) -> str:
    response = client.post(
        "/api/v1/workflows/",
        json={"workflow": WORKFLOW_BODY},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text
    return response.json()["workflow_id"]


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


def test_list_workflows_requires_auth(enable_multiuser: Any, client: TestClient):
    response = client.get("/api/v1/workflows/")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_create_workflow_requires_auth(enable_multiuser: Any, client: TestClient):
    response = client.post("/api/v1/workflows/", json={"workflow": WORKFLOW_BODY})
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


# ---------------------------------------------------------------------------
# Ownership isolation
# ---------------------------------------------------------------------------


def test_workflows_are_isolated_between_users(client: TestClient, user1_token: str, user2_token: str):
    """Users should only see their own workflows in list."""
    # user1 creates a workflow
    create_workflow(client, user1_token)

    # user1 can see it
    r1 = client.get("/api/v1/workflows/?categories=user", headers={"Authorization": f"Bearer {user1_token}"})
    assert r1.status_code == 200
    assert r1.json()["total"] == 1

    # user2 cannot see user1's workflow
    r2 = client.get("/api/v1/workflows/?categories=user", headers={"Authorization": f"Bearer {user2_token}"})
    assert r2.status_code == 200
    assert r2.json()["total"] == 0


def test_user_cannot_delete_another_users_workflow(client: TestClient, user1_token: str, user2_token: str):
    workflow_id = create_workflow(client, user1_token)
    response = client.delete(
        f"/api/v1/workflows/i/{workflow_id}",
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_user_cannot_update_another_users_workflow(client: TestClient, user1_token: str, user2_token: str):
    workflow_id = create_workflow(client, user1_token)
    updated = {**WORKFLOW_BODY, "id": workflow_id, "name": "Hijacked"}
    response = client.patch(
        f"/api/v1/workflows/i/{workflow_id}",
        json={"workflow": updated},
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_owner_can_delete_own_workflow(client: TestClient, user1_token: str):
    workflow_id = create_workflow(client, user1_token)
    response = client.delete(
        f"/api/v1/workflows/i/{workflow_id}",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert response.status_code == 200


def test_admin_can_delete_any_workflow(client: TestClient, admin_token: str, user1_token: str):
    workflow_id = create_workflow(client, user1_token)
    response = client.delete(
        f"/api/v1/workflows/i/{workflow_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Shared workflow (is_public)
# ---------------------------------------------------------------------------


def test_update_is_public_owner_succeeds(client: TestClient, user1_token: str):
    workflow_id = create_workflow(client, user1_token)
    response = client.patch(
        f"/api/v1/workflows/i/{workflow_id}/is_public",
        json={"is_public": True},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert response.status_code == 200
    assert response.json()["is_public"] is True


def test_update_is_public_other_user_forbidden(client: TestClient, user1_token: str, user2_token: str):
    workflow_id = create_workflow(client, user1_token)
    response = client.patch(
        f"/api/v1/workflows/i/{workflow_id}/is_public",
        json={"is_public": True},
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_public_workflow_visible_to_other_users(client: TestClient, user1_token: str, user2_token: str):
    """A shared (is_public=True) workflow should appear when filtering with is_public=true."""
    workflow_id = create_workflow(client, user1_token)
    # Make it public
    client.patch(
        f"/api/v1/workflows/i/{workflow_id}/is_public",
        json={"is_public": True},
        headers={"Authorization": f"Bearer {user1_token}"},
    )

    # user2 can see it through is_public=true filter
    response = client.get(
        "/api/v1/workflows/?categories=user&is_public=true",
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert response.status_code == 200
    ids = [w["workflow_id"] for w in response.json()["items"]]
    assert workflow_id in ids


def test_private_workflow_not_visible_to_other_users(client: TestClient, user1_token: str, user2_token: str):
    """A private (is_public=False) user workflow should NOT appear for another user."""
    workflow_id = create_workflow(client, user1_token)

    # user2 lists 'yours' style (their own workflows)
    response = client.get(
        "/api/v1/workflows/?categories=user",
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert response.status_code == 200
    ids = [w["workflow_id"] for w in response.json()["items"]]
    assert workflow_id not in ids


def test_public_workflow_still_in_owners_list(client: TestClient, user1_token: str):
    """A shared workflow should still appear in the owner's own workflow list."""
    workflow_id = create_workflow(client, user1_token)
    client.patch(
        f"/api/v1/workflows/i/{workflow_id}/is_public",
        json={"is_public": True},
        headers={"Authorization": f"Bearer {user1_token}"},
    )

    # owner's 'yours' list (no is_public filter)
    response = client.get(
        "/api/v1/workflows/?categories=user",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert response.status_code == 200
    ids = [w["workflow_id"] for w in response.json()["items"]]
    assert workflow_id in ids


def test_workflow_has_user_id_and_is_public_fields(client: TestClient, user1_token: str):
    """Created workflow should return user_id and is_public fields."""
    response = client.post(
        "/api/v1/workflows/",
        json={"workflow": WORKFLOW_BODY},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "user_id" in data
    assert "is_public" in data
    assert data["is_public"] is False


# ---------------------------------------------------------------------------
# System-owned workflow visibility (regression tests for migration 30 fix)
# ---------------------------------------------------------------------------


def _insert_system_workflow(mock_invoker: Invoker, name: str = "Legacy Workflow", is_public: bool = True) -> str:
    """Insert a workflow owned by 'system' directly via the service layer, then set is_public."""
    from invokeai.app.services.workflow_records.workflow_records_common import WorkflowWithoutID

    wf = WorkflowWithoutID(**{**WORKFLOW_BODY, "name": name})
    record = mock_invoker.services.workflow_records.create(workflow=wf, user_id="system")
    if is_public:
        mock_invoker.services.workflow_records.update_is_public(workflow_id=record.workflow_id, is_public=True)
    return record.workflow_id


def test_system_public_workflow_visible_in_shared_listing(client: TestClient, user1_token: str, mock_invoker: Invoker):
    """After migration 30, system-owned public workflows should appear in the shared workflows listing."""
    wf_id = _insert_system_workflow(mock_invoker, "Legacy Workflow")

    response = client.get(
        "/api/v1/workflows/?categories=user&is_public=true",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert response.status_code == 200
    ids = [w["workflow_id"] for w in response.json()["items"]]
    assert wf_id in ids


def test_system_public_workflow_not_in_your_workflows(client: TestClient, user1_token: str, mock_invoker: Invoker):
    """System-owned public workflows should NOT appear in 'Your Workflows' listing."""
    wf_id = _insert_system_workflow(mock_invoker, "Legacy Workflow")

    response = client.get(
        "/api/v1/workflows/?categories=user",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert response.status_code == 200
    ids = [w["workflow_id"] for w in response.json()["items"]]
    assert wf_id not in ids


def test_admin_can_list_system_workflows(client: TestClient, admin_token: str, mock_invoker: Invoker):
    """Admins should see system-owned workflows in their listing."""
    wf_id = _insert_system_workflow(mock_invoker, "Admin Visible Workflow")

    response = client.get(
        "/api/v1/workflows/?categories=user",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200
    ids = [w["workflow_id"] for w in response.json()["items"]]
    assert wf_id in ids


def test_admin_can_update_system_workflow(client: TestClient, admin_token: str, mock_invoker: Invoker):
    """Admins should be able to update a system-owned workflow."""
    wf_id = _insert_system_workflow(mock_invoker, "Editable Legacy")

    # Get the full workflow to update it
    get_resp = client.get(
        f"/api/v1/workflows/i/{wf_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert get_resp.status_code == 200
    workflow_data = get_resp.json()["workflow"]
    workflow_data["name"] = "Updated by Admin"

    update_resp = client.patch(
        f"/api/v1/workflows/i/{wf_id}",
        json={"workflow": workflow_data},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["workflow"]["name"] == "Updated by Admin"


def test_admin_can_delete_system_workflow(client: TestClient, admin_token: str, mock_invoker: Invoker):
    """Admins should be able to delete a system-owned workflow."""
    wf_id = _insert_system_workflow(mock_invoker, "Deletable Legacy")

    response = client.delete(
        f"/api/v1/workflows/i/{wf_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200


def test_regular_user_cannot_update_system_workflow(client: TestClient, user1_token: str, mock_invoker: Invoker):
    """Regular users should NOT be able to update a system-owned workflow."""
    wf_id = _insert_system_workflow(mock_invoker, "Protected Legacy")

    get_resp = client.get(
        f"/api/v1/workflows/i/{wf_id}",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert get_resp.status_code == 200
    workflow_data = get_resp.json()["workflow"]
    workflow_data["name"] = "Hijacked"

    update_resp = client.patch(
        f"/api/v1/workflows/i/{wf_id}",
        json={"workflow": workflow_data},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert update_resp.status_code == status.HTTP_403_FORBIDDEN


def test_regular_user_cannot_delete_system_workflow(client: TestClient, user1_token: str, mock_invoker: Invoker):
    """Regular users should NOT be able to delete a system-owned workflow."""
    wf_id = _insert_system_workflow(mock_invoker, "Undeletable Legacy")

    response = client.delete(
        f"/api/v1/workflows/i/{wf_id}",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN


# ---------------------------------------------------------------------------
# Single-user mode: default ownership + sharing on create
# ---------------------------------------------------------------------------


@pytest.fixture
def single_user_mode(monkeypatch: Any, mock_invoker: Invoker):
    """Configure the app for single-user (legacy) mode."""
    mock_invoker.services.configuration.multiuser = False
    mock_workflow_thumbnails = MagicMock()
    mock_workflow_thumbnails.get_url.return_value = None
    mock_invoker.services.workflow_thumbnails = mock_workflow_thumbnails

    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.workflows.ApiDependencies", mock_deps)
    yield


def test_single_user_create_workflow_owned_by_system_and_public(single_user_mode: Any, client: TestClient):
    """In single-user mode, newly created workflows should be owned by 'system' and shared (is_public=True)."""
    response = client.post("/api/v1/workflows/", json={"workflow": WORKFLOW_BODY})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["user_id"] == "system"
    assert payload["is_public"] is True


def test_multiuser_create_workflow_owned_by_user_and_private(client: TestClient, user1_token: str):
    """In multiuser mode, newly created workflows should be owned by the creator and private (is_public=False)."""
    response = client.post(
        "/api/v1/workflows/",
        json={"workflow": WORKFLOW_BODY},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["user_id"] != "system"
    assert payload["is_public"] is False
