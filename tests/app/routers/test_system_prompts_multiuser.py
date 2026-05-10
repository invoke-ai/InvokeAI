"""Multi-user permission tests for the /api/v1/system_prompts router.

Verifies:
- list scopes to own + public for non-admins
- non-owner PATCH/DELETE returns 403
- owner can flip is_public
- admin sees and mutates everything
"""

import logging
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.users.users_common import UserCreateRequest
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
    from invokeai.app.services.system_prompt_records.system_prompt_records_sqlite import (
        SqliteSystemPromptRecordsStorage,
    )
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
        workflow_records=None,  # type: ignore
        tensors=None,  # type: ignore
        conditioning=None,  # type: ignore
        style_preset_records=None,  # type: ignore
        style_preset_image_files=None,  # type: ignore
        system_prompt_records=SqliteSystemPromptRecordsStorage(db=db),
        workflow_thumbnails=None,  # type: ignore
        model_relationship_records=None,  # type: ignore
        model_relationships=None,  # type: ignore
        client_state_persistence=ClientStatePersistenceSqlite(db=db),
        users=UserService(db),
        external_generation=None,  # type: ignore
    )


@pytest.fixture
def mock_invoker(mock_services: InvocationServices) -> Invoker:
    return Invoker(services=mock_services)


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
    assert response.status_code == 200, response.text
    return response.json()["token"]


@pytest.fixture
def enable_multiuser(monkeypatch: Any, mock_invoker: Invoker):
    mock_invoker.services.configuration.multiuser = True
    mock_deps = MockApiDependencies(mock_invoker)
    # Patch every module that imports ApiDependencies for symbol resolution.
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.system_prompts.ApiDependencies", mock_deps)
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


def _create_prompt(client: TestClient, token: str, name: str = "p", content: str = "c") -> dict:
    r = client.post(
        "/api/v1/system_prompts/",
        json={"name": name, "content": content},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    return r.json()


def _list(client: TestClient, token: str) -> list:
    r = client.get("/api/v1/system_prompts/", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    return r.json()


# ---------------------------------------------------------------------------
# Auth gate
# ---------------------------------------------------------------------------


def test_list_requires_auth_in_multiuser(enable_multiuser: Any, client: TestClient):
    r = client.get("/api/v1/system_prompts/")
    assert r.status_code == status.HTTP_401_UNAUTHORIZED


# ---------------------------------------------------------------------------
# Visibility scoping
# ---------------------------------------------------------------------------


def test_user2_does_not_see_user1_private_prompt(client: TestClient, user1_token: str, user2_token: str):
    p = _create_prompt(client, user1_token, name="user1 private")
    assert p["is_public"] is False  # multiuser create defaults to private
    user2_ids = {row["id"] for row in _list(client, user2_token)}
    assert p["id"] not in user2_ids


def test_user2_sees_user1_public_prompt(client: TestClient, user1_token: str, user2_token: str):
    created = _create_prompt(client, user1_token, name="user1 created")
    # flip to public via PATCH
    r = client.patch(
        f"/api/v1/system_prompts/i/{created['id']}",
        json={"is_public": True},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == 200
    assert r.json()["is_public"] is True

    user2_ids = {row["id"] for row in _list(client, user2_token)}
    assert created["id"] in user2_ids


def test_seeded_defaults_visible_to_non_admin(client: TestClient, user1_token: str):
    rows = _list(client, user1_token)
    # The Default seed has fixed UUID …0000.
    default_id = "0f8f5b2e-1c9e-4f2a-9a4e-1f1f1f1f0000"
    assert any(r["id"] == default_id for r in rows)


# ---------------------------------------------------------------------------
# Mutation permissions
# ---------------------------------------------------------------------------


def test_user2_cannot_patch_user1_prompt(client: TestClient, user1_token: str, user2_token: str):
    p = _create_prompt(client, user1_token, name="locked")
    r = client.patch(
        f"/api/v1/system_prompts/i/{p['id']}",
        json={"name": "hijacked"},
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert r.status_code == status.HTTP_403_FORBIDDEN


def test_user2_cannot_delete_user1_prompt(client: TestClient, user1_token: str, user2_token: str):
    p = _create_prompt(client, user1_token, name="locked")
    r = client.delete(
        f"/api/v1/system_prompts/i/{p['id']}",
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert r.status_code == status.HTTP_403_FORBIDDEN

    # Confirm the row still exists by fetching it as the owner.
    r2 = client.get(
        f"/api/v1/system_prompts/i/{p['id']}",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r2.status_code == 200


def test_user2_cannot_get_user1_private_prompt(client: TestClient, user1_token: str, user2_token: str):
    p = _create_prompt(client, user1_token, name="locked")
    r = client.get(
        f"/api/v1/system_prompts/i/{p['id']}",
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert r.status_code == status.HTTP_403_FORBIDDEN


def test_admin_can_patch_any_users_prompt(client: TestClient, user1_token: str, admin_token: str):
    p = _create_prompt(client, user1_token, name="user1 owned")
    r = client.patch(
        f"/api/v1/system_prompts/i/{p['id']}",
        json={"name": "renamed by admin"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == 200
    assert r.json()["name"] == "renamed by admin"


def test_admin_can_delete_any_users_prompt(client: TestClient, user1_token: str, admin_token: str):
    p = _create_prompt(client, user1_token, name="user1 owned")
    r = client.delete(
        f"/api/v1/system_prompts/i/{p['id']}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == 200

    # Confirm gone (owner should get 404 now).
    r2 = client.get(
        f"/api/v1/system_prompts/i/{p['id']}",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r2.status_code == status.HTTP_404_NOT_FOUND
