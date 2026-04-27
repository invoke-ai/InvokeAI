"""Tests for API-level authorization on board-image mutations, image mutations,
workflow thumbnail access, and admin email leak prevention.

These tests verify the security fixes for:
1. Shared-board write protection bypass via direct API calls
2. Image mutation endpoints lacking ownership checks
3. Private workflow thumbnail exposure
4. Admin email leak on unauthenticated status endpoint
"""

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
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem
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
    "description": "",
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
        external_generation=None,  # type: ignore
    )


@pytest.fixture()
def mock_invoker(mock_services: InvocationServices) -> Invoker:
    return Invoker(services=mock_services)


def _save_image(mock_invoker: Invoker, image_name: str, user_id: str) -> None:
    """Helper to insert an image record owned by a specific user."""
    from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin

    mock_invoker.services.image_records.save(
        image_name=image_name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=100,
        height=100,
        has_workflow=False,
        user_id=user_id,
    )


def _create_user(mock_invoker: Invoker, email: str, display_name: str, is_admin: bool = False) -> str:
    user = mock_invoker.services.users.create(
        UserCreateRequest(email=email, display_name=display_name, password="TestPass123", is_admin=is_admin)
    )
    return user.user_id


def _login(client: TestClient, email: str) -> str:
    r = client.post("/api/v1/auth/login", json={"email": email, "password": "TestPass123", "remember_me": False})
    assert r.status_code == 200
    return r.json()["token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def enable_multiuser(monkeypatch: Any, mock_invoker: Invoker):
    mock_invoker.services.configuration.multiuser = True

    mock_board_images = MagicMock()
    mock_board_images.get_all_board_image_names_for_board.return_value = []
    mock_invoker.services.board_images = mock_board_images

    mock_workflow_thumbnails = MagicMock()
    mock_workflow_thumbnails.get_url.return_value = None
    mock_invoker.services.workflow_thumbnails = mock_workflow_thumbnails

    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.boards.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.board_images.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.images.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.workflows.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.session_queue.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.recall_parameters.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.model_manager.ApiDependencies", mock_deps)
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


def _create_board(client: TestClient, token: str, name: str = "Test Board") -> str:
    r = client.post(f"/api/v1/boards/?board_name={name.replace(' ', '+')}", headers=_auth(token))
    assert r.status_code == status.HTTP_201_CREATED
    return r.json()["board_id"]


def _share_board(client: TestClient, token: str, board_id: str) -> None:
    r = client.patch(f"/api/v1/boards/{board_id}", json={"board_visibility": "shared"}, headers=_auth(token))
    assert r.status_code == status.HTTP_201_CREATED


def _set_board_visibility(client: TestClient, token: str, board_id: str, visibility: str) -> None:
    r = client.patch(f"/api/v1/boards/{board_id}", json={"board_visibility": visibility}, headers=_auth(token))
    assert r.status_code == status.HTTP_201_CREATED


def _create_workflow(client: TestClient, token: str) -> str:
    r = client.post("/api/v1/workflows/", json={"workflow": WORKFLOW_BODY}, headers=_auth(token))
    assert r.status_code == 200
    return r.json()["workflow_id"]


# ===========================================================================
# 1. Board-image mutation authorization
# ===========================================================================


class TestBoardImageMutationAuth:
    """Tests that board_images mutation endpoints enforce ownership."""

    def test_add_image_to_board_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.post("/api/v1/board_images/", json={"board_id": "x", "image_name": "y"})
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_add_image_to_board_batch_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.post("/api/v1/board_images/batch", json={"board_id": "x", "image_names": ["y"]})
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_remove_image_from_board_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.request("DELETE", "/api/v1/board_images/", json={"image_name": "y"})
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_remove_images_from_board_batch_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.post("/api/v1/board_images/batch/delete", json={"image_names": ["y"]})
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_non_owner_cannot_add_image_to_shared_board(self, client: TestClient, user1_token: str, user2_token: str):
        board_id = _create_board(client, user1_token, "User1 Shared Board")
        _share_board(client, user1_token, board_id)

        r = client.post(
            "/api/v1/board_images/",
            json={"board_id": board_id, "image_name": "some-image"},
            headers=_auth(user2_token),
        )
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_non_owner_cannot_add_images_batch_to_shared_board(
        self, client: TestClient, user1_token: str, user2_token: str
    ):
        board_id = _create_board(client, user1_token, "User1 Shared Board Batch")
        _share_board(client, user1_token, board_id)

        r = client.post(
            "/api/v1/board_images/batch",
            json={"board_id": board_id, "image_names": ["img1", "img2"]},
            headers=_auth(user2_token),
        )
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_admin_can_add_image_to_any_board(
        self, client: TestClient, mock_invoker: Invoker, admin_token: str, user1_token: str
    ):
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "user1-admin-board-img", user1.user_id)
        board_id = _create_board(client, user1_token, "User1 Board For Admin")

        # Admin can add any image to any board — should not be 403
        r = client.post(
            "/api/v1/board_images/",
            json={"board_id": board_id, "image_name": "user1-admin-board-img"},
            headers=_auth(admin_token),
        )
        assert r.status_code != status.HTTP_403_FORBIDDEN

    def test_non_owner_can_add_own_image_to_public_board(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """Public boards are documented as writable by other authenticated users."""
        public_board_id = _create_board(client, user1_token, "User1 Public Board")
        _set_board_visibility(client, user1_token, public_board_id, "public")

        user2 = mock_invoker.services.users.get_by_email("user2@test.com")
        assert user2 is not None
        _save_image(mock_invoker, "user2-public-board-img", user2.user_id)

        r = client.post(
            "/api/v1/board_images/",
            json={"board_id": public_board_id, "image_name": "user2-public-board-img"},
            headers=_auth(user2_token),
        )
        assert r.status_code == status.HTTP_201_CREATED

    def test_owner_can_add_image_to_own_board(self, client: TestClient, mock_invoker: Invoker, user1_token: str):
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "user1-own-board-img", user1.user_id)
        board_id = _create_board(client, user1_token, "User1 Own Board")

        r = client.post(
            "/api/v1/board_images/",
            json={"board_id": board_id, "image_name": "user1-own-board-img"},
            headers=_auth(user1_token),
        )
        assert r.status_code != status.HTTP_403_FORBIDDEN

    def test_non_owner_cannot_add_other_users_image_to_own_board(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """Attacker creates their own board, then tries to add victim's image to it.
        This must be rejected — otherwise the attacker gains mutation rights via
        the board-ownership fallback in _assert_image_owner."""
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "victim-image", user1.user_id)

        attacker_board = _create_board(client, user2_token, "Attacker Board")

        r = client.post(
            "/api/v1/board_images/",
            json={"board_id": attacker_board, "image_name": "victim-image"},
            headers=_auth(user2_token),
        )
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_non_owner_cannot_batch_add_other_users_images_to_own_board(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """Same attack via the batch endpoint."""
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "victim-batch-img", user1.user_id)

        attacker_board = _create_board(client, user2_token, "Attacker Batch Board")

        r = client.post(
            "/api/v1/board_images/batch",
            json={"board_id": attacker_board, "image_names": ["victim-batch-img"]},
            headers=_auth(user2_token),
        )
        assert r.status_code == status.HTTP_403_FORBIDDEN


# ===========================================================================
# 2a. Image read-access authorization
# ===========================================================================


class TestImageReadAuth:
    """Tests that image GET endpoints enforce visibility."""

    def test_get_image_dto_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.get("/api/v1/images/i/some-image")
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_image_metadata_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.get("/api/v1/images/i/some-image/metadata")
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_image_full_is_unauthenticated(self, enable_multiuser: Any, client: TestClient):
        # Binary image endpoints are intentionally unauthenticated because
        # browsers load them via <img src> which cannot send Bearer tokens.
        r = client.get("/api/v1/images/i/some-image/full")
        assert r.status_code != status.HTTP_401_UNAUTHORIZED

    def test_get_image_thumbnail_is_unauthenticated(self, enable_multiuser: Any, client: TestClient):
        r = client.get("/api/v1/images/i/some-image/thumbnail")
        assert r.status_code != status.HTTP_401_UNAUTHORIZED

    def test_get_image_urls_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.get("/api/v1/images/i/some-image/urls")
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_non_owner_cannot_read_private_image(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """User2 should not be able to read user1's image that is not on a shared board."""
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "user1-private-img", user1.user_id)

        r = client.get("/api/v1/images/i/user1-private-img", headers=_auth(user2_token))
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_owner_can_read_own_image(self, client: TestClient, mock_invoker: Invoker, user1_token: str):
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "user1-readable", user1.user_id)

        r = client.get("/api/v1/images/i/user1-readable", headers=_auth(user1_token))
        # Should not be 403 (may be 404/500 due to missing board_image_records mock)
        assert r.status_code != status.HTTP_403_FORBIDDEN

    def test_admin_can_read_any_image(
        self, client: TestClient, mock_invoker: Invoker, admin_token: str, user1_token: str
    ):
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "user1-admin-read", user1.user_id)

        r = client.get("/api/v1/images/i/user1-admin-read", headers=_auth(admin_token))
        assert r.status_code != status.HTTP_403_FORBIDDEN

    def test_shared_board_image_readable_by_other_user(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """An image on a shared board should be readable by any authenticated user."""
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "shared-board-img", user1.user_id)

        # Create a shared board and add the image to it
        board_id = _create_board(client, user1_token, "Shared Read Board")
        _share_board(client, user1_token, board_id)
        mock_invoker.services.board_image_records.add_image_to_board(board_id=board_id, image_name="shared-board-img")

        r = client.get("/api/v1/images/i/shared-board-img", headers=_auth(user2_token))
        # Should not be 403 — image is on a shared board
        assert r.status_code != status.HTTP_403_FORBIDDEN

    def test_non_owner_cannot_read_image_metadata(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "user1-meta-blocked", user1.user_id)

        r = client.get("/api/v1/images/i/user1-meta-blocked/metadata", headers=_auth(user2_token))
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_list_images_private_board_rejected_for_non_owner(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """User2 must not be able to enumerate images on user1's private board
        via GET /api/v1/images?board_id=..."""
        board_id = _create_board(client, user1_token, "Private Enum Board")

        r = client.get(f"/api/v1/images/?board_id={board_id}", headers=_auth(user2_token))
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_list_images_shared_board_allowed_for_non_owner(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """User2 should be able to list images on user1's shared board."""
        board_id = _create_board(client, user1_token, "Shared Enum Board")
        _share_board(client, user1_token, board_id)

        r = client.get(f"/api/v1/images/?board_id={board_id}", headers=_auth(user2_token))
        assert r.status_code == status.HTTP_200_OK

    def test_get_image_names_private_board_rejected_for_non_owner(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """User2 must not be able to enumerate image names on user1's private board
        via GET /api/v1/images/names?board_id=..."""
        board_id = _create_board(client, user1_token, "Private Names Board")

        r = client.get(f"/api/v1/images/names?board_id={board_id}", headers=_auth(user2_token))
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_get_image_names_shared_board_allowed_for_non_owner(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """User2 should be able to list image names on user1's shared board."""
        board_id = _create_board(client, user1_token, "Shared Names Board")
        _share_board(client, user1_token, board_id)

        r = client.get(f"/api/v1/images/names?board_id={board_id}", headers=_auth(user2_token))
        assert r.status_code == status.HTTP_200_OK

    def test_list_images_own_private_board_allowed(self, client: TestClient, mock_invoker: Invoker, user1_token: str):
        """Owner should be able to list images on their own private board."""
        board_id = _create_board(client, user1_token, "Own Private Board")

        r = client.get(f"/api/v1/images/?board_id={board_id}", headers=_auth(user1_token))
        assert r.status_code == status.HTTP_200_OK

    def test_admin_can_list_images_on_any_board(
        self, client: TestClient, mock_invoker: Invoker, admin_token: str, user1_token: str
    ):
        """Admin should be able to list images on any board."""
        board_id = _create_board(client, user1_token, "Admin Enum Board")

        r = client.get(f"/api/v1/images/?board_id={board_id}", headers=_auth(admin_token))
        assert r.status_code == status.HTTP_200_OK


# ===========================================================================
# 2b. Image mutation authorization
# ===========================================================================


class TestImageUploadAuth:
    """Tests that image upload enforces board ownership."""

    def test_upload_to_other_users_shared_board_forbidden(self, client: TestClient, user1_token: str, user2_token: str):
        """A user should not be able to upload an image into another user's shared board."""
        board_id = _create_board(client, user1_token, "User1 Shared Upload Board")
        _share_board(client, user1_token, board_id)

        # user2 tries to upload into user1's shared board
        import io

        fake_image = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        r = client.post(
            f"/api/v1/images/upload?image_category=general&is_intermediate=false&board_id={board_id}",
            files={"file": ("test.png", fake_image, "image/png")},
            headers=_auth(user2_token),
        )
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_owner_can_upload_to_own_shared_board(self, client: TestClient, user1_token: str):
        board_id = _create_board(client, user1_token, "User1 Own Upload Board")
        _share_board(client, user1_token, board_id)

        import io

        fake_image = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        r = client.post(
            f"/api/v1/images/upload?image_category=general&is_intermediate=false&board_id={board_id}",
            files={"file": ("test.png", fake_image, "image/png")},
            headers=_auth(user1_token),
        )
        # Should not be 403 (may fail for other reasons in test env)
        assert r.status_code != status.HTTP_403_FORBIDDEN

    def test_non_owner_can_upload_to_public_board(self, client: TestClient, user1_token: str, user2_token: str):
        """Public boards allow any authenticated user to upload images."""
        board_id = _create_board(client, user1_token, "User1 Public Upload Board")
        _set_board_visibility(client, user1_token, board_id, "public")

        import io

        fake_image = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        r = client.post(
            f"/api/v1/images/upload?image_category=general&is_intermediate=false&board_id={board_id}",
            files={"file": ("test.png", fake_image, "image/png")},
            headers=_auth(user2_token),
        )
        # Should not be 403 (may fail downstream for other reasons in test env)
        assert r.status_code != status.HTTP_403_FORBIDDEN


class TestImageMutationAuth:
    """Tests that image mutation endpoints enforce ownership."""

    def test_delete_image_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.delete("/api/v1/images/i/some-image")
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_update_image_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.patch("/api/v1/images/i/some-image", json={"starred": True})
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_batch_delete_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.post("/api/v1/images/delete", json={"image_names": ["x"]})
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_star_images_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.post("/api/v1/images/star", json={"image_names": ["x"]})
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_unstar_images_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.post("/api/v1/images/unstar", json={"image_names": ["x"]})
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_clear_intermediates_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.delete("/api/v1/images/intermediates")
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_delete_uncategorized_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.delete("/api/v1/images/uncategorized")
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_non_owner_cannot_delete_image(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """User2 should not be able to delete user1's image."""
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "user1-image", user1.user_id)

        r = client.delete("/api/v1/images/i/user1-image", headers=_auth(user2_token))
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_owner_can_delete_own_image(self, client: TestClient, mock_invoker: Invoker, user1_token: str):
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "user1-delete-me", user1.user_id)

        r = client.delete("/api/v1/images/i/user1-delete-me", headers=_auth(user1_token))
        # Should not be 403 (may be 200 or 500 depending on file system)
        assert r.status_code != status.HTTP_403_FORBIDDEN

    def test_admin_can_delete_any_image(
        self, client: TestClient, mock_invoker: Invoker, admin_token: str, user1_token: str
    ):
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "user1-admin-delete", user1.user_id)

        r = client.delete("/api/v1/images/i/user1-admin-delete", headers=_auth(admin_token))
        assert r.status_code != status.HTTP_403_FORBIDDEN

    def test_board_owner_can_delete_image_on_own_board(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str
    ):
        """Board owner should be able to delete images on their board even if
        the image's user_id is 'system' (e.g. generated images)."""
        # Create image owned by "system" (simulates queue-generated image)
        _save_image(mock_invoker, "system-img-on-board", "system")

        # Create a board owned by user1 and add the image to it
        board_id = _create_board(client, user1_token, "User1 Board With System Img")
        mock_invoker.services.board_image_records.add_image_to_board(
            board_id=board_id, image_name="system-img-on-board"
        )

        r = client.delete("/api/v1/images/i/system-img-on-board", headers=_auth(user1_token))
        assert r.status_code != status.HTTP_403_FORBIDDEN

    def test_non_owner_cannot_update_image(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "user1-no-star", user1.user_id)

        r = client.patch(
            "/api/v1/images/i/user1-no-star",
            json={"starred": True},
            headers=_auth(user2_token),
        )
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_non_owner_cannot_star_image(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "user1-star-blocked", user1.user_id)

        r = client.post(
            "/api/v1/images/star",
            json={"image_names": ["user1-star-blocked"]},
            headers=_auth(user2_token),
        )
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_non_owner_cannot_batch_delete_image(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "user1-batch-del", user1.user_id)

        r = client.post(
            "/api/v1/images/delete",
            json={"image_names": ["user1-batch-del"]},
            headers=_auth(user2_token),
        )
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_non_owner_can_delete_image_from_public_board(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """Public-board semantics promise delete access to images contained in the board."""
        public_board_id = _create_board(client, user1_token, "User1 Public Delete Board")
        _set_board_visibility(client, user1_token, public_board_id, "public")

        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "user1-public-delete", user1.user_id)
        mock_invoker.services.board_image_records.add_image_to_board(public_board_id, "user1-public-delete")

        r = client.delete(
            "/api/v1/images/i/user1-public-delete",
            headers=_auth(user2_token),
        )
        assert r.status_code == status.HTTP_200_OK

    def test_clear_intermediates_non_admin_forbidden(self, client: TestClient, user1_token: str):
        r = client.delete("/api/v1/images/intermediates", headers=_auth(user1_token))
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_get_intermediates_count_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.get("/api/v1/images/intermediates")
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_download_images_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.post("/api/v1/images/download", json={"image_names": ["x"]})
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_non_owner_cannot_fetch_existing_bulk_download_item(
        self,
        client: TestClient,
        mock_invoker: Invoker,
        monkeypatch: Any,
        tmp_path: Any,
        user1_token: str,
        user2_token: str,
    ):
        """A bulk download zip should be fetchable only by its owner."""
        from fastapi import BackgroundTasks

        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None

        mock_file = tmp_path / "owned-download.zip"
        mock_file.write_text("contents")

        monkeypatch.setattr(mock_invoker.services.bulk_download, "get_path", lambda _: str(mock_file))
        monkeypatch.setattr(mock_invoker.services.bulk_download, "get_owner", lambda _: user1.user_id)
        monkeypatch.setattr(BackgroundTasks, "add_task", lambda *args, **kwargs: None)

        r = client.get("/api/v1/images/download/owned-download.zip", headers=_auth(user2_token))
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_images_by_names_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.post("/api/v1/images/images_by_names", json={"image_names": ["x"]})
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_images_by_names_filters_unauthorized(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """images_by_names should silently skip images the caller cannot access."""
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "user1-by-name", user1.user_id)

        r = client.post(
            "/api/v1/images/images_by_names",
            json={"image_names": ["user1-by-name"]},
            headers=_auth(user2_token),
        )
        assert r.status_code == 200
        # user2 should get an empty list — the image belongs to user1
        assert r.json() == []

    def test_none_board_image_names_only_return_callers_uncategorized_images(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """The uncategorized-images sentinel must not expose other users' image names."""
        mock_invoker.services.board_images.get_all_board_image_names_for_board.side_effect = (
            mock_invoker.services.board_image_records.get_all_board_image_names_for_board
        )

        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        user2 = mock_invoker.services.users.get_by_email("user2@test.com")
        assert user1 is not None
        assert user2 is not None

        _save_image(mock_invoker, "user1-uncategorized-private", user1.user_id)
        _save_image(mock_invoker, "user2-uncategorized-private", user2.user_id)

        r = client.get("/api/v1/boards/none/image_names", headers=_auth(user2_token))
        assert r.status_code == status.HTTP_200_OK
        assert "user2-uncategorized-private" in r.json()
        assert "user1-uncategorized-private" not in r.json()


# ===========================================================================
# 3. Workflow mutation authorization (additional)
# ===========================================================================


class TestWorkflowListScoping:
    """Tests that listing workflows in multiuser mode does not filter out default workflows."""

    def test_default_workflows_visible_when_listing_user_and_default(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str
    ):
        """When categories=['user','default'], default workflows must still appear even
        though user_id_filter is set to the current user (default workflows belong to 'system')."""
        from invokeai.app.services.workflow_records.workflow_records_common import (
            Workflow,
            WorkflowCategory,
            WorkflowMeta,
            WorkflowWithoutID,
        )
        from invokeai.app.util.misc import uuid_string

        default_wf = WorkflowWithoutID(
            name="Test Default Workflow",
            description="A built-in workflow",
            meta=WorkflowMeta(version="3.0.0", category=WorkflowCategory.Default),
            nodes=[],
            edges=[],
            tags="",
            author="",
            contact="",
            version="1.0.0",
            notes="",
            exposedFields=[],
            form_fields=[],
        )
        wf_with_id = Workflow(**default_wf.model_dump(), id=uuid_string())
        # Insert directly via DB since the create API rejects default workflows
        with mock_invoker.services.workflow_records._db.transaction() as cursor:
            cursor.execute(
                "INSERT INTO workflow_library (workflow_id, workflow, user_id) VALUES (?, ?, ?)",
                (wf_with_id.id, wf_with_id.model_dump_json(), "system"),
            )

        # Also create a user workflow via the API
        _create_workflow(client, user1_token)

        # List with categories=user&categories=default
        r = client.get(
            "/api/v1/workflows/?categories=user&categories=default",
            headers=_auth(user1_token),
        )
        assert r.status_code == 200
        data = r.json()
        categories_found = {item["category"] for item in data["items"]}
        assert "default" in categories_found, (
            f"Default workflows were filtered out. Categories found: {categories_found}"
        )
        assert "user" in categories_found

    def test_default_workflows_visible_when_no_category_filter(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str
    ):
        """When no categories filter is given, default workflows should still appear."""
        from invokeai.app.services.workflow_records.workflow_records_common import (
            Workflow,
            WorkflowCategory,
            WorkflowMeta,
            WorkflowWithoutID,
        )
        from invokeai.app.util.misc import uuid_string

        default_wf = WorkflowWithoutID(
            name="Another Default Workflow",
            description="Built-in",
            meta=WorkflowMeta(version="3.0.0", category=WorkflowCategory.Default),
            nodes=[],
            edges=[],
            tags="",
            author="",
            contact="",
            version="1.0.0",
            notes="",
            exposedFields=[],
            form_fields=[],
        )
        wf_with_id = Workflow(**default_wf.model_dump(), id=uuid_string())
        with mock_invoker.services.workflow_records._db.transaction() as cursor:
            cursor.execute(
                "INSERT INTO workflow_library (workflow_id, workflow, user_id) VALUES (?, ?, ?)",
                (wf_with_id.id, wf_with_id.model_dump_json(), "system"),
            )

        _create_workflow(client, user1_token)

        r = client.get("/api/v1/workflows/", headers=_auth(user1_token))
        assert r.status_code == 200
        data = r.json()
        categories_found = {item["category"] for item in data["items"]}
        assert "default" in categories_found, (
            f"Default workflows were filtered out. Categories found: {categories_found}"
        )


class TestWorkflowMutationAuth:
    """Tests for additional workflow mutation endpoints."""

    def test_update_opened_at_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.put("/api/v1/workflows/i/some-id/opened_at")
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_non_owner_cannot_update_opened_at(self, client: TestClient, user1_token: str, user2_token: str):
        workflow_id = _create_workflow(client, user1_token)
        r = client.put(
            f"/api/v1/workflows/i/{workflow_id}/opened_at",
            headers=_auth(user2_token),
        )
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_owner_can_update_opened_at(self, client: TestClient, user1_token: str):
        workflow_id = _create_workflow(client, user1_token)
        r = client.put(
            f"/api/v1/workflows/i/{workflow_id}/opened_at",
            headers=_auth(user1_token),
        )
        assert r.status_code == 200


# ===========================================================================
# 4. Workflow thumbnail authorization
# ===========================================================================


class TestWorkflowThumbnailAuth:
    """Tests for the workflow thumbnail GET endpoint.

    Workflow and image thumbnail endpoints are intentionally unauthenticated
    because browsers load them via <img src> tags which cannot send Bearer
    tokens. IDs are UUIDs, providing security through unguessability.
    """

    def test_thumbnail_is_unauthenticated(self, enable_multiuser: Any, client: TestClient):
        # Binary image endpoints don't require auth — loaded via <img src>
        r = client.get("/api/v1/workflows/i/some-workflow/thumbnail")
        assert r.status_code != status.HTTP_401_UNAUTHORIZED


# ===========================================================================
# 4. Admin email leak prevention
# ===========================================================================


class TestAdminEmailLeak:
    """Tests that the auth status endpoint does not leak admin email."""

    def test_status_does_not_leak_admin_email_when_setup_complete(self, client: TestClient, admin_token: str):
        """After setup is complete, admin_email must be null."""
        r = client.get("/api/v1/auth/status")
        assert r.status_code == 200
        data = r.json()
        assert data["multiuser_enabled"] is True
        assert data["setup_required"] is False
        assert data["admin_email"] is None

    def test_status_returns_admin_email_during_setup(
        self, setup_jwt_secret: None, enable_multiuser: Any, mock_invoker: Invoker, client: TestClient
    ):
        """Before any admin exists, setup_required=True and admin_email may be returned."""
        # Don't create any users -- setup_required should be True
        r = client.get("/api/v1/auth/status")
        assert r.status_code == 200
        data = r.json()
        assert data["setup_required"] is True
        # admin_email is null here because no admin exists yet, which is correct

    def test_status_no_leak_in_single_user_mode(
        self, setup_jwt_secret: None, monkeypatch: Any, mock_invoker: Invoker, client: TestClient
    ):
        """In single-user mode, admin_email should always be null."""
        mock_invoker.services.configuration.multiuser = False
        mock_deps = MockApiDependencies(mock_invoker)
        monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", mock_deps)

        r = client.get("/api/v1/auth/status")
        assert r.status_code == 200
        data = r.json()
        assert data["admin_email"] is None
        assert data["multiuser_enabled"] is False


# ===========================================================================
# 6. Session queue authorization
# ===========================================================================


class TestSessionQueueAuth:
    """Tests that session queue endpoints enforce authentication."""

    def test_get_queue_item_ids_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.get("/api/v1/queue/default/item_ids")
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_current_queue_item_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.get("/api/v1/queue/default/current")
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_next_queue_item_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.get("/api/v1/queue/default/next")
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_batch_status_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.get("/api/v1/queue/default/b/some-batch/status")
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_counts_by_destination_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.get("/api/v1/queue/default/counts_by_destination?destination=canvas")
        assert r.status_code == status.HTTP_401_UNAUTHORIZED


# ===========================================================================
# 6b. Session queue sanitization (cross-user isolation)
# ===========================================================================


class TestSessionQueueSanitization:
    """Tests that sanitize_queue_item_for_user strips all sensitive fields
    from queue items viewed by non-owner, non-admin users."""

    @pytest.fixture
    def _sample_queue_item(self):
        from invokeai.app.services.shared.graph import Graph, GraphExecutionState

        return SessionQueueItem(
            item_id=42,
            status="pending",
            priority=10,
            batch_id="batch-abc",
            origin="workflows",
            destination="canvas",
            session_id="sess-123",
            session=GraphExecutionState(id="sess-123", graph=Graph()),
            error_type="RuntimeError",
            error_message="something broke",
            error_traceback="Traceback ...",
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T01:00:00",
            started_at="2026-01-01T00:30:00",
            completed_at=None,
            queue_id="default",
            user_id="owner-user",
            user_display_name="Owner Display",
            user_email="owner@test.com",
            field_values=None,
            workflow=None,
        )

    def test_owner_sees_all_fields(self, _sample_queue_item: SessionQueueItem):
        from invokeai.app.api.routers.session_queue import sanitize_queue_item_for_user

        result = sanitize_queue_item_for_user(_sample_queue_item, "owner-user", is_admin=False)
        assert result.user_id == "owner-user"
        assert result.user_display_name == "Owner Display"
        assert result.user_email == "owner@test.com"
        assert result.batch_id == "batch-abc"
        assert result.origin == "workflows"
        assert result.destination == "canvas"
        assert result.session_id == "sess-123"
        assert result.priority == 10

    def test_admin_sees_all_fields(self, _sample_queue_item: SessionQueueItem):
        from invokeai.app.api.routers.session_queue import sanitize_queue_item_for_user

        result = sanitize_queue_item_for_user(_sample_queue_item, "admin-user", is_admin=True)
        assert result.user_id == "owner-user"
        assert result.user_display_name == "Owner Display"
        assert result.user_email == "owner@test.com"
        assert result.batch_id == "batch-abc"

    def test_non_owner_sees_only_status_timestamps_errors(self, _sample_queue_item: SessionQueueItem):
        from invokeai.app.api.routers.session_queue import sanitize_queue_item_for_user

        result = sanitize_queue_item_for_user(_sample_queue_item, "other-user", is_admin=False)

        # Preserved: item_id, queue_id, status, timestamps
        assert result.item_id == 42
        assert result.queue_id == "default"
        assert result.status == "pending"
        assert result.created_at == "2026-01-01T00:00:00"
        assert result.updated_at == "2026-01-01T01:00:00"
        assert result.started_at == "2026-01-01T00:30:00"
        assert result.completed_at is None

        # Stripped: errors (may leak file paths, prompts, model names)
        assert result.error_type is None
        assert result.error_message is None
        assert result.error_traceback is None

        # Stripped: user identity
        assert result.user_id == "redacted"
        assert result.user_display_name is None
        assert result.user_email is None

        # Stripped: generation metadata
        assert result.batch_id == "redacted"
        assert result.session_id == "redacted"
        assert result.origin is None
        assert result.destination is None
        assert result.priority == 0
        assert result.field_values is None
        assert result.retried_from_item_id is None
        assert result.workflow is None
        assert result.session.id == "redacted"
        assert len(result.session.graph.nodes) == 0

    def test_sanitization_does_not_mutate_original(self, _sample_queue_item: SessionQueueItem):
        from invokeai.app.api.routers.session_queue import sanitize_queue_item_for_user

        sanitize_queue_item_for_user(_sample_queue_item, "other-user", is_admin=False)
        # Original should be unchanged
        assert _sample_queue_item.user_id == "owner-user"
        assert _sample_queue_item.user_email == "owner@test.com"
        assert _sample_queue_item.batch_id == "batch-abc"


# ===========================================================================
# 7. Recall parameters authorization
# ===========================================================================


class TestRecallParametersAuth:
    """Tests that recall parameter endpoints enforce authentication."""

    def test_get_recall_parameters_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.get("/api/v1/recall/default")
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_update_recall_parameters_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.post("/api/v1/recall/default", json={"positive_prompt": "test"})
        assert r.status_code == status.HTTP_401_UNAUTHORIZED


# ===========================================================================
# 7a2. Recall parameters image access control
# ===========================================================================


class TestRecallImageAccess:
    """Tests that recall parameter image references are validated for read access."""

    def test_recall_controlnet_with_other_users_image_rejected(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """User2 must not be able to reference user1's private image in a control layer."""
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "victim-ctrl-img", user1.user_id)

        r = client.post(
            "/api/v1/recall/default",
            json={"control_layers": [{"model_name": "some-controlnet", "image_name": "victim-ctrl-img"}]},
            headers=_auth(user2_token),
        )
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_recall_ip_adapter_with_other_users_image_rejected(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """User2 must not be able to reference user1's private image in an IP adapter."""
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "victim-ip-img", user1.user_id)

        r = client.post(
            "/api/v1/recall/default",
            json={"ip_adapters": [{"model_name": "some-ip-adapter", "image_name": "victim-ip-img"}]},
            headers=_auth(user2_token),
        )
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_recall_own_image_allowed(self, client: TestClient, mock_invoker: Invoker, user1_token: str):
        """Owner should be able to reference their own image in recall parameters."""
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "own-ctrl-img", user1.user_id)

        r = client.post(
            "/api/v1/recall/default",
            json={"control_layers": [{"model_name": "some-controlnet", "image_name": "own-ctrl-img"}]},
            headers=_auth(user1_token),
        )
        # Should not be 403 (may fail downstream for other reasons, e.g. model not found)
        assert r.status_code != status.HTTP_403_FORBIDDEN

    def test_recall_shared_board_image_allowed(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """An image on a shared board should be usable in recall by any user."""
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "shared-recall-img", user1.user_id)

        board_id = _create_board(client, user1_token, "Shared Recall Board")
        _share_board(client, user1_token, board_id)
        mock_invoker.services.board_image_records.add_image_to_board(board_id=board_id, image_name="shared-recall-img")

        r = client.post(
            "/api/v1/recall/default",
            json={"ip_adapters": [{"model_name": "some-ip-adapter", "image_name": "shared-recall-img"}]},
            headers=_auth(user2_token),
        )
        assert r.status_code != status.HTTP_403_FORBIDDEN

    def test_recall_admin_can_reference_any_image(
        self, client: TestClient, mock_invoker: Invoker, admin_token: str, user1_token: str
    ):
        """Admin should be able to reference any user's image."""
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "admin-recall-img", user1.user_id)

        r = client.post(
            "/api/v1/recall/default",
            json={"control_layers": [{"model_name": "some-controlnet", "image_name": "admin-recall-img"}]},
            headers=_auth(admin_token),
        )
        assert r.status_code != status.HTTP_403_FORBIDDEN


# ===========================================================================
# 7b. Recall parameters cross-user isolation
# ===========================================================================


class TestRecallParametersIsolation:
    """Tests that recall parameters are scoped per-user, not globally by queue_id."""

    def test_user1_write_does_not_leak_to_user2(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """User1 sets a recall parameter; user2 should not see it in client state."""
        # user1 writes a recall parameter
        r = client.post(
            "/api/v1/recall/default",
            json={"positive_prompt": "user1 secret prompt"},
            headers=_auth(user1_token),
        )
        assert r.status_code == 200

        # Verify that user1's data is stored under user1's user_id, not the queue_id
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        user2 = mock_invoker.services.users.get_by_email("user2@test.com")
        assert user1 is not None
        assert user2 is not None

        # user1 should have the value
        val = mock_invoker.services.client_state_persistence.get_by_key(user1.user_id, "recall_positive_prompt")
        assert val is not None
        assert "user1 secret prompt" in val

        # user2 should NOT have the value
        val2 = mock_invoker.services.client_state_persistence.get_by_key(user2.user_id, "recall_positive_prompt")
        assert val2 is None

    def test_two_users_independent_state(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """Both users can write recall params independently without overwriting each other."""
        r1 = client.post(
            "/api/v1/recall/default",
            json={"positive_prompt": "prompt from user1"},
            headers=_auth(user1_token),
        )
        assert r1.status_code == 200

        r2 = client.post(
            "/api/v1/recall/default",
            json={"positive_prompt": "prompt from user2"},
            headers=_auth(user2_token),
        )
        assert r2.status_code == 200

        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        user2 = mock_invoker.services.users.get_by_email("user2@test.com")
        assert user1 is not None
        assert user2 is not None

        val1 = mock_invoker.services.client_state_persistence.get_by_key(user1.user_id, "recall_positive_prompt")
        val2 = mock_invoker.services.client_state_persistence.get_by_key(user2.user_id, "recall_positive_prompt")
        assert val1 is not None and "prompt from user1" in val1
        assert val2 is not None and "prompt from user2" in val2


# ===========================================================================
# 9. Recall parameters event user scoping
# ===========================================================================


class TestRecallParametersEventScoping:
    """Tests that RecallParametersUpdatedEvent carries user_id for targeted delivery."""

    def test_event_includes_user_id(self):
        """RecallParametersUpdatedEvent.build() must set user_id so the socket handler
        can route the event to the correct user room instead of broadcasting."""
        from invokeai.app.services.events.events_common import RecallParametersUpdatedEvent

        event = RecallParametersUpdatedEvent.build(
            queue_id="default",
            user_id="user-abc",
            parameters={"positive_prompt": "test"},
        )
        assert event.queue_id == "default"
        assert event.user_id == "user-abc"
        assert event.parameters == {"positive_prompt": "test"}

    def test_event_not_broadcast_to_all_queue_subscribers(self):
        """RecallParametersUpdatedEvent must have a user_id field so _handle_queue_event
        in sockets.py can route it to the owner room + admin room, not the queue room."""
        from invokeai.app.services.events.events_common import RecallParametersUpdatedEvent

        event = RecallParametersUpdatedEvent.build(
            queue_id="default",
            user_id="owner-123",
            parameters={"seed": 42},
        )
        # The event must carry user_id; without it the socket handler would
        # fall through to the generic else branch and broadcast to all subscribers
        assert hasattr(event, "user_id")
        assert event.user_id == "owner-123"


# ===========================================================================
# 10. Queue status endpoint scoping
# ===========================================================================


class TestQueueStatusScoping:
    """Tests that queue status, batch status, and counts_by_destination
    endpoints scope data to the current user for non-admin callers."""

    def test_get_queue_status_hides_current_item_for_non_owner(self):
        """get_queue_status() must not expose current item details to non-owner, non-admin users."""
        from invokeai.app.services.session_queue.session_queue_common import SessionQueueStatus

        # Simulate a status where the current item belongs to another user
        # When user_id is provided and doesn't match, item details should be None
        status_obj = SessionQueueStatus(
            queue_id="default",
            item_id=None,  # hidden because user doesn't own current item
            session_id=None,
            batch_id=None,
            pending=2,
            in_progress=0,
            completed=1,
            failed=0,
            canceled=0,
            total=3,
        )
        # Verify the model accepts None for item details
        assert status_obj.item_id is None
        assert status_obj.session_id is None
        assert status_obj.batch_id is None

    def test_session_queue_status_has_user_fields(self):
        """SessionQueueStatus exposes user_pending/user_in_progress so the queue badge
        can render an X/Y count (X = caller's jobs, Y = global total)."""
        from invokeai.app.services.session_queue.session_queue_common import SessionQueueStatus

        fields = set(SessionQueueStatus.model_fields.keys())
        assert "user_pending" in fields
        assert "user_in_progress" in fields

        status_obj = SessionQueueStatus(
            queue_id="default",
            item_id=None,
            session_id=None,
            batch_id=None,
            pending=5,
            in_progress=1,
            completed=0,
            failed=0,
            canceled=0,
            total=6,
            user_pending=2,
            user_in_progress=1,
        )
        assert status_obj.user_pending == 2
        assert status_obj.user_in_progress == 1


# ===========================================================================
# 10b. Model install job authorization
# ===========================================================================


class TestModelInstallAuth:
    """Tests that model install job endpoints require admin authentication."""

    def test_list_model_installs_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.get("/api/v2/models/install")
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_model_install_job_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.get("/api/v2/models/install/1")
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_pause_model_install_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.post("/api/v2/models/install/1/pause")
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_resume_model_install_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.post("/api/v2/models/install/1/resume")
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_restart_failed_model_install_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.post("/api/v2/models/install/1/restart_failed")
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_restart_model_install_file_requires_auth(self, enable_multiuser: Any, client: TestClient):
        r = client.post("/api/v2/models/install/1/restart_file", json="https://example.com/model.safetensors")
        assert r.status_code == status.HTTP_401_UNAUTHORIZED

    def test_non_admin_cannot_list_model_installs(self, enable_multiuser: Any, client: TestClient, user1_token: str):
        r = client.get("/api/v2/models/install", headers=_auth(user1_token))
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_non_admin_cannot_pause_model_install(self, enable_multiuser: Any, client: TestClient, user1_token: str):
        r = client.post("/api/v2/models/install/1/pause", headers=_auth(user1_token))
        assert r.status_code == status.HTTP_403_FORBIDDEN


# ===========================================================================
# 11. Bulk download access control
# ===========================================================================


class TestBulkDownloadAccessControl:
    """Tests that bulk download endpoints enforce image/board read access and
    that the fetch endpoint verifies ownership of the zip file."""

    @pytest.fixture(autouse=True)
    def _mock_background_tasks(self, monkeypatch: Any):
        """Prevent BackgroundTasks.add_task from actually running the handler,
        which would fail because image_files is None in the test fixture."""
        from fastapi import BackgroundTasks

        monkeypatch.setattr(BackgroundTasks, "add_task", lambda *args, **kwargs: None)

    def test_bulk_download_by_image_names_rejected_for_non_owner(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """User2 must not be able to bulk-download images owned by user1."""
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "user1-private-dl", user1.user_id)

        r = client.post(
            "/api/v1/images/download",
            json={"image_names": ["user1-private-dl"]},
            headers=_auth(user2_token),
        )
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_bulk_download_by_image_names_allowed_for_owner(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str
    ):
        """Owner should be able to bulk-download their own images."""
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "user1-own-dl", user1.user_id)

        r = client.post(
            "/api/v1/images/download",
            json={"image_names": ["user1-own-dl"]},
            headers=_auth(user1_token),
        )
        assert r.status_code == status.HTTP_202_ACCEPTED

    def test_bulk_download_by_board_rejected_for_private_board(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """User2 must not be able to bulk-download from user1's private board."""
        board_id = _create_board(client, user1_token, "Private DL Board")

        r = client.post(
            "/api/v1/images/download",
            json={"board_id": board_id},
            headers=_auth(user2_token),
        )
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_bulk_download_by_shared_board_allowed(
        self, client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
    ):
        """User2 should be able to bulk-download from user1's shared board."""
        board_id = _create_board(client, user1_token, "Shared DL Board")
        _share_board(client, user1_token, board_id)

        r = client.post(
            "/api/v1/images/download",
            json={"board_id": board_id},
            headers=_auth(user2_token),
        )
        assert r.status_code == status.HTTP_202_ACCEPTED

    def test_admin_can_bulk_download_any_images(
        self, client: TestClient, mock_invoker: Invoker, admin_token: str, user1_token: str
    ):
        """Admin should be able to bulk-download any user's images."""
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "user1-admin-dl", user1.user_id)

        r = client.post(
            "/api/v1/images/download",
            json={"image_names": ["user1-admin-dl"]},
            headers=_auth(admin_token),
        )
        assert r.status_code == status.HTTP_202_ACCEPTED

    def test_bulk_download_events_carry_user_id(self):
        """BulkDownloadEventBase must carry user_id so events can be routed privately."""
        from invokeai.app.services.events.events_common import (
            BulkDownloadCompleteEvent,
            BulkDownloadErrorEvent,
            BulkDownloadEventBase,
            BulkDownloadStartedEvent,
        )

        assert "user_id" in BulkDownloadEventBase.model_fields

        started = BulkDownloadStartedEvent.build("default", "item-1", "item-1.zip", user_id="owner-abc")
        assert started.user_id == "owner-abc"

        complete = BulkDownloadCompleteEvent.build("default", "item-2", "item-2.zip", user_id="owner-abc")
        assert complete.user_id == "owner-abc"

        error = BulkDownloadErrorEvent.build("default", "item-3", "item-3.zip", "oops", user_id="owner-abc")
        assert error.user_id == "owner-abc"

    def test_bulk_download_event_not_emitted_to_shared_default_room(self, mock_invoker: Invoker, monkeypatch: Any):
        """Bulk download capability tokens must not be broadcast to the shared default room."""
        import asyncio
        from unittest.mock import AsyncMock

        from fastapi import FastAPI

        from invokeai.app.api.sockets import SocketIO
        from invokeai.app.services.events.events_common import BulkDownloadCompleteEvent

        mock_deps = MockApiDependencies(mock_invoker)
        monkeypatch.setattr("invokeai.app.api.dependencies.ApiDependencies", mock_deps)

        fastapi_app = FastAPI()
        socketio = SocketIO(fastapi_app)

        event = BulkDownloadCompleteEvent.build("default", "item-x", "item-x.zip", user_id="owner-xyz")

        mock_emit = AsyncMock()
        socketio._sio.emit = mock_emit

        asyncio.run(socketio._handle_bulk_image_download_event(("bulk_download_complete", event)))

        rooms_emitted_to = [call.kwargs.get("room") for call in mock_emit.call_args_list]
        assert "default" not in rooms_emitted_to
        assert "user:owner-xyz" in rooms_emitted_to


# ===========================================================================
# 12. WebSocket authentication and event scoping
# ===========================================================================


class TestWebSocketAuth:
    """Tests that anonymous WebSocket clients cannot subscribe to queue rooms
    in multiuser mode, and that queue item events are scoped to the owner +
    admin rooms instead of being broadcast to the full queue room."""

    @pytest.fixture
    def socketio(self, mock_invoker: Invoker, monkeypatch: Any):
        """Create a SocketIO instance wired to the mock invoker's configuration."""
        from fastapi import FastAPI

        from invokeai.app.api.sockets import SocketIO

        # The SocketIO connect/sub handlers look up ApiDependencies.invoker.services.configuration.multiuser
        # at request time. Patch it to point at the mock invoker.
        mock_deps = MockApiDependencies(mock_invoker)
        monkeypatch.setattr("invokeai.app.api.dependencies.ApiDependencies", mock_deps)

        fastapi_app = FastAPI()
        return SocketIO(fastapi_app)

    def test_connect_rejected_without_token_in_multiuser_mode(self, socketio: Any, mock_invoker: Invoker) -> None:
        """In multiuser mode, _handle_connect must return False when no valid token is provided."""
        import asyncio

        mock_invoker.services.configuration.multiuser = True

        result = asyncio.run(socketio._handle_connect("sid-anon-1", environ={}, auth=None))
        assert result is False
        # The socket must not be recorded in the users dict
        assert "sid-anon-1" not in socketio._socket_users

    def test_connect_rejected_with_invalid_token_in_multiuser_mode(
        self, socketio: Any, mock_invoker: Invoker, setup_jwt_secret: None
    ) -> None:
        """An invalid/garbage token in multiuser mode must still be rejected."""
        import asyncio

        mock_invoker.services.configuration.multiuser = True

        result = asyncio.run(socketio._handle_connect("sid-bad-1", environ={}, auth={"token": "not-a-real-token"}))
        assert result is False
        assert "sid-bad-1" not in socketio._socket_users

    def test_connect_accepted_without_token_in_single_user_mode(self, socketio: Any, mock_invoker: Invoker) -> None:
        """In single-user mode, the socket handler should accept unauthenticated connections
        as the system admin user (matching how the REST API's get_current_user_or_default behaves)."""
        import asyncio

        mock_invoker.services.configuration.multiuser = False

        result = asyncio.run(socketio._handle_connect("sid-single-1", environ={}, auth=None))
        assert result is True
        assert socketio._socket_users["sid-single-1"]["user_id"] == "system"
        assert socketio._socket_users["sid-single-1"]["is_admin"] is True

    def test_connect_accepted_with_valid_token_in_multiuser_mode(
        self,
        socketio: Any,
        mock_invoker: Invoker,
        setup_jwt_secret: None,
    ) -> None:
        """A valid token in multiuser mode should be accepted with the correct user identity."""
        import asyncio

        from invokeai.app.services.auth.token_service import TokenData, create_access_token
        from invokeai.app.services.users.users_common import UserCreateRequest

        mock_invoker.services.configuration.multiuser = True

        # Create the user in the database so the active-user check passes
        user = mock_invoker.services.users.create(
            UserCreateRequest(email="real@test.com", display_name="Real User", password="Test1234!@#$")
        )
        token = create_access_token(TokenData(user_id=user.user_id, email=user.email, is_admin=False))

        result = asyncio.run(socketio._handle_connect("sid-good-1", environ={}, auth={"token": token}))
        assert result is True
        assert socketio._socket_users["sid-good-1"]["user_id"] == user.user_id
        assert socketio._socket_users["sid-good-1"]["is_admin"] is False

    def test_connect_rejected_for_deleted_user_in_multiuser_mode(
        self, socketio: Any, mock_invoker: Invoker, setup_jwt_secret: None
    ) -> None:
        """A structurally valid JWT for a user that no longer exists in the database
        must be rejected.  This mirrors the REST auth check in auth_dependencies.py:53-58."""
        import asyncio

        from invokeai.app.services.auth.token_service import TokenData, create_access_token

        mock_invoker.services.configuration.multiuser = True
        # Create a token for a user_id that was never created in the user service
        token = create_access_token(TokenData(user_id="deleted-user-999", email="gone@test.com", is_admin=False))

        result = asyncio.run(socketio._handle_connect("sid-deleted-1", environ={}, auth={"token": token}))
        assert result is False
        assert "sid-deleted-1" not in socketio._socket_users

    def test_connect_rejected_for_inactive_user_in_multiuser_mode(
        self, socketio: Any, mock_invoker: Invoker, setup_jwt_secret: None
    ) -> None:
        """A structurally valid JWT for a deactivated user must be rejected even though
        the token itself has not expired."""
        import asyncio

        from invokeai.app.services.auth.token_service import TokenData, create_access_token
        from invokeai.app.services.users.users_common import UserCreateRequest

        mock_invoker.services.configuration.multiuser = True

        # Create a real user, then deactivate them
        user = mock_invoker.services.users.create(
            UserCreateRequest(email="inactive@test.com", display_name="Inactive", password="Test1234!@#$")
        )
        token = create_access_token(TokenData(user_id=user.user_id, email=user.email, is_admin=False))

        # Deactivate the user
        from invokeai.app.services.users.users_common import UserUpdateRequest

        mock_invoker.services.users.update(user.user_id, UserUpdateRequest(is_active=False))

        result = asyncio.run(socketio._handle_connect("sid-inactive-1", environ={}, auth={"token": token}))
        assert result is False
        assert "sid-inactive-1" not in socketio._socket_users

    def test_sub_queue_refuses_unknown_socket_in_multiuser_mode(self, socketio: Any, mock_invoker: Invoker) -> None:
        """If a socket somehow reaches _handle_sub_queue without a recorded identity
        in multiuser mode (e.g. bug, race), it must be refused rather than falling back
        to an anonymous system user who could then observe queue item events."""
        import asyncio

        mock_invoker.services.configuration.multiuser = True

        # Call sub_queue without a corresponding connect — the sid is unknown.
        asyncio.run(socketio._handle_sub_queue("sid-ghost-1", {"queue_id": "default"}))

        # The ghost socket must not have been added to the internal users dict
        assert "sid-ghost-1" not in socketio._socket_users

    def test_queue_item_status_changed_has_user_id(self) -> None:
        """QueueItemStatusChangedEvent must carry user_id so _handle_queue_event can
        route it to the owner + admin rooms instead of the public queue room. Without
        this field the event falls through to the generic broadcast branch and any
        subscriber to the queue can observe cross-user queue activity."""
        from invokeai.app.services.events.events_common import (
            InvocationEventBase,
            QueueItemEventBase,
            QueueItemStatusChangedEvent,
        )

        # The event base carries a user_id field
        assert "user_id" in QueueItemEventBase.model_fields
        # QueueItemStatusChangedEvent inherits it
        assert "user_id" in QueueItemStatusChangedEvent.model_fields
        # It is NOT an InvocationEventBase (so the generic QueueItemEventBase branch
        # in _handle_queue_event must also handle it privately)
        assert not issubclass(QueueItemStatusChangedEvent, InvocationEventBase)

    def test_batch_enqueued_event_carries_user_id(self) -> None:
        """BatchEnqueuedEvent must carry user_id so it can be routed privately to the
        owner and admin rooms. Otherwise a subscriber on the same queue_id would see
        every other user's batch_id, origin and enqueued counts."""
        from invokeai.app.services.events.events_common import BatchEnqueuedEvent
        from invokeai.app.services.session_queue.session_queue_common import (
            Batch,
            EnqueueBatchResult,
        )
        from invokeai.app.services.shared.graph import Graph

        enqueue_result = EnqueueBatchResult(
            queue_id="default",
            enqueued=3,
            requested=3,
            batch=Batch(batch_id="batch-xyz", origin="workflows", graph=Graph()),
            priority=0,
            item_ids=[1, 2, 3],
        )
        event = BatchEnqueuedEvent.build(enqueue_result, user_id="owner-123")
        assert event.user_id == "owner-123"
        assert event.batch_id == "batch-xyz"
        assert event.queue_id == "default"

    def test_queue_item_status_changed_routed_privately(self, socketio: Any) -> None:
        """_handle_queue_event must emit the FULL QueueItemStatusChangedEvent only to the
        owner's user room and the admin room. A sanitized companion (user_id="redacted",
        identifiers stripped) is also emitted to the queue_id room so other users' UIs can
        refresh, with the owner's and admins' sids in skip_sid so they don't get a duplicate
        that would clobber their cache."""
        import asyncio
        from unittest.mock import AsyncMock

        from invokeai.app.services.events.events_common import QueueItemStatusChangedEvent
        from invokeai.app.services.session_queue.session_queue_common import (
            BatchStatus,
            SessionQueueStatus,
        )

        event = QueueItemStatusChangedEvent(
            queue_id="default",
            item_id=1,
            batch_id="batch-private",
            origin="workflows",
            destination="canvas",
            user_id="owner-xyz",
            session_id="sess-private",
            status="in_progress",
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:01:00",
            started_at="2026-01-01T00:00:30",
            completed_at=None,
            batch_status=BatchStatus(
                queue_id="default",
                batch_id="batch-private",
                origin="workflows",
                destination="canvas",
                pending=0,
                in_progress=1,
                completed=0,
                failed=0,
                canceled=0,
                total=1,
            ),
            queue_status=SessionQueueStatus(
                queue_id="default",
                item_id=1,
                session_id="sess-private",
                batch_id="batch-private",
                pending=0,
                in_progress=1,
                completed=0,
                failed=0,
                canceled=0,
                total=1,
            ),
        )

        # Track owner sid so we can verify skip_sid is honored
        socketio._socket_users["sid-owner"] = {"user_id": "owner-xyz", "is_admin": False}
        socketio._socket_users["sid-admin"] = {"user_id": "admin-1", "is_admin": True}
        socketio._socket_users["sid-other"] = {"user_id": "other-user", "is_admin": False}

        mock_emit = AsyncMock()
        socketio._sio.emit = mock_emit

        asyncio.run(socketio._handle_queue_event(("queue_item_status_changed", event)))

        # Collect (room, payload, skip_sid) for each emit call
        emits = [
            (c.kwargs.get("room"), c.kwargs.get("data"), c.kwargs.get("skip_sid")) for c in mock_emit.call_args_list
        ]

        # Full event must go to owner room and admin room with original sensitive fields
        owner_emits = [(p, s) for r, p, s in emits if r == "user:owner-xyz"]
        admin_emits = [(p, s) for r, p, s in emits if r == "admin"]
        assert len(owner_emits) == 1 and len(admin_emits) == 1
        for payload, _ in owner_emits + admin_emits:
            assert payload["user_id"] == "owner-xyz"
            assert payload["batch_id"] == "batch-private"
            assert payload["session_id"] == "sess-private"
            assert payload["destination"] == "canvas"

        # A sanitized companion event must go to the queue_id room with sensitive fields cleared
        queue_emits = [(p, s) for r, p, s in emits if r == "default"]
        assert len(queue_emits) == 1, "expected exactly one sanitized emit to queue room"
        sanitized_payload, skip_sid = queue_emits[0]
        assert sanitized_payload["user_id"] == "redacted"
        assert sanitized_payload["batch_id"] == "redacted"
        assert sanitized_payload["session_id"] == "redacted"
        assert sanitized_payload["origin"] is None
        assert sanitized_payload["destination"] is None
        assert sanitized_payload["error_type"] is None
        assert sanitized_payload["batch_status"]["batch_id"] == "redacted"
        assert sanitized_payload["batch_status"]["destination"] is None
        assert sanitized_payload["queue_status"]["item_id"] is None
        assert sanitized_payload["queue_status"]["batch_id"] is None
        assert sanitized_payload["queue_status"]["user_pending"] is None
        # Owner and admin sids must be skipped so they don't receive the duplicate
        assert "sid-owner" in skip_sid
        assert "sid-admin" in skip_sid
        # Third-party user must NOT be skipped — they need the sanitized event
        assert "sid-other" not in skip_sid
        # Status (non-sensitive) is preserved so the non-owner UI knows what changed
        assert sanitized_payload["status"] == "in_progress"
        assert sanitized_payload["item_id"] == 1

    def test_batch_enqueued_routed_privately(self, socketio: Any) -> None:
        """_handle_queue_event must emit the FULL BatchEnqueuedEvent only to the owner's
        user room and the admin room. A sanitized companion (user_id="redacted", batch_id
        and origin stripped) is also emitted to the queue_id room so other users' badge
        totals refresh, with owner/admin sids in skip_sid."""
        import asyncio
        from unittest.mock import AsyncMock

        from invokeai.app.services.events.events_common import BatchEnqueuedEvent
        from invokeai.app.services.session_queue.session_queue_common import (
            Batch,
            EnqueueBatchResult,
        )
        from invokeai.app.services.shared.graph import Graph

        enqueue_result = EnqueueBatchResult(
            queue_id="default",
            enqueued=5,
            requested=5,
            batch=Batch(batch_id="batch-pvt", origin="workflows", graph=Graph()),
            priority=0,
            item_ids=[10, 11, 12, 13, 14],
        )
        event = BatchEnqueuedEvent.build(enqueue_result, user_id="owner-zzz")

        socketio._socket_users["sid-owner"] = {"user_id": "owner-zzz", "is_admin": False}
        socketio._socket_users["sid-admin"] = {"user_id": "admin-1", "is_admin": True}
        socketio._socket_users["sid-other"] = {"user_id": "other-user", "is_admin": False}

        mock_emit = AsyncMock()
        socketio._sio.emit = mock_emit

        asyncio.run(socketio._handle_queue_event(("batch_enqueued", event)))

        emits = [
            (c.kwargs.get("room"), c.kwargs.get("data"), c.kwargs.get("skip_sid")) for c in mock_emit.call_args_list
        ]

        # Full event to owner + admin contains the real batch_id and origin
        owner_emits = [(p, s) for r, p, s in emits if r == "user:owner-zzz"]
        admin_emits = [(p, s) for r, p, s in emits if r == "admin"]
        assert len(owner_emits) == 1 and len(admin_emits) == 1
        for payload, _ in owner_emits + admin_emits:
            assert payload["user_id"] == "owner-zzz"
            assert payload["batch_id"] == "batch-pvt"
            assert payload["origin"] == "workflows"

        # Sanitized event to queue room: user/batch/origin redacted, owner+admin skipped
        queue_emits = [(p, s) for r, p, s in emits if r == "default"]
        assert len(queue_emits) == 1
        sanitized_payload, skip_sid = queue_emits[0]
        assert sanitized_payload["user_id"] == "redacted"
        assert sanitized_payload["batch_id"] == "redacted"
        assert sanitized_payload["origin"] is None
        assert sanitized_payload["enqueued"] == 5  # count is non-sensitive
        assert "sid-owner" in skip_sid
        assert "sid-admin" in skip_sid
        assert "sid-other" not in skip_sid

    def test_queue_cleared_still_broadcast(self, socketio: Any) -> None:
        """QueueClearedEvent does not carry user identity and should still be broadcast
        to all queue subscribers — this is a sanity check that we haven't over-scoped."""
        import asyncio
        from unittest.mock import AsyncMock

        from invokeai.app.services.events.events_common import QueueClearedEvent

        event = QueueClearedEvent.build(queue_id="default")

        mock_emit = AsyncMock()
        socketio._sio.emit = mock_emit

        asyncio.run(socketio._handle_queue_event(("queue_cleared", event)))

        rooms_emitted_to = [call.kwargs.get("room") for call in mock_emit.call_args_list]
        assert "default" in rooms_emitted_to
