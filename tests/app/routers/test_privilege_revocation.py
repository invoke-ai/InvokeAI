"""Tests that database role/status changes take effect immediately for existing JWTs.

The JWT proves identity only. Authorization (`is_admin`, `is_active`) is derived from
the database on every authenticated request, and sliding-window refresh mints the new
token from the database record. Without this, a demoted administrator could keep admin
rights until token expiry — renewing the stale claim (and media cookie) indefinitely
with every mutation.
"""

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.auth.token_service import verify_token
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.events.events_common import UserAccessChangedEvent
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
    from invokeai.app.services.board_video_records.board_video_records_sqlite import SqliteBoardVideoRecordStorage
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
    from invokeai.app.services.video_records.video_records_sqlite import SqliteVideoRecordStorage
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
        videos=None,  # type: ignore
        video_files=None,  # type: ignore
        video_records=SqliteVideoRecordStorage(db=db),
        board_video_records=SqliteBoardVideoRecordStorage(db=db),
        gallery=None,  # type: ignore
    )


@pytest.fixture()
def mock_invoker(mock_services: InvocationServices) -> Invoker:
    return Invoker(services=mock_services)


def _save_image(mock_invoker: Invoker, image_name: str, user_id: str) -> None:
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
def enable_multiuser(monkeypatch: Any, mock_invoker: Invoker, setup_jwt_secret: None):
    mock_invoker.services.configuration.multiuser = True

    mock_board_images = MagicMock()
    mock_board_images.get_all_board_image_names_for_board.return_value = []
    mock_invoker.services.board_images = mock_board_images

    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.images.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers._access.ApiDependencies", mock_deps)
    # The sliding-window middleware resolves ApiDependencies from its defining module
    # at request time; patch it there so refresh assertions exercise the real logic.
    monkeypatch.setattr("invokeai.app.api.dependencies.ApiDependencies", mock_deps)
    yield


@pytest.fixture
def admin_token(enable_multiuser: Any, mock_invoker: Invoker, client: TestClient):
    _create_user(mock_invoker, "admin@test.com", "Admin", is_admin=True)
    return _login(client, "admin@test.com")


def _demote(client: TestClient, admin_token: str, mock_invoker: Invoker, email: str) -> None:
    user = mock_invoker.services.users.get_by_email(email)
    assert user is not None
    r = client.patch(f"/api/v1/auth/users/{user.user_id}", json={"is_admin": False}, headers=_auth(admin_token))
    assert r.status_code == 200


class TestDbDerivedAuthorization:
    """Authorization fields come from the database record, not the token's claims."""

    def test_demoted_admin_old_token_gets_403_and_no_refresh(
        self, client: TestClient, mock_invoker: Invoker, admin_token: str
    ) -> None:
        _create_user(mock_invoker, "admin2@test.com", "Admin Two", is_admin=True)
        admin2_token = _login(client, "admin2@test.com")
        _demote(client, admin_token, mock_invoker, "admin2@test.com")

        # Admin-only mutation with the pre-demotion token: rejected, and the stale
        # admin claim is not renewed (no refreshed bearer token, no media cookie).
        r = client.post(
            "/api/v1/auth/users",
            json={"email": "new@test.com", "display_name": "New", "password": "TestPass123", "is_admin": False},
            headers=_auth(admin2_token),
        )
        assert r.status_code == status.HTTP_403_FORBIDDEN
        assert "X-Refreshed-Token" not in r.headers
        assert "set-cookie" not in r.headers

    def test_demoted_admin_old_token_cannot_read_other_users_private_image(
        self, client: TestClient, mock_invoker: Invoker, admin_token: str
    ) -> None:
        _create_user(mock_invoker, "admin2@test.com", "Admin Two", is_admin=True)
        _create_user(mock_invoker, "user1@test.com", "User One")
        admin2_token = _login(client, "admin2@test.com")
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None
        _save_image(mock_invoker, "user1-private-img", user1.user_id)
        # The DTO route resolves URLs through the urls service, which is None in
        # the test harness.
        mock_urls = MagicMock()
        mock_urls.get_image_url.return_value = "http://test/image.png"
        mock_invoker.services.urls = mock_urls

        # Pre-demotion the token works (admin may read any image)...
        r = client.get("/api/v1/images/i/user1-private-img", headers=_auth(admin2_token))
        assert r.status_code == status.HTTP_200_OK

        _demote(client, admin_token, mock_invoker, "admin2@test.com")

        # ...post-demotion the same token is denied.
        r = client.get("/api/v1/images/i/user1-private-img", headers=_auth(admin2_token))
        assert r.status_code == status.HTTP_403_FORBIDDEN

    def test_promoted_user_old_token_gains_admin(
        self, client: TestClient, mock_invoker: Invoker, admin_token: str
    ) -> None:
        _create_user(mock_invoker, "user1@test.com", "User One")
        user1_token = _login(client, "user1@test.com")
        user1 = mock_invoker.services.users.get_by_email("user1@test.com")
        assert user1 is not None

        r = client.get("/api/v1/auth/users", headers=_auth(user1_token))
        assert r.status_code == status.HTTP_403_FORBIDDEN

        r = client.patch(f"/api/v1/auth/users/{user1.user_id}", json={"is_admin": True}, headers=_auth(admin_token))
        assert r.status_code == 200

        # The pre-promotion token now carries admin rights (derived from the DB).
        r = client.get("/api/v1/auth/users", headers=_auth(user1_token))
        assert r.status_code == status.HTTP_200_OK

    def test_unchanged_admin_mutation_refreshes_with_admin_claim(
        self, client: TestClient, mock_invoker: Invoker, admin_token: str
    ) -> None:
        r = client.post(
            "/api/v1/auth/users",
            json={"email": "new@test.com", "display_name": "New", "password": "TestPass123", "is_admin": False},
            headers=_auth(admin_token),
        )
        assert r.status_code == status.HTTP_201_CREATED
        refreshed = verify_token(r.headers["X-Refreshed-Token"])
        assert refreshed is not None
        assert refreshed.is_admin is True

    def test_demoted_admin_allowed_mutation_refreshes_with_demoted_claim(
        self, client: TestClient, mock_invoker: Invoker, admin_token: str
    ) -> None:
        """A successful non-admin mutation by a demoted user renews the token with the
        database's is_admin=False — the stale admin claim does not survive refresh."""
        _create_user(mock_invoker, "admin2@test.com", "Admin Two", is_admin=True)
        admin2_token = _login(client, "admin2@test.com")
        _demote(client, admin_token, mock_invoker, "admin2@test.com")

        r = client.patch("/api/v1/auth/me", json={"display_name": "Renamed"}, headers=_auth(admin2_token))
        assert r.status_code == 200
        refreshed = verify_token(r.headers["X-Refreshed-Token"])
        assert refreshed is not None
        assert refreshed.is_admin is False


class TestUserAccessChangedEmission:
    """Role/status changes emit the internal event that re-authorizes live connections."""

    def _access_events(self, mock_invoker: Invoker) -> list[UserAccessChangedEvent]:
        return [e for e in mock_invoker.services.events.events if isinstance(e, UserAccessChangedEvent)]

    def test_demotion_emits_event(self, client: TestClient, mock_invoker: Invoker, admin_token: str) -> None:
        _create_user(mock_invoker, "admin2@test.com", "Admin Two", is_admin=True)
        _demote(client, admin_token, mock_invoker, "admin2@test.com")

        events = self._access_events(mock_invoker)
        assert len(events) == 1
        assert events[0].is_admin is False
        assert events[0].is_active is True

    def test_deactivation_emits_event(self, client: TestClient, mock_invoker: Invoker, admin_token: str) -> None:
        user_id = _create_user(mock_invoker, "user1@test.com", "User One")

        r = client.patch(f"/api/v1/auth/users/{user_id}", json={"is_active": False}, headers=_auth(admin_token))
        assert r.status_code == 200

        events = self._access_events(mock_invoker)
        assert len(events) == 1
        assert events[0].user_id == user_id
        assert events[0].is_active is False

    def test_display_name_change_does_not_emit_event(
        self, client: TestClient, mock_invoker: Invoker, admin_token: str
    ) -> None:
        user_id = _create_user(mock_invoker, "user1@test.com", "User One")

        r = client.patch(f"/api/v1/auth/users/{user_id}", json={"display_name": "Renamed"}, headers=_auth(admin_token))
        assert r.status_code == 200

        assert self._access_events(mock_invoker) == []

    def test_deletion_emits_inactive_event(self, client: TestClient, mock_invoker: Invoker, admin_token: str) -> None:
        user_id = _create_user(mock_invoker, "user1@test.com", "User One")

        r = client.delete(f"/api/v1/auth/users/{user_id}", headers=_auth(admin_token))
        assert r.status_code == status.HTTP_204_NO_CONTENT

        events = self._access_events(mock_invoker)
        assert len(events) == 1
        assert events[0].user_id == user_id
        assert events[0].is_active is False
        assert events[0].is_admin is False
