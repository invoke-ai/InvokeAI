"""Multiuser regression tests for the /v1/videos/ routes.

Covers JPPhoto's code-review finding (PR #9163): the list endpoints accepted
an explicit ``board_id`` with no read-access check, so a non-admin user could
enumerate videos on someone else's private board if they happened to know its
id. The fix added ``_assert_board_read_access`` to both ``list_video_dtos``
and ``get_video_names``.

These tests exercise the HTTP layer end-to-end (auth + route guards) using the
same fixture pattern as test_boards_multiuser. The storage-level user_id
filter is covered separately in tests/app/services/video_records.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.users.users_common import UserCreateRequest


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


def setup_test_user(
    mock_invoker: Invoker,
    email: str,
    display_name: str,
    password: str = "TestPass123",
    is_admin: bool = False,
) -> str:
    user_service = mock_invoker.services.users
    user = user_service.create(
        UserCreateRequest(email=email, display_name=display_name, password=password, is_admin=is_admin)
    )
    return user.user_id


def get_user_token(client: TestClient, email: str, password: str = "TestPass123") -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password, "remember_me": False},
    )
    assert response.status_code == 200
    return response.json()["token"]


@pytest.fixture
def enable_multiuser_for_videos(monkeypatch: Any, mock_invoker: Invoker):
    """Enable multiuser and stub services the video routes touch."""
    mock_invoker.services.configuration.multiuser = True

    # The list routes call services.videos.get_many / get_video_names. We don't care about
    # the payloads here — only whether the route runs the board-access guard *before* the
    # service call. A return value of "any non-error response" is enough.
    mock_videos = MagicMock()
    mock_videos.get_many.return_value = {"items": [], "offset": 0, "limit": 10, "total": 0}
    mock_videos.get_video_names.return_value = {"video_names": [], "starred_count": 0, "total_count": 0}
    mock_invoker.services.videos = mock_videos

    # board_video_records is touched by remove_video_from_board; not exercised by the
    # list tests but stub it defensively so unrelated routes don't blow up.
    mock_invoker.services.board_video_records = MagicMock()
    mock_invoker.services.video_records = MagicMock()
    mock_invoker.services.board_images = MagicMock()
    mock_invoker.services.board_images.get_all_board_image_names_for_board.return_value = []

    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.boards.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.videos.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.images.ApiDependencies", mock_deps)
    yield


@pytest.fixture
def admin_token(setup_jwt_secret: None, enable_multiuser_for_videos: Any, mock_invoker: Invoker, client: TestClient):
    setup_test_user(mock_invoker, "admin@test.com", "Test Admin", is_admin=True)
    return get_user_token(client, "admin@test.com")


@pytest.fixture
def user1_token(enable_multiuser_for_videos: Any, mock_invoker: Invoker, client: TestClient, admin_token: str):
    setup_test_user(mock_invoker, "user1@test.com", "User One", is_admin=False)
    return get_user_token(client, "user1@test.com")


@pytest.fixture
def user2_token(enable_multiuser_for_videos: Any, mock_invoker: Invoker, client: TestClient, admin_token: str):
    setup_test_user(mock_invoker, "user2@test.com", "User Two", is_admin=False)
    return get_user_token(client, "user2@test.com")


@pytest.fixture
def user1_private_board(client: TestClient, user1_token: str) -> str:
    response = client.post(
        "/api/v1/boards/?board_name=User1+Private+Board",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert response.status_code == status.HTTP_201_CREATED
    return response.json()["board_id"]


# ---------------------------------------------------------------------------
# Auth requirement
# ---------------------------------------------------------------------------


def test_list_video_dtos_requires_auth(enable_multiuser_for_videos: Any, client: TestClient):
    response = client.get("/api/v1/videos/")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_get_video_names_requires_auth(enable_multiuser_for_videos: Any, client: TestClient):
    response = client.get("/api/v1/videos/names")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


# ---------------------------------------------------------------------------
# Explicit board_id with no read access (the JPPhoto finding)
# ---------------------------------------------------------------------------


def test_list_video_dtos_forbidden_for_other_users_private_board(
    client: TestClient, user1_private_board: str, user2_token: str
):
    """user2 cannot list videos on user1's private board even if they know the board_id."""
    response = client.get(
        f"/api/v1/videos/?board_id={user1_private_board}",
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_get_video_names_forbidden_for_other_users_private_board(
    client: TestClient, user1_private_board: str, user2_token: str
):
    response = client.get(
        f"/api/v1/videos/names?board_id={user1_private_board}",
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_owner_can_list_videos_on_their_private_board(
    client: TestClient, user1_private_board: str, user1_token: str
):
    response = client.get(
        f"/api/v1/videos/?board_id={user1_private_board}",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert response.status_code == status.HTTP_200_OK


def test_admin_can_list_videos_on_any_private_board(
    client: TestClient, user1_private_board: str, admin_token: str
):
    response = client.get(
        f"/api/v1/videos/?board_id={user1_private_board}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == status.HTTP_200_OK


# ---------------------------------------------------------------------------
# Omitted board_id: route should not blow up; isolation enforced at SQL layer
# ---------------------------------------------------------------------------


def test_list_video_dtos_no_board_id_succeeds_for_any_authed_user(
    client: TestClient, user2_token: str
):
    """The route allows omitted board_id (the SQL layer filters by user_id) — no 403 here."""
    response = client.get(
        "/api/v1/videos/",
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert response.status_code == status.HTTP_200_OK


def test_list_video_dtos_none_board_succeeds_for_any_authed_user(
    client: TestClient, user2_token: str
):
    response = client.get(
        "/api/v1/videos/?board_id=none",
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert response.status_code == status.HTTP_200_OK
