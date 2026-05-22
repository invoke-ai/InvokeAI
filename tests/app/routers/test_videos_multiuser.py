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

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

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


def test_owner_can_list_videos_on_their_private_board(client: TestClient, user1_private_board: str, user1_token: str):
    response = client.get(
        f"/api/v1/videos/?board_id={user1_private_board}",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert response.status_code == status.HTTP_200_OK


def test_admin_can_list_videos_on_any_private_board(client: TestClient, user1_private_board: str, admin_token: str):
    response = client.get(
        f"/api/v1/videos/?board_id={user1_private_board}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == status.HTTP_200_OK


# ---------------------------------------------------------------------------
# Omitted board_id: route should not blow up; isolation enforced at SQL layer
# ---------------------------------------------------------------------------


def test_list_video_dtos_no_board_id_succeeds_for_any_authed_user(client: TestClient, user2_token: str):
    """The route allows omitted board_id (the SQL layer filters by user_id) — no 403 here."""
    response = client.get(
        "/api/v1/videos/",
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert response.status_code == status.HTTP_200_OK


def test_list_video_dtos_none_board_succeeds_for_any_authed_user(client: TestClient, user2_token: str):
    response = client.get(
        "/api/v1/videos/?board_id=none",
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert response.status_code == status.HTTP_200_OK


# ---------------------------------------------------------------------------
# POST /videos/delete must not re-raise mid-loop (PR #9163 review fix)
# ---------------------------------------------------------------------------


def test_delete_videos_from_list_skips_foreign_items_and_returns_owned(
    client: TestClient, mock_invoker: Invoker, user1_token: str
):
    """A non-admin batch delete that includes a video owned by another user must keep going
    and return 200 with the owned items in ``deleted_videos``. Previously the route raised
    403 mid-loop, throwing away the response payload so the frontend cache never learned
    about already-deleted records and the UI showed stale entries until the next refresh.
    """
    # Resolve user1's id from the token claim so we can wire up the ownership stub
    # without depending on test-internal user state.
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    user1_id = user1.user_id

    def fake_get_user_id(video_name: str):
        # Names beginning with 'mine_' belong to user1, anything else to a stranger.
        return user1_id if video_name.startswith("mine_") else "other-user-id"

    mock_invoker.services.video_records.get_user_id.side_effect = fake_get_user_id
    # When _assert_video_owner falls back to the board check, return no board so the public
    # fallback path doesn't relax permissions for the foreign video.
    mock_invoker.services.board_video_records.get_board_for_video.return_value = None

    fake_dto = MagicMock()
    fake_dto.board_id = None
    mock_invoker.services.videos.get_dto.return_value = fake_dto

    response = client.post(
        "/api/v1/videos/delete",
        json={"video_names": ["mine_a.mp4", "foreign.mp4", "mine_b.mp4"]},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    # Both owned items must appear; the foreign one must be skipped silently.
    assert set(body["deleted_videos"]) == {"mine_a.mp4", "mine_b.mp4"}
    # The service must have been told to delete the owned names but not the foreign one.
    delete_calls = {call.args[0] for call in mock_invoker.services.videos.delete.call_args_list}
    assert delete_calls == {"mine_a.mp4", "mine_b.mp4"}


# ---------------------------------------------------------------------------
# POST /videos/upload must reject malformed MP4 payloads with 415 (residual
# verification flagged in JPPhoto's PR #9163 review)
# ---------------------------------------------------------------------------


def test_upload_video_malformed_mp4_returns_415_and_cleans_up_tmp(
    client: TestClient, mock_invoker: Invoker, user1_token: str, tmp_path: Path
):
    """An upload that looks like an MP4 on the surface (``.mp4`` extension or video MIME
    type) but contains bytes ``probe_video`` can't decode must:

      1. Reach ``probe_video`` (the extension/MIME gate is intentionally permissive — the
         real validation is the decode probe).
      2. Surface a 415 to the caller.
      3. Unlink the streamed-to-disk temp file so the server doesn't leak storage on every
         garbage upload.
    """
    # Capture the tmp path the route created so we can prove it was unlinked after the
    # 415 response. ``tempfile.NamedTemporaryFile(..., delete=False)`` is invoked inside
    # the route, so we wrap the real call and stash the resulting path.
    captured_paths: list[Path] = []

    import tempfile as _tempfile

    real_named_tmp = _tempfile.NamedTemporaryFile

    def spying_named_tmp(*args: Any, **kwargs: Any):
        handle = real_named_tmp(*args, **kwargs)
        captured_paths.append(Path(handle.name))
        return handle

    # The fixture's videos mock would no-op the service call; we explicitly do NOT want
    # that path to fire because we're asserting probe_video runs and rejects.
    mock_invoker.services.videos.create.side_effect = AssertionError(
        "videos.create should not be called when probe_video rejects the upload"
    )

    with (
        patch("invokeai.app.api.routers.videos.tempfile.NamedTemporaryFile", side_effect=spying_named_tmp),
        patch(
            "invokeai.app.api.routers.videos.probe_video",
            side_effect=RuntimeError("not a decodable mp4"),
        ),
    ):
        response = client.post(
            "/api/v1/videos/upload",
            params={"video_category": "general", "is_intermediate": False},
            files={"file": ("renamed_text.mp4", b"this is not an mp4 payload at all", "video/mp4")},
            headers={"Authorization": f"Bearer {user1_token}"},
        )

    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    # The route should have allocated exactly one tmp file and then unlinked it.
    assert len(captured_paths) == 1, f"expected one tmp file, got {captured_paths}"
    tmp_file = captured_paths[0]
    assert not tmp_file.exists(), f"tmp file leaked after 415: {tmp_file}"


# ---------------------------------------------------------------------------
# GET /videos/i/{video_name}/thumbnail must return 404 when the thumbnail file
# is missing on disk (JPPhoto PR #9163 follow-up). Video saves are allowed
# without a thumbnail in video_files_disk.save, so this is reachable.
# ---------------------------------------------------------------------------


def test_get_video_thumbnail_missing_file_returns_404(
    enable_multiuser_for_videos: Any,
    client: TestClient,
    mock_invoker: Invoker,
    tmp_path: Path,
):
    """If videos.get_path resolves successfully but the file doesn't exist, the route must
    return 404 up front. Previously it returned FileResponse and the missing-file error was
    raised by Starlette *after* the route's try/except, so callers saw a 500-class failure
    instead of the documented 404.
    """
    missing_path = tmp_path / "does_not_exist.webp"
    assert not missing_path.exists()
    mock_invoker.services.videos.get_path.return_value = str(missing_path)

    response = client.get("/api/v1/videos/i/some_video.mp4/thumbnail")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    mock_invoker.services.videos.get_path.assert_called_once_with("some_video.mp4", thumbnail=True)
