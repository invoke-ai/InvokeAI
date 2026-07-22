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
from invokeai.app.api.routers.videos import _is_mp4_file
from invokeai.app.api_app import app
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.videos.videos_common import VideoDTO


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
    # The board service computes video_count + cover_video_name on every get_dto/update;
    # an unconfigured MagicMock returns nested MagicMocks that fail Pydantic validation and
    # the boards route swallows the exception as a 404. Pin sane defaults.
    mock_invoker.services.board_video_records.get_video_count_for_board.return_value = 0
    mock_invoker.services.video_records = MagicMock()
    mock_invoker.services.video_records.get_most_recent_video_for_board.return_value = None
    mock_invoker.services.board_images = MagicMock()
    mock_invoker.services.board_images.get_all_board_image_names_for_board.return_value = []

    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api_app.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.boards.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.videos.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.images.ApiDependencies", mock_deps)
    # _access.assert_board_read_access is called from list_video_dtos and get_video_names
    # via the videos router; it uses ApiDependencies from its own module scope.
    monkeypatch.setattr("invokeai.app.api.routers._access.ApiDependencies", mock_deps)
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
    assert body["failed_videos"] == ["foreign.mp4"]
    # The service must have been told to delete the owned names but not the foreign one.
    delete_calls = {call.args[0] for call in mock_invoker.services.videos.delete.call_args_list}
    assert delete_calls == {"mine_a.mp4", "mine_b.mp4"}


# ---------------------------------------------------------------------------
# POST /videos/star and /videos/unstar must not re-raise mid-loop either
# (PR #9163 review fix — same partial-mutation-then-403 pattern as bulk delete)
# ---------------------------------------------------------------------------


def _setup_mixed_ownership_batch(mock_invoker: Invoker, user1_id: str) -> None:
    """Names beginning with 'mine_' belong to user1, anything else to a stranger."""

    def fake_get_user_id(video_name: str):
        return user1_id if video_name.startswith("mine_") else "other-user-id"

    mock_invoker.services.video_records.get_user_id.side_effect = fake_get_user_id
    # When _assert_video_owner falls back to the board check, return no board so the public
    # fallback path doesn't relax permissions for the foreign video.
    mock_invoker.services.board_video_records.get_board_for_video.return_value = None

    # The route reads ``updated.board_id`` to build ``affected_boards``; a bare MagicMock
    # there would fail the response model's Pydantic validation.
    fake_updated = MagicMock()
    fake_updated.board_id = None
    mock_invoker.services.videos.update.return_value = fake_updated


def test_star_videos_from_list_skips_foreign_items_and_returns_owned(
    client: TestClient, mock_invoker: Invoker, user1_token: str
):
    """A batch star that includes a video owned by another user must keep going and return
    200 with the owned items in ``starred_videos``. Previously the route raised 403
    mid-loop: earlier videos were already mutated, but the error-shaped response carried no
    payload, so the client never invalidated caches for the successful updates.
    """
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    _setup_mixed_ownership_batch(mock_invoker, user1.user_id)

    response = client.post(
        "/api/v1/videos/star",
        json={"video_names": ["mine_a.mp4", "foreign.mp4", "mine_b.mp4"]},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert set(body["starred_videos"]) == {"mine_a.mp4", "mine_b.mp4"}
    # The service must have been asked to update the owned names but not the foreign one.
    update_calls = {call.args[0] for call in mock_invoker.services.videos.update.call_args_list}
    assert update_calls == {"mine_a.mp4", "mine_b.mp4"}


def test_unstar_videos_from_list_skips_foreign_items_and_returns_owned(
    client: TestClient, mock_invoker: Invoker, user1_token: str
):
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    _setup_mixed_ownership_batch(mock_invoker, user1.user_id)

    response = client.post(
        "/api/v1/videos/unstar",
        json={"video_names": ["mine_a.mp4", "foreign.mp4", "mine_b.mp4"]},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert set(body["unstarred_videos"]) == {"mine_a.mp4", "mine_b.mp4"}
    update_calls = {call.args[0] for call in mock_invoker.services.videos.update.call_args_list}
    assert update_calls == {"mine_a.mp4", "mine_b.mp4"}


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


def test_upload_video_rejects_non_mp4_container_with_spoofed_mime(
    client: TestClient, mock_invoker: Invoker, user1_token: str
):
    mock_invoker.services.videos.create.side_effect = AssertionError("non-MP4 payload reached video creation")
    with patch("invokeai.app.api.routers.videos._probe_decodable_video", return_value=(64, 64, 1.0, 8.0)):
        response = client.post(
            "/api/v1/videos/upload",
            params={"video_category": "general", "is_intermediate": False},
            files={"file": ("spoofed.mp4", b"\x1aE\xdf\xa3webm payload", "video/mp4")},
            headers={"Authorization": f"Bearer {user1_token}"},
        )

    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    mock_invoker.services.videos.create.assert_not_called()


def _uploaded_video_dto() -> VideoDTO:
    return VideoDTO.model_validate(
        {
            "video_name": "uploaded.mp4",
            "video_origin": "external",
            "video_category": "general",
            "width": 64,
            "height": 64,
            "duration": 1.0,
            "fps": 8.0,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "is_intermediate": False,
            "starred": False,
            "has_workflow": False,
            "video_subfolder": "",
            "video_url": "/api/v1/videos/i/uploaded.mp4/full",
            "thumbnail_url": "/api/v1/videos/i/uploaded.mp4/thumbnail",
        }
    )


@pytest.mark.parametrize("metadata", ["not json", '["not", "an", "object"]'])
def test_upload_video_rejects_malformed_metadata_before_create(
    client: TestClient, mock_invoker: Invoker, user1_token: str, metadata: str
):
    mock_invoker.services.videos.create.return_value = _uploaded_video_dto()
    mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 12

    with patch("invokeai.app.api.routers.videos._probe_decodable_video", return_value=(64, 64, 1.0, 8.0)):
        response = client.post(
            "/api/v1/videos/upload",
            params={"video_category": "general", "is_intermediate": False},
            files={"file": ("video.mp4", mp4, "video/mp4")},
            data={"metadata": metadata},
            headers={"Authorization": f"Bearer {user1_token}"},
        )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    mock_invoker.services.videos.create.assert_not_called()


def test_upload_video_accepts_object_metadata(client: TestClient, mock_invoker: Invoker, user1_token: str):
    mock_invoker.services.videos.create.return_value = _uploaded_video_dto()
    mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 12
    metadata = '{"seed": 123}'

    with patch("invokeai.app.api.routers.videos._probe_decodable_video", return_value=(64, 64, 1.0, 8.0)):
        response = client.post(
            "/api/v1/videos/upload",
            params={"video_category": "general", "is_intermediate": False},
            files={"file": ("video.mp4", mp4, "video/mp4")},
            data={"metadata": metadata},
            headers={"Authorization": f"Bearer {user1_token}"},
        )

    assert response.status_code == status.HTTP_201_CREATED
    assert mock_invoker.services.videos.create.call_args.kwargs["metadata"] == metadata


def test_mp4_validation_allows_boxes_before_file_type(tmp_path: Path) -> None:
    path = tmp_path / "valid.mp4"
    path.write_bytes(b"\x00\x00\x00\x08free" + b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 12)

    assert _is_mp4_file(path)


def test_mp4_validation_rejects_quicktime_brand(tmp_path: Path) -> None:
    path = tmp_path / "quicktime.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypqt  " + b"\x00" * 12)

    assert not _is_mp4_file(path)


@pytest.mark.parametrize("suffix,thumbnail", [("full", False), ("thumbnail", True)])
def test_video_media_requires_auth_in_multiuser_mode(
    enable_multiuser_for_videos: Any,
    client: TestClient,
    mock_invoker: Invoker,
    tmp_path: Path,
    suffix: str,
    thumbnail: bool,
):
    client.cookies.clear()
    media_path = tmp_path / ("video.webp" if thumbnail else "video.mp4")
    media_path.write_bytes(b"media")
    mock_invoker.services.videos.get_path.return_value = str(media_path)

    response = client.get(f"/api/v1/videos/i/private.mp4/{suffix}")

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    mock_invoker.services.videos.get_path.assert_not_called()


@pytest.mark.parametrize("suffix,thumbnail", [("full", False), ("thumbnail", True)])
def test_video_owner_can_load_media_with_login_cookie(
    client: TestClient,
    mock_invoker: Invoker,
    user1_token: str,
    tmp_path: Path,
    suffix: str,
    thumbnail: bool,
):
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    mock_invoker.services.video_records.get_user_id.return_value = user1.user_id
    media_path = tmp_path / ("video.webp" if thumbnail else "video.mp4")
    media_path.write_bytes(b"media")
    mock_invoker.services.videos.get_path.return_value = str(media_path)
    client.cookies.clear()
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "user1@test.com", "password": "TestPass123", "remember_me": False},
    )
    assert login.status_code == status.HTTP_200_OK

    response = client.get(f"/api/v1/videos/i/private.mp4/{suffix}")

    assert response.status_code == status.HTTP_200_OK
    assert response.headers["cache-control"] == "private, no-store"
    mock_invoker.services.videos.get_path.assert_called_once_with("private.mp4", thumbnail=thumbnail)


def test_foreign_user_cannot_load_private_video_media(
    client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
):
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    mock_invoker.services.video_records.get_user_id.return_value = user1.user_id
    mock_invoker.services.board_video_records.get_board_for_video.return_value = None

    response = client.get(
        "/api/v1/videos/i/private.mp4/full",
        headers={"Authorization": f"Bearer {user2_token}"},
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN
    mock_invoker.services.videos.get_path.assert_not_called()


# ---------------------------------------------------------------------------
# GET /videos/i/{video_name}/thumbnail must return 404 when the thumbnail file
# is missing on disk (JPPhoto PR #9163 follow-up). Video saves are allowed
# without a thumbnail in video_files_disk.save, so this is reachable.
# ---------------------------------------------------------------------------


def test_get_video_thumbnail_missing_file_returns_404(
    client: TestClient,
    mock_invoker: Invoker,
    user1_token: str,
    tmp_path: Path,
):
    """If videos.get_path resolves successfully but the file doesn't exist, the route must
    return 404 up front. Previously it returned FileResponse and the missing-file error was
    raised by Starlette *after* the route's try/except, so callers saw a 500-class failure
    instead of the documented 404.
    """
    missing_path = tmp_path / "does_not_exist.webp"
    assert not missing_path.exists()
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    mock_invoker.services.video_records.get_user_id.return_value = user1.user_id
    mock_invoker.services.videos.get_path.return_value = str(missing_path)

    response = client.get(
        "/api/v1/videos/i/some_video.mp4/thumbnail",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND
    mock_invoker.services.videos.get_path.assert_called_once_with("some_video.mp4", thumbnail=True)


# ---------------------------------------------------------------------------
# DELETE /videos/board: stranded-contributor recovery (JPPhoto PR #9163 May-22 follow-up)
#
# Scenario: user2 uploads to user1's Public board, user1 later flips the board to
# Shared/Private. Without a fallback path, neither the uploader nor the board owner
# can detach the video — _assert_video_direct_owner rejects user1, and
# _assert_board_write_access rejects user2 because the board is no longer Public.
# The route must accept removal from either the video owner OR a user with write
# access to the destination board (mirrors remove_image_from_board).
# ---------------------------------------------------------------------------


def test_remove_video_from_board_succeeds_for_video_owner_on_foreign_private_board(
    client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
):
    """user2 owns the video; the video sits on user1's now-private board. user2 must still
    be able to detach it via its direct ownership."""
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    user2 = mock_invoker.services.users.get_by_email("user2@test.com")
    assert user1 is not None and user2 is not None

    # user2 owns the video.
    mock_invoker.services.video_records.get_user_id.return_value = user2.user_id
    # The video lives on user1's now-private board.
    mock_invoker.services.board_video_records.get_board_for_video.return_value = "user1-private-board"

    response = client.request(
        "DELETE",
        "/api/v1/videos/board",
        json={"video_name": "uploaded.mp4"},
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert response.status_code == status.HTTP_200_OK
    mock_invoker.services.board_video_records.remove_video_from_board.assert_called_once_with(video_name="uploaded.mp4")


def test_remove_video_from_board_succeeds_for_board_owner_of_non_owned_video(
    client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
):
    """user1 owns the board; user2 owns the video sitting on it. user1 must be able to
    detach the foreign video from their board even though they are not the video owner."""
    from invokeai.app.services.board_records.board_records_common import BoardVisibility

    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    user2 = mock_invoker.services.users.get_by_email("user2@test.com")
    assert user1 is not None and user2 is not None

    mock_invoker.services.video_records.get_user_id.return_value = user2.user_id
    mock_invoker.services.board_video_records.get_board_for_video.return_value = "user1-board"

    # _assert_board_write_access reads the board DTO to check ownership/visibility.
    fake_board = MagicMock()
    fake_board.user_id = user1.user_id
    fake_board.board_visibility = BoardVisibility.Private
    with patch.object(mock_invoker.services.boards, "get_dto", return_value=fake_board):
        response = client.request(
            "DELETE",
            "/api/v1/videos/board",
            json={"video_name": "stranded.mp4"},
            headers={"Authorization": f"Bearer {user1_token}"},
        )
    assert response.status_code == status.HTTP_200_OK
    mock_invoker.services.board_video_records.remove_video_from_board.assert_called_once_with(video_name="stranded.mp4")


def test_remove_video_from_board_rejects_third_party(
    client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str
):
    """A user who is neither the video owner nor a board write-access holder must be
    rejected — the relaxed path is a stranded-contributor escape hatch, not an open door."""
    from invokeai.app.services.board_records.board_records_common import BoardVisibility

    admin = mock_invoker.services.users.get_by_email("admin@test.com")
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert admin is not None and user1 is not None

    # Video is owned by admin; board is owned by user1 and is Private.
    mock_invoker.services.video_records.get_user_id.return_value = admin.user_id
    mock_invoker.services.board_video_records.get_board_for_video.return_value = "user1-board"

    fake_board = MagicMock()
    fake_board.user_id = user1.user_id
    fake_board.board_visibility = BoardVisibility.Private

    # user2 has no claim to either resource.
    with patch.object(mock_invoker.services.boards, "get_dto", return_value=fake_board):
        response = client.request(
            "DELETE",
            "/api/v1/videos/board",
            json={"video_name": "not_mine.mp4"},
            headers={"Authorization": f"Bearer {user2_token}"},
        )
    assert response.status_code == status.HTTP_403_FORBIDDEN
    mock_invoker.services.board_video_records.remove_video_from_board.assert_not_called()


# ---------------------------------------------------------------------------
# GET /videos/i/{video_name}/workflow (PR #9163 review fix — generated video
# workflows were persisted but had no retrieval endpoint)
# ---------------------------------------------------------------------------


def test_get_video_workflow_returns_workflow_and_graph_for_owner(
    client: TestClient, mock_invoker: Invoker, user1_token: str
):
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    mock_invoker.services.video_records.get_user_id.return_value = user1.user_id
    mock_invoker.services.videos.get_workflow.return_value = '{"nodes": []}'
    mock_invoker.services.videos.get_graph.return_value = '{"edges": []}'

    response = client.get(
        "/api/v1/videos/i/mine_a.mp4/workflow",
        headers={"Authorization": f"Bearer {user1_token}"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"workflow": '{"nodes": []}', "graph": '{"edges": []}'}


def test_get_video_workflow_returns_nulls_when_video_has_none(
    client: TestClient, mock_invoker: Invoker, user1_token: str
):
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    mock_invoker.services.video_records.get_user_id.return_value = user1.user_id
    mock_invoker.services.videos.get_workflow.return_value = None
    mock_invoker.services.videos.get_graph.return_value = None

    response = client.get(
        "/api/v1/videos/i/mine_a.mp4/workflow",
        headers={"Authorization": f"Bearer {user1_token}"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"workflow": None, "graph": None}


def test_get_video_workflow_missing_video_returns_404(client: TestClient, mock_invoker: Invoker, user1_token: str):
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    mock_invoker.services.video_records.get_user_id.return_value = user1.user_id
    mock_invoker.services.videos.get_workflow.side_effect = Exception("video file not found")

    response = client.get(
        "/api/v1/videos/i/mine_a.mp4/workflow",
        headers={"Authorization": f"Bearer {user1_token}"},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_get_video_workflow_forbidden_for_foreign_private_video(
    client: TestClient, mock_invoker: Invoker, user2_token: str
):
    mock_invoker.services.video_records.get_user_id.return_value = "other-user-id"
    mock_invoker.services.board_video_records.get_board_for_video.return_value = None

    response = client.get(
        "/api/v1/videos/i/not_mine.mp4/workflow",
        headers={"Authorization": f"Bearer {user2_token}"},
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN
    mock_invoker.services.videos.get_workflow.assert_not_called()


def test_get_video_workflow_requires_auth(enable_multiuser_for_videos: Any, client: TestClient):
    response = client.get("/api/v1/videos/i/a.mp4/workflow")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


# ---------------------------------------------------------------------------
# DELETE /videos/uncategorized (PR #9163 review fix — "Delete All Uncategorized
# Images/Videos" previously deleted only images)
# ---------------------------------------------------------------------------


def test_delete_uncategorized_videos_deletes_only_owned(client: TestClient, mock_invoker: Invoker, user1_token: str):
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    _setup_mixed_ownership_batch(mock_invoker, user1.user_id)

    names_result = MagicMock()
    names_result.video_names = ["mine_a.mp4", "foreign.mp4", "mine_b.mp4"]
    mock_invoker.services.videos.get_video_names.return_value = names_result

    response = client.delete(
        "/api/v1/videos/uncategorized",
        headers={"Authorization": f"Bearer {user1_token}"},
    )

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert set(body["deleted_videos"]) == {"mine_a.mp4", "mine_b.mp4"}
    assert body["failed_videos"] == ["foreign.mp4"]
    # The service must be scoped to the caller's uncategorized bucket.
    assert mock_invoker.services.videos.get_video_names.call_args.kwargs["board_id"] == "none"
    assert mock_invoker.services.videos.get_video_names.call_args.kwargs["user_id"] == user1.user_id
    delete_calls = {call.args[0] for call in mock_invoker.services.videos.delete.call_args_list}
    assert delete_calls == {"mine_a.mp4", "mine_b.mp4"}


def test_delete_uncategorized_videos_requires_auth(enable_multiuser_for_videos: Any, client: TestClient):
    response = client.delete("/api/v1/videos/uncategorized")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
