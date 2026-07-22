"""Router-level tests for /api/v1/virtual_boards.

These routes already use CurrentUserOrDefault, but until now had no tests pinning
the anonymous-rejection + per-user filtering behavior.
"""

from typing import Any

from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.invoker import Invoker


def _save_image(mock_invoker: Invoker, image_name: str, user_id: str) -> None:
    mock_invoker.services.image_records.save(
        image_name=image_name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=10,
        height=10,
        has_workflow=False,
        user_id=user_id,
    )


def _save_video(mock_invoker: Invoker, video_name: str, user_id: str) -> None:
    mock_invoker.services.video_records.save(
        video_name=video_name,
        video_origin=ResourceOrigin.INTERNAL,
        video_category=ImageCategory.GENERAL,
        width=10,
        height=10,
        duration=1.0,
        fps=8.0,
        has_workflow=False,
        is_intermediate=False,
        user_id=user_id,
    )


def test_list_by_date_requires_auth(enable_multiuser: Any, client: TestClient):
    r = client.get("/api/v1/virtual_boards/by_date")
    assert r.status_code == status.HTTP_401_UNAUTHORIZED


def test_image_names_by_date_requires_auth(enable_multiuser: Any, client: TestClient):
    r = client.get("/api/v1/virtual_boards/by_date/2026-05-18/image_names")
    assert r.status_code == status.HTTP_401_UNAUTHORIZED


def test_user_sees_only_own_dates(client: TestClient, user1_token: str, user2_token: str, mock_invoker: Invoker):
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    user2 = mock_invoker.services.users.get_by_email("user2@test.com")
    assert user1 is not None and user2 is not None

    _save_image(mock_invoker, "u1-img-a.png", user1.user_id)
    _save_image(mock_invoker, "u2-img-a.png", user2.user_id)

    r1 = client.get(
        "/api/v1/virtual_boards/by_date",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r1.status_code == status.HTTP_200_OK
    user1_counts = sum(b.get("image_count", 0) for b in r1.json())

    r2 = client.get(
        "/api/v1/virtual_boards/by_date",
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert r2.status_code == status.HTTP_200_OK
    user2_counts = sum(b.get("image_count", 0) for b in r2.json())

    # Each user sees only their single image — not the other user's.
    assert user1_counts == 1
    assert user2_counts == 1


def test_admin_sees_all_dates(
    client: TestClient, admin_token: str, user1_token: str, user2_token: str, mock_invoker: Invoker
):
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    user2 = mock_invoker.services.users.get_by_email("user2@test.com")
    assert user1 is not None and user2 is not None

    _save_image(mock_invoker, "u1-shared.png", user1.user_id)
    _save_image(mock_invoker, "u2-shared.png", user2.user_id)

    r = client.get(
        "/api/v1/virtual_boards/by_date",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    total = sum(b.get("image_count", 0) for b in r.json())
    assert total >= 2  # admin sees images from both users


def test_item_names_by_date_requires_auth(enable_multiuser: Any, client: TestClient):
    r = client.get("/api/v1/virtual_boards/by_date/2026-05-18/item_names")
    assert r.status_code == status.HTTP_401_UNAUTHORIZED


def test_video_only_date_appears_as_virtual_board(client: TestClient, user1_token: str, mock_invoker: Invoker):
    """A date containing only videos must still surface as a virtual board (video_count > 0)."""
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None

    _save_video(mock_invoker, "u1-vid-only.mp4", user1.user_id)

    r = client.get(
        "/api/v1/virtual_boards/by_date",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    boards = r.json()
    assert len(boards) == 1
    assert boards[0]["image_count"] == 0
    assert boards[0]["video_count"] == 1
    assert boards[0]["cover_video_name"] == "u1-vid-only.mp4"
    assert boards[0]["cover_image_name"] is None


def test_item_names_by_date_returns_video_refs(client: TestClient, user1_token: str, mock_invoker: Invoker):
    """Selecting a virtual date must return polymorphic refs — videos included, per-user filtered."""
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None

    _save_image(mock_invoker, "u1-mixed.png", user1.user_id)
    _save_video(mock_invoker, "u1-mixed.mp4", user1.user_id)

    r = client.get(
        "/api/v1/virtual_boards/by_date",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    date = r.json()[0]["date"]

    r = client.get(
        f"/api/v1/virtual_boards/by_date/{date}/item_names",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    body = r.json()
    refs = {(item["kind"], item["name"]) for item in body["items"]}
    assert refs == {("image", "u1-mixed.png"), ("video", "u1-mixed.mp4")}
    assert body["total_count"] == 2
