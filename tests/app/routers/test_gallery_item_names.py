"""Router-level tests for GET /api/v1/gallery/item_names.

This endpoint replaces two deprecated ones — `/gallery/items/names` and
`/virtual_boards/by_date/{date}/item_names` — with a flat name list. The deprecated shape
wrapped every name in an object carrying a `kind` discriminator, which cost one model per row
(~800ms on a 200k-item library) for a value callers derive from the file extension.

What matters here is that the replacement is faithful: same order, same filtering, same
per-user isolation, and videos still present. Where possible that is asserted *against* the
deprecated endpoint rather than against hardcoded expectations, so the two cannot drift while
both are still served.
"""

from typing import Any

from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.invoker import Invoker

ITEM_NAMES_URL = "/api/v1/gallery/item_names"
DEPRECATED_URL = "/api/v1/gallery/items/names"


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


def test_requires_auth_in_multiuser_mode(enable_multiuser: Any, client: TestClient):
    r = client.get(ITEM_NAMES_URL)
    assert r.status_code == status.HTTP_401_UNAUTHORIZED


def test_returns_flat_names_including_videos(client: TestClient, user1_token: str, mock_invoker: Invoker):
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None

    _save_image(mock_invoker, "img-a.png", user1.user_id)
    _save_video(mock_invoker, "vid-a.mp4", user1.user_id)

    r = client.get(ITEM_NAMES_URL, headers={"Authorization": f"Bearer {user1_token}"})
    assert r.status_code == status.HTTP_200_OK
    body = r.json()

    assert sorted(body["item_names"]) == ["img-a.png", "vid-a.mp4"]
    assert body["total_count"] == 2
    # The names are plain strings — no per-item object wrapper.
    assert all(isinstance(name, str) for name in body["item_names"])


def test_matches_the_deprecated_endpoint_order(client: TestClient, user1_token: str, mock_invoker: Invoker):
    """The replacement must not reorder anything, or virtualized selection silently breaks."""
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None

    for i in range(5):
        _save_image(mock_invoker, f"img-{i}.png", user1.user_id)
    _save_video(mock_invoker, "vid-mid.mp4", user1.user_id)

    headers = {"Authorization": f"Bearer {user1_token}"}
    new = client.get(ITEM_NAMES_URL, headers=headers)
    old = client.get(DEPRECATED_URL, headers=headers)
    assert new.status_code == status.HTTP_200_OK
    assert old.status_code == status.HTTP_200_OK

    assert new.json()["item_names"] == [item["name"] for item in old.json()["items"]]
    assert new.json()["total_count"] == old.json()["total_count"]
    assert new.json()["starred_count"] == old.json()["starred_count"]


def test_created_date_matches_the_deprecated_virtual_board_endpoint(
    client: TestClient, user1_token: str, mock_invoker: Invoker
):
    """`created_date` is the replacement for the by-date virtual-board route."""
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None

    _save_image(mock_invoker, "dated-img.png", user1.user_id)
    _save_video(mock_invoker, "dated-vid.mp4", user1.user_id)

    headers = {"Authorization": f"Bearer {user1_token}"}
    # Read the date the records actually landed on rather than assuming today's date, so the
    # test cannot fail when it runs across a midnight boundary.
    dates = client.get("/api/v1/virtual_boards/by_date", headers=headers).json()
    assert len(dates) == 1
    date = dates[0]["date"]

    new = client.get(ITEM_NAMES_URL, params={"created_date": date, "is_intermediate": "false"}, headers=headers)
    old = client.get(f"/api/v1/virtual_boards/by_date/{date}/item_names", headers=headers)
    assert new.status_code == status.HTTP_200_OK
    assert old.status_code == status.HTTP_200_OK

    assert new.json()["item_names"] == [item["name"] for item in old.json()["items"]]
    assert "dated-vid.mp4" in new.json()["item_names"]


def test_created_date_excludes_other_dates(client: TestClient, user1_token: str, mock_invoker: Invoker):
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None

    _save_image(mock_invoker, "dated-img.png", user1.user_id)

    r = client.get(
        ITEM_NAMES_URL,
        params={"created_date": "1999-01-01", "is_intermediate": "false"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert r.status_code == status.HTTP_200_OK
    assert r.json()["item_names"] == []
    assert r.json()["total_count"] == 0


def test_non_admin_sees_only_own_items(client: TestClient, user1_token: str, user2_token: str, mock_invoker: Invoker):
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    user2 = mock_invoker.services.users.get_by_email("user2@test.com")
    assert user1 is not None and user2 is not None

    _save_image(mock_invoker, "u1-only.png", user1.user_id)
    _save_image(mock_invoker, "u2-only.png", user2.user_id)

    r = client.get(ITEM_NAMES_URL, headers={"Authorization": f"Bearer {user1_token}"})
    assert r.status_code == status.HTTP_200_OK
    assert r.json()["item_names"] == ["u1-only.png"]


def test_marked_deprecated_in_the_openapi_schema(client: TestClient):
    """External integrations rely on the old routes, so they stay served — but signposted."""
    schema = client.get("/openapi.json").json()

    assert schema["paths"][ITEM_NAMES_URL]["get"].get("deprecated") is not True
    for path in (
        DEPRECATED_URL,
        "/api/v1/images/names",
        "/api/v1/videos/names",
        "/api/v1/virtual_boards/by_date/{date}/image_names",
        "/api/v1/virtual_boards/by_date/{date}/item_names",
    ):
        assert schema["paths"][path]["get"]["deprecated"] is True, f"{path} should be marked deprecated"
