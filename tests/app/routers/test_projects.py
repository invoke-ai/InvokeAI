"""Router-level tests for projects, focused on the board every project owns.

The storage-level rules live in `tests/app/services/project_records/test_project_records.py`; what
these pin is the HTTP contract on top of them — which failure is a 404 and which is a 409, and that
a board handed to a project really does leave the generic board routes' reach.
"""

from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.services.invoker import Invoker


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _create_board(client: TestClient, token: str, name: str = "Loose+Board") -> str:
    response = client.post(f"/api/v1/boards/?board_name={name}", headers=_auth(token))
    assert response.status_code == status.HTTP_201_CREATED
    return response.json()["board_id"]


def _create_project(client: TestClient, token: str, **body: Any):
    return client.post("/api/v1/projects/", json={"name": "Project", "data": {}, **body}, headers=_auth(token))


def test_creating_a_project_creates_and_returns_its_board(client: TestClient, user1_token: str):
    response = _create_project(client, user1_token, name="Fresh")

    assert response.status_code == status.HTTP_201_CREATED
    board_id = response.json()["board_id"]
    assert board_id

    board = client.get(f"/api/v1/boards/{board_id}", headers=_auth(user1_token))
    assert board.status_code == status.HTTP_200_OK
    assert board.json()["board_name"] == "Fresh"
    assert board.json()["board_visibility"] == "private"


def test_listing_projects_carries_the_board(client: TestClient, user1_token: str):
    created = _create_project(client, user1_token).json()

    listed = client.get("/api/v1/projects/", headers=_auth(user1_token))

    assert listed.status_code == status.HTTP_200_OK
    (summary,) = [p for p in listed.json() if p["project_id"] == created["project_id"]]
    assert summary["board_id"] == created["board_id"]


def test_creating_a_project_claims_the_board_it_is_given(client: TestClient, user1_token: str):
    board_id = _create_board(client, user1_token, "Staging")

    response = _create_project(client, user1_token, name="Imported", board_id=board_id)

    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["board_id"] == board_id
    # Claiming renames the board to the project.
    board = client.get(f"/api/v1/boards/{board_id}", headers=_auth(user1_token)).json()
    assert board["board_name"] == "Imported"
    assert board["project_id"] == response.json()["project_id"]


def test_claiming_a_board_that_is_missing_or_someone_elses_is_a_404(
    client: TestClient, user1_token: str, user2_token: str
):
    theirs = _create_board(client, user2_token, "Theirs")

    assert _create_project(client, user1_token, board_id="no-such-board").status_code == status.HTTP_404_NOT_FOUND
    # Someone else's board must be indistinguishable from one that does not exist.
    assert _create_project(client, user1_token, board_id=theirs).status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.parametrize("visibility", ["shared", "public"])
def test_claiming_a_non_private_board_is_a_409(client: TestClient, user1_token: str, visibility: str):
    board_id = _create_board(client, user1_token, "Visible")
    patched = client.patch(
        f"/api/v1/boards/{board_id}", json={"board_visibility": visibility}, headers=_auth(user1_token)
    )
    assert patched.status_code == status.HTTP_201_CREATED

    assert _create_project(client, user1_token, board_id=board_id).status_code == status.HTTP_409_CONFLICT


def test_a_board_can_only_be_claimed_once(client: TestClient, user1_token: str):
    board_id = _create_board(client, user1_token, "Contested")
    assert _create_project(client, user1_token, name="First", board_id=board_id).status_code == (
        status.HTTP_201_CREATED
    )

    second = _create_project(client, user1_token, name="Second", board_id=board_id)

    assert second.status_code == status.HTTP_409_CONFLICT
    board = client.get(f"/api/v1/boards/{board_id}", headers=_auth(user1_token)).json()
    assert board["board_name"] == "First"


def test_a_duplicate_project_id_is_a_409_and_leaves_no_orphan_board(client: TestClient, user1_token: str):
    assert _create_project(client, user1_token, project_id="p1").status_code == status.HTTP_201_CREATED
    before = len(client.get("/api/v1/boards/?all=true", headers=_auth(user1_token)).json())

    response = _create_project(client, user1_token, project_id="p1")

    assert response.status_code == status.HTTP_409_CONFLICT
    after = len(client.get("/api/v1/boards/?all=true", headers=_auth(user1_token)).json())
    assert after == before


def test_saving_a_project_renames_its_board_but_a_stale_save_does_not(client: TestClient, user1_token: str):
    created = _create_project(client, user1_token, name="Before").json()
    project_id, board_id = created["project_id"], created["board_id"]

    saved = client.put(
        f"/api/v1/projects/{project_id}",
        json={"name": "After", "data": {}, "expected_revision": 1},
        headers=_auth(user1_token),
    )
    assert saved.status_code == status.HTTP_200_OK
    assert client.get(f"/api/v1/boards/{board_id}", headers=_auth(user1_token)).json()["board_name"] == "After"

    stale = client.put(
        f"/api/v1/projects/{project_id}",
        json={"name": "Loser", "data": {}, "expected_revision": 1},
        headers=_auth(user1_token),
    )
    assert stale.status_code == status.HTTP_409_CONFLICT
    assert client.get(f"/api/v1/boards/{board_id}", headers=_auth(user1_token)).json()["board_name"] == "After"


def test_deleting_a_project_deletes_its_board(client: TestClient, user1_token: str):
    created = _create_project(client, user1_token).json()

    deleted = client.delete(f"/api/v1/projects/{created['project_id']}", headers=_auth(user1_token))

    assert deleted.status_code == status.HTTP_204_NO_CONTENT
    board = client.get(f"/api/v1/boards/{created['board_id']}", headers=_auth(user1_token))
    assert board.status_code == status.HTTP_404_NOT_FOUND


def test_deleting_a_project_leaves_its_media_uncategorized(
    client: TestClient, mock_invoker: Invoker, user1_token: str
):
    created = _create_project(client, user1_token).json()
    with mock_invoker.services.board_records._db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO images (image_name, image_origin, image_category, width, height)"
            " VALUES ('kept.png', 'internal', 'general', 64, 64);"
        )
        cursor.execute(
            "INSERT INTO board_images (board_id, image_name) VALUES (?, 'kept.png');", (created["board_id"],)
        )

    client.delete(f"/api/v1/projects/{created['project_id']}", headers=_auth(user1_token))

    with mock_invoker.services.board_records._db.transaction() as cursor:
        cursor.execute("SELECT COUNT(*) FROM images WHERE image_name = 'kept.png';")
        assert cursor.fetchone()[0] == 1
        cursor.execute("SELECT COUNT(*) FROM board_images WHERE image_name = 'kept.png';")
        assert cursor.fetchone()[0] == 0


def test_the_board_snapshot_lists_the_projects_visible_media(
    client: TestClient, mock_invoker: Invoker, user1_token: str
):
    created = _create_project(client, user1_token).json()
    with mock_invoker.services.board_records._db.transaction() as cursor:
        for name, category, intermediate in (
            ("shown.png", "general", False),
            ("asset.png", "control", False),
            ("canvas.png", "other", False),
            ("scratch.png", "general", True),
        ):
            cursor.execute(
                "INSERT INTO images (image_name, image_origin, image_category, width, height, is_intermediate)"
                " VALUES (?, 'internal', ?, 64, 64, ?);",
                (name, category, intermediate),
            )
            cursor.execute(
                "INSERT INTO board_images (board_id, image_name) VALUES (?, ?);", (created["board_id"], name)
            )

    response = client.get(f"/api/v1/projects/{created['project_id']}/board-snapshot", headers=_auth(user1_token))

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "items": [
            {"category": "control", "kind": "image", "name": "asset.png", "starred": False},
            {"category": "general", "kind": "image", "name": "shown.png", "starred": False},
        ]
    }


def test_the_board_snapshot_of_an_empty_project_is_empty(client: TestClient, user1_token: str):
    created = _create_project(client, user1_token).json()

    response = client.get(f"/api/v1/projects/{created['project_id']}/board-snapshot", headers=_auth(user1_token))

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"items": []}


def test_the_board_snapshot_of_a_missing_or_foreign_project_is_a_404(
    client: TestClient, user1_token: str, user2_token: str
):
    created = _create_project(client, user1_token).json()

    assert client.get("/api/v1/projects/nope/board-snapshot", headers=_auth(user1_token)).status_code == (
        status.HTTP_404_NOT_FOUND
    )
    assert client.get(
        f"/api/v1/projects/{created['project_id']}/board-snapshot", headers=_auth(user2_token)
    ).status_code == status.HTTP_404_NOT_FOUND


def test_projects_stay_private_to_their_owner(client: TestClient, user1_token: str, user2_token: str):
    created = _create_project(client, user1_token).json()
    project_id = created["project_id"]

    assert client.get(f"/api/v1/projects/{project_id}", headers=_auth(user2_token)).status_code == (
        status.HTTP_404_NOT_FOUND
    )
    assert client.put(
        f"/api/v1/projects/{project_id}",
        json={"name": "Stolen", "data": {}, "expected_revision": 1},
        headers=_auth(user2_token),
    ).status_code == status.HTTP_404_NOT_FOUND

    # Deleting someone else's project is a silent no-op, and must not take their board with it.
    client.delete(f"/api/v1/projects/{project_id}", headers=_auth(user2_token))
    assert client.get(f"/api/v1/projects/{project_id}", headers=_auth(user1_token)).status_code == status.HTTP_200_OK
    assert client.get(f"/api/v1/boards/{created['board_id']}", headers=_auth(user1_token)).status_code == (
        status.HTTP_200_OK
    )
