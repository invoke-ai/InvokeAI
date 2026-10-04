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
from invokeai.app.services.project_records import project_records_sqlite
from tests.app.routers.conftest import _auth, _create_board


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


def test_oversized_project_documents_return_a_structured_413(
    client: TestClient, user1_token: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(project_records_sqlite, "PROJECT_DOCUMENT_MAX_BYTES", 16)

    response = _create_project(client, user1_token, data={"value": "x" * 32})

    assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE
    assert response.json()["detail"] == {
        "actual_bytes": 44,
        "code": "project_document_too_large",
        "max_bytes": 16,
        "message": "Project document is 44 bytes; the limit is 16 bytes",
    }


def test_oversized_project_updates_return_a_structured_413(
    client: TestClient, user1_token: str, monkeypatch: pytest.MonkeyPatch
):
    created = _create_project(client, user1_token).json()
    monkeypatch.setattr(project_records_sqlite, "PROJECT_DOCUMENT_MAX_BYTES", 16)

    response = client.put(
        f"/api/v1/projects/{created['project_id']}",
        json={"data": {"value": "x" * 32}, "expected_revision": created["revision"], "name": "Too large"},
        headers=_auth(user1_token),
    )

    assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE
    assert response.json()["detail"]["code"] == "project_document_too_large"
    assert response.json()["detail"]["actual_bytes"] == 44


def test_non_finite_project_values_return_a_structured_422(client: TestClient, user1_token: str) -> None:
    response = client.post(
        "/api/v1/projects/",
        content=b'{"name":"Invalid","data":{"value":NaN}}',
        headers={**_auth(user1_token), "Content-Type": "application/json"},
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert response.json()["detail"]["code"] == "project_document_invalid"


def test_project_canvas_schema_negotiation_defaults_legacy_clients_to_v2(client: TestClient, user1_token: str):
    created = _create_project(client, user1_token)

    assert created.status_code == status.HTTP_201_CREATED
    assert created.json()["minimum_canvas_schema_version"] == 2

    fetched = client.get(f"/api/v1/projects/{created.json()['project_id']}", headers=_auth(user1_token))
    assert fetched.status_code == status.HTTP_200_OK


def test_project_document_reads_require_a_capable_canvas_client(client: TestClient, user1_token: str):
    created = _create_project(
        client,
        user1_token,
        data={"canvas": {"version": 3}},
        minimum_canvas_schema_version=3,
        max_canvas_schema_version=3,
    ).json()

    refused = client.get(f"/api/v1/projects/{created['project_id']}", headers=_auth(user1_token))
    assert refused.status_code == status.HTTP_412_PRECONDITION_FAILED
    assert refused.json()["detail"] == {
        "code": "canvas_schema_unsupported",
        "message": (
            f"Project {created['project_id']} requires canvas schema 3, but this client supports up to schema 2."
            " Update the client before opening or editing this project."
        ),
        "minimum_canvas_schema_version": 3,
        "max_canvas_schema_version": 2,
    }

    supported = client.get(
        f"/api/v1/projects/{created['project_id']}?max_canvas_schema_version=3", headers=_auth(user1_token)
    )
    assert supported.status_code == status.HTTP_200_OK
    assert supported.json()["data"] == {"canvas": {"version": 3}}


def test_a_save_atomically_raises_the_project_canvas_schema_floor(client: TestClient, user1_token: str):
    created = _create_project(client, user1_token, data={"canvas": {"version": 2}}).json()

    upgraded = client.put(
        f"/api/v1/projects/{created['project_id']}",
        json={
            "name": "Upgraded",
            "data": {"canvas": {"version": 3}},
            "expected_revision": 1,
            "minimum_canvas_schema_version": 3,
            "max_canvas_schema_version": 3,
        },
        headers=_auth(user1_token),
    )

    assert upgraded.status_code == status.HTTP_200_OK
    assert upgraded.json()["minimum_canvas_schema_version"] == 3
    assert upgraded.json()["data"] == {"canvas": {"version": 3}}
    refused = client.get(f"/api/v1/projects/{created['project_id']}", headers=_auth(user1_token))
    assert refused.status_code == status.HTTP_412_PRECONDITION_FAILED


def test_a_save_cannot_lower_the_project_canvas_schema_floor(client: TestClient, user1_token: str):
    created = _create_project(
        client,
        user1_token,
        data={"canvas": {"version": 3}},
        minimum_canvas_schema_version=3,
        max_canvas_schema_version=3,
    ).json()

    refused = client.put(
        f"/api/v1/projects/{created['project_id']}",
        json={
            "name": "Downgraded",
            "data": {"canvas": {"version": 2}},
            "expected_revision": 1,
            "minimum_canvas_schema_version": 2,
            "max_canvas_schema_version": 3,
        },
        headers=_auth(user1_token),
    )

    assert refused.status_code == status.HTTP_400_BAD_REQUEST
    assert refused.json()["detail"]["code"] == "canvas_schema_downgrade"
    preserved = client.get(
        f"/api/v1/projects/{created['project_id']}?max_canvas_schema_version=3", headers=_auth(user1_token)
    )
    assert preserved.json()["data"] == {"canvas": {"version": 3}}
    assert preserved.json()["revision"] == 1


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


def test_deleting_a_project_leaves_its_media_uncategorized(client: TestClient, mock_invoker: Invoker, user1_token: str):
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
    assert (
        client.get(f"/api/v1/projects/{created['project_id']}/board-snapshot", headers=_auth(user2_token)).status_code
        == status.HTTP_404_NOT_FOUND
    )


def test_projects_stay_private_to_their_owner(client: TestClient, user1_token: str, user2_token: str):
    created = _create_project(client, user1_token).json()
    project_id = created["project_id"]

    assert client.get(f"/api/v1/projects/{project_id}", headers=_auth(user2_token)).status_code == (
        status.HTTP_404_NOT_FOUND
    )
    assert (
        client.put(
            f"/api/v1/projects/{project_id}",
            json={"name": "Stolen", "data": {}, "expected_revision": 1},
            headers=_auth(user2_token),
        ).status_code
        == status.HTTP_404_NOT_FOUND
    )

    # Deleting someone else's project is a silent no-op, and must not take their board with it.
    client.delete(f"/api/v1/projects/{project_id}", headers=_auth(user2_token))
    assert client.get(f"/api/v1/projects/{project_id}", headers=_auth(user1_token)).status_code == status.HTTP_200_OK
    assert client.get(f"/api/v1/boards/{created['board_id']}", headers=_auth(user1_token)).status_code == (
        status.HTTP_200_OK
    )
