"""Router-level tests for projects, focused on the board every project owns.

The storage-level rules live in `tests/app/services/project_records/test_project_records.py`; what
these pin is the HTTP contract on top of them — which failure is a 404 and which is a 409, and that
a board handed to a project really does leave the generic board routes' reach.
"""

from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import insert

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.project_records import project_records_default
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.projects import project_workspaces
from tests.app.routers.conftest import _auth, _create_board
from tests.fixtures.sqlite_database import sqlite_cursor


def _create_project(client: TestClient, token: str, **body: Any):
    return client.post("/api/v1/projects/", json={"name": "Project", "data": {}, **body}, headers=_auth(token))


def test_creating_a_project_creates_and_returns_its_board(client: TestClient, user1_token: str):
    response = _create_project(client, user1_token, name="Fresh")

    assert response.status_code == status.HTTP_201_CREATED
    board_id = response.json()["board_id"]
    assert board_id
    assert response.json()["workspace"] is None

    board = client.get(f"/api/v1/boards/{board_id}", headers=_auth(user1_token))
    assert board.status_code == status.HTTP_200_OK
    assert board.json()["board_name"] == "Fresh"
    assert board.json()["board_visibility"] == "private"


def test_oversized_project_documents_return_a_structured_413(
    client: TestClient, user1_token: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(project_records_default, "PROJECT_DOCUMENT_MAX_BYTES", 16)

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
    monkeypatch.setattr(project_records_default, "PROJECT_DOCUMENT_MAX_BYTES", 16)

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


def test_deleting_a_project_leaves_its_media_uncategorized(
    client: TestClient, mock_invoker: Invoker, mock_sqlite_database: Database, user1_token: str
):
    created = _create_project(client, user1_token).json()
    with sqlite_cursor(mock_sqlite_database) as cursor:
        cursor.execute(
            "INSERT INTO images (image_name, image_origin, image_category, width, height)"
            " VALUES ('kept.png', 'internal', 'general', 64, 64);"
        )
        cursor.execute(
            "INSERT INTO board_images (board_id, image_name) VALUES (?, 'kept.png');", (created["board_id"],)
        )

    client.delete(f"/api/v1/projects/{created['project_id']}", headers=_auth(user1_token))

    with sqlite_cursor(mock_sqlite_database) as cursor:
        cursor.execute("SELECT COUNT(*) FROM images WHERE image_name = 'kept.png';")
        assert cursor.fetchone()[0] == 1
        cursor.execute("SELECT COUNT(*) FROM board_images WHERE image_name = 'kept.png';")
        assert cursor.fetchone()[0] == 0


def test_the_board_snapshot_lists_the_projects_visible_media(
    client: TestClient, mock_invoker: Invoker, mock_sqlite_database: Database, user1_token: str
):
    created = _create_project(client, user1_token).json()
    with sqlite_cursor(mock_sqlite_database) as cursor:
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
        "boards": [
            {
                "archived": False,
                "board_id": created["board_id"],
                "is_inbox": True,
                "items": [
                    {"category": "control", "kind": "image", "name": "asset.png", "starred": False},
                    {"category": "general", "kind": "image", "name": "shown.png", "starred": False},
                ],
                "name": "Project",
            }
        ]
    }


def test_the_board_snapshot_of_an_empty_project_is_empty(client: TestClient, user1_token: str):
    created = _create_project(client, user1_token).json()

    response = client.get(f"/api/v1/projects/{created['project_id']}/board-snapshot", headers=_auth(user1_token))

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "boards": [
            {"archived": False, "board_id": created["board_id"], "is_inbox": True, "items": [], "name": "Project"}
        ]
    }


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


def test_admins_cannot_open_someone_elses_project(client: TestClient, user1_token: str, admin_token: str):
    project_id = _create_project(client, user1_token).json()["project_id"]

    assert client.get(f"/api/v1/projects/{project_id}", headers=_auth(admin_token)).status_code == (
        status.HTTP_404_NOT_FOUND
    )


def test_a_project_id_another_account_holds_is_a_409_that_changes_nothing(
    client: TestClient, user1_token: str, user2_token: str
):
    """Project ids are global. Taking one that is already held is refused as a conflict; the holder's project and
    the refused caller's boards are as they were."""
    held = _create_project(client, user1_token, name="Held").json()
    adoptable = _create_board(client, user2_token, "Imported")
    boards_before = client.get("/api/v1/boards/?all=true", headers=_auth(user2_token)).json()

    for claim in ({}, {"board_id": adoptable}):
        refused = _create_project(client, user2_token, name="Taken", project_id=held["project_id"], **claim)
        assert refused.status_code == status.HTTP_409_CONFLICT, claim

    assert client.get(f"/api/v1/projects/{held['project_id']}", headers=_auth(user2_token)).status_code == (
        status.HTTP_404_NOT_FOUND
    )
    kept = client.get(f"/api/v1/projects/{held['project_id']}", headers=_auth(user1_token)).json()
    assert (kept["name"], kept["revision"]) == ("Held", held["revision"])
    assert client.get("/api/v1/boards/?all=true", headers=_auth(user2_token)).json() == boards_before


def _create_board_in_project(client: TestClient, token: str, project_id: str, name: str) -> str:
    response = client.post(f"/api/v1/boards/?board_name={name}&project_id={project_id}", headers=_auth(token))
    assert response.status_code == status.HTTP_201_CREATED
    return response.json()["board_id"]


def test_deleting_a_project_releases_its_other_boards_to_the_library(client: TestClient, user1_token: str):
    created = _create_project(client, user1_token).json()
    member = _create_board_in_project(client, user1_token, created["project_id"], "Member")

    deleted = client.delete(f"/api/v1/projects/{created['project_id']}", headers=_auth(user1_token))

    assert deleted.status_code == status.HTTP_204_NO_CONTENT
    assert client.get(f"/api/v1/boards/{created['board_id']}", headers=_auth(user1_token)).status_code == 404
    released = client.get(f"/api/v1/boards/{member}", headers=_auth(user1_token)).json()
    assert released.get("project_id") is None
    assert released["is_inbox"] is False


def test_deleting_a_project_can_delete_its_other_boards(client: TestClient, user1_token: str):
    created = _create_project(client, user1_token).json()
    member = _create_board_in_project(client, user1_token, created["project_id"], "Member")

    deleted = client.delete(f"/api/v1/projects/{created['project_id']}?boards=delete", headers=_auth(user1_token))

    assert deleted.status_code == status.HTTP_204_NO_CONTENT
    assert client.get(f"/api/v1/boards/{member}", headers=_auth(user1_token)).status_code == 404


def test_an_unknown_boards_mode_is_rejected_before_anything_is_deleted(client: TestClient, user1_token: str):
    created = _create_project(client, user1_token).json()

    refused = client.delete(f"/api/v1/projects/{created['project_id']}?boards=burn", headers=_auth(user1_token))

    assert refused.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert client.get(f"/api/v1/projects/{created['project_id']}", headers=_auth(user1_token)).status_code == 200


def test_the_board_snapshot_lists_member_boards_after_the_inbox(client: TestClient, user1_token: str):
    created = _create_project(client, user1_token).json()
    member = _create_board_in_project(client, user1_token, created["project_id"], "Member")

    response = client.get(f"/api/v1/projects/{created['project_id']}/board-snapshot", headers=_auth(user1_token))

    assert [(board["board_id"], board["is_inbox"], board["name"]) for board in response.json()["boards"]] == [
        (created["board_id"], True, "Project"),
        (member, False, "Member"),
    ]


# region workspaces


def _split_project(client: TestClient, token: str) -> dict[str, Any]:
    response = _create_project(
        client, token, data={"canvas": {}}, minimum_document_schema_version=4, max_document_schema_version=4
    )
    assert response.status_code == status.HTTP_201_CREATED
    return response.json()


def _put_workspace(client: TestClient, token: str, project_id: str, expected: int | None, data: dict[str, Any]):
    return client.put(
        f"/api/v1/projects/{project_id}/workspace",
        json={"data": data, "expected_revision": expected, "max_document_schema_version": 4},
        headers=_auth(token),
    )


def test_a_project_is_read_with_the_callers_workspace(client: TestClient, user1_token: str):
    project = _split_project(client, user1_token)
    saved = _put_workspace(client, user1_token, project["project_id"], None, {"layout": "mine"})

    fetched = client.get(
        f"/api/v1/projects/{project['project_id']}?max_document_schema_version=4", headers=_auth(user1_token)
    ).json()

    assert saved.status_code == status.HTTP_200_OK
    assert fetched["data"] == {"canvas": {}}
    assert fetched["workspace"] == saved.json()
    assert (saved.json()["revision"], saved.json()["data"]) == (1, {"layout": "mine"})


def test_a_stale_workspace_save_is_a_409_with_the_current_revision(client: TestClient, user1_token: str):
    project_id = _split_project(client, user1_token)["project_id"]
    _put_workspace(client, user1_token, project_id, None, {"v": 1})

    stale = _put_workspace(client, user1_token, project_id, None, {"v": "lost"})

    assert stale.status_code == status.HTTP_409_CONFLICT
    assert stale.json()["detail"]["current_revision"] == 1


def test_a_client_that_predates_the_split_gets_a_412_naming_the_document_schema(client: TestClient, user1_token: str):
    project_id = _split_project(client, user1_token)["project_id"]

    refused = client.get(f"/api/v1/projects/{project_id}", headers=_auth(user1_token))

    assert refused.status_code == status.HTTP_412_PRECONDITION_FAILED
    assert refused.json()["detail"]["code"] == "document_schema_unsupported"
    assert (
        refused.json()["detail"]["minimum_document_schema_version"],
        refused.json()["detail"]["max_document_schema_version"],
    ) == (4, 3)


def test_a_save_from_a_client_that_predates_the_split_is_a_412(client: TestClient, user1_token: str):
    """Both halves: the content save today's client sends, with no `max_document_schema_version`, and a workspace
    save that declares too old a schema."""
    project_id = _split_project(client, user1_token)["project_id"]

    content = client.put(
        f"/api/v1/projects/{project_id}",
        json={"name": "Whole", "data": {"whole": True}, "expected_revision": 1},
        headers=_auth(user1_token),
    )
    workspace = client.put(
        f"/api/v1/projects/{project_id}/workspace",
        json={"data": {}, "expected_revision": None, "max_document_schema_version": 3},
        headers=_auth(user1_token),
    )

    for refused in (content, workspace):
        assert refused.status_code == status.HTTP_412_PRECONDITION_FAILED
        assert refused.json()["detail"]["code"] == "document_schema_unsupported"


def test_lowering_the_document_schema_floor_is_a_400(client: TestClient, user1_token: str):
    project_id = _split_project(client, user1_token)["project_id"]

    lowered = client.put(
        f"/api/v1/projects/{project_id}",
        json={
            "name": "Project",
            "data": {},
            "expected_revision": 1,
            "minimum_document_schema_version": 3,
            "max_document_schema_version": 4,
        },
        headers=_auth(user1_token),
    )

    assert lowered.status_code == status.HTTP_400_BAD_REQUEST
    assert lowered.json()["detail"]["code"] == "document_schema_downgrade"


def test_no_one_else_can_save_a_workspace_in_a_project(client: TestClient, user1_token: str, user2_token: str):
    project_id = _split_project(client, user1_token)["project_id"]

    assert _put_workspace(client, user2_token, project_id, None, {}).status_code == status.HTTP_404_NOT_FOUND


def test_a_workspace_save_below_the_canvas_floor_is_a_412(client: TestClient, user1_token: str):
    created = _create_project(
        client,
        user1_token,
        data={"canvas": {"version": 3}},
        minimum_canvas_schema_version=3,
        max_canvas_schema_version=3,
        minimum_document_schema_version=4,
        max_document_schema_version=4,
    ).json()

    # `_put_workspace` names no canvas schema, so the client is taken for a v2 one.
    refused = _put_workspace(client, user1_token, created["project_id"], None, {"stagingArea": {}})
    assert refused.status_code == status.HTTP_412_PRECONDITION_FAILED
    assert refused.json()["detail"]["code"] == "canvas_schema_unsupported"

    saved = client.put(
        f"/api/v1/projects/{created['project_id']}/workspace",
        json={
            "data": {"stagingArea": {}},
            "expected_revision": None,
            "max_document_schema_version": 4,
            "max_canvas_schema_version": 3,
        },
        headers=_auth(user1_token),
    )
    assert saved.status_code == status.HTTP_200_OK
    assert saved.json()["revision"] == 1


def _split_whole_project(client: TestClient, token: str, project_id: str, workspace: dict[str, Any]):
    """The save that splits a whole project: content, raised floor and the caller's workspace in one request."""
    return client.put(
        f"/api/v1/projects/{project_id}",
        json={
            "name": "Project",
            "data": {"canvas": {}},
            "expected_revision": 1,
            "minimum_document_schema_version": 4,
            "max_document_schema_version": 4,
            "workspace": workspace,
        },
        headers=_auth(token),
    )


def test_the_split_saves_content_floor_and_workspace_in_one_request(client: TestClient, user1_token: str):
    created = _create_project(client, user1_token, data={"whole": True}).json()

    saved = _split_whole_project(
        client, user1_token, created["project_id"], {"data": {"layout": "mine"}, "expected_revision": None}
    )

    assert saved.status_code == status.HTTP_200_OK
    assert (saved.json()["revision"], saved.json()["minimum_document_schema_version"]) == (2, 4)
    assert (saved.json()["workspace"]["revision"], saved.json()["workspace"]["data"]) == (1, {"layout": "mine"})
    fetched = client.get(
        f"/api/v1/projects/{created['project_id']}?max_document_schema_version=4", headers=_auth(user1_token)
    ).json()
    assert fetched["data"] == {"canvas": {}}
    assert fetched["workspace"] == saved.json()["workspace"]


def test_a_stale_workspace_revision_in_the_split_is_a_409_that_saves_nothing(
    client: TestClient, mock_sqlite_database: Database, user1_token: str
):
    created = _create_project(client, user1_token, data={"whole": True}).json()
    # A workspace row beside the whole project, as builds that saved workspaces before the split left them; no
    # request writes one there now.
    user_id = client.get("/api/v1/auth/me", headers=_auth(user1_token)).json()["user_id"]
    with mock_sqlite_database.begin(write=True) as conn:
        conn.execute(
            insert(project_workspaces).values(project_id=created["project_id"], user_id=user_id, data='{"v":1}')
        )

    refused = _split_whole_project(
        client, user1_token, created["project_id"], {"data": {"v": "lost"}, "expected_revision": None}
    )

    assert refused.status_code == status.HTTP_409_CONFLICT
    assert refused.json()["detail"]["current_revision"] == 1
    fetched = client.get(
        f"/api/v1/projects/{created['project_id']}?max_document_schema_version=4", headers=_auth(user1_token)
    ).json()
    assert (fetched["revision"], fetched["minimum_document_schema_version"], fetched["data"]) == (
        1,
        3,
        {"whole": True},
    )
    assert fetched["workspace"]["data"] == {"v": 1}


def test_a_workspace_beside_a_whole_document_is_a_400(client: TestClient, user1_token: str):
    created = _create_project(client, user1_token, data={"whole": True}).json()

    refused = client.put(
        f"/api/v1/projects/{created['project_id']}",
        json={
            "name": "Project",
            "data": {"whole": True},
            "expected_revision": 1,
            "max_document_schema_version": 4,
            "workspace": {"data": {"layout": "mine"}, "expected_revision": None},
        },
        headers=_auth(user1_token),
    )

    assert refused.status_code == status.HTTP_400_BAD_REQUEST
    assert refused.json()["detail"]["code"] == "workspace_needs_split_document"
    assert refused.json()["detail"]["minimum_document_schema_version"] == 3
    fetched = client.get(f"/api/v1/projects/{created['project_id']}", headers=_auth(user1_token)).json()
    assert (fetched["revision"], fetched["workspace"]) == (1, None)


def test_a_workspace_save_alone_on_a_whole_document_is_a_400(client: TestClient, user1_token: str):
    created = _create_project(client, user1_token, data={"whole": True}).json()

    refused = _put_workspace(client, user1_token, created["project_id"], None, {"layout": "mine"})

    assert refused.status_code == status.HTTP_400_BAD_REQUEST
    assert refused.json()["detail"]["code"] == "workspace_needs_split_document"
    assert refused.json()["detail"]["minimum_document_schema_version"] == 3
    fetched = client.get(f"/api/v1/projects/{created['project_id']}", headers=_auth(user1_token)).json()
    assert (fetched["revision"], fetched["workspace"]) == (1, None)


def test_the_split_without_a_workspace_is_a_400(client: TestClient, user1_token: str):
    created = _create_project(client, user1_token, data={"whole": True}).json()

    refused = client.put(
        f"/api/v1/projects/{created['project_id']}",
        json={
            "name": "Project",
            "data": {"canvas": {}},
            "expected_revision": 1,
            "minimum_document_schema_version": 4,
            "max_document_schema_version": 4,
        },
        headers=_auth(user1_token),
    )

    assert refused.status_code == status.HTTP_400_BAD_REQUEST
    assert refused.json()["detail"]["code"] == "split_needs_workspace"
    assert refused.json()["detail"]["minimum_document_schema_version"] == 3
    fetched = client.get(f"/api/v1/projects/{created['project_id']}", headers=_auth(user1_token)).json()
    assert (fetched["revision"], fetched["minimum_document_schema_version"], fetched["data"]) == (1, 3, {"whole": True})


def test_a_save_without_a_workspace_answers_workspace_null(client: TestClient, user1_token: str):
    created = _create_project(client, user1_token).json()

    saved = client.put(
        f"/api/v1/projects/{created['project_id']}",
        json={"name": "Project", "data": {}, "expected_revision": 1},
        headers=_auth(user1_token),
    )

    assert saved.status_code == status.HTTP_200_OK
    assert saved.json()["workspace"] is None


def test_creating_a_project_with_a_workspace_stores_both_in_one_request(client: TestClient, user1_token: str):
    created = _create_project(
        client,
        user1_token,
        data={"canvas": {}},
        minimum_document_schema_version=4,
        max_document_schema_version=4,
        workspace={"layout": "mine"},
    )

    assert created.status_code == status.HTTP_201_CREATED
    body = created.json()
    assert (body["revision"], body["data"]) == (1, {"canvas": {}})
    assert (body["workspace"]["revision"], body["workspace"]["data"]) == (1, {"layout": "mine"})
    fetched = client.get(
        f"/api/v1/projects/{body['project_id']}?max_document_schema_version=4", headers=_auth(user1_token)
    )
    assert fetched.json() == body


def test_creating_a_whole_project_with_a_workspace_is_a_400_that_creates_nothing(client: TestClient, user1_token: str):
    refused = _create_project(
        client, user1_token, data={"whole": True}, max_document_schema_version=4, workspace={"layout": "mine"}
    )

    assert refused.status_code == status.HTTP_400_BAD_REQUEST
    assert refused.json()["detail"]["code"] == "workspace_needs_split_document"
    assert refused.json()["detail"]["minimum_document_schema_version"] == 3
    assert client.get("/api/v1/projects/", headers=_auth(user1_token)).json() == []
    assert client.get("/api/v1/boards/?all=true", headers=_auth(user1_token)).json() == []


def test_a_refused_create_with_a_workspace_leaves_its_staged_board_unclaimed(client: TestClient, user1_token: str):
    """An import stages its uploads on a Library board before the create: the refusal must leave that board as the
    import can still find it, neither renamed nor claimed."""
    board_id = _create_board(client, user1_token, "Staging")

    refused = _create_project(
        client,
        user1_token,
        name="Imported",
        data={"whole": True},
        max_document_schema_version=4,
        board_id=board_id,
        workspace={"layout": "mine"},
    )

    assert refused.status_code == status.HTTP_400_BAD_REQUEST
    assert refused.json()["detail"]["code"] == "workspace_needs_split_document"
    assert client.get("/api/v1/projects/", headers=_auth(user1_token)).json() == []
    board = client.get(f"/api/v1/boards/{board_id}", headers=_auth(user1_token)).json()
    assert (board["board_name"], board["project_id"]) == ("Staging", None)


# endregion
