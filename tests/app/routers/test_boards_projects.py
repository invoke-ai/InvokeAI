"""The HTTP contract for boards that live in projects.

A board belongs to exactly one of its owner's projects, or to the Library. Every project has an
inbox, which only the project routes may rename, archive, move or delete; its other boards are
ordinary boards that happen to be members. The storage rules live in the board records service;
what these pin is which failure is a 404 and which a 409, and that members and inboxes are told
apart on the wire.
"""

from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import insert

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.boards import shared_boards
from tests.app.routers.conftest import _auth, _create_board


def _create_project(client: TestClient, token: str, name: str = "Project") -> dict[str, Any]:
    response = client.post("/api/v1/projects/", json={"name": name, "data": {}}, headers=_auth(token))
    assert response.status_code == status.HTTP_201_CREATED
    return response.json()


def _create_member(client: TestClient, token: str, project_id: str, name: str = "Member") -> dict[str, Any]:
    response = client.post(f"/api/v1/boards/?board_name={name}&project_id={project_id}", headers=_auth(token))
    assert response.status_code == status.HTTP_201_CREATED
    return response.json()


def _get_board(client: TestClient, token: str, board_id: str) -> dict[str, Any]:
    response = client.get(f"/api/v1/boards/{board_id}", headers=_auth(token))
    assert response.status_code == status.HTTP_200_OK
    return response.json()


def _move(client: TestClient, token: str, board_id: str, project_id: str | None):
    return client.patch(f"/api/v1/boards/{board_id}", json={"project_id": project_id}, headers=_auth(token))


def test_reading_a_board_is_a_404_only_when_the_board_is_gone(
    client: TestClient, user1_token: str, mock_invoker: Invoker, monkeypatch: pytest.MonkeyPatch
):
    """Clients take a 404 here as the board being deleted and clear what names it, so a failure is never one."""
    board_id = _create_board(client, user1_token, "Kept")
    assert client.delete(f"/api/v1/boards/{board_id}", headers=_auth(user1_token)).status_code == status.HTTP_200_OK
    assert client.get(f"/api/v1/boards/{board_id}", headers=_auth(user1_token)).status_code == 404

    kept = _create_board(client, user1_token, "Kept")

    def fail(board_id: str) -> None:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(mock_invoker.services.boards, "get_dto", fail)
    unraised = TestClient(client.app, raise_server_exceptions=False)
    response = unraised.get(f"/api/v1/boards/{kept}", headers=_auth(user1_token))

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# --- creating ---------------------------------------------------------------------------------


def test_a_board_created_in_a_project_is_a_member_but_not_the_inbox(client: TestClient, user1_token: str):
    project = _create_project(client, user1_token)

    member = _create_member(client, user1_token, project["project_id"])

    assert member["project_id"] == project["project_id"]
    assert member["is_inbox"] is False
    assert member["board_visibility"] == "private"
    inbox = _get_board(client, user1_token, project["board_id"])
    assert inbox["project_id"] == project["project_id"]
    assert inbox["is_inbox"] is True


def test_a_library_board_carries_neither_membership_nor_the_inbox_flag(client: TestClient, user1_token: str):
    board = _get_board(client, user1_token, _create_board(client, user1_token, "Library+Board"))

    assert board.get("project_id") is None
    assert board["is_inbox"] is False


def test_creating_a_board_in_someone_elses_or_a_missing_project_is_a_404(
    client: TestClient, user1_token: str, user2_token: str
):
    theirs = _create_project(client, user2_token)

    for project_id in (theirs["project_id"], "no-such-project"):
        response = client.post(
            f"/api/v1/boards/?board_name=Intruder&project_id={project_id}", headers=_auth(user1_token)
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    assert all(
        b["board_name"] != "Intruder" for b in client.get("/api/v1/boards/?all=true", headers=_auth(user2_token)).json()
    )


# --- moving -----------------------------------------------------------------------------------


def test_a_board_moves_between_the_library_and_projects(client: TestClient, user1_token: str):
    first = _create_project(client, user1_token, "First")
    second = _create_project(client, user1_token, "Second")
    board_id = _create_board(client, user1_token, "Roaming")

    into_first = _move(client, user1_token, board_id, first["project_id"])
    assert into_first.status_code == status.HTTP_201_CREATED
    assert into_first.json()["project_id"] == first["project_id"]

    into_second = _move(client, user1_token, board_id, second["project_id"])
    assert into_second.json()["project_id"] == second["project_id"]

    to_library = _move(client, user1_token, board_id, None)
    assert to_library.status_code == status.HTTP_201_CREATED
    assert to_library.json().get("project_id") is None


def test_omitting_project_id_leaves_a_member_where_it_is(client: TestClient, user1_token: str):
    project = _create_project(client, user1_token)
    member = _create_member(client, user1_token, project["project_id"])

    renamed = client.patch(
        f"/api/v1/boards/{member['board_id']}", json={"board_name": "Still here"}, headers=_auth(user1_token)
    )

    assert renamed.status_code == status.HTTP_201_CREATED
    assert renamed.json()["board_name"] == "Still here"
    assert renamed.json()["project_id"] == project["project_id"]


def test_moving_into_someone_elses_project_is_a_404_and_moves_nothing(
    client: TestClient, user1_token: str, user2_token: str, admin_token: str
):
    theirs = _create_project(client, user2_token)
    board_id = _create_board(client, user1_token, "Mine")
    admins_project = _create_project(client, admin_token, "Admin's")

    assert _move(client, user1_token, board_id, theirs["project_id"]).status_code == status.HTTP_404_NOT_FOUND
    assert _move(client, user1_token, board_id, "no-such-project").status_code == status.HTTP_404_NOT_FOUND
    # An admin may edit the board, but only the board's owner's projects can hold it.
    assert _move(client, admin_token, board_id, admins_project["project_id"]).status_code == status.HTTP_404_NOT_FOUND

    assert _get_board(client, user1_token, board_id).get("project_id") is None


@pytest.mark.parametrize("changes", [{"board_name": "Renamed"}, {"archived": True}, {"board_visibility": "public"}])
def test_an_inbox_cannot_be_changed_or_moved_through_the_board_routes(
    client: TestClient, user1_token: str, admin_token: str, changes: dict[str, Any]
):
    project = _create_project(client, user1_token, "Owning project")
    other = _create_project(client, user1_token, "Other")

    for token in (user1_token, admin_token):
        for body in (changes, {"project_id": None}, {"project_id": other["project_id"]}):
            response = client.patch(f"/api/v1/boards/{project['board_id']}", json=body, headers=_auth(token))
            assert response.status_code == status.HTTP_409_CONFLICT, body

    inbox = _get_board(client, user1_token, project["board_id"])
    assert inbox["board_name"] == "Owning project"
    assert inbox["archived"] is False
    assert inbox["board_visibility"] == "private"
    assert inbox["project_id"] == project["project_id"]


@pytest.mark.parametrize("visibility", ["shared", "public"])
def test_a_non_private_board_cannot_enter_a_project(client: TestClient, user1_token: str, visibility: str):
    project = _create_project(client, user1_token)
    board_id = _create_board(client, user1_token, "Visible")
    assert (
        client.patch(
            f"/api/v1/boards/{board_id}", json={"board_visibility": visibility}, headers=_auth(user1_token)
        ).status_code
        == status.HTTP_201_CREATED
    )

    refused = _move(client, user1_token, board_id, project["project_id"])

    assert refused.status_code == status.HTTP_409_CONFLICT
    assert _get_board(client, user1_token, board_id).get("project_id") is None


def test_an_explicitly_shared_board_cannot_enter_a_project(
    client: TestClient, mock_sqlite_database: Database, user1_token: str, user2_token: str
):
    project = _create_project(client, user1_token)
    board_id = _create_board(client, user1_token, "Shared+with+two")
    recipient = client.get("/api/v1/auth/me", headers=_auth(user2_token)).json()["user_id"]
    with mock_sqlite_database.begin(write=True) as conn:
        conn.execute(insert(shared_boards).values(board_id=board_id, user_id=recipient))

    assert _move(client, user1_token, board_id, project["project_id"]).status_code == status.HTTP_409_CONFLICT


@pytest.mark.parametrize("visibility", ["shared", "public"])
def test_a_member_cannot_be_published(client: TestClient, user1_token: str, visibility: str):
    project = _create_project(client, user1_token)
    member = _create_member(client, user1_token, project["project_id"])

    refused = client.patch(
        f"/api/v1/boards/{member['board_id']}", json={"board_visibility": visibility}, headers=_auth(user1_token)
    )

    assert refused.status_code == status.HTTP_409_CONFLICT
    assert _get_board(client, user1_token, member["board_id"])["board_visibility"] == "private"


def test_publishing_and_leaving_a_project_in_one_request_is_allowed(client: TestClient, user1_token: str):
    """The rules apply to the board as it will be, not as it is."""
    project = _create_project(client, user1_token)
    member = _create_member(client, user1_token, project["project_id"])

    response = client.patch(
        f"/api/v1/boards/{member['board_id']}",
        json={"board_visibility": "public", "project_id": None},
        headers=_auth(user1_token),
    )

    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["board_visibility"] == "public"
    assert response.json().get("project_id") is None


# --- members are ordinary boards -------------------------------------------------------------


def test_a_member_can_be_renamed_archived_and_deleted(client: TestClient, user1_token: str):
    project = _create_project(client, user1_token)
    member = _create_member(client, user1_token, project["project_id"])

    changed = client.patch(
        f"/api/v1/boards/{member['board_id']}",
        json={"board_name": "Kitchen", "archived": True},
        headers=_auth(user1_token),
    )
    assert changed.status_code == status.HTTP_201_CREATED
    assert changed.json()["board_name"] == "Kitchen"
    assert changed.json()["archived"] is True

    deleted = client.delete(f"/api/v1/boards/{member['board_id']}", headers=_auth(user1_token))
    assert deleted.status_code == status.HTTP_200_OK
    assert client.get(f"/api/v1/boards/{member['board_id']}", headers=_auth(user1_token)).status_code == 404
    # The project and its inbox are untouched.
    assert _get_board(client, user1_token, project["board_id"])["is_inbox"] is True


def test_the_listing_tells_inboxes_and_members_apart(client: TestClient, user1_token: str):
    project = _create_project(client, user1_token)
    member = _create_member(client, user1_token, project["project_id"])
    loose = _create_board(client, user1_token, "Loose")

    listed = {b["board_id"]: b for b in client.get("/api/v1/boards/?all=true", headers=_auth(user1_token)).json()}

    assert listed[project["board_id"]]["is_inbox"] is True
    assert listed[project["board_id"]]["project_id"] == project["project_id"]
    assert listed[member["board_id"]]["is_inbox"] is False
    assert listed[member["board_id"]]["project_id"] == project["project_id"]
    assert listed[loose]["is_inbox"] is False
    assert listed[loose].get("project_id") is None
