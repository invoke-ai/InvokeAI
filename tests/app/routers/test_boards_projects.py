"""The HTTP contract for boards that live in projects.

A board belongs to exactly one of its owner's projects, or to the Library. Every project has an
inbox, which only the project routes may rename, archive, move or delete; its other boards are
ordinary boards that happen to be members. The storage rules live in the board records service;
what these pin is which failure is a 404 and which a 409, and that members and inboxes are told
apart on the wire.
"""

import inspect
import typing
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import status
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sqlalchemy import insert

from invokeai.app.api_app import app
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.boards import board_members
from invokeai.app.services.urls.urls_default import LocalUrlService
from invokeai.app.services.videos.videos_common import VideoDTO
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


@pytest.mark.parametrize(
    "changes",
    [{"board_name": "Renamed"}, {"archived": True}, {"board_visibility": "public"}, {"general_access": "viewer"}],
)
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
    # Clients are told up front which changes the project owns.
    assert set(inbox["allowed_actions"]) == {"read", "add_media", "manage_items", "set_cover"}
    assert inbox["board_name"] == "Owning project"
    assert inbox["archived"] is False
    assert inbox["board_visibility"] == "private"
    assert inbox["project_id"] == project["project_id"]


@pytest.mark.parametrize("general_access", ["viewer", "contributor"])
def test_a_board_keeps_its_general_access_when_it_enters_a_project(
    client: TestClient, user1_token: str, general_access: str
):
    project = _create_project(client, user1_token)
    board_id = _create_board(client, user1_token, "Visible")
    assert (
        client.patch(
            f"/api/v1/boards/{board_id}", json={"general_access": general_access}, headers=_auth(user1_token)
        ).status_code
        == status.HTTP_201_CREATED
    )

    moved = _move(client, user1_token, board_id, project["project_id"])

    assert moved.status_code == status.HTTP_201_CREATED
    board = _get_board(client, user1_token, board_id)
    assert (board["project_id"], board["general_access"]) == (project["project_id"], general_access)


def test_a_board_keeps_its_members_when_it_enters_a_project(
    client: TestClient, mock_sqlite_database: Database, user1_token: str, user2_token: str
):
    project = _create_project(client, user1_token)
    board_id = _create_board(client, user1_token, "Shared+with+two")
    recipient = client.get("/api/v1/auth/me", headers=_auth(user2_token)).json()["user_id"]
    with mock_sqlite_database.begin(write=True) as conn:
        conn.execute(insert(board_members).values(board_id=board_id, user_id=recipient, role="viewer"))

    assert _move(client, user1_token, board_id, project["project_id"]).status_code == status.HTTP_201_CREATED
    # The member still reads it; the project it is in stays the owner's to know.
    shared = _get_board(client, user2_token, board_id)
    assert (shared["my_role"], shared.get("project_id")) == ("viewer", None)

    assert _move(client, user1_token, board_id, None).status_code == status.HTTP_201_CREATED
    assert _get_board(client, user2_token, board_id)["my_role"] == "viewer"


def test_a_board_released_by_a_deleted_project_keeps_its_members(
    client: TestClient, mock_sqlite_database: Database, user1_token: str, user2_token: str
):
    project = _create_project(client, user1_token)
    member = _create_member(client, user1_token, project["project_id"])
    recipient = client.get("/api/v1/auth/me", headers=_auth(user2_token)).json()["user_id"]
    with mock_sqlite_database.begin(write=True) as conn:
        conn.execute(insert(board_members).values(board_id=member["board_id"], user_id=recipient, role="editor"))

    deleted = client.delete(f"/api/v1/projects/{project['project_id']}?boards=release", headers=_auth(user1_token))

    assert deleted.status_code == status.HTTP_204_NO_CONTENT
    assert _get_board(client, user2_token, member["board_id"])["my_role"] == "editor"


@pytest.mark.parametrize("general_access", ["viewer", "contributor"])
def test_a_member_can_be_shared(client: TestClient, user1_token: str, general_access: str):
    project = _create_project(client, user1_token)
    member = _create_member(client, user1_token, project["project_id"])

    shared = client.patch(
        f"/api/v1/boards/{member['board_id']}", json={"general_access": general_access}, headers=_auth(user1_token)
    )

    assert shared.status_code == status.HTTP_201_CREATED
    assert _get_board(client, user1_token, member["board_id"])["general_access"] == general_access


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


def test_an_images_project_is_shown_only_to_its_uploader_and_admins(
    client: TestClient,
    mock_invoker: Invoker,
    mock_sqlite_database: Database,
    user1_token: str,
    user2_token: str,
    admin_token: str,
):
    """A member of a shared board learns nothing about the project an image was made in, on the single, listed,
    by-names and starring reads alike: `project_id` is theirs to see only when it names one of their own projects.
    Confirming a foreign id here, with the 409 on creating a project under an existing id, would tell them the
    project exists."""
    mock_invoker.services.urls = LocalUrlService()
    mock_invoker.services.images.start(mock_invoker)
    project = _create_project(client, user1_token, "Mine")
    board = _create_board(client, user1_token, "Shared")
    client.patch(f"/api/v1/boards/{board}", json={"general_access": "viewer"}, headers=_auth(user1_token))
    user1 = mock_invoker.services.users.get_by_email("user1@test.com")
    assert user1 is not None
    # An editor: starring another member's item is a board role's to do, and answers with the item's DTO.
    user2_id = client.get("/api/v1/auth/me", headers=_auth(user2_token)).json()["user_id"]
    with mock_sqlite_database.begin(write=True) as conn:
        conn.execute(insert(board_members).values(board_id=board, user_id=user2_id, role="editor"))
    mock_invoker.services.image_records.save(
        image_name="made-in-a-project.png",
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=8,
        height=8,
        has_workflow=False,
        user_id=user1.user_id,
        project_id=project["project_id"],
    )
    mock_invoker.services.board_image_records.add_image_to_board(board, "made-in-a-project.png")

    def shown_to(token: str) -> tuple[str | None, ...]:
        single = client.get("/api/v1/images/i/made-in-a-project.png", headers=_auth(token))
        listed = client.get(f"/api/v1/images/?board_id={board}&offset=0&limit=10", headers=_auth(token))
        by_names = client.post(
            "/api/v1/images/images_by_names", json={"image_names": ["made-in-a-project.png"]}, headers=_auth(token)
        )
        starred = client.patch("/api/v1/images/i/made-in-a-project.png", json={"starred": True}, headers=_auth(token))
        assert (single.status_code, listed.status_code, by_names.status_code, starred.status_code) == (200,) * 4
        return (
            single.json().get("project_id"),
            listed.json()["items"][0].get("project_id"),
            by_names.json()[0].get("project_id"),
            starred.json().get("project_id"),
        )

    assert shown_to(user1_token) == (project["project_id"],) * 4
    assert shown_to(admin_token) == (project["project_id"],) * 4
    assert shown_to(user2_token) == (None,) * 4


# Routes that answer with a media DTO but need no redaction, and why. A new entry needs the same kind of reason.
_MEDIA_ROUTES_WITHOUT_REDACTION = {
    "upload_image": "the project is the caller's own: assert_project_owned refuses any other before the upload",
    "upload_video": "the project is the caller's own: assert_project_owned refuses any other before the upload",
    "recall_initial_video_upload": "the video is the caller's own upload, ingested with no project at all",
    "recall_reference_video_upload": "the video is the caller's own upload, ingested with no project at all",
    "recall_conditioning_video_upload": "the video is the caller's own upload, ingested with no project at all",
    "create_image_upload_entry": "a stub that answers 501 and returns no DTO; an implementation must redact",
}


def _media_dto_in(model: Any, seen: set[int] | None = None) -> bool:
    """Whether a route's response model carries an image or video DTO: alone, in a list or a page, or as a field of a
    wrapper model, however deep. A parametrized pydantic generic (`OffsetPaginatedResults[ImageDTO]`) is a concrete
    class, so its arguments come from pydantic; a wrapper's fields come from `model_fields`."""
    if model in (ImageDTO, VideoDTO):
        return True
    seen = set() if seen is None else seen
    if id(model) in seen:
        return False
    seen.add(id(model))
    arguments = typing.get_args(model) or getattr(model, "__pydantic_generic_metadata__", {}).get("args", ())
    if any(_media_dto_in(argument, seen) for argument in arguments):
        return True
    if isinstance(model, type) and issubclass(model, BaseModel):
        return any(_media_dto_in(field.annotation, seen) for field in model.model_fields.values())
    return False


def _redacts(endpoint: Any) -> bool:
    """Whether the endpoint calls `redact_project_ids`, itself or in a function of its own module that it names: the
    gallery recall routes answer through one placement helper."""
    named = (endpoint.__globals__.get(name) for name in endpoint.__code__.co_names)
    helpers = [
        function for function in named if inspect.isfunction(function) and function.__module__ == endpoint.__module__
    ]
    return any("redact_project_ids(" in inspect.getsource(function) for function in (endpoint, *helpers))


def _api_routes(routes: Any) -> Iterator[APIRoute]:
    """Every API route of the app. Included routers are mounted lazily and keep their routes on `original_router`."""
    for route in routes:
        if isinstance(route, APIRoute):
            yield route
        else:
            inner = getattr(route, "original_router", None) or getattr(route, "routes", None)
            if inner is not None:
                yield from _api_routes(getattr(inner, "routes", inner))


@pytest.mark.platform_independent
def test_every_route_answering_a_media_dto_redacts_its_project() -> None:
    """The next route that returns an image or video DTO must redact `project_id` too, or say here why not."""
    media_routes = {
        route.endpoint.__name__: route.endpoint
        for route in _api_routes(app.routes)
        if _media_dto_in(route.response_model)
    }
    assert set(media_routes) >= {
        "get_image_dto",
        "update_image",
        "list_image_dtos",
        "get_images_by_names",
        "get_video_dto",
        "update_video",
        "list_video_dtos",
        "recall_initial_video",
        "recall_reference_video",
        "recall_conditioning_video",
    }, "the route enumeration lost routes it is meant to guard"

    unredacted = sorted(
        name
        for name, endpoint in media_routes.items()
        if name not in _MEDIA_ROUTES_WITHOUT_REDACTION and not _redacts(endpoint)
    )
    assert unredacted == []
    assert set(_MEDIA_ROUTES_WITHOUT_REDACTION) <= set(media_routes), "an exemption names a route that is gone"
