"""`POST /api/v1/gallery/usage`: what still uses media before the gallery deletes it, per caller."""

import json
import uuid
from typing import Any, Optional

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.services.gallery import gallery_default
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import intermediates as intermediate_queries
from invokeai.app.services.shared.database.queries import media_references as media_reference_queries
from invokeai.app.services.shared.media_references import MediaReferences
from invokeai.app.services.workflow_records.workflow_records_common import (
    WorkflowCategory,
    WorkflowMeta,
    WorkflowWithoutID,
)
from tests.app.routers.conftest import _auth, _create_board
from tests.fixtures.sqlite_database import sqlite_cursor

URL = "/api/v1/gallery/usage"


def _user_id(invoker: Invoker, email: str) -> str:
    user = invoker.services.users.get_by_email(email)
    assert user is not None
    return user.user_id


@pytest.fixture
def user1(mock_invoker: Invoker, user1_token: str) -> str:
    return _user_id(mock_invoker, "user1@test.com")


@pytest.fixture
def user2(mock_invoker: Invoker, user2_token: str) -> str:
    return _user_id(mock_invoker, "user2@test.com")


def _image(invoker: Invoker, name: str, user_id: str, board_id: Optional[str] = None) -> None:
    invoker.services.image_records.save(
        image_name=name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=8,
        height=8,
        has_workflow=False,
        user_id=user_id,
    )
    if board_id is not None:
        invoker.services.board_image_records.add_image_to_board(board_id, name)


def _video(invoker: Invoker, name: str, user_id: str, board_id: Optional[str] = None) -> None:
    invoker.services.video_records.save(
        video_name=name,
        video_origin=ResourceOrigin.INTERNAL,
        video_category=ImageCategory.GENERAL,
        width=8,
        height=8,
        duration=1.0,
        fps=24.0,
        has_workflow=False,
        user_id=user_id,
    )
    if board_id is not None:
        invoker.services.board_video_records.add_video_to_board(board_id, name)


def _naming(*images: str, videos: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "images": [{"imageName": name} for name in images],
        "videos": [{"videoName": name} for name in videos],
    }


def _project(invoker: Invoker, user_id: str, name: str, content: dict, workspace: Optional[dict] = None) -> str:
    return invoker.services.project_records.create(
        user_id,
        name,
        content,
        minimum_document_schema_version=4,
        max_document_schema_version=4,
        workspace=workspace,
    ).project_id


def _workflow(invoker: Invoker, user_id: str, name: str, *images: str) -> str:
    workflow = WorkflowWithoutID(
        name=name,
        author="",
        description="",
        version="1.0.0",
        contact="",
        tags="",
        notes="",
        exposedFields=[],
        meta=WorkflowMeta(version="3.0.0", category=WorkflowCategory.User),
        nodes=[{"id": f"n{index}", "data": {"image": {"image_name": image}}} for index, image in enumerate(images)],
        edges=[],
    )
    return invoker.services.workflow_records.create(workflow, user_id=user_id).workflow_id


def _quarantined(database: Database, user_id: str, project_id: str, name: str, *images: str) -> None:
    with sqlite_cursor(database) as cursor:
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS orphaned_projects_2026_08_06 (project_id TEXT, user_id TEXT, name TEXT,"
            " data TEXT, PRIMARY KEY (user_id, project_id));"
        )
        cursor.execute("INSERT INTO orphaned_projects_2026_08_06 VALUES (?, ?, ?, '{}');", (project_id, user_id, name))
    database.queries.media_references.replace(
        owner_kind="quarantined_project",
        user_id=user_id,
        owner_id=project_id,
        references=MediaReferences(images=set(images)),
    )


def _queue_item(database: Database, user_id: Optional[str], status: str, *images: str) -> None:
    nodes = {f"n{index}": {"image": {"image_name": image}} for index, image in enumerate(images)}
    with sqlite_cursor(database) as cursor:
        cursor.execute(
            "INSERT INTO session_queue (queue_id, session, session_id, batch_id, priority, user_id, status)"
            " VALUES ('default', ?, ?, 'batch', 0, ?, ?);",
            (json.dumps({"graph": {"nodes": nodes}}), uuid.uuid4().hex, user_id, status),
        )


def _usage(client: TestClient, token: str, **body: Any) -> dict[str, Any]:
    response = client.post(URL, json=body, headers=_auth(token))
    assert response.status_code == status.HTTP_200_OK, response.text
    return response.json()


def _documents(usage: dict[str, Any]) -> list[tuple[str, Optional[str], list[str], list[str]]]:
    return [(d["kind"], d["name"], d["image_names"], d["video_names"]) for d in usage["documents"]]


NO_QUEUE = {"item_count": 0, "image_names": [], "video_names": [], "image_count": 0, "video_count": 0}
NO_OTHERS = {"document_count": 0, "queue_item_count": 0, "image_count": 0, "video_count": 0}


def test_names_every_kind_of_the_callers_own_use(
    client: TestClient, mock_invoker: Invoker, mock_sqlite_database: Database, user1_token: str, user1: str
) -> None:
    for name in ("content.png", "workspace.png", "flow.png", "canvas.png", "ref.png", "kept.png", "queued.png"):
        _image(mock_invoker, name, user1)
    for name in ("done.png", "free.png"):
        _image(mock_invoker, name, user1)
    _video(mock_invoker, "clip.mp4", user1)
    project = _project(
        mock_invoker,
        user1,
        "Poster",
        _naming("content.png", videos=("clip.mp4",)),
        workspace=_naming("workspace.png"),
    )
    workflow = _workflow(mock_invoker, user1, "Upscale", "flow.png")
    client_state = mock_invoker.services.client_state_persistence
    client_state.set_by_key(user1, "canvas", '{"layers": [{"image": {"image_name": "canvas.png"}}]}')
    client_state.set_by_key(user1, "refs", '{"refs": [{"imageName": "ref.png"}]}')
    _quarantined(mock_sqlite_database, user1, "old-project", "Before boards", "kept.png")
    _queue_item(mock_sqlite_database, user1, "pending", "queued.png")
    _queue_item(mock_sqlite_database, user1, "completed", "done.png")

    usage = _usage(
        client,
        user1_token,
        image_names=[
            "content.png",
            "workspace.png",
            "flow.png",
            "canvas.png",
            "ref.png",
            "kept.png",
            "queued.png",
            "done.png",
            "free.png",
        ],
        video_names=["clip.mp4"],
    )

    assert usage["checked_image_count"] == 9
    assert usage["checked_video_count"] == 1
    assert usage["in_use_image_count"] == 7
    assert usage["in_use_video_count"] == 1
    assert _documents(usage) == [
        ("legacy_editor", None, ["canvas.png", "ref.png"], []),
        ("project", "Poster", ["content.png", "workspace.png"], ["clip.mp4"]),
        ("quarantined_project", "Before boards", ["kept.png"], []),
        ("workflow", "Upscale", ["flow.png"], []),
    ]
    assert [d["id"] for d in usage["documents"]] == ["legacy_editor", project, "old-project", workflow]
    assert usage["documents_total"] == 4
    assert usage["queue"] == {
        "item_count": 1,
        "image_names": ["queued.png"],
        "video_names": [],
        "image_count": 1,
        "video_count": 0,
    }
    assert usage["other_accounts"] == NO_OTHERS


def test_other_accounts_use_is_counted_without_naming_it(
    client: TestClient,
    mock_invoker: Invoker,
    mock_sqlite_database: Database,
    user1_token: str,
    user1: str,
    user2: str,
) -> None:
    _image(mock_invoker, "shared.png", user1)
    _image(mock_invoker, "queued.png", user1)
    _video(mock_invoker, "clip.mp4", user1)
    own_project = _project(mock_invoker, user1, "Mine", _naming("shared.png"))
    _project(mock_invoker, user2, "Secret plans", _naming("shared.png", videos=("clip.mp4",)))
    _workflow(mock_invoker, user2, "Secret flow", "shared.png")
    # Another account's workspace in the caller's project is that account's document.
    mock_sqlite_database.queries.media_references.replace(
        owner_kind="project_workspace",
        user_id=user2,
        owner_id=own_project,
        references=MediaReferences(images={"shared.png"}),
    )
    _queue_item(mock_sqlite_database, user2, "in_progress", "queued.png")

    response = client.post(
        URL, json={"image_names": ["shared.png", "queued.png"], "video_names": ["clip.mp4"]}, headers=_auth(user1_token)
    )

    assert response.status_code == status.HTTP_200_OK
    usage = response.json()
    assert _documents(usage) == [("project", "Mine", ["shared.png"], [])]
    assert usage["queue"] == NO_QUEUE
    assert usage["other_accounts"] == {"document_count": 3, "queue_item_count": 1, "image_count": 2, "video_count": 1}
    assert usage["in_use_image_count"] == 2
    assert usage["in_use_video_count"] == 1
    assert "Secret" not in response.text
    assert user2 not in response.text


def test_queue_work_without_an_owner_counts_as_another_accounts(
    client: TestClient,
    mock_invoker: Invoker,
    mock_sqlite_database: Database,
    admin_token: str,
    user1_token: str,
    user1: str,
) -> None:
    # The queue's own access rules match an item's owner by equality, so a row stored without one is no caller's.
    _image(mock_invoker, "orphan.png", user1)
    _queue_item(mock_sqlite_database, None, "pending", "orphan.png")

    for token in (user1_token, admin_token):
        usage = _usage(client, token, image_names=["orphan.png"])

        assert usage["in_use_image_count"] == 1
        assert usage["queue"] == NO_QUEUE
        assert usage["other_accounts"] == {
            "document_count": 0,
            "queue_item_count": 1,
            "image_count": 1,
            "video_count": 0,
        }


def test_names_the_caller_may_not_delete_are_not_checked(
    client: TestClient,
    mock_invoker: Invoker,
    mock_sqlite_database: Database,
    admin_token: str,
    user1_token: str,
    user2: str,
) -> None:
    _image(mock_invoker, "theirs.png", user2)
    _video(mock_invoker, "theirs.mp4", user2)
    _project(mock_invoker, user2, "Theirs", _naming("theirs.png", "missing.png", videos=("theirs.mp4",)))
    _queue_item(mock_sqlite_database, user2, "pending", "theirs.png")

    usage = _usage(
        client, user1_token, image_names=["theirs.png", "missing.png"], video_names=["theirs.mp4", "missing.mp4"]
    )

    assert (usage["checked_image_count"], usage["checked_video_count"]) == (0, 0)
    assert (usage["in_use_image_count"], usage["in_use_video_count"]) == (0, 0)
    assert usage["documents"] == []
    assert usage["other_accounts"] == NO_OTHERS

    # An administrator may delete anyone's media, so it is checked, but the owner's documents stay counts.
    usage = _usage(client, admin_token, image_names=["theirs.png", "missing.png"], video_names=["theirs.mp4"])

    assert (usage["checked_image_count"], usage["checked_video_count"]) == (1, 1)
    assert usage["documents"] == []
    assert usage["other_accounts"] == {"document_count": 1, "queue_item_count": 1, "image_count": 1, "video_count": 1}


def test_board_scope_checks_what_deleting_the_board_would_delete(
    client: TestClient,
    mock_invoker: Invoker,
    admin_token: str,
    user1_token: str,
    user1: str,
    user2: str,
) -> None:
    board_id = _create_board(client, user1_token)
    _image(mock_invoker, "mine.png", user1, board_id)
    _video(mock_invoker, "mine.mp4", user1, board_id)
    # Another account's contribution survives the owner's board deletion, so the owner does not check it.
    _image(mock_invoker, "contributed.png", user2, board_id)
    _project(mock_invoker, user1, "Poster", _naming("mine.png", videos=("mine.mp4",)))
    _workflow(mock_invoker, user2, "Theirs", "contributed.png")

    usage = _usage(client, user1_token, board_id=board_id)

    assert (usage["checked_image_count"], usage["checked_video_count"]) == (1, 1)
    assert _documents(usage) == [("project", "Poster", ["mine.png"], ["mine.mp4"])]
    assert usage["other_accounts"] == NO_OTHERS

    # An administrator deleting someone else's board deletes everyone's media on it.
    usage = _usage(client, admin_token, board_id=board_id)

    assert (usage["checked_image_count"], usage["checked_video_count"]) == (2, 1)
    assert usage["in_use_image_count"] == 2
    assert usage["documents"] == []
    assert usage["other_accounts"] == {"document_count": 2, "queue_item_count": 0, "image_count": 2, "video_count": 1}


def test_board_scope_spanning_query_chunks_reports_every_use(
    client: TestClient,
    mock_invoker: Invoker,
    mock_sqlite_database: Database,
    monkeypatch: pytest.MonkeyPatch,
    user1_token: str,
    user1: str,
    user2: str,
) -> None:
    # Document names are read in chunks of ids per kind, active queue sessions in chunks of item ids.
    monkeypatch.setattr(media_reference_queries, "IN_CHUNK", 2)
    monkeypatch.setattr(intermediate_queries, "IN_CHUNK", 2)
    board_id = _create_board(client, user1_token)
    images = [f"board-{index}.png" for index in range(7)]
    for name in images:
        _image(mock_invoker, name, user1, board_id)
    for index, name in enumerate(images[:5]):
        _workflow(mock_invoker, user1, f"flow {index}", name)
    for index, name in enumerate(images[:3]):
        _project(mock_invoker, user1, f"project {index}", _naming(name, images[6]))
    for name in images[:5]:
        _queue_item(mock_sqlite_database, user1, "pending", name)
    _workflow(mock_invoker, user2, "theirs", images[5])

    usage = _usage(client, user1_token, board_id=board_id)

    assert usage["checked_image_count"] == 7
    assert usage["in_use_image_count"] == 7
    assert _documents(usage) == [
        *((("project", f"project {index}", sorted([images[index], images[6]]), []) for index in range(3))),
        *((("workflow", f"flow {index}", [images[index]], []) for index in range(5))),
    ]
    assert usage["documents_total"] == 8
    assert usage["queue"]["item_count"] == 5
    assert usage["queue"]["image_names"] == sorted(images[:5])
    assert usage["other_accounts"] == {"document_count": 1, "queue_item_count": 0, "image_count": 1, "video_count": 0}


def test_board_scope_follows_the_board_deletion_errors(
    client: TestClient, mock_invoker: Invoker, user1_token: str, user2_token: str, user1: str
) -> None:
    board_id = _create_board(client, user1_token)

    assert client.post(URL, json={"board_id": board_id}, headers=_auth(user2_token)).status_code == 403
    assert client.post(URL, json={"board_id": "no-such-board"}, headers=_auth(user1_token)).status_code == 404

    project = _project(mock_invoker, user1, "Poster", {})
    inbox = mock_invoker.services.project_records.get_board_id(user1, project)
    response = client.post(URL, json={"board_id": inbox}, headers=_auth(user1_token))
    assert response.status_code == status.HTTP_409_CONFLICT


@pytest.mark.parametrize(
    "body",
    [
        {"board_id": "b", "image_names": ["a.png"]},
        {"board_id": "b", "video_names": ["a.mp4"]},
        {"image_names": [""]},
        {"image_names": ["x" * 256]},
        {"image_names": [f"{index}.png" for index in range(1001)]},
        {"video_names": [f"{index}.mp4" for index in range(1001)]},
    ],
)
def test_malformed_requests_are_refused(client: TestClient, user1_token: str, body: dict[str, Any]) -> None:
    response = client.post(URL, json=body, headers=_auth(user1_token))
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, response.text


def test_lists_are_capped_and_counts_are_exact(
    client: TestClient,
    mock_invoker: Invoker,
    mock_sqlite_database: Database,
    monkeypatch: pytest.MonkeyPatch,
    user1_token: str,
    user1: str,
) -> None:
    monkeypatch.setattr(gallery_default, "MAX_USAGE_DOCUMENTS", 2)
    monkeypatch.setattr(gallery_default, "MAX_USAGE_NAMES", 2)
    names = ["a.png", "b.png", "c.png"]
    for name in names:
        _image(mock_invoker, name, user1)
    for workflow_name in ("charlie", "Bravo", "alpha"):
        _workflow(mock_invoker, user1, workflow_name, *names)
    _queue_item(mock_sqlite_database, user1, "pending", *names)

    usage = _usage(client, user1_token, image_names=names)

    assert [(d["name"], d["image_names"], d["image_count"]) for d in usage["documents"]] == [
        ("alpha", ["a.png", "b.png"], 3),
        ("Bravo", ["a.png", "b.png"], 3),
    ]
    assert usage["documents_total"] == 3
    assert usage["queue"]["image_names"] == ["a.png", "b.png"]
    assert usage["queue"]["image_count"] == 3


def test_projects_report_the_revisions_their_names_were_read_at(
    client: TestClient, mock_invoker: Invoker, user1_token: str, user1: str
) -> None:
    _image(mock_invoker, "a.png", user1)
    projects = mock_invoker.services.project_records
    bare = _project(mock_invoker, user1, "Bare", _naming("a.png"))
    worked = _project(mock_invoker, user1, "Worked", _naming("a.png"), workspace=_naming("a.png"))
    _workflow(mock_invoker, user1, "Flow", "a.png")

    def revisions() -> dict[Optional[str], tuple[Optional[int], Optional[int]]]:
        usage = _usage(client, user1_token, image_names=["a.png"])
        return {d["name"]: (d["revision"], d["workspace_revision"]) for d in usage["documents"]}

    def as_read(project_id: str) -> tuple[int, Optional[int]]:
        record = projects.get(user1, project_id, max_document_schema_version=4)
        return record.revision, record.workspace.revision if record.workspace else None

    assert revisions() == {"Bare": as_read(bare), "Worked": as_read(worked), "Flow": (None, None)}
    assert as_read(bare) == (1, None)

    projects.update(
        user1, bare, 1, "Bare", _naming("a.png"), minimum_document_schema_version=4, max_document_schema_version=4
    )
    projects.save_workspace(user1, bare, None, _naming(), max_document_schema_version=4)
    projects.save_workspace(user1, worked, 1, _naming("a.png"), max_document_schema_version=4)

    assert revisions() == {"Bare": (2, 1), "Worked": (1, 2), "Flow": (None, None)}


def test_an_empty_request_checks_nothing(client: TestClient, user1_token: str) -> None:
    assert client.post(URL, json={}).status_code == status.HTTP_401_UNAUTHORIZED

    assert _usage(client, user1_token) == {
        "checked_image_count": 0,
        "checked_video_count": 0,
        "in_use_image_count": 0,
        "in_use_video_count": 0,
        "documents": [],
        "documents_total": 0,
        "queue": NO_QUEUE,
        "other_accounts": NO_OTHERS,
    }
