"""Projects become keyed by id alone: a shared id is re-keyed everywhere, and memberships no project resolves go."""

import json
import logging
from functools import partial
from pathlib import Path

import pytest
from sqlalchemy import column, insert, inspect, select, table, update

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.users import users
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import MigrationBase
from tests.fixtures.migration_state import (
    alembic_operations,
    build_application_migrations,
    migrate_to_before,
    run_migrations,
)

MIGRATION_ID = "2026_10_10_global_project_ids"
AT = "2026-01-02 03:04:05.000"

# As the application and the web client write these documents: compact, with non-ASCII text as itself.
_dumps = partial(json.dumps, ensure_ascii=False, separators=(",", ":"))

ALICE_DOCUMENT = {"id": "dup", "name": "Été ☕", "canvas": {"layers": [1]}}
BOB_DOCUMENT = {"id": "dup", "name": "dup"}
# As the web client's `JSON.stringify` writes a draft cut in the middle of an emoji: the lone surrogate as its escape,
# which `json.loads` turns into a character no UTF-8 column can hold as itself.
ALICE_DRAFT_TEXT = '{"activeProjectId":"dup","note":"\\ud83d"}'

# The tables as they were before the migration, as far as the rows need.
_boards = table(
    "boards",
    column("board_id"),
    column("board_name"),
    column("user_id"),
    column("project_id"),
    column("created_at"),
    column("updated_at"),
)
_projects = table(
    "projects",
    column("project_id"),
    column("user_id"),
    column("name"),
    column("data"),
    column("board_id"),
    column("revision"),
    column("created_at"),
    column("updated_at"),
    column("minimum_canvas_schema_version"),
)
_images = table(
    "images",
    column("image_name"),
    column("image_origin"),
    column("image_category"),
    column("width"),
    column("height"),
    column("user_id"),
    column("project_id"),
    column("created_at"),
    column("updated_at"),
)
_videos = table(
    "videos",
    column("video_name"),
    column("video_origin"),
    column("video_category"),
    column("width"),
    column("height"),
    column("user_id"),
    column("project_id"),
    column("created_at"),
    column("updated_at"),
)
_session_queue = table(
    "session_queue",
    column("batch_id"),
    column("queue_id"),
    column("session_id"),
    column("session"),
    column("user_id"),
    column("project_id"),
    column("origin"),
    column("created_at"),
    column("updated_at"),
)
_media_references = table(
    "media_references",
    column("owner_kind"),
    column("user_id"),
    column("owner_id"),
    column("media_kind"),
    column("media_name"),
)
_client_state = table("client_state", column("user_id"), column("key"), column("value"), column("updated_at"))


@pytest.fixture
def migrations(tmp_path: Path) -> list[MigrationBase]:
    return build_application_migrations(tmp_path)


@pytest.fixture
def before(empty_database: Database, migrations: list[MigrationBase]) -> Database:
    """Alice and Bob each hold a project called `dup`. Bob's is the older one although Alice sorts first and her row
    was written first; a board of Alice's names a project that went with a deleted account. Alice's client state also
    holds a draft with a lone surrogate in it."""
    migrate_to_before(empty_database, migrations, MIGRATION_ID)
    with empty_database.begin(write=True) as conn:
        conn.execute(
            insert(users),
            [
                {"user_id": "alice", "email": "alice@example.com", "display_name": "Alice", "password_hash": "x"},
                {"user_id": "bob", "email": "bob@example.com", "display_name": "Bob", "password_hash": "x"},
            ],
        )
        conn.execute(
            insert(_boards),
            [
                {
                    "board_id": board_id,
                    "board_name": name,
                    "user_id": owner,
                    "project_id": project_id,
                    "created_at": AT,
                    "updated_at": AT,
                }
                for board_id, name, owner, project_id in (
                    ("alice-inbox", "Dup", "alice", "dup"),
                    ("bob-inbox", "Dup", "bob", "dup"),
                    ("bob-member", "Refs", "bob", "dup"),
                    ("alice-stray", "Stray", "alice", "gone"),
                )
            ],
        )
        conn.execute(
            insert(_projects),
            [
                {
                    "project_id": "dup",
                    "user_id": "alice",
                    "name": "Alice's dup",
                    "data": _dumps(ALICE_DOCUMENT),
                    "board_id": "alice-inbox",
                    "revision": 7,
                    "created_at": "2026-02-01 00:00:00.000",
                    "updated_at": "2026-03-01 00:00:00.000",
                    "minimum_canvas_schema_version": 3,
                },
                {
                    "project_id": "dup",
                    "user_id": "bob",
                    "name": "Bob's dup",
                    "data": _dumps(BOB_DOCUMENT),
                    "board_id": "bob-inbox",
                    "revision": 4,
                    "created_at": "2026-01-01 00:00:00.000",
                    "updated_at": "2026-01-02 00:00:00.000",
                    "minimum_canvas_schema_version": 2,
                },
            ],
        )
        for user_id in ("alice", "bob"):
            media = {
                "width": 1,
                "height": 1,
                "user_id": user_id,
                "project_id": "dup",
                "created_at": AT,
                "updated_at": AT,
            }
            conn.execute(
                insert(_images).values(
                    image_name=f"{user_id}.png", image_origin="internal", image_category="general", **media
                )
            )
            conn.execute(
                insert(_videos).values(
                    video_name=f"{user_id}.mp4", video_origin="internal", video_category="general", **media
                )
            )
            conn.execute(
                insert(_session_queue).values(
                    batch_id="b",
                    queue_id="default",
                    session_id=f"{user_id}-session",
                    session="{}",
                    user_id=user_id,
                    project_id="dup",
                    origin="webv2:p:dup:q:1",
                    created_at=AT,
                    updated_at=AT,
                )
            )
            conn.execute(
                insert(_media_references).values(
                    owner_kind="project",
                    user_id=user_id,
                    owner_id="dup",
                    media_kind="image",
                    media_name=f"{user_id}.png",
                )
            )
            conn.execute(
                insert(_client_state),
                [
                    {
                        "user_id": user_id,
                        "key": "webv2:workbench-account",
                        "value": _dumps({"activeProjectId": "dup", "openProjectIds": ["dup", "other"]}),
                        "updated_at": AT,
                    },
                    {
                        "user_id": user_id,
                        "key": "webv2:project-covers",
                        "value": _dumps({"coverImageNames": {"dup": f"{user_id}.png"}}),
                        "updated_at": AT,
                    },
                ],
            )
        conn.execute(
            insert(_client_state).values(
                user_id="alice", key="webv2:prompt-drafts", value=ALICE_DRAFT_TEXT, updated_at=AT
            )
        )
    return empty_database


def _projects_by_owner(db: Database) -> dict[str, tuple[object, ...]]:
    with db.begin(write=False) as conn:
        rows = conn.execute(
            select(
                _projects.c.user_id,
                _projects.c.project_id,
                _projects.c.name,
                _projects.c.data,
                _projects.c.board_id,
                _projects.c.revision,
                _projects.c.created_at,
                _projects.c.updated_at,
                _projects.c.minimum_canvas_schema_version,
            )
        ).all()
    return {row[0]: tuple(row[1:]) for row in rows}


def _client_state_of(db: Database, user_id: str) -> dict[str, str]:
    with db.begin(write=False) as conn:
        return dict(
            conn.execute(
                select(_client_state.c.key, _client_state.c.value).where(_client_state.c.user_id == user_id)
            ).all()
        )


def _primary_key(db: Database) -> list[str]:
    with db.begin(write=False) as conn:
        return inspect(conn).get_pk_constraint("projects")["constrained_columns"]


def test_a_shared_id_stays_with_the_oldest_project_and_the_other_is_rekeyed_everywhere(
    before: Database, migrations: list[MigrationBase], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        assert run_migrations(before, migrations)

    projects = _projects_by_owner(before)
    alice_id = projects["alice"][0]
    assert projects["bob"][0] == "dup"
    assert isinstance(alice_id, str) and alice_id.startswith("project-")
    # Every value survives the rebuild; the re-keyed document names its new id and is written as the application
    # writes documents: compact, its non-ASCII text as itself, so it grows by the new id's length and no more.
    alice_document = _dumps({**ALICE_DOCUMENT, "id": alice_id})
    assert projects == {
        "alice": (
            alice_id,
            "Alice's dup",
            alice_document,
            "alice-inbox",
            7,
            "2026-02-01 00:00:00.000",
            "2026-03-01 00:00:00.000",
            3,
        ),
        "bob": (
            "dup",
            "Bob's dup",
            _dumps(BOB_DOCUMENT),
            "bob-inbox",
            4,
            "2026-01-01 00:00:00.000",
            "2026-01-02 00:00:00.000",
            2,
        ),
    }
    assert len(alice_document.encode("utf-8")) == len(_dumps(ALICE_DOCUMENT).encode("utf-8")) + len(alice_id) - len(
        "dup"
    )

    with before.begin(write=False) as conn:
        for user_id, expected in (("alice", alice_id), ("bob", "dup")):
            for statement in (
                select(_images.c.project_id).where(_images.c.user_id == user_id),
                select(_videos.c.project_id).where(_videos.c.user_id == user_id),
                select(_session_queue.c.project_id).where(_session_queue.c.user_id == user_id),
                select(_media_references.c.owner_id).where(_media_references.c.user_id == user_id),
            ):
                assert conn.execute(statement).scalar_one() == expected, str(statement)
            origin = conn.execute(
                select(_session_queue.c.origin).where(_session_queue.c.user_id == user_id)
            ).scalar_one()
            assert origin == f"webv2:p:{expected}:q:1"
        boards = dict(conn.execute(select(_boards.c.board_id, _boards.c.project_id)).all())
    alice_draft = _client_state_of(before, "alice")["webv2:prompt-drafts"]
    for user_id, expected in (("alice", alice_id), ("bob", "dup")):
        assert _client_state_of(before, user_id) == {
            "webv2:workbench-account": _dumps({"activeProjectId": expected, "openProjectIds": [expected, "other"]}),
            "webv2:project-covers": _dumps({"coverImageNames": {expected: f"{user_id}.png"}}),
            **({"webv2:prompt-drafts": alice_draft} if user_id == "alice" else {}),
        }
    # The draft names the new id and keeps its lone surrogate as the escape the client wrote: raw, it could not be
    # stored, and a row skipped over it would leave the account's active project pointing at the old id.
    assert alice_draft.isascii()
    assert json.loads(alice_draft) == {"activeProjectId": alice_id, "note": "\ud83d"}
    assert boards == {"alice-inbox": alice_id, "bob-inbox": "dup", "bob-member": "dup", "alice-stray": None}
    assert _primary_key(before) == ["project_id"]
    assert f"the copy owned by alice is now {alice_id}" in caplog.text
    assert "1 boards named a project that no longer exists" in caplog.text
    assert not run_migrations(before, migrations)


def test_a_run_that_stopped_after_the_data_changes_finishes_the_work(
    before: Database, migrations: list[MigrationBase]
) -> None:
    """On a server the data changes commit with the first DDL: a failed run can leave the owner index created and
    the per-account key still in place, with no duplicate left to re-key."""
    with before.begin(write=True) as conn:
        conn.execute(update(_projects).where(_projects.c.user_id == "alice").values(project_id="renamed"))
        conn.execute(update(_boards).where(_boards.c.board_id == "alice-inbox").values(project_id="renamed"))
        alembic_operations(conn).create_index("idx_projects_user_id", "projects", ["user_id"])

    assert run_migrations(before, migrations)

    assert {owner: values[0] for owner, values in _projects_by_owner(before).items()} == {
        "alice": "renamed",
        "bob": "dup",
    }
    assert _primary_key(before) == ["project_id"]
