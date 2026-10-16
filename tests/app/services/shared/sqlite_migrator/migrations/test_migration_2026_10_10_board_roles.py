"""Board visibility and shares become general access and members, from 6.13 and alpha.1 data."""

from pathlib import Path

import pytest
from sqlalchemy import Column, column, insert, inspect, select, table, update

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import CheckViolation
from invokeai.app.services.shared.database.schema.boards import board_members
from invokeai.app.services.shared.database.schema.metadata import default
from invokeai.app.services.shared.database.schema.users import users
from invokeai.app.services.shared.database.types import LongText
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import MigrationBase
from tests.fixtures.migration_state import (
    alembic_operations,
    build_application_migrations,
    migrate_to_before,
    run_migrations,
)

MIGRATION_ID = "2026_10_10_board_roles"
SHARED_AT = "2026-01-02 03:04:05.000"

# The tables as 6.13 and the alpha left them; the schema metadata no longer has these columns.
_boards = table(
    "boards",
    column("board_id"),
    column("board_name"),
    column("user_id"),
    column("is_public"),
    column("board_visibility"),
    column("general_access"),
    column("created_at"),
    column("updated_at"),
)
_shared_boards = table("shared_boards", column("board_id"), column("user_id"), column("can_edit"), column("shared_at"))


@pytest.fixture
def migrations(tmp_path: Path) -> list[MigrationBase]:
    return build_application_migrations(tmp_path)


@pytest.fixture
def before(empty_database: Database, migrations: list[MigrationBase]) -> Database:
    """A database just before this migration, holding boards as 6.13 and the alpha left them."""
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
                    "board_name": board_id.capitalize(),
                    "user_id": owner,
                    "is_public": is_public,
                    "board_visibility": visibility,
                    "created_at": SHARED_AT,
                    "updated_at": SHARED_AT,
                }
                for board_id, owner, is_public, visibility in (
                    ("private", "alice", False, "private"),
                    ("shared", "alice", False, "shared"),
                    ("public", "alice", True, "public"),
                    ("lent", "alice", False, "private"),
                    # Left by a deleted account: boards keep no foreign key to their owner.
                    ("orphan", None, False, "private"),
                )
            ],
        )
        conn.execute(
            insert(_shared_boards),
            [
                {"board_id": "lent", "user_id": "bob", "can_edit": True, "shared_at": SHARED_AT},
                # A share naming the board's own owner grants nothing.
                {"board_id": "lent", "user_id": "alice", "can_edit": False, "shared_at": SHARED_AT},
                {"board_id": "orphan", "user_id": "bob", "can_edit": False, "shared_at": SHARED_AT},
            ],
        )
    return empty_database


def _general_access(db: Database) -> dict[str, str]:
    with db.begin(write=False) as conn:
        return dict(conn.execute(select(_boards.c.board_id, _boards.c.general_access)).all())


def _members(db: Database) -> list[tuple[str, str, str, str | None, str]]:
    columns = board_members.c
    with db.begin(write=False) as conn:
        rows = conn.execute(
            select(columns.board_id, columns.user_id, columns.role, columns.granted_by, columns.granted_at).order_by(
                columns.board_id, columns.user_id
            )
        ).all()
    return [tuple(row) for row in rows]


def _columns(db: Database) -> set[str]:
    with db.begin(write=False) as conn:
        return {entry["name"] for entry in inspect(conn).get_columns("boards")}


def _tables(db: Database) -> set[str]:
    with db.begin(write=False) as conn:
        return set(inspect(conn).get_table_names())


EXPECTED_MEMBERS = [
    ("lent", "bob", "viewer", None, SHARED_AT),
    ("orphan", "bob", "viewer", None, SHARED_AT),
]


def test_visibility_becomes_general_access_and_shares_become_viewer_members(
    before: Database, migrations: list[MigrationBase]
) -> None:
    assert run_migrations(before, migrations)

    assert _general_access(before) == {
        "private": "restricted",
        "shared": "viewer",
        "public": "contributor",
        "lent": "restricted",
        "orphan": "restricted",
    }
    assert _members(before) == EXPECTED_MEMBERS
    assert not {"board_visibility", "is_public"} & _columns(before)
    assert "shared_boards" not in _tables(before)
    assert not run_migrations(before, migrations)


def test_general_access_and_roles_are_checked(before: Database, migrations: list[MigrationBase]) -> None:
    assert run_migrations(before, migrations)

    with pytest.raises(CheckViolation):
        with before.begin(write=True) as conn:
            conn.execute(update(_boards).where(_boards.c.board_id == "private").values(general_access="editor"))
    with pytest.raises(CheckViolation):
        with before.begin(write=True) as conn:
            conn.execute(insert(board_members).values(board_id="private", user_id="bob", role="owner"))


def test_a_run_that_stopped_halfway_finishes_the_work(before: Database, migrations: list[MigrationBase]) -> None:
    """On a server each DDL statement commits on its own: a failed run can leave the new column behind, still at
    its default, with everything else left to do."""
    with before.begin(write=True) as conn:
        alembic_operations(conn).add_column(
            "boards", Column("general_access", LongText(), nullable=False, server_default=default("restricted"))
        )

    assert run_migrations(before, migrations)

    assert _general_access(before)["public"] == "contributor"
    assert _members(before) == EXPECTED_MEMBERS
    assert not {"board_visibility", "is_public"} & _columns(before)


def test_a_run_that_stopped_after_copying_members_does_not_copy_them_twice(
    before: Database, migrations: list[MigrationBase]
) -> None:
    """On a server the new table commits as it is created: a run that failed after the copy leaves the members in
    it and `shared_boards` still there."""
    with before.begin(write=True) as conn:
        board_members.create(conn)
        conn.execute(insert(board_members).values(board_id="lent", user_id="bob", role="viewer", granted_at=SHARED_AT))

    assert run_migrations(before, migrations)

    assert _members(before) == EXPECTED_MEMBERS
