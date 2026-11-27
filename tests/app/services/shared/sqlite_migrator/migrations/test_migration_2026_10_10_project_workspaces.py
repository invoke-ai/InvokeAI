"""Existing projects need document schema 3 and have no workspaces yet; workspaces go with their project and account."""

from pathlib import Path

import pytest
from sqlalchemy import column, delete, func, insert, select, table, update

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import CheckViolation
from invokeai.app.services.shared.database.schema.projects import project_workspaces, projects
from invokeai.app.services.shared.database.schema.users import users
from tests.fixtures.migration_state import build_application_migrations, migrate_to_before, run_migrations

MIGRATION_ID = "2026_10_10_project_workspaces"
AT = "2026-01-02 03:04:05.000"

# As they were before the migration, as far as a row needs.
_boards = table(
    "boards", column("board_id"), column("board_name"), column("user_id"), column("created_at"), column("updated_at")
)
_projects = table(
    "projects",
    column("project_id"),
    column("user_id"),
    column("name"),
    column("data"),
    column("board_id"),
    column("created_at"),
    column("updated_at"),
)


@pytest.fixture
def migrated(empty_database: Database, tmp_path: Path) -> Database:
    """A project of Alice's from before this migration, then the migration."""
    migrations = build_application_migrations(tmp_path)
    migrate_to_before(empty_database, migrations, MIGRATION_ID)
    with empty_database.begin(write=True) as conn:
        conn.execute(
            insert(users).values(user_id="alice", email="alice@example.com", display_name="Alice", password_hash="x")
        )
        conn.execute(
            insert(_boards).values(board_id="inbox", board_name="P", user_id="alice", created_at=AT, updated_at=AT)
        )
        conn.execute(
            insert(_projects).values(
                project_id="p", user_id="alice", name="P", data="{}", board_id="inbox", created_at=AT, updated_at=AT
            )
        )
    assert run_migrations(empty_database, migrations)
    assert not run_migrations(empty_database, migrations)
    return empty_database


def _workspaces(db: Database) -> int:
    with db.begin(write=False) as conn:
        return conn.execute(select(func.count()).select_from(project_workspaces)).scalar_one()


def test_existing_projects_need_document_schema_3_and_have_no_workspaces(migrated: Database) -> None:
    with migrated.begin(write=False) as conn:
        assert conn.execute(select(projects.c.minimum_document_schema_version)).scalar_one() == 3
    assert _workspaces(migrated) == 0
    with pytest.raises(CheckViolation):
        with migrated.begin(write=True) as conn:
            conn.execute(update(projects).values(minimum_document_schema_version=0))


@pytest.mark.parametrize("deleted", [delete(projects), delete(users).where(users.c.user_id == "alice")])
def test_a_workspace_goes_with_its_project_and_its_account(migrated: Database, deleted: object) -> None:
    with migrated.begin(write=True) as conn:
        conn.execute(insert(project_workspaces).values(project_id="p", user_id="alice", data="{}"))
    assert _workspaces(migrated) == 1

    with migrated.begin(write=True) as conn:
        conn.execute(deleted)

    assert _workspaces(migrated) == 0
