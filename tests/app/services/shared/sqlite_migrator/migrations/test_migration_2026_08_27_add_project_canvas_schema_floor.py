import sqlite3

import pytest

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_08_27_add_project_canvas_schema_floor import (
    AddProjectCanvasSchemaFloorCallback,
    build_migration,
)


def test_migration_backfills_existing_projects_to_canvas_schema_v2() -> None:
    db = sqlite3.connect(":memory:")
    db.execute(
        """--sql
        CREATE TABLE projects (
            project_id TEXT NOT NULL PRIMARY KEY,
            data TEXT NOT NULL
        );
        """
    )
    db.execute("INSERT INTO projects (project_id, data) VALUES ('existing', '{}');")

    AddProjectCanvasSchemaFloorCallback()(db.cursor())

    row = db.execute("SELECT minimum_canvas_schema_version FROM projects WHERE project_id = 'existing';").fetchone()
    assert row == (2,)

    db.execute("INSERT INTO projects (project_id, data) VALUES ('new', '{}');")
    row = db.execute("SELECT minimum_canvas_schema_version FROM projects WHERE project_id = 'new';").fetchone()
    assert row == (2,)


def test_migration_rejects_invalid_schema_floors() -> None:
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE projects (project_id TEXT NOT NULL PRIMARY KEY, data TEXT NOT NULL);")
    AddProjectCanvasSchemaFloorCallback()(db.cursor())

    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO projects (project_id, data, minimum_canvas_schema_version) VALUES ('bad', '{}', 0);")


def test_migration_runs_after_the_project_table_rebuild() -> None:
    migration = build_migration()

    assert migration.id == "2026_08_27_add_project_canvas_schema_floor"
    assert migration.depends_on == "2026_08_06_add_project_boards"
