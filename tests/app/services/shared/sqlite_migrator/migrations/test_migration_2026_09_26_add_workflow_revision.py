"""Existing library workflows start at revision 1 and the column is added once."""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_09_26_add_workflow_revision import (
    build_migration,
)


def test_existing_workflows_start_at_revision_one() -> None:
    connection = sqlite3.connect(":memory:")
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE workflow_library (workflow_id TEXT NOT NULL PRIMARY KEY, workflow TEXT NOT NULL);")
    cursor.execute("INSERT INTO workflow_library VALUES ('wf-1', '{}');")

    build_migration().callback(cursor)
    build_migration().callback(cursor)

    cursor.execute("SELECT revision FROM workflow_library WHERE workflow_id = 'wf-1';")
    assert cursor.fetchone() == (1,)
    cursor.execute("INSERT INTO workflow_library (workflow_id, workflow) VALUES ('wf-2', '{}');")
    cursor.execute("SELECT revision FROM workflow_library WHERE workflow_id = 'wf-2';")
    assert cursor.fetchone() == (1,)
    cursor.execute("PRAGMA table_info(workflow_library);")
    assert [row[1] for row in cursor.fetchall()].count("revision") == 1
