import sqlite3


def v0(cursor: sqlite3.Cursor) -> None:
    """
    Migration for `workflows` table v0
    https://github.com/invoke-ai/InvokeAI/pull/4686

    Creates the `workflows` table for the workflow_records service & a trigger for updated_at.

    Note: `workflow_id` gets an implicit index. We don't need to make one for this column.
    """
    cursor.execute(
        """--sql
        CREATE TABLE IF NOT EXISTS workflows (
            workflow TEXT NOT NULL,
            workflow_id TEXT GENERATED ALWAYS AS (json_extract(workflow, '$.id')) VIRTUAL NOT NULL UNIQUE, -- gets implicit index
            created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
            updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')) -- updated via trigger
        );
        """
    )

    cursor.execute(
        """--sql
        CREATE TRIGGER IF NOT EXISTS tg_workflows_updated_at
        AFTER UPDATE
        ON workflows FOR EACH ROW
        BEGIN
            UPDATE workflows
            SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
            WHERE workflow_id = old.workflow_id;
        END;
        """
    )
