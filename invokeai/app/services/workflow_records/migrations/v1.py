import sqlite3


def v1(cursor: sqlite3.Cursor) -> None:
    """
    Migration for `workflows` table v1
    https://github.com/invoke-ai/InvokeAI/pull/5148

    Drops the `workflow_images` table and empties the `workflows` table.

    Prior to v3.5.0, all workflows were associated with images. They were stored in the image files
    themselves, and in v3.4.0 we started storing them in the DB. This turned out to be a bad idea -
    you end up with *many* image workflows, most of which are duplicates.

    The purpose of workflows DB storage was to provide a workflow library. Library workflows are
    different from image workflows. They are only saved when the user requests they be saved.

    Moving forward, the storage for image workflows and library workflows will be separate. Image
    workflows are store only in the image files. Library workflows are stored only in the DB.

    To give ourselves a clean slate, we need to delete all existing workflows in the DB (all of which)
    are image workflows. We also need to delete the workflow_images table, which is no longer needed.
    """
    cursor.execute(
        """--sql
        DROP TABLE IF EXISTS workflow_images;
        """
    )
    cursor.execute(
        """--sql
        DELETE FROM workflows;
        """
    )
