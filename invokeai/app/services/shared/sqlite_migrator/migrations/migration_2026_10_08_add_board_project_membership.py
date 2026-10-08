"""Let a board belong to a project, so a project can hold more than its inbox.

`boards.project_id` says which of its owner's projects a board lives in; `NULL` is the Library,
the tier every project can see. `projects.board_id` keeps its job of naming the one board that is
the project's inbox, which the database already guarantees exists exactly once and cannot be
deleted out from under the project.

Membership is a plain nullable column rather than a foreign key, like `boards.user_id` before it.
`projects` is keyed by `(user_id, project_id)`, so a foreign key would have to be composite and
therefore a table constraint, which SQLite can only add by rebuilding `boards` — and rebuilding a
table that `board_images`, `board_videos`, `shared_boards` and `projects` all reference, with
foreign keys enforced, runs every one of those cascades. The services keep membership consistent
instead, and the backfill below makes every inbox a member of its own project. Deleting an account
cascades its projects but, as before, not its boards (`boards.user_id` has no foreign key either), so
those orphans keep a `project_id` nothing resolves; readers treat an unresolvable membership as any
other board, and `is_inbox` is derived from `projects`, so it turns false and admins can delete them.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class AddBoardProjectMembershipCallback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("PRAGMA table_info(boards);")
        if not any(row[1] == "project_id" for row in cursor.fetchall()):
            cursor.execute("ALTER TABLE boards ADD COLUMN project_id TEXT;")

        cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_boards_project_id ON boards(project_id);
            """
        )
        # Every inbox is a member of its own project. Rows that already say so are left alone, so
        # a board the user has since moved is not dragged back.
        cursor.execute(
            """--sql
            UPDATE boards
            SET project_id = (SELECT projects.project_id FROM projects WHERE projects.board_id = boards.board_id)
            WHERE project_id IS NULL
              AND EXISTS (SELECT 1 FROM projects WHERE projects.board_id = boards.board_id);
            """
        )


def build_migration() -> Migration:
    return Migration(
        id="2026_10_08_add_board_project_membership",
        depends_on="2026_08_27_add_project_canvas_schema_floor",
        callback=AddBoardProjectMembershipCallback(),
    )
