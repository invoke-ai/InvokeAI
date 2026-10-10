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

from sqlalchemy import Column, column, exists, inspect, select, table, update

from invokeai.app.services.shared.database.types import Key
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import (
    PortableMigration,
    PortableMigrationContext,
)

_boards = table("boards", column("board_id"), column("project_id"))
_projects = table("projects", column("board_id"), column("project_id"))


def _add_board_project_membership(context: PortableMigrationContext) -> None:
    if "project_id" not in {column["name"] for column in inspect(context.conn).get_columns("boards")}:
        context.op.add_column("boards", Column("project_id", Key()))
    if "idx_boards_project_id" not in {index["name"] for index in inspect(context.conn).get_indexes("boards")}:
        context.op.create_index("idx_boards_project_id", "boards", ["project_id"])
    # Safe after an interrupted DDL run, and on a database already upgraded by this feature branch.
    inbox_project = select(_projects.c.project_id).where(_projects.c.board_id == _boards.c.board_id)
    context.conn.execute(
        update(_boards)
        .where(_boards.c.project_id.is_(None), exists(inbox_project))
        .values(project_id=inbox_project.scalar_subquery())
    )


def build_migration() -> PortableMigration:
    return PortableMigration(
        id="2026_10_08_add_board_project_membership",
        depends_on="2026_10_07_add_session_queue_admission_lock",
        callback=_add_board_project_membership,
    )
