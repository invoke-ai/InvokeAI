"""Give each account its own workspace in a project, and record which document schema a project needs.

`project_workspaces` holds one account's layout, widgets and generation settings in a project, saved apart from
the project content in `projects.data` so that the two never conflict. Existing projects get no rows: the client
splits a project's document into content and workspace the first time it saves it.

`projects.minimum_document_schema_version` is the oldest document schema that can read a project, as
`minimum_canvas_schema_version` is for the canvas. Existing projects are at 3; the split raises a project to 4, and
a client that understands less is refused it.
"""

from sqlalchemy import Column, ForeignKey, Table, inspect

from invokeai.app.services.shared.database.schema.metadata import USER_ID_LENGTH, default, updated_at
from invokeai.app.services.shared.database.types import BigInt, Key, LongText
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import (
    PortableMigration,
    PortableMigrationContext,
)

_FLOOR = "minimum_document_schema_version"


def _project_workspaces(context: PortableMigrationContext) -> None:
    if "project_workspaces" not in set(inspect(context.conn).get_table_names()):
        # The tables its foreign keys name, as far as the CREATE TABLE needs them.
        Table("projects", context.metadata, Column("project_id", Key(), primary_key=True))
        Table("users", context.metadata, Column("user_id", Key(USER_ID_LENGTH), primary_key=True))
        context.create_table(
            "project_workspaces",
            Column("project_id", Key(), ForeignKey("projects.project_id", ondelete="CASCADE"), primary_key=True),
            Column("user_id", Key(USER_ID_LENGTH), ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True),
            Column("data", LongText(), nullable=False),
            Column("revision", BigInt(), nullable=False, server_default=default(1)),
            updated_at(),
        )
    if "idx_project_workspaces_user_id" not in {
        index["name"] for index in inspect(context.conn).get_indexes("project_workspaces")
    }:
        context.op.create_index("idx_project_workspaces_user_id", "project_workspaces", ["user_id"])

    if _FLOOR not in {entry["name"] for entry in inspect(context.conn).get_columns("projects")}:
        if context.conn.dialect.name == "sqlite":
            # SQLite adds a column with its CHECK in place; Alembic would copy the whole table to add the CHECK.
            context.conn.exec_driver_sql(
                f"ALTER TABLE projects ADD COLUMN {_FLOOR} INTEGER NOT NULL DEFAULT 3 "
                f"CONSTRAINT ck_projects_{_FLOOR} CHECK ({_FLOOR} >= 1)"
            )
        else:
            context.op.add_column("projects", Column(_FLOOR, BigInt(), nullable=False, server_default=default(3)))
    if f"ck_projects_{_FLOOR}" not in {
        check["name"] for check in inspect(context.conn).get_check_constraints("projects")
    }:
        # On a server the CHECK is a statement of its own, which a failed run may not have reached.
        context.op.create_check_constraint(_FLOOR, "projects", f"{_FLOOR} >= 1")


def build_migration() -> PortableMigration:
    return PortableMigration(
        id="2026_10_10_project_workspaces",
        depends_on="2026_10_10_global_project_ids",
        callback=_project_workspaces,
    )
