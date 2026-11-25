"""Give images and videos the marks of culling: a rating, a rejected flag and a color label.

`starred` stays the pick and `rejected` is its opposite; a CHECK keeps an item from being both. `rating` runs from 0,
unrated, to 5. `color_label` names a color, or is NULL; the application validates the names, so that another color
needs no table rebuild. Existing rows read as unrated, not rejected and unlabelled.
"""

from sqlalchemy import Boolean, Column, inspect

from invokeai.app.services.shared.database.schema.metadata import default
from invokeai.app.services.shared.database.types import BigInt, LongText
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import (
    PortableMigration,
    PortableMigrationContext,
)

# Literals rather than imports: a migration keeps meaning what it meant when it was written.
_CHECKS = {"rating": "rating >= 0 AND rating <= 5", "flag": "NOT (starred AND rejected)"}


def _sqlite_columns(table_name: str) -> dict[str, str]:
    # SQLite adds a column with its CHECK in place; Alembic would copy the whole table to add the CHECK.
    return {
        "rating": f"rating INTEGER NOT NULL DEFAULT 0 CONSTRAINT ck_{table_name}_rating CHECK ({_CHECKS['rating']})",
        "rejected": f"rejected BOOLEAN NOT NULL DEFAULT FALSE CONSTRAINT ck_{table_name}_flag CHECK ({_CHECKS['flag']})",
        "color_label": "color_label TEXT",
    }


def _server_columns() -> dict[str, Column[object]]:
    return {
        "rating": Column("rating", BigInt(), nullable=False, server_default=default(0)),
        "rejected": Column("rejected", Boolean(), nullable=False, server_default=default(False)),
        "color_label": Column("color_label", LongText()),
    }


def _add_media_marks(context: PortableMigrationContext) -> None:
    on_sqlite = context.conn.dialect.name == "sqlite"
    for table_name in ("images", "videos"):
        existing = {entry["name"] for entry in inspect(context.conn).get_columns(table_name)}
        if on_sqlite:
            for name, definition in _sqlite_columns(table_name).items():
                if name not in existing:
                    context.conn.exec_driver_sql(f"ALTER TABLE {table_name} ADD COLUMN {definition}")
            continue
        for name, column in _server_columns().items():
            if name not in existing:
                context.op.add_column(table_name, column)
        # On a server each CHECK is a statement of its own, which a failed run may not have reached.
        checks = {check["name"] for check in inspect(context.conn).get_check_constraints(table_name)}
        for name, condition in _CHECKS.items():
            if f"ck_{table_name}_{name}" not in checks:
                context.op.create_check_constraint(name, table_name, condition)


def build_migration() -> PortableMigration:
    return PortableMigration(
        id="2026_11_24_add_media_marks",
        depends_on="2026_10_10_project_workspaces",
        callback=_add_media_marks,
    )
