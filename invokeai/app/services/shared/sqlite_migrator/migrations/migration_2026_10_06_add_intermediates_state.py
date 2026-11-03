"""Keep the intermediates cleanup's state in tables, and let its windows page by creation time.

SQLite's TEMP tables belong to the one connection that created them; MySQL and MariaDB run each transaction on a
connection of a pool. The holds of cached outputs and the marks of unmeasurable files move into ordinary tables,
which the cleanup service empties when it starts. A `media_protection` lock serialises a cleanup's deleting
transaction with the saves that reference media. The cleanup pages intermediates by (creation time, name), since a
server has no rowid, so its indexes end with those columns.
"""

from typing import Optional

from sqlalchemy import Column, Index, column, insert, inspect, select, table, text

from invokeai.app.services.shared.database.types import Key, Timestamp
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import (
    PortableMigration,
    PortableMigrationContext,
)

_db_locks = table("db_locks", column("name"))

# Literals rather than imports: a migration keeps meaning what it meant when it was written.
_MEDIA_PROTECTION = "media_protection"
_ENUM_LENGTH = 32
_INTERMEDIATE = "is_intermediate = TRUE"
_UNMEASURED = "is_intermediate = TRUE AND file_size_bytes IS NULL"


def _index(context: PortableMigrationContext, name: str, table_name: str, columns: list[str], where: str) -> None:
    """Gives the table the index on SQLite (partial, on `where`) or on a server (leading with `where`'s columns)."""
    on_sqlite = context.conn.dialect.name == "sqlite"
    if not on_sqlite:
        leading = ["is_intermediate", "file_size_bytes"] if "file_size_bytes" in where else ["is_intermediate"]
        columns = [*leading, *columns]
    existing: Optional[list[str]] = next(
        (index["column_names"] for index in inspect(context.conn).get_indexes(table_name) if index["name"] == name),
        None,
    )
    if existing == columns:
        return
    if existing is not None:
        context.op.drop_index(name, table_name=table_name)
    if on_sqlite:
        context.op.create_index(name, table_name, columns, sqlite_where=text(where))
    else:
        context.op.create_index(name, table_name, columns)


def _add_intermediates_state(context: PortableMigrationContext) -> None:
    tables = inspect(context.conn).get_table_names()
    if "intermediates_session_holds" not in tables:
        context.create_table(
            "intermediates_session_holds",
            Column("session_id", Key(), primary_key=True),
            Column("media_kind", Key(_ENUM_LENGTH), primary_key=True),
            Column("media_name", Key(), primary_key=True),
            Column("released_at", Timestamp("TEXT")),
            Index("idx_intermediates_session_holds_media", "media_kind", "media_name", "released_at"),
        )
    if "intermediates_unmeasurable" not in tables:
        context.create_table(
            "intermediates_unmeasurable",
            Column("media_kind", Key(_ENUM_LENGTH), primary_key=True),
            Column("media_name", Key(), primary_key=True),
        )
    if context.conn.execute(select(_db_locks.c.name).where(_db_locks.c.name == _MEDIA_PROTECTION)).first() is None:
        context.conn.execute(insert(_db_locks).values(name=_MEDIA_PROTECTION))

    for kind, name_column in (("images", "image_name"), ("videos", "video_name")):
        keyset = ["created_at", name_column]
        _index(context, f"idx_{kind}_intermediate_scope", kind, ["user_id", "project_id", *keyset], _INTERMEDIATE)
        _index(context, f"idx_{kind}_intermediates_owner", kind, ["user_id", *keyset], _INTERMEDIATE)
        _index(context, f"idx_{kind}_intermediates_created", kind, keyset, _INTERMEDIATE)
        _index(context, f"idx_{kind}_unmeasured_intermediates", kind, keyset, _UNMEASURED)


def build_migration() -> PortableMigration:
    return PortableMigration(
        id="2026_10_06_add_intermediates_state",
        depends_on="2026_10_04_add_db_locks",
        callback=_add_intermediates_state,
    )
