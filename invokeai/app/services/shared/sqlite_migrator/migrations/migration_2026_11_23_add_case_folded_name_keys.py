"""Keep the case-folded names the workflow library and the font catalog sort by.

SQLite sorted names ignoring the case of ASCII letters only (`COLLATE NOCASE`), a server ignoring the case of every
letter (`lower()`), so a list sorted differently per backend. Every backend now sorts by a stored key, written by
whoever writes the name: `fold_case(name)` (`workflow_library.name_key`, `fonts.family_key`, `fonts.label_key`).

The keys are nullable, because SQLite cannot add a NOT NULL column without a default, and are filled here from the
names the rows hold. Running again fills in the same keys. On SQLite the font catalog's listing index leads with the
family key instead of the family under NOCASE.
"""

from collections.abc import Sequence

from sqlalchemy import Column, Connection, TableClause, bindparam, column, inspect, select, table, update

from invokeai.app.services.shared.database.dialect import fold_case
from invokeai.app.services.shared.database.types import LongText
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import (
    PortableMigration,
    PortableMigrationContext,
)

_WORKFLOW_LIBRARY = table("workflow_library", column("workflow_id"), column("name"), column("name_key"))
_FONTS = table("fonts", column("id"), column("family"), column("label"), column("family_key"), column("label_key"))
_BACKFILL_CHUNK = 1000


def _add_missing_columns(context: PortableMigrationContext, table_name: str, names: Sequence[str]) -> None:
    existing = {found["name"] for found in inspect(context.conn).get_columns(table_name)}
    for name in names:
        if name not in existing:
            context.op.add_column(table_name, Column(name, LongText()))


def _backfill(conn: Connection, target: TableClause, primary_key: str, keyed: dict[str, str]) -> None:
    """Sets each key column of `target` to the fold of its source column (`keyed`: key column -> source column),
    where it differs."""
    key_names = list(keyed)
    rows = conn.execute(
        select(
            target.c[primary_key],
            *(target.c[source] for source in keyed.values()),
            *(target.c[key] for key in key_names),
        )
    ).all()
    changes = []
    for row in rows:
        wanted = [fold_case(value) for value in row[1 : 1 + len(key_names)]]
        if list(row[1 + len(key_names) :]) != wanted:
            changes.append(
                {"target_key": row[0], **{f"new_{key}": value for key, value in zip(key_names, wanted, strict=False)}}
            )
    statement = (
        update(target)
        .where(target.c[primary_key] == bindparam("target_key"))
        .values({key: bindparam(f"new_{key}") for key in key_names})
    )
    for start in range(0, len(changes), _BACKFILL_CHUNK):
        conn.execute(statement, changes[start : start + _BACKFILL_CHUNK])


def _add_case_folded_name_keys(context: PortableMigrationContext) -> None:
    _add_missing_columns(context, "workflow_library", ["name_key"])
    _add_missing_columns(context, "fonts", ["family_key", "label_key"])
    _backfill(context.conn, _WORKFLOW_LIBRARY, "workflow_id", {"name_key": "name"})
    _backfill(context.conn, _FONTS, "id", {"family_key": "family", "label_key": "label"})
    if context.conn.dialect.name == "sqlite":
        # Only SQLite has this index (a server cannot sort by a prefix index).
        context.op.execute("DROP INDEX IF EXISTS idx_fonts_visible_private")
        context.op.create_index("idx_fonts_visible_private", "fonts", ["source", "scope", "owner_id", "family_key"])


def build_migration() -> PortableMigration:
    return PortableMigration(
        id="2026_11_23_add_case_folded_name_keys",
        depends_on="2026_09_07_add_fonts",
        callback=_add_case_folded_name_keys,
    )
