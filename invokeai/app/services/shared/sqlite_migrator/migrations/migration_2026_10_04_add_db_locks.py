"""Add `db_locks`, the rows a transaction locks to serialise work with other connections.

SQLite serialises all writes through its single connection; on MySQL and MariaDB, invariants that span rows
(the last active administrator, for one) need a lock that every transaction changing them takes first. A
lock is a row: `SELECT ... FOR UPDATE` on it waits for the transaction that holds it to end.
"""

from sqlalchemy import Column, column, insert, inspect, select, table

from invokeai.app.services.shared.database.types import Key
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import (
    PortableMigration,
    PortableMigrationContext,
)

_db_locks = table("db_locks", column("name"))

# Literal rather than imported: a migration keeps meaning what it meant when it was written.
_ADMIN_ACCOUNTS = "admin_accounts"


def _add_db_locks(context: PortableMigrationContext) -> None:
    if not inspect(context.conn).has_table("db_locks"):
        context.create_table("db_locks", Column("name", Key(32), primary_key=True))
    # On a server the table may exist without the row: its creation committed, and the run then failed.
    if context.conn.execute(select(_db_locks.c.name).where(_db_locks.c.name == _ADMIN_ACCOUNTS)).first() is None:
        context.conn.execute(insert(_db_locks).values(name=_ADMIN_ACCOUNTS))


def build_migration() -> PortableMigration:
    return PortableMigration(
        id="2026_10_04_add_db_locks",
        depends_on="2026_10_01_add_anima_variant",
        callback=_add_db_locks,
    )
