"""Add the `session_queue_admission` lock, which every write that adds items to the session queue takes first.

The queue holds at most `max_queue_size` pending items, and an enqueue request with an idempotency key is settled by
its receipt. On MySQL and MariaDB two enqueues running side by side would each count the pending items and look for
the receipt before the other inserts, so both could fit into the last free place, or both write the same receipt.
"""

from sqlalchemy import column, insert, select, table

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import (
    PortableMigration,
    PortableMigrationContext,
)

_db_locks = table("db_locks", column("name"))

# Literal rather than imported: a migration keeps meaning what it meant when it was written.
_SESSION_QUEUE_ADMISSION = "session_queue_admission"


def _add_session_queue_admission_lock(context: PortableMigrationContext) -> None:
    lock = select(_db_locks.c.name).where(_db_locks.c.name == _SESSION_QUEUE_ADMISSION)
    if context.conn.execute(lock).first() is None:
        context.conn.execute(insert(_db_locks).values(name=_SESSION_QUEUE_ADMISSION))


def build_migration() -> PortableMigration:
    return PortableMigration(
        id="2026_10_07_add_session_queue_admission_lock",
        depends_on="2026_10_07_add_image_index_vocabulary_lock",
        callback=_add_session_queue_admission_lock,
    )
