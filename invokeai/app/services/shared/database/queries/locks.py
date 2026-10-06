"""Named locks, for invariants that span rows.

On MySQL and MariaDB transactions run side by side, so a check followed by a write ("is another active
administrator left?") can interleave with another transaction's. Every transaction that changes what such a
check reads takes the check's lock first, and the second waits for the first to end. Its reads then see what
the first committed, because writers run at READ COMMITTED (see `engines.py`): each statement reads the newest
committed rows. On SQLite all transactions are serialised already; the lock rows are still read there, so a
missing one is noticed on every backend.
"""

from enum import StrEnum

from sqlalchemy import Connection, bindparam, select

from invokeai.app.services.shared.database.queries.base import QueryModule, SharedTransaction, write
from invokeai.app.services.shared.database.schema.locks import db_locks


class DatabaseLock(StrEnum):
    """A lock, by the work it serialises. Each has a row in `db_locks`, which a migration adds."""

    # Changes to who is an active administrator: counting them and changing one is one step.
    ADMIN_ACCOUNTS = "admin_accounts"
    # A replace of the image index's custom vocabulary: deleting every term and inserting the new ones is one step.
    IMAGE_INDEX_VOCABULARY = "image_index_vocabulary"
    # What protects media from the intermediates cleanup: a cleanup takes it exclusively for its check and delete,
    # a write that makes media protected (a reference, a hold) shares it.
    MEDIA_PROTECTION = "media_protection"


# In name order, so that two transactions that take several of the same locks take them in the same order.
_LOCKS = (
    select(db_locks.c.name).where(db_locks.c.name.in_(bindparam("names", expanding=True))).order_by(db_locks.c.name)
)
_LOCK = _LOCKS.with_for_update()
_SHARE = _LOCKS.with_for_update(read=True)


class LockQueries(QueryModule):
    @write
    def acquire(self, conn: Connection, *locks: DatabaseLock, shared: bool = False) -> None:
        """Holds `locks` until the transaction ends.

        It must be the first call of a `transaction()`: a lock taken outside one would be released at once, and
        one taken after other rows were read or locked would guard reads made without it and could deadlock
        against a transaction that holds it. Raises `RuntimeError` otherwise, and when a lock has no row:
        locking nothing would guard nothing, silently.

        `shared` locks share with each other and wait only for, and hold off, an exclusive lock of the same name.
        """
        if not isinstance(self._scope, SharedTransaction) or self._scope.calls != 1:
            raise RuntimeError("Database locks are taken by the first call of a transaction, all at once")
        names = sorted({lock.value for lock in locks})
        locked: set[str] = set(conn.execute(_SHARE if shared else _LOCK, {"names": names}).scalars().all())
        if missing := [name for name in names if name not in locked]:
            raise RuntimeError(f"No row in db_locks for {', '.join(missing)}: the migration that adds it has not run")
