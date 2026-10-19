"""Named locks, for invariants that span rows.

On MySQL and MariaDB transactions run side by side, so a check followed by a write ("is another active
administrator left?") can interleave with another transaction's. Every transaction that changes what such a
check reads takes the check's lock first, and the second waits for the first to end. Its reads then see what
the first committed, because writers run at READ COMMITTED (see `engines.py`): each statement reads the newest
committed rows. On SQLite all transactions are serialised already; the lock rows are still read there, so a
missing one is noticed on every backend.
"""

import functools
from collections.abc import Iterable
from enum import StrEnum
from typing import Any

from sqlalchemy import Connection, Select, literal, select

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
    # What the session queue admits: counting its pending items and adding some is one step, as is finding an
    # enqueue request's receipt and writing it.
    SESSION_QUEUE_ADMISSION = "session_queue_admission"


@functools.cache
def _lock(names: tuple[str, ...], shared: bool) -> Select[Any]:
    """Locks the named rows, in name order, so that two transactions that take several of the same locks take them in
    the same order. One statement per set of names, of which there are few: an IN list bound value by value costs half
    what an expanding one does, on every transaction that takes a lock."""
    statement = select(db_locks.c.name).where(db_locks.c.name.in_([literal(name) for name in names]))
    return statement.order_by(db_locks.c.name).with_for_update(read=shared)


class LockQueries(QueryModule):
    @write
    def acquire(
        self, conn: Connection, *locks: DatabaseLock, shared: bool = False, also_shared: Iterable[DatabaseLock] = ()
    ) -> None:
        """Holds `locks` until the transaction ends.

        It must be the first call of a `transaction()`: a lock taken outside one would be released at once, and
        one taken after other rows were read or locked would guard reads made without it and could deadlock
        against a transaction that holds it. Raises `RuntimeError` otherwise, and when a lock has no row:
        locking nothing would guard nothing, silently.

        `shared` locks share with each other and wait only for, and hold off, an exclusive lock of the same name.
        `also_shared` adds shared locks to exclusive ones; every lock is still taken in name order.
        """
        if not isinstance(self._scope, SharedTransaction) or self._scope.calls != 1:
            raise RuntimeError("Database locks are taken by the first call of a transaction, all at once")
        modes = {lock.value: shared for lock in locks}
        modes.update({lock.value: True for lock in also_shared if lock.value not in modes})
        names = tuple(sorted(modes))
        if not names:
            raise ValueError("No lock to acquire")
        # SQLite has no row locks (its transaction excludes every other): one read finds the rows there.
        if len(set(modes.values())) == 1 or conn.dialect.name == "sqlite":
            locked: set[str] = set(conn.execute(_lock(names, modes[names[0]])).scalars().all())
        else:
            locked = set()
            for name in names:
                locked.update(conn.execute(_lock((name,), modes[name])).scalars().all())
        if missing := [name for name in names if name not in locked]:
            raise RuntimeError(f"No row in db_locks for {', '.join(missing)}: the migration that adds it has not run")
