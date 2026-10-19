"""Database locks: named locks (`queries/locks.py`) and the locks of single rows. What they serialise is tested with
the work they guard."""

import pytest
from sqlalchemy import delete

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries.locks import DatabaseLock
from invokeai.app.services.shared.database.schema.locks import db_locks


def test_every_lock_has_its_row_in_a_new_database(database: Database) -> None:
    with database.queries.transaction() as q:
        q.locks.acquire(*DatabaseLock)


def test_a_lock_without_its_row_is_refused_rather_than_locking_nothing(database: Database) -> None:
    with database.begin(write=True) as conn:
        conn.execute(delete(db_locks))

    with pytest.raises(RuntimeError, match="admin_accounts"):
        with database.queries.transaction() as q:
            q.locks.acquire(DatabaseLock.ADMIN_ACCOUNTS)


def test_a_lock_outside_a_transaction_is_refused(database: Database) -> None:
    # It would be released as soon as it was taken.
    with pytest.raises(RuntimeError, match="first call of a transaction"):
        database.queries.locks.acquire(DatabaseLock.ADMIN_ACCOUNTS)


def test_a_lock_after_other_work_in_the_transaction_is_refused(database: Database) -> None:
    # What the transaction read before would not be guarded.
    with pytest.raises(RuntimeError, match="first call of a transaction"):
        with database.queries.transaction() as q:
            q.users.count_active_admins()
            q.locks.acquire(DatabaseLock.ADMIN_ACCOUNTS)


@pytest.mark.parametrize("row", ["board", "project"])
def test_a_row_lock_outside_a_transaction_is_refused(database: Database, row: str) -> None:
    # It would be released as soon as it was taken.
    with pytest.raises(RuntimeError, match="only a transaction"):
        if row == "board":
            database.queries.boards.lock("board")
        else:
            database.queries.projects.lock("system", "project")
