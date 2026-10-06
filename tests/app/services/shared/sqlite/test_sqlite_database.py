"""Regression test: clean() must serialize with transaction() users.

Manual VACUUM may run while services and their worker threads are live; without
the shared lock it can fail with "cannot VACUUM - SQL statements in progress".
"""

import threading

from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.backend.util.logging import InvokeAILogger


def test_clean_waits_for_in_flight_transactions(tmp_path) -> None:
    db = SqliteDatabase(db_path=tmp_path / "test.db", logger=InvokeAILogger.get_logger())
    in_transaction = threading.Event()
    release = threading.Event()
    errors: list[Exception] = []

    def hold_transaction() -> None:
        with db.transaction() as cursor:
            cursor.execute("CREATE TABLE t (x INTEGER);")
            cursor.execute("INSERT INTO t VALUES (1);")
            in_transaction.set()
            release.wait(10)

    def run_clean() -> None:
        try:
            db.clean()
        except Exception as e:
            errors.append(e)

    holder = threading.Thread(target=hold_transaction)
    holder.start()
    assert in_transaction.wait(10)

    # clean() must block on the shared lock until the transaction finishes,
    # not fail against its open statement.
    cleaner = threading.Thread(target=run_clean)
    cleaner.start()
    release.set()
    holder.join(10)
    cleaner.join(10)

    assert not holder.is_alive()
    assert not cleaner.is_alive()
    assert errors == []
