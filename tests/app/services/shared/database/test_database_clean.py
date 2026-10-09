"""Regression test: clean() must serialize with transactions.

Manual VACUUM may run while services and their worker threads are live; without
the shared lock it can fail with "cannot VACUUM - SQL statements in progress".
"""

import sqlite3
import threading

import pytest

from invokeai.app.services.shared.database.database import Database
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import sqlite_cursor


def test_clean_waits_for_in_flight_transactions(tmp_path) -> None:
    db = Database.open_sqlite(tmp_path / "test.db", InvokeAILogger.get_logger())
    in_transaction = threading.Event()
    release = threading.Event()
    errors: list[Exception] = []

    def hold_transaction() -> None:
        with sqlite_cursor(db) as cursor:
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


def test_a_backup_holds_committed_wal_data_and_overwrites_no_file(tmp_path) -> None:
    source = tmp_path / "source.db"
    backup = tmp_path / "backups" / "backup.db"
    db = Database.open_sqlite(source, InvokeAILogger.get_logger())
    try:
        with sqlite_cursor(db) as cursor:
            cursor.execute("CREATE TABLE values_table (value TEXT NOT NULL);")
            cursor.execute("INSERT INTO values_table VALUES ('committed in WAL');")

        # The directory is created; the WAL's committed pages are in the copy.
        db.backup(backup)

        with sqlite3.connect(backup) as connection:
            assert connection.execute("SELECT value FROM values_table;").fetchall() == [("committed in WAL",)]

        with pytest.raises(FileExistsError):
            db.backup(backup)
    finally:
        db.dispose()
