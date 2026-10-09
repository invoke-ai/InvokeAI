"""Regression test: clean() must serialize with transaction() users.

Manual VACUUM may run while services and their worker threads are live; without
the shared lock it can fail with "cannot VACUUM - SQL statements in progress".
"""

import sqlite3
import threading

import pytest

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


def test_backup_to_captures_committed_wal_data_and_refuses_overwrite(tmp_path) -> None:
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    db = SqliteDatabase(db_path=source, logger=InvokeAILogger.get_logger())
    with db.transaction() as cursor:
        cursor.execute("CREATE TABLE values_table (value TEXT NOT NULL);")
        cursor.execute("INSERT INTO values_table VALUES ('committed in WAL');")

    db.backup_to(backup)

    with sqlite3.connect(backup) as connection:
        assert connection.execute("SELECT value FROM values_table;").fetchall() == [("committed in WAL",)]

    with pytest.raises(FileExistsError):
        db.backup_to(backup)
