"""The transitional cursor facade, which services not yet ported to the query layer still use."""

import sqlite3

import pytest

from invokeai.app.services.shared.database.errors import NestedTransactionError
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from tests.fixtures.database_probe import ProbeQueries

pytestmark = pytest.mark.sqlite_only


def _names(facade: SqliteDatabase) -> list[str]:
    with facade.transaction() as cursor:
        return [row["name"] for row in cursor.execute("SELECT name FROM probe_items ORDER BY id")]


def test_a_nested_legacy_transaction_no_longer_commits_the_outer_one_early(facade: SqliteDatabase) -> None:
    with pytest.raises(RuntimeError):
        with facade.transaction() as outer:
            outer.execute("INSERT INTO probe_items (id, name, size) VALUES (1, 'outer', 0)")
            with facade.transaction() as inner:
                inner.execute("INSERT INTO probe_items (id, name, size) VALUES (2, 'inner', 0)")
            raise RuntimeError("the outer block fails after the inner block finished")

    assert _names(facade) == []


def test_legacy_and_query_transactions_do_not_nest(facade: SqliteDatabase) -> None:
    queries = ProbeQueries(facade.database)

    with facade.transaction():
        with pytest.raises(NestedTransactionError):
            queries.items.names()

    with queries.transaction():
        with pytest.raises(NestedTransactionError):
            with facade.transaction():
                pass


def test_a_legacy_read_is_not_held_up_by_another_process_writing(facade: SqliteDatabase) -> None:
    # Database tools and the user-management commands write to the same file. A read must not wait for them,
    # and with it everything queued behind the lock it holds.
    facade._conn.execute("PRAGMA busy_timeout = 20;")
    assert facade._db_path is not None
    other_process = sqlite3.connect(facade._db_path, timeout=0)
    try:
        other_process.execute("BEGIN IMMEDIATE")
        other_process.execute("INSERT INTO probe_items (id, name, size) VALUES (1, 'uncommitted', 0)")

        assert _names(facade) == []
    finally:
        other_process.close()


def test_a_write_left_uncommitted_on_the_raw_connection_is_reported_and_kept(facade: SqliteDatabase) -> None:
    queries = ProbeQueries(facade.database)
    facade._conn.execute("INSERT INTO probe_items (id, name, size) VALUES (1, 'raw', 0)")

    with pytest.raises(RuntimeError, match="raw connection without being committed"):
        with facade.transaction():
            pass
    with pytest.raises(RuntimeError, match="raw connection without being committed"):
        queries.items.names()

    facade._conn.commit()
    assert queries.items.names() == ["raw"]


def test_legacy_rows_are_read_by_name_and_query_rows_are_plain(facade: SqliteDatabase) -> None:
    # Queries read their rows without the driver's `sqlite3.Row`; the facade keeps it for the cursor code.
    queries = ProbeQueries(facade.database)
    queries.items.add(1, "a")

    assert _names(facade) == ["a"]
    assert queries.items.names() == ["a"]
    assert _names(facade) == ["a"]


def test_legacy_writes_are_visible_to_queries(facade: SqliteDatabase) -> None:
    with facade.transaction() as cursor:
        cursor.execute("INSERT INTO probe_items (id, name, size) VALUES (1, 'legacy', 0)")

    assert ProbeQueries(facade.database).items.names() == ["legacy"]
