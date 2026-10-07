"""Transactions of the database layer: what commits together, what may nest, and how each backend isolates."""

import logging
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Optional

import pytest
from sqlalchemy import URL, Connection, Row, select
from sqlalchemy.exc import DBAPIError, InvalidRequestError

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import (
    ConflictError,
    LockTimeoutError,
    NestedTransactionError,
    ReadOnlyTransactionError,
)
from invokeai.app.services.shared.database.queries.base import OwnTransaction, QueryModule, mapped, read
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.database import external_test_db_url
from tests.fixtures.database_probe import ProbeQueries, probe_items

server_only = pytest.mark.skipif(
    external_test_db_url() is None, reason="needs a MySQL or MariaDB server (INVOKEAI_TEST_DB_URL)"
)


@contextmanager
def _retries(database: Database, on_retry: Optional[Callable[[], None]] = None) -> Iterator[list[str]]:
    """Collects the database's retry warnings, calling `on_retry` (if given) as each is logged."""
    retries: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "Retrying a database transaction" in record.getMessage():
                retries.append(record.getMessage())
                if on_retry is not None:
                    on_retry()

    handler = Collect()
    database.logger.addHandler(handler)
    try:
        yield retries
    finally:
        database.logger.removeHandler(handler)


def test_a_single_call_commits_on_its_own(probe: ProbeQueries) -> None:
    probe.items.add(1, "a")

    assert probe.items.names() == ["a"]


def test_calls_in_one_transaction_commit_together(probe: ProbeQueries) -> None:
    with probe.transaction() as q:
        q.items.add(1, "a")
        q.items.add(2, "b")

    assert probe.items.names() == ["a", "b"]


def test_a_failing_transaction_leaves_no_trace(probe: ProbeQueries) -> None:
    with pytest.raises(RuntimeError):
        with probe.transaction() as q:
            q.items.add(1, "a")
            raise RuntimeError("the unit of work fails after its first write")

    assert probe.items.names() == []


class TestNesting:
    def test_a_call_on_the_unbound_queries_inside_a_transaction_is_refused(self, probe: ProbeQueries) -> None:
        with probe.transaction() as q:
            q.items.add(1, "a")
            with pytest.raises(NestedTransactionError):
                probe.items.add(2, "b")
            # Refusing one call does not release the transaction's claim on the thread.
            with pytest.raises(NestedTransactionError):
                probe.items.add(3, "c")

        # The refused calls never reached the database, so the transaction around them still commits.
        assert probe.items.names() == ["a"]

    def test_a_transaction_inside_a_transaction_is_refused(self, probe: ProbeQueries) -> None:
        with probe.transaction() as q:
            with pytest.raises(NestedTransactionError):
                with probe.transaction():
                    pass
            with pytest.raises(NestedTransactionError):
                with q.transaction():
                    pass

    def test_a_transaction_on_another_thread_is_not_nesting(self, probe: ProbeQueries) -> None:
        outer_wrote = threading.Event()
        errors: list[BaseException] = []

        def write_from_another_thread() -> None:
            try:
                outer_wrote.wait(10)
                probe.items.add(2, "b")
            except BaseException as error:
                errors.append(error)

        other = threading.Thread(target=write_from_another_thread)
        other.start()
        with probe.transaction() as q:
            q.items.add(1, "a")
            outer_wrote.set()
            # On SQLite the other thread waits for this transaction; on a server it commits meanwhile.
            other.join(0.3)
        other.join(10)

        assert errors == []
        assert probe.items.names() == ["a", "b"]


class TestReadOnlyTransactions:
    def test_a_read_only_transaction_refuses_writes(self, probe: ProbeQueries) -> None:
        with probe.transaction(read_only=True) as q:
            with pytest.raises(ReadOnlyTransactionError):
                q.items.add(1, "a")

        assert probe.items.names() == []

    def test_a_read_only_transaction_reads_one_snapshot(self, probe: ProbeQueries) -> None:
        probe.items.add(1, "a")
        first_read_done = threading.Event()
        writer_done = threading.Event()

        def write_concurrently() -> None:
            first_read_done.wait(10)
            probe.items.add(2, "b")
            writer_done.set()

        writer = threading.Thread(target=write_concurrently)
        writer.start()
        with probe.transaction(read_only=True) as q:
            before = q.items.count()
            first_read_done.set()
            # A server commits the write meanwhile; SQLite makes the writer wait for this transaction.
            writer_done.wait(0.3)
            after = q.items.count()
        writer.join(10)

        assert before == after == 1
        assert probe.items.count() == 2


def test_a_mapper_runs_once_the_transaction_of_its_call_has_ended(
    probe: ProbeQueries, empty_database: Database
) -> None:
    # On SQLite the transaction holds the process-wide lock, which building DTOs need not hold.
    probe.items.add(1, "a")
    in_a_transaction: list[bool] = []

    def note_whether_in_a_transaction(rows: list[Row[Any]]) -> list[str]:
        try:
            with empty_database.begin(write=False):
                in_a_transaction.append(False)
        except NestedTransactionError:
            in_a_transaction.append(True)
        return [name for (name,) in rows]

    class Names(QueryModule):
        @mapped(note_whether_in_a_transaction)
        @read
        def names(self, conn: Connection) -> list[Row[Any]]:
            return list(conn.execute(select(probe_items.c.name)).all())

    assert Names(OwnTransaction(empty_database)).names() == ["a"]
    assert in_a_transaction == [False]


def test_queries_of_an_ended_transaction_are_refused(probe: ProbeQueries) -> None:
    with probe.transaction() as q:
        pass

    with pytest.raises(RuntimeError, match="already ended"):
        q.items.names()


@server_only
def test_a_server_database_is_left_to_its_operator_to_back_up(empty_database: Database, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no SQLite connection"):
        empty_database.backup(tmp_path / "backup.db")
    assert not (tmp_path / "backup.db").exists()


def test_a_disposed_database_refuses_further_work(empty_database: Database) -> None:
    empty_database.dispose()
    empty_database.dispose()  # Repeating it is harmless.

    with pytest.raises(RuntimeError, match="closed"):
        with empty_database.queries.transaction():
            pass


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://user:secret@localhost/invokeai",
        "sqlite:///invokeai.db",
        # The right server through another driver, whose errors `translate_error` would not understand.
        "mysql+mysqldb://user:secret@localhost/invokeai",
    ],
)
def test_only_mysql_and_mariadb_through_pymysql_can_be_opened_by_url(url: str) -> None:
    with pytest.raises(ValueError, match="Unsupported database URL scheme"):
        Database.open_url(url, InvokeAILogger.get_logger("test_database"))


@pytest.mark.sqlite_only
class TestSqlite:
    """The SQLite file is shared with other processes (the user-management commands, database tools)."""

    @pytest.fixture
    def other_process(self, sqlite_file_database: Database) -> Iterator[sqlite3.Connection]:
        assert sqlite_file_database.sqlite.path is not None
        conn = sqlite3.connect(sqlite_file_database.sqlite.path, timeout=0)
        try:
            yield conn
        finally:
            conn.close()

    def test_a_write_transaction_holds_the_write_lock_before_it_writes(
        self, sqlite_file_database: Database, other_process: sqlite3.Connection
    ) -> None:
        with ProbeQueries(sqlite_file_database).transaction() as q:
            q.items.names()
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                other_process.execute("BEGIN IMMEDIATE")

    def test_a_read_transaction_leaves_other_processes_free_to_write(
        self, sqlite_file_database: Database, other_process: sqlite3.Connection
    ) -> None:
        queries = ProbeQueries(sqlite_file_database)
        with queries.transaction(read_only=True) as q:
            assert q.items.names() == []
            other_process.execute("BEGIN IMMEDIATE")
            other_process.execute("INSERT INTO probe_items (id, name, size) VALUES (1, 'other', 0)")
            other_process.commit()
            assert q.items.names() == []

        assert queries.items.names() == ["other"]

    def test_a_backup_holds_what_only_the_write_ahead_log_holds(
        self, sqlite_file_database: Database, tmp_path: Path
    ) -> None:
        ProbeQueries(sqlite_file_database).items.add(1, "logged")
        live = sqlite_file_database.sqlite.path
        assert live is not None
        # Not checkpointed yet: a copy of the database file alone would miss the row.
        assert live.with_name(live.name + "-wal").stat().st_size > 0

        sqlite_file_database.backup(tmp_path / "backup.db")

        with closing(sqlite3.connect(tmp_path / "backup.db")) as copy:
            assert copy.execute("SELECT name FROM probe_items").fetchall() == [("logged",)]

    def test_a_backup_never_replaces_a_file_nor_copies_a_transaction_in_flight(
        self, sqlite_file_database: Database, tmp_path: Path
    ) -> None:
        earlier = tmp_path / "earlier.db"
        earlier.write_bytes(b"an earlier backup")
        with pytest.raises(FileExistsError):
            sqlite_file_database.backup(earlier)
        assert earlier.read_bytes() == b"an earlier backup"

        with ProbeQueries(sqlite_file_database).transaction() as q:
            q.items.add(1, "uncommitted")
            with pytest.raises(NestedTransactionError):
                sqlite_file_database.backup(tmp_path / "during.db")
        assert not (tmp_path / "during.db").exists()

    def _commit_elsewhere(self, other_process: sqlite3.Connection) -> None:
        other_process.execute("BEGIN IMMEDIATE")
        other_process.execute("UPDATE probe_items SET size = size + 1")
        other_process.commit()

    def test_a_single_call_that_loses_a_race_is_retried(
        self, sqlite_file_database: Database, other_process: sqlite3.Connection
    ) -> None:
        queries = ProbeQueries(sqlite_file_database)
        queries.items.add(1, "a")
        first_attempt = iter([True])

        def commit_elsewhere_once() -> None:
            if next(first_attempt, False):
                self._commit_elsewhere(other_process)

        with _retries(sqlite_file_database) as retries:
            queries.items.read_then_write(commit_elsewhere_once)

        assert len(retries) == 1

    def test_a_unit_run_by_run_that_loses_a_race_is_retried(
        self, sqlite_file_database: Database, other_process: sqlite3.Connection
    ) -> None:
        queries = ProbeQueries(sqlite_file_database)
        queries.items.add(1, "a")
        first_attempt = iter([True])

        def commit_elsewhere_once() -> None:
            if next(first_attempt, False):
                self._commit_elsewhere(other_process)

        with _retries(sqlite_file_database) as retries:
            queries.run(lambda q: q.items.read_then_write(commit_elsewhere_once), read_only=True)

        assert len(retries) == 1

    def test_a_call_that_keeps_losing_races_gives_up_after_three_attempts(
        self, sqlite_file_database: Database, other_process: sqlite3.Connection
    ) -> None:
        queries = ProbeQueries(sqlite_file_database)
        queries.items.add(1, "a")

        with _retries(sqlite_file_database) as retries:
            with pytest.raises(ConflictError):
                queries.items.read_then_write(lambda: self._commit_elsewhere(other_process))

        assert len(retries) == 2

    def test_a_write_lock_held_too_long_elsewhere_fails_the_work_without_a_retry(
        self, sqlite_file_database: Database, other_process: sqlite3.Connection
    ) -> None:
        # A retry would wait the whole busy timeout again, holding up every other thread meanwhile.
        sqlite_file_database.sqlite.conn.execute("PRAGMA busy_timeout = 20;")
        other_process.execute("BEGIN IMMEDIATE")
        queries = ProbeQueries(sqlite_file_database)

        with _retries(sqlite_file_database) as retries:
            with pytest.raises(LockTimeoutError):
                queries.items.add(1, "a")
            with pytest.raises(LockTimeoutError):
                with queries.transaction() as q:
                    q.items.add(1, "a")
        other_process.rollback()

        assert retries == []
        assert queries.items.names() == []

    def test_a_failed_commit_leaves_the_database_usable(
        self, sqlite_file_database: Database, other_process: sqlite3.Connection
    ) -> None:
        # With a rollback journal (where WAL could not engage), a COMMIT needs every reader gone. A reader
        # in another process that stays past the busy timeout makes the COMMIT itself fail.
        sqlite_file_database.sqlite.conn.execute("PRAGMA journal_mode = DELETE;")
        sqlite_file_database.sqlite.conn.execute("PRAGMA busy_timeout = 20;")
        other_process.execute("BEGIN")
        other_process.execute("SELECT COUNT(*) FROM probe_items").fetchone()
        queries = ProbeQueries(sqlite_file_database)

        with pytest.raises(LockTimeoutError):
            with queries.transaction() as q:
                q.items.add(1, "a")
        other_process.rollback()

        # The failed transaction was rolled back: it neither blocks the next one nor left its write behind.
        queries.items.add(2, "b")
        assert queries.items.names() == ["b"]
        other_process.execute("BEGIN IMMEDIATE")
        other_process.rollback()

    def test_a_second_connection_through_the_engine_is_refused_and_harms_no_transaction(
        self, sqlite_file_database: Database
    ) -> None:
        # Another connection object over the shared driver connection would run outside the lock, and closing
        # it would roll back whatever was open on that connection -- here, another thread's transaction.
        queries = ProbeQueries(sqlite_file_database)
        attempted = threading.Event()
        refused: list[BaseException] = []

        def connect_meanwhile() -> None:
            try:
                sqlite_file_database.engine.connect()
            except BaseException as error:
                refused.append(error)
            attempted.set()

        with sqlite_file_database.legacy_cursor() as cursor:
            cursor.execute("INSERT INTO probe_items (id, name, size) VALUES (1, 'a', 0)")
            intruder = threading.Thread(target=connect_meanwhile)
            intruder.start()
            assert attempted.wait(10)
        intruder.join(10)

        assert len(refused) == 1
        assert queries.items.names() == ["a"]

    def test_an_interrupt_during_a_statement_leaves_the_database_open(
        self, sqlite_file_database: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dialect = sqlite_file_database.engine.dialect
        execute = dialect.do_execute

        def interrupted_once(*args: Any, **kwargs: Any) -> None:
            monkeypatch.setattr(dialect, "do_execute", execute)
            raise KeyboardInterrupt

        monkeypatch.setattr(dialect, "do_execute", interrupted_once)
        queries = ProbeQueries(sqlite_file_database)

        with pytest.raises(KeyboardInterrupt):
            queries.items.add(1, "a")

        # The statement's transaction rolled back, and the one connection is still there.
        queries.items.add(2, "b")
        assert queries.items.names() == ["b"]


@server_only
class TestServerBackends:
    def test_a_deadlock_fails_a_transaction_with_a_conflict(self, probe: ProbeQueries) -> None:
        probe.items.add(1, "a")
        probe.items.add(2, "b")
        first_rows_locked = threading.Barrier(2, timeout=10)
        outcomes: dict[str, str] = {}

        def lock_in_order(name: str, first: int, second: int) -> None:
            try:
                with probe.transaction() as q:
                    q.items.resize(first, 1)
                    first_rows_locked.wait()
                    q.items.resize(second, 1)
                outcomes[name] = "committed"
            except ConflictError:
                outcomes[name] = "conflict"

        workers = [
            threading.Thread(target=lock_in_order, args=("forward", 1, 2)),
            threading.Thread(target=lock_in_order, args=("backward", 2, 1)),
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(30)

        assert sorted(outcomes.values()) == ["committed", "conflict"]

    def test_a_single_call_that_deadlocks_is_retried(self, probe: ProbeQueries, empty_database: Database) -> None:
        probe.items.add(1, "a")
        probe.items.add(2, "b")
        first_rows_locked = threading.Barrier(2, timeout=10)
        attempted = threading.local()

        def meet_on_the_first_attempt() -> None:
            # Both calls hold their first row before either asks for its second: a deadlock. The retry of
            # the call that loses it runs straight through.
            if not getattr(attempted, "once", False):
                attempted.once = True
                first_rows_locked.wait()

        errors: list[BaseException] = []

        def resize(first: int, second: int) -> None:
            try:
                probe.items.resize_both(first, second, meet_on_the_first_attempt)
            except BaseException as error:
                errors.append(error)

        with _retries(empty_database) as retries:
            workers = [threading.Thread(target=resize, args=(1, 2)), threading.Thread(target=resize, args=(2, 1))]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(30)

        assert errors == []
        assert len(retries) == 1

    def test_a_unit_run_by_run_that_deadlocks_is_retried(self, probe: ProbeQueries, empty_database: Database) -> None:
        probe.items.add(1, "a")
        probe.items.add(2, "b")
        first_rows_locked = threading.Barrier(2, timeout=10)
        attempted = threading.local()

        def resize_in_one_unit(first: int, second: int) -> Callable[[ProbeQueries], None]:
            def work(q: ProbeQueries) -> None:
                q.items.resize(first, 1)
                # Both units hold their first row before either asks for its second: a deadlock. The retry of
                # the unit that loses it runs straight through.
                if not getattr(attempted, "once", False):
                    attempted.once = True
                    first_rows_locked.wait()
                q.items.resize(second, 1)

            return work

        errors: list[BaseException] = []

        def resize(first: int, second: int) -> None:
            try:
                probe.run(resize_in_one_unit(first, second))
            except BaseException as error:
                errors.append(error)

        with _retries(empty_database) as retries:
            workers = [threading.Thread(target=resize, args=(1, 2)), threading.Thread(target=resize, args=(2, 1))]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(30)

        assert errors == []
        assert len(retries) == 1

    def test_a_write_transaction_reads_what_others_committed_meanwhile(self, probe: ProbeQueries) -> None:
        # Writers run at READ COMMITTED: they take no gap locks, and a guard they evaluate sees current rows.
        # (Readers keep one snapshot; see test_a_read_only_transaction_reads_one_snapshot.)
        probe.items.add(1, "a")

        with probe.transaction() as q:
            before = q.items.count()
            other = threading.Thread(target=probe.items.add, args=(2, "b"))
            other.start()
            other.join(10)
            after = q.items.count()

        assert (before, after) == (1, 2)

    @pytest.mark.uses_database
    def test_a_url_naming_the_other_server_flavor_is_refused(self, _external_test_schema: URL) -> None:
        # Tables are created with the collations and column definitions of the flavor the URL names.
        other = "mysql+pymysql" if _external_test_schema.get_backend_name() == "mariadb" else "mariadb+pymysql"
        database = Database.open_url(
            _external_test_schema.set(drivername=other).render_as_string(hide_password=False),
            InvokeAILogger.get_logger("test_database"),
        )
        try:
            with pytest.raises((ValueError, InvalidRequestError), match="MariaDB"):
                with database.begin(write=False):
                    pass
        finally:
            database.dispose()

    def test_literals_compare_byte_for_byte(self, empty_database: Database) -> None:
        # The connection's collation, which comparisons of parameters and literals use: case and trailing
        # spaces matter, as in SQLite.
        with empty_database.begin(write=False) as conn:
            assert tuple(conn.exec_driver_sql("SELECT 'A' = 'a', 'a ' = 'a'").one()) == (0, 0)

    @pytest.fixture
    def lax_session_database(self, probe: ProbeQueries, _external_test_schema: URL) -> Iterator[Database]:
        """The schema under test, through connections that start with an empty sql_mode.

        Both servers default to strict modes, so only a session that starts lax shows that the layer
        pins its own mode instead of inheriting the server's.
        """
        lax = _external_test_schema.update_query_dict({"init_command": "SET SESSION sql_mode = ''"})
        database = Database.open_url(lax.render_as_string(hide_password=False), InvokeAILogger.get_logger("lax"))
        try:
            yield database
        finally:
            database.dispose()

    def test_a_value_too_long_for_its_column_is_rejected(self, lax_session_database: Database) -> None:
        queries = ProbeQueries(lax_session_database)

        with pytest.raises(DBAPIError) as raised:
            queries.items.add(1, "x" * 40)

        assert raised.value.orig is not None
        assert raised.value.orig.args[0] == 1406  # ER_DATA_TOO_LONG, instead of silent truncation
        assert queries.items.names() == []

    def test_an_ambiguous_group_by_is_rejected(self, lax_session_database: Database) -> None:
        queries = ProbeQueries(lax_session_database)
        queries.items.add(1, "a", size=1)
        queries.items.add(2, "b", size=1)

        with pytest.raises(DBAPIError) as raised:
            with lax_session_database.begin(write=False) as conn:
                conn.exec_driver_sql("SELECT name, COUNT(*) FROM probe_items GROUP BY size").all()

        assert raised.value.orig is not None
        assert raised.value.orig.args[0] == 1055  # ER_WRONG_FIELD_WITH_GROUP
