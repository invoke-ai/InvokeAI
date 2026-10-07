import random
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, closing, contextmanager, nullcontext
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Optional, TypeVar, get_args

from sqlalchemy import Connection, Engine, make_url
from sqlalchemy.exc import DBAPIError

from invokeai.app.services.config.config_default import DB_SYNCHRONOUS
from invokeai.app.services.shared.database.engines import (
    SERVER_DIALECTS,
    WRITE_INTENT,
    create_mysql_engine,
    create_sqlite_engine,
)
from invokeai.app.services.shared.database.errors import ConflictError, NestedTransactionError, translate_error
from invokeai.app.services.shared.database.queries import Queries

R = TypeVar("R")

# A transaction that lost a race (a deadlock) is run again from the start this many times in all before the
# error reaches the caller.
_CONFLICT_ATTEMPTS = 3
_CONFLICT_BASE_DELAY_SECONDS = 0.05

# The one server driver, whose error numbers `translate_error` reads.
_SUPPORTED_SERVER_DRIVER = "pymysql"


@dataclass(frozen=True)
class SqliteConnection:
    """The single connection of a SQLite database, the lock that serialises all work on it, and its file."""

    conn: sqlite3.Connection
    lock: threading.RLock
    path: Optional[Path]


def _refuse_stray_transaction(sqlite: SqliteConnection) -> None:
    """Names the cause when a statement left a transaction open on the raw SQLite connection.

    Python's sqlite3 begins a transaction before a data-modifying statement and keeps it open until
    commit(), so a write made on the raw connection without committing would otherwise make the next
    BEGIN fail with "cannot start a transaction within a transaction" -- and its rollback discard that write.
    """
    if sqlite.conn.in_transaction:
        raise RuntimeError(
            "The SQLite connection has a transaction open that no database transaction owns: a statement ran "
            "on the raw connection without being committed"
        )


class Database:
    """The application database: owns its connections and every transaction on them.

    Queries run through `queries`, either one call per transaction (`db.queries.<domain>.<method>(...)`)
    or several calls in one unit of work (`with db.queries.transaction() as q: ...`).

    SQLite is a single connection behind a process-wide re-entrant lock that is held for each whole
    transaction, which serialises all database work exactly as before this layer existed. MySQL and MariaDB
    use a connection pool and no process lock, so their invariants rest on the SQL itself.

    A thread may have one transaction open per database at a time; opening a second raises
    `NestedTransactionError` (see there for why nesting is not joined).
    """

    def __init__(self, engine: Engine, logger: Logger, *, sqlite: Optional[SqliteConnection] = None) -> None:
        self._engine = engine
        self._logger = logger
        self._sqlite = sqlite
        # The SQLite lock serialises every transaction, so one connection object serves them all. Checking
        # one out of the pool per transaction would cost about as much as the statement of a point read.
        self._sqlite_connection: Optional[Connection] = engine.connect() if sqlite is not None else None
        self._thread_state = threading.local()
        self._disposed = False
        self.queries = Queries(self)

    @classmethod
    def open_sqlite(
        cls,
        db_path: Optional[Path],
        logger: Logger,
        *,
        verbose: bool = False,
        synchronous: DB_SYNCHRONOUS = "full",
    ) -> "Database":
        """Opens a SQLite database file, or an in-memory database when `db_path` is None.

        :param verbose: Log every SQL statement at debug level.
        :param synchronous: SQLite `synchronous` setting. Defaults to `full`, what InvokeAI has always used.
            `normal` is refused, with a warning, for a database file whose journal mode did not become WAL.
        """
        # Called directly by the user-management commands and tests, not only through the validated
        # config, and `synchronous` is interpolated into a PRAGMA below. `DB_SYNCHRONOUS` is a static
        # annotation, so the value is checked here, before anything is opened.
        if synchronous not in get_args(DB_SYNCHRONOUS):
            raise ValueError(f"Invalid synchronous setting {synchronous!r}, expected one of {get_args(DB_SYNCHRONOUS)}")

        if db_path is None:
            logger.info("Initializing in-memory database")
        else:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initializing database at {db_path}")

        conn = sqlite3.connect(database=db_path or ":memory:", check_same_thread=False)
        # Rows of the transitional cursor API; queries get plain tuples (see `_begin_sqlite`).
        conn.row_factory = sqlite3.Row

        if verbose:
            # On the connection rather than the engine, so it also covers the transitional cursor API.
            conn.set_trace_callback(logger.debug)

        conn.execute("PRAGMA foreign_keys = ON;")

        # Enable Write-Ahead Logging (WAL) mode for better concurrency. The statement reports the mode
        # that was actually established: WAL needs shared memory, so it can fail to engage -- quietly --
        # on network filesystems, some container volume drivers and `SQLITE_OMIT_WAL` builds.
        journal_mode = str(conn.execute("PRAGMA journal_mode = WAL;").fetchone()[0]).lower()

        # Set a busy timeout to prevent database lockups during writes
        conn.execute("PRAGMA busy_timeout = 5000;")  # 5 seconds

        # Durability. SQLite's own default is `full`, which fsyncs on every commit; `normal` trades the
        # last transactions on a power loss or OS crash for roughly 12x shorter commits. Shorter commits
        # matter twice here, because every write holds the lock that serialises all database work.
        #
        # `normal` cannot corrupt the database only *because* of WAL; on a rollback journal it can, so a
        # database file that did not get WAL keeps `full` instead. In-memory databases report `memory`
        # and have no durability to trade, so the requirement does not apply to them.
        #
        # Refused rather than fatal: the setting is a performance preference, and starting with stronger
        # durability than asked for costs commit time, where starting with weaker durability than the
        # documented guarantee risks the user's database. A warning names the mode so the cause is
        # visible instead of merely slow.
        if synchronous == "normal" and db_path is not None and journal_mode != "wal":
            logger.warning(
                f"Journal mode is '{journal_mode}', not WAL, so db_synchronous='normal' was refused and "
                "'full' used instead: without WAL that setting can corrupt the database on power loss."
            )
            synchronous = "full"

        # PRAGMA does not take bind parameters, so the value is interpolated. It is checked against
        # DB_SYNCHRONOUS at the top of this method.
        conn.execute(f"PRAGMA synchronous = {synchronous.upper()};")

        return cls(
            create_sqlite_engine(conn), logger, sqlite=SqliteConnection(conn=conn, lock=threading.RLock(), path=db_path)
        )

    @classmethod
    def open_url(cls, url: str, logger: Logger) -> "Database":
        """Opens a MySQL or MariaDB database through PyMySQL, e.g. `mariadb+pymysql://user:password@host/invokeai`."""
        parsed = make_url(url)
        if parsed.get_backend_name() not in SERVER_DIALECTS or parsed.get_driver_name() != _SUPPORTED_SERVER_DRIVER:
            raise ValueError(
                f"Unsupported database URL scheme '{parsed.drivername}', expected "
                f"{' or '.join(f'{backend}+{_SUPPORTED_SERVER_DRIVER}' for backend in SERVER_DIALECTS)}"
            )
        logger.info(f"Connecting to database {parsed.render_as_string(hide_password=True)}")
        return cls(create_mysql_engine(parsed), logger)

    @property
    def dialect_name(self) -> str:
        """`sqlite`, `mysql` or `mariadb`."""
        return self._engine.dialect.name

    @property
    def engine(self) -> Engine:
        """The SQLAlchemy engine, for instrumentation (event listeners) in tests and benchmarks, and for a
        connection of its own outside any transaction, as the migrator's lock needs on a server.

        On SQLite, connecting through it raises: the one connection is held by this database. Use `begin()`.
        """
        return self._engine

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def sqlite(self) -> SqliteConnection:
        """Transitional: the raw SQLite connection, its lock and its file, for the migrator and the cursor facade.

        Raises on any other backend.
        """
        if self._sqlite is None:
            raise RuntimeError(f"The {self.dialect_name} database has no SQLite connection")
        return self._sqlite

    @contextmanager
    def begin(self, *, write: bool) -> Iterator[Connection]:
        """One transaction on one connection, committed when the block exits normally and rolled back otherwise.

        For the database layer itself (query modules, the migrator, the copy tool); services run their
        queries through `queries`. Driver errors with a backend-neutral meaning are raised as the matching
        `DatabaseError` subclass.

        :param write: Whether the transaction may write. On SQLite a writer begins with BEGIN IMMEDIATE.
        """
        self._claim_thread("query")
        try:
            with self._exclusive():
                with self._connection() as conn:
                    began_sqlite = False
                    try:
                        if self._sqlite is not None:
                            self._begin_sqlite(write)
                            began_sqlite = True
                        else:
                            conn.execution_options(**{WRITE_INTENT: write})
                        with conn.begin():
                            yield conn
                    except (DBAPIError, sqlite3.Error) as error:
                        translated = translate_error(error, self.dialect_name)
                        if translated is None:
                            raise
                        raise translated from error
                    finally:
                        if began_sqlite:
                            self._end_sqlite()
        finally:
            self._release_thread()

    def _begin_sqlite(self, write: bool) -> None:
        """Begins the transaction that the SQLAlchemy transaction around it then commits or rolls back.

        The driver begins a transaction only before data-modifying statements: a read would see a new
        snapshot per statement, and DDL would commit as it runs. So it is begun here, explicitly, and on the
        driver connection: SQLAlchemy's own begin is a no-op for this driver, and issuing the BEGIN through
        SQLAlchemy would cost as much again as the point read it begins.
        """
        assert self._sqlite is not None
        _refuse_stray_transaction(self._sqlite)
        self._sqlite.conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        # SQLAlchemy builds its own rows; the driver need not build `sqlite3.Row` objects first. (A third
        # of the time of a large fetch, measured.) The lock is held, so no cursor of the facade runs meanwhile.
        self._sqlite.conn.row_factory = None

    def _end_sqlite(self) -> None:
        assert self._sqlite is not None
        self._sqlite.conn.row_factory = sqlite3.Row
        if self._sqlite.conn.in_transaction:
            # When the COMMIT itself fails, SQLAlchemy skips the driver rollback, and this connection never
            # returns to a pool whose reset would roll it back instead.
            self._sqlite.conn.rollback()

    def run(self, work: Callable[[Connection], R], *, write: bool) -> R:
        """Runs `work` in a transaction of its own, again from the start when it loses a race (`ConflictError`).

        `work` must have no effects outside the transaction, since a retry repeats it.
        """

        def transaction() -> R:
            with self.begin(write=write) as conn:
                return work(conn)

        return self.retry_conflicts(transaction)

    def retry_conflicts(self, transaction: Callable[[], R]) -> R:
        """Calls `transaction`, which runs one transaction, again when it loses a race (`ConflictError`): three
        attempts in all, with a short random pause between them."""
        attempt = 1
        while True:
            try:
                return transaction()
            except ConflictError as error:
                if attempt >= _CONFLICT_ATTEMPTS:
                    raise
                self._logger.warning(
                    f"Retrying a database transaction that lost a race "
                    f"(attempt {attempt + 1} of {_CONFLICT_ATTEMPTS}): {error}"
                )
                time.sleep(_CONFLICT_BASE_DELAY_SECONDS * attempt * random.uniform(0.5, 1.5))
                attempt += 1

    @contextmanager
    def legacy_cursor(self) -> Iterator[sqlite3.Cursor]:
        """Transitional: a raw cursor in a transaction, for services not yet ported to `queries`.

        The transaction begins as the driver begins it, implicitly before the first data-modifying
        statement, so the reads before it take no lock and see the latest committed data -- as they always
        have. Code that must hold the write lock from its first read issues BEGIN IMMEDIATE itself.

        A legacy transaction opened on a thread that already has one open joins it: the inner block neither
        commits nor rolls back, so it no longer commits the outer block's work early. Nesting a legacy
        transaction with a `queries` transaction, in either order, raises `NestedTransactionError`.
        """
        sqlite = self.sqlite
        with sqlite.lock:
            if getattr(self._thread_state, "open", None) == "legacy":
                joined = sqlite.conn.cursor()
                try:
                    yield joined
                finally:
                    joined.close()
                return
            self._claim_thread("legacy")
            try:
                _refuse_stray_transaction(sqlite)
                cursor = sqlite.conn.cursor()
                try:
                    yield cursor
                    sqlite.conn.commit()
                except BaseException:
                    sqlite.conn.rollback()
                    raise
                finally:
                    cursor.close()
            finally:
                self._release_thread()

    def clean(self) -> None:
        """Reclaims the free pages of a SQLite database file, reporting the freed space.

        A no-op for in-memory and server databases.
        """
        if self._sqlite is None or self._sqlite.path is None:
            return
        sqlite = self._sqlite
        assert sqlite.path is not None
        try:
            # Serialize with transactions: services with their own threads (e.g. the image index worker)
            # may have a statement in progress on this shared connection, and VACUUM refuses to run
            # concurrently with one ("cannot VACUUM - SQL statements in progress").
            with sqlite.lock:
                initial_db_size = sqlite.path.stat().st_size
                sqlite.conn.execute("VACUUM;")
                final_db_size = sqlite.path.stat().st_size
            freed_space_in_mb = round((initial_db_size - final_db_size) / 1024 / 1024, 2)
            if freed_space_in_mb > 0:
                self._logger.info(f"Cleaned database (freed {freed_space_in_mb}MB)")
        except Exception as e:
            self._logger.error(f"Error cleaning database: {e}")
            raise

    def backup(self, destination: Path) -> None:
        """Copies a SQLite database into the file `destination` with SQLite's online backup, which is consistent
        while the database is in use and includes what its write-ahead log holds. (A copy of the database file is
        not.) A server database is backed up by its operator; this raises there.

        It holds the database's lock while it copies, so it is for startup and for tools, not for a running app's
        requests. It refuses a destination that exists, and a caller inside a transaction of its own, whose
        uncommitted changes the copy would hold."""
        sqlite = self.sqlite
        if destination.exists():
            raise FileExistsError(f"Backup destination {destination} exists")
        with sqlite.lock:
            if sqlite.conn.in_transaction:
                raise NestedTransactionError("A backup cannot be taken inside a transaction")
            with closing(sqlite3.connect(destination)) as target:
                sqlite.conn.backup(target)

    def dispose(self) -> None:
        """Closes every connection; transactions begun afterwards are refused. Safe to call more than once.

        Closing a SQLite connection also checkpoints its write-ahead log into the database file.
        """
        with self._exclusive():
            self._disposed = True
            if self._sqlite_connection is not None:
                self._sqlite_connection.close()
            self._engine.dispose()
            if self._sqlite is not None:
                self._sqlite.conn.close()

    def _exclusive(self) -> AbstractContextManager[object]:
        return self._sqlite.lock if self._sqlite is not None else nullcontext()

    def _connection(self) -> AbstractContextManager[Connection]:
        """The SQLite connection object, held under the lock; on a server, a connection from the pool."""
        if self._sqlite_connection is not None:
            return nullcontext(self._sqlite_connection)
        return self._engine.connect()

    def _claim_thread(self, kind: str) -> None:
        if self._disposed:
            raise RuntimeError("The database has been closed")
        open_kind = getattr(self._thread_state, "open", None)
        if open_kind is not None:
            raise NestedTransactionError(
                f"This thread already has a {open_kind} transaction open on this database; "
                "run the work on that transaction instead of opening another"
            )
        self._thread_state.open = kind

    def _release_thread(self) -> None:
        self._thread_state.open = None
