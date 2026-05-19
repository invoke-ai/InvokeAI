import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from logging import Logger
from pathlib import Path
from uuid import uuid4

from sqlalchemy import event
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from invokeai.app.services.shared.sqlite.sqlite_common import sqlite_memory


class SqliteDatabase:
    """
    Manages a connection to an SQLite database.

    :param db_path: Path to the database file. If None, an in-memory database is used.
    :param logger: Logger to use for logging.
    :param verbose: Whether to log SQL statements. Provides `logger.debug` as the SQLite trace callback.

    This is a light wrapper around the `sqlite3` module, providing a few conveniences:
    - The database file is written to disk if it does not exist.
    - Foreign key constraints are enabled by default.
    - The connection is configured to use the `sqlite3.Row` row factory.

    In addition to the constructor args, the instance provides the following attributes and methods:
    - `conn`: A `sqlite3.Connection` object. Note that the connection must never be closed if the database is in-memory.
    - `lock`: A shared re-entrant lock, used to approximate thread safety.
    - `clean()`: Runs the SQL `VACUUM;` command and reports on the freed space.
    - `get_session()`: Returns a SQLModel Session for ORM-based queries.
    """

    def __init__(self, db_path: Path | None, logger: Logger, verbose: bool = False) -> None:
        """Initializes the database. This is used internally by the class constructor."""
        self._logger = logger
        self._db_path = db_path
        self._verbose = verbose
        self._lock = threading.RLock()

        if not self._db_path:
            logger.info("Initializing in-memory database")
        else:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"Initializing database at {self._db_path}")

        self._conn = sqlite3.connect(database=self._db_path or sqlite_memory, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        if self._verbose:
            self._conn.set_trace_callback(self._logger.debug)

        # Enable foreign key constraints
        self._conn.execute("PRAGMA foreign_keys = ON;")

        # Enable Write-Ahead Logging (WAL) mode for better concurrency
        self._conn.execute("PRAGMA journal_mode = WAL;")

        # Set a busy timeout to prevent database lockups during writes
        self._conn.execute("PRAGMA busy_timeout = 5000;")  # 5 seconds

        # Set up the SQLAlchemy engine for SQLModel-based queries.
        # For file-based DBs, both connections point to the same file.
        # For in-memory DBs, we use a named shared cache so both connections
        # see the same database.
        if self._db_path:
            db_uri = f"sqlite:///{self._db_path}"
            # StaticPool reuses a single connection — ideal for SQLite which
            # serializes writes anyway. Avoids the overhead of creating a new
            # connection for every Session.
            self._engine = create_engine(
                db_uri,
                echo=self._verbose,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            # Use a shared in-memory database via URI with shared cache.
            # The raw sqlite3 connection above already created ":memory:",
            # so we re-create it with the shared URI instead.
            shared_uri = f"file:invokeai_memdb_{uuid4().hex}?mode=memory&cache=shared"
            self._conn.close()
            self._conn = sqlite3.connect(shared_uri, uri=True, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            if self._verbose:
                self._conn.set_trace_callback(self._logger.debug)
            self._conn.execute("PRAGMA foreign_keys = ON;")

            self._engine = create_engine(
                "sqlite+pysqlite://",
                echo=self._verbose,
                creator=lambda: sqlite3.connect(shared_uri, uri=True, check_same_thread=False),
                poolclass=StaticPool,
            )

        # Apply the same PRAGMAs to all SQLAlchemy connections
        @event.listens_for(self._engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, connection_record):  # type: ignore
            cursor = dbapi_connection.cursor()
            # Note: We intentionally skip PRAGMA foreign_keys for the SQLAlchemy engine.
            # Migration 22 renames the `models` table which corrupts FK references in
            # `model_relationships`. The raw sqlite3 connection already enforces FKs
            # for the migration phase. The SQLAlchemy engine is used only for queries
            # after migrations are complete.
            if self._db_path:
                cursor.execute("PRAGMA journal_mode = WAL;")
                cursor.execute("PRAGMA busy_timeout = 5000;")
            cursor.close()

    def clean(self) -> None:
        """
        Cleans the database by running the VACUUM command, reporting on the freed space.
        """
        # No need to clean in-memory database
        if not self._db_path:
            return
        try:
            with self._conn as conn:
                initial_db_size = Path(self._db_path).stat().st_size
                conn.execute("VACUUM;")
                conn.commit()
                final_db_size = Path(self._db_path).stat().st_size
                freed_space_in_mb = round((initial_db_size - final_db_size) / 1024 / 1024, 2)
                if freed_space_in_mb > 0:
                    self._logger.info(f"Cleaned database (freed {freed_space_in_mb}MB)")
        except Exception as e:
            self._logger.error(f"Error cleaning database: {e}")
            raise

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager that yields a SQLModel Session for write operations.
        Commits on success, rolls back on exception.

        Uses expire_on_commit=False so that model attributes remain accessible
        after commit without triggering lazy-loads or DetachedInstanceError.
        """
        with Session(self._engine, expire_on_commit=False) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    @contextmanager
    def get_readonly_session(self) -> Generator[Session, None, None]:
        """
        Context manager that yields a lightweight read-only SQLModel Session.

        Optimized for SELECT queries:
        - autoflush=False: skips the automatic flush before every query
        - no commit/rollback: avoids transaction overhead for reads
        - expire_on_commit=False: attributes stay accessible after close
        """
        with Session(self._engine, expire_on_commit=False, autoflush=False) as session:
            yield session

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """
        Thread-safe context manager for DB work.
        Acquires the RLock, yields a Cursor, then commits or rolls back.
        """
        with self._lock:
            cursor = self._conn.cursor()
            try:
                yield cursor
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cursor.close()
