import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from logging import Logger
from pathlib import Path

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
            except:
                self._conn.rollback()
                raise
            finally:
                cursor.close()
