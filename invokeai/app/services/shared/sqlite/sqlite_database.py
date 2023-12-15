import sqlite3
import threading
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
        self.logger = logger
        self.db_path = db_path
        self.verbose = verbose

        if not self.db_path:
            logger.info("Initializing in-memory database")
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Initializing database at {self.db_path}")

        self.conn = sqlite3.connect(database=self.db_path or sqlite_memory, check_same_thread=False)
        self.lock = threading.RLock()
        self.conn.row_factory = sqlite3.Row

        if self.verbose:
            self.conn.set_trace_callback(self.logger.debug)

        self.conn.execute("PRAGMA foreign_keys = ON;")

    def clean(self) -> None:
        """
        Cleans the database by running the VACUUM command, reporting on the freed space.
        """
        # No need to clean in-memory database
        if not self.db_path:
            return
        with self.lock:
            try:
                initial_db_size = Path(self.db_path).stat().st_size
                self.conn.execute("VACUUM;")
                self.conn.commit()
                final_db_size = Path(self.db_path).stat().st_size
                freed_space_in_mb = round((initial_db_size - final_db_size) / 1024 / 1024, 2)
                if freed_space_in_mb > 0:
                    self.logger.info(f"Cleaned database (freed {freed_space_in_mb}MB)")
            except Exception as e:
                self.logger.error(f"Error cleaning database: {e}")
                raise
