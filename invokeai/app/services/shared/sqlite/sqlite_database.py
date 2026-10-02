import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from logging import Logger
from pathlib import Path
from typing import get_args

from invokeai.app.services.config.config_default import DB_SYNCHRONOUS
from invokeai.app.services.shared.sqlite.sqlite_common import sqlite_memory


class SqliteDatabase:
    """
    Manages a connection to an SQLite database.

    :param db_path: Path to the database file. If None, an in-memory database is used.
    :param logger: Logger to use for logging.
    :param verbose: Whether to log SQL statements. Provides `logger.debug` as the SQLite trace callback.
    :param synchronous: SQLite `synchronous` setting. Defaults to `full`, what InvokeAI has always used.
        `normal` is refused, with a warning, for a database file whose journal mode did not become WAL.

    This is a light wrapper around the `sqlite3` module, providing a few conveniences:
    - The database file is written to disk if it does not exist.
    - Foreign key constraints are enabled by default.
    - The connection is configured to use the `sqlite3.Row` row factory.

    In addition to the constructor args, the instance provides the following attributes and methods:
    - `conn`: A `sqlite3.Connection` object. Note that the connection must never be closed if the database is in-memory.
    - `lock`: A shared re-entrant lock, used to approximate thread safety.
    - `clean()`: Runs the SQL `VACUUM;` command and reports on the freed space.
    """

    def __init__(
        self,
        db_path: Path | None,
        logger: Logger,
        verbose: bool = False,
        synchronous: DB_SYNCHRONOUS = "full",
    ) -> None:
        """Initializes the database. This is used internally by the class constructor."""
        # This constructor is called directly (user management commands, tests), not only through the
        # validated config, and `synchronous` is interpolated into a PRAGMA below. `DB_SYNCHRONOUS` is
        # a static annotation, so the value is checked here, before anything is opened.
        if synchronous not in get_args(DB_SYNCHRONOUS):
            raise ValueError(f"Invalid synchronous setting {synchronous!r}, expected one of {get_args(DB_SYNCHRONOUS)}")

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

        # Enable Write-Ahead Logging (WAL) mode for better concurrency. The statement reports the mode
        # that was actually established: WAL needs shared memory, so it can fail to engage -- quietly --
        # on network filesystems, some container volume drivers and `SQLITE_OMIT_WAL` builds.
        journal_mode = str(self._conn.execute("PRAGMA journal_mode = WAL;").fetchone()[0]).lower()

        # Set a busy timeout to prevent database lockups during writes
        self._conn.execute("PRAGMA busy_timeout = 5000;")  # 5 seconds

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
        if synchronous == "normal" and self._db_path is not None and journal_mode != "wal":
            self._logger.warning(
                f"Journal mode is '{journal_mode}', not WAL, so db_synchronous='normal' was refused and "
                "'full' used instead: without WAL that setting can corrupt the database on power loss."
            )
            synchronous = "full"

        # PRAGMA does not take bind parameters, so the value is interpolated. It is checked against
        # DB_SYNCHRONOUS at the top of this constructor.
        self._conn.execute(f"PRAGMA synchronous = {synchronous.upper()};")

    def clean(self) -> None:
        """
        Cleans the database by running the VACUUM command, reporting on the freed space.
        """
        # No need to clean in-memory database
        if not self._db_path:
            return
        try:
            # Serialize with transaction() users: services with their own
            # threads (e.g. the image index worker) may have a statement in
            # progress on this shared connection, and VACUUM refuses to run
            # concurrently with one ("cannot VACUUM - SQL statements in
            # progress").
            with self._lock, self._conn as conn:
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
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cursor.close()
