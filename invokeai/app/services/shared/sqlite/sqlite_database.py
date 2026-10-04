import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from logging import Logger
from pathlib import Path
from typing import Optional

from invokeai.app.services.config.config_default import DB_SYNCHRONOUS
from invokeai.app.services.shared.database.database import Database


class SqliteDatabase:
    """
    Cursor-level access to the SQLite database, for services that are not yet ported to the query layer.

    Transitional: this wraps a `Database` -- the same connection under the same lock -- and goes away once
    every service queries through `Database.queries`. New code uses `Database` and its queries instead.

    :param db_path: Path to the database file. If None, an in-memory database is used.
    :param logger: Logger to use for logging.
    :param verbose: Whether to log SQL statements. Provides `logger.debug` as the SQLite trace callback.
    :param synchronous: SQLite `synchronous` setting. Defaults to `full`, what InvokeAI has always used.
        `normal` is refused, with a warning, for a database file whose journal mode did not become WAL.

    In addition to the constructor args, the instance provides:
    - `database`: the wrapped `Database`.
    - `transaction()`: a cursor in a transaction; see `Database.legacy_cursor` for how transactions nest.
    - `clean()`: Runs the SQL `VACUUM;` command and reports on the freed space.
    """

    database: Database

    def __init__(
        self,
        db_path: Optional[Path],
        logger: Logger,
        verbose: bool = False,
        synchronous: DB_SYNCHRONOUS = "full",
    ) -> None:
        self.database = Database.open_sqlite(db_path, logger, verbose=verbose, synchronous=synchronous)

    @property
    def _conn(self) -> sqlite3.Connection:
        return self.database.sqlite.conn

    @property
    def _db_path(self) -> Optional[Path]:
        return self.database.sqlite.path

    @property
    def _logger(self) -> Logger:
        return self.database.logger

    def clean(self) -> None:
        """
        Cleans the database by running the VACUUM command, reporting on the freed space.
        """
        self.database.clean()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """
        Thread-safe context manager for DB work.
        Acquires the RLock, yields a Cursor, then commits or rolls back. Opened again on a thread that already
        has one open, it joins that transaction instead of committing it early (see `Database.legacy_cursor`).
        """
        with self.database.legacy_cursor() as cursor:
            yield cursor
