import shutil
import sqlite3
import threading
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Callable, Optional, TypeAlias

from invokeai.app.services.shared.sqlite.sqlite_common import sqlite_memory

MigrateCallback: TypeAlias = Callable[[sqlite3.Cursor], None]


class MigrationError(Exception):
    """Raised when a migration fails."""


class MigrationVersionError(ValueError, MigrationError):
    """Raised when a migration version is invalid."""


class Migration:
    """Represents a migration for a SQLite database.

    :param db_version: The database schema version this migration results in.
    :param app_version: The app version this migration is introduced in.
    :param migrate: The callback to run to perform the migration. The callback will be passed a
    cursor to the database. The migrator will manage locking database access and committing the
    transaction; the callback should not do either of these things.
    """

    def __init__(
        self,
        db_version: int,
        app_version: str,
        migrate: MigrateCallback,
    ) -> None:
        self.db_version = db_version
        self.app_version = app_version
        self.migrate = migrate


class SQLiteMigrator:
    """
    Manages migrations for a SQLite database.

    :param conn: The database connection.
    :param database: The path to the database file, or ":memory:" for an in-memory database.
    :param lock: A lock to use when accessing the database.
    :param logger: The logger to use.

    Migrations should be registered with :meth:`register_migration`. Migrations will be run in
    order of their version number. If the database is already at the latest version, no migrations
    will be run.
    """

    def __init__(self, conn: sqlite3.Connection, database: Path | str, lock: threading.RLock, logger: Logger) -> None:
        self._logger = logger
        self._conn = conn
        self._cursor = self._conn.cursor()
        self._lock = lock
        self._database = database
        self._migrations: set[Migration] = set()

    def register_migration(self, migration: Migration) -> None:
        """Registers a migration."""
        if not isinstance(migration.db_version, int) or migration.db_version < 1:
            raise MigrationVersionError(f"Invalid migration version {migration.db_version}")
        if any(m.db_version == migration.db_version for m in self._migrations):
            raise MigrationVersionError(f"Migration version {migration.db_version} already registered")
        self._migrations.add(migration)
        self._logger.debug(f"Registered migration {migration.db_version}")

    def run_migrations(self) -> None:
        """Migrates the database to the latest version."""
        with self._lock:
            self._create_version_table()
            sorted_migrations = sorted(self._migrations, key=lambda m: m.db_version)
            current_version = self._get_current_version()

            if len(sorted_migrations) == 0:
                self._logger.debug("No migrations registered")
                return

            latest_version = sorted_migrations[-1].db_version
            if current_version == latest_version:
                self._logger.debug("Database is up to date, no migrations to run")
                return

            if current_version > latest_version:
                raise MigrationError(
                    f"Database version {current_version} is greater than the latest migration version {latest_version}"
                )

            self._logger.info("Database update needed")

            # Only make a backup if using a file database (not memory)
            backup_path: Optional[Path] = None
            if isinstance(self._database, Path):
                backup_path = self._backup_db(self._database)
            else:
                self._logger.info("Using in-memory database, skipping backup")

            for migration in sorted_migrations:
                try:
                    self._run_migration(migration)
                except MigrationError:
                    if backup_path is not None:
                        self._logger.error(f" Restoring from {backup_path}")
                        self._restore_db(backup_path)
                    raise
            self._logger.info("Database updated successfully")

    def _run_migration(self, migration: Migration) -> None:
        """Runs a single migration."""
        with self._lock:
            current_version = self._get_current_version()
            try:
                if current_version >= migration.db_version:
                    return
                migration.migrate(self._cursor)
                # Migration callbacks only get a cursor; they cannot commit the transaction.
                self._conn.commit()
                self._set_version(db_version=migration.db_version, app_version=migration.app_version)
                self._logger.debug(f"Successfully migrated database from {current_version} to {migration.db_version}")
            except Exception as e:
                msg = f"Error migrating database from {current_version} to {migration.db_version}: {e}"
                self._conn.rollback()
                self._logger.error(msg)
                raise MigrationError(msg) from e

    def _create_version_table(self) -> None:
        """Creates a version table for the database, if one does not already exist."""
        with self._lock:
            try:
                self._cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='version';")
                if self._cursor.fetchone() is not None:
                    return
                self._cursor.execute(
                    """--sql
                    CREATE TABLE IF NOT EXISTS version (
                        db_version INTEGER PRIMARY KEY,
                        app_version TEXT NOT NULL,
                        migrated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW'))
                    );
                    """
                )
                self._cursor.execute("INSERT INTO version (db_version, app_version) VALUES (?,?);", (0, "0.0.0"))
                self._conn.commit()
                self._logger.debug("Created version table")
            except sqlite3.Error as e:
                msg = f"Problem creation version table: {e}"
                self._logger.error(msg)
                self._conn.rollback()
                raise MigrationError(msg) from e

    def _get_current_version(self) -> int:
        """Gets the current version of the database, or 0 if the version table does not exist."""
        with self._lock:
            try:
                self._cursor.execute("SELECT MAX(db_version) FROM version;")
                version = self._cursor.fetchone()[0]
                if version is None:
                    return 0
                return version
            except sqlite3.OperationalError as e:
                if "no such table" in str(e):
                    return 0
                raise

    def _set_version(self, db_version: int, app_version: str) -> None:
        """Adds a version entry to the table's version table."""
        with self._lock:
            try:
                self._cursor.execute(
                    "INSERT INTO version (db_version, app_version) VALUES (?,?);", (db_version, app_version)
                )
                self._conn.commit()
            except sqlite3.Error as e:
                msg = f"Problem setting database version: {e}"
                self._logger.error(msg)
                self._conn.rollback()
                raise MigrationError(msg) from e

    def _backup_db(self, db_path: Path | str) -> Path:
        """Backs up the databse, returning the path to the backup file."""
        if db_path == sqlite_memory:
            raise MigrationError("Cannot back up memory database")
        if not isinstance(db_path, Path):
            raise MigrationError(f'Database path must be "{sqlite_memory}" or a Path')
        with self._lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = db_path.parent / f"{db_path.stem}_{timestamp}.db"
            self._logger.info(f"Backing up database to {backup_path}")
            backup_conn = sqlite3.connect(backup_path)
            with backup_conn:
                self._conn.backup(backup_conn)
            backup_conn.close()
            return backup_path

    def _restore_db(self, backup_path: Path) -> None:
        """Restores the database from a backup file, unless the database is a memory database."""
        if self._database == sqlite_memory:
            return
        with self._lock:
            self._logger.info(f"Restoring database from {backup_path}")
            self._conn.close()
            if not Path(backup_path).is_file():
                raise FileNotFoundError(f"Backup file {backup_path} does not exist")
            shutil.copy2(backup_path, self._database)
