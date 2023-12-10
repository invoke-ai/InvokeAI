import shutil
import sqlite3
import threading
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Callable, Optional, TypeAlias

from pydantic import BaseModel, Field, model_validator

from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.shared.sqlite.sqlite_common import sqlite_memory

MigrateCallback: TypeAlias = Callable[[sqlite3.Cursor, ImageFileStorageBase, Logger], None]


class MigrationError(Exception):
    """Raised when a migration fails."""


class MigrationVersionError(ValueError, MigrationError):
    """Raised when a migration version is invalid."""


class Migration(BaseModel):
    """
    Represents a migration for a SQLite database.

    Migration callbacks will be provided an instance of SqliteDatabase.
    Migration callbacks should not commit; the migrator will commit the transaction.
    """

    from_version: int = Field(ge=0, strict=True, description="The database version on which this migration may be run")
    to_version: int = Field(ge=1, strict=True, description="The database version that results from this migration")
    migrate: MigrateCallback = Field(description="The callback to run to perform the migration")
    pre_migrate: list[MigrateCallback] = Field(
        default=[], description="A list of callbacks to run before the migration"
    )
    post_migrate: list[MigrateCallback] = Field(
        default=[], description="A list of callbacks to run after the migration"
    )

    @model_validator(mode="after")
    def validate_to_version(self) -> "Migration":
        if self.to_version <= self.from_version:
            raise ValueError("to_version must be greater than from_version")
        return self

    def __hash__(self) -> int:
        # Callables are not hashable, so we need to implement our own __hash__ function to use this class in a set.
        return hash((self.from_version, self.to_version))


class MigrationSet:
    """A set of Migrations. Performs validation during migration registration and provides utility methods."""

    def __init__(self) -> None:
        self._migrations: set[Migration] = set()

    def register(self, migration: Migration) -> None:
        """Registers a migration."""
        if any(m.from_version == migration.from_version for m in self._migrations):
            raise MigrationVersionError(f"Migration from {migration.from_version} already registered")
        if any(m.to_version == migration.to_version for m in self._migrations):
            raise MigrationVersionError(f"Migration to {migration.to_version} already registered")
        self._migrations.add(migration)

    def get(self, from_version: int) -> Optional[Migration]:
        """Gets the migration that may be run on the given database version."""
        # register() ensures that there is only one migration with a given from_version, so this is safe.
        return next((m for m in self._migrations if m.from_version == from_version), None)

    @property
    def count(self) -> int:
        """The count of registered migrations."""
        return len(self._migrations)

    @property
    def latest_version(self) -> int:
        """Gets latest to_version among registered migrations. Returns 0 if there are no migrations registered."""
        if self.count == 0:
            return 0
        return sorted(self._migrations, key=lambda m: m.to_version)[-1].to_version


class SQLiteMigrator:
    """
    Manages migrations for a SQLite database.

    :param db: The SqliteDatabase, representing the database on which to run migrations.
    :param image_files: An instance of ImageFileStorageBase. Migrations may need to access image files.

    Migrations should be registered with :meth:`register_migration`. Migrations will be run in
    order of their version number. If the database is already at the latest version, no migrations
    will be run.
    """

    backup_path: Optional[Path] = None

    def __init__(
        self,
        database: Path | str,
        lock: threading.RLock,
        logger: Logger,
        image_files: ImageFileStorageBase,
    ) -> None:
        self._lock = lock
        self._database = database
        self._is_memory = database == sqlite_memory
        self._image_files = image_files
        self._logger = logger
        self._conn = sqlite3.connect(database)
        self._cursor = self._conn.cursor()
        self._migrations = MigrationSet()

        # Use a lock file to indicate that a migration is in progress. Should only exist in the event of catastrophic failure.
        self._migration_lock_file_path = (
            self._database.parent / ".migration_in_progress" if isinstance(self._database, Path) else None
        )

        if self._unlink_lock_file():
            self._logger.warning("Previous migration failed! Trying again...")

    def register_migration(self, migration: Migration) -> None:
        """
        Registers a migration.
        Migration callbacks should not commit any changes to the database; the migrator will commit the transaction.
        """
        self._migrations.register(migration)
        self._logger.debug(f"Registered migration {migration.from_version} -> {migration.to_version}")

    def run_migrations(self) -> None:
        """Migrates the database to the latest version."""
        with self._lock:
            self._create_version_table()
            current_version = self._get_current_version()

            if self._migrations.count == 0:
                self._logger.debug("No migrations registered")
                return

            latest_version = self._migrations.latest_version
            if current_version == latest_version:
                self._logger.debug("Database is up to date, no migrations to run")
                return

            self._logger.info("Database update needed")

            # Only make a backup if using a file database (not memory)
            self._backup_db()

            next_migration = self._migrations.get(from_version=current_version)
            while next_migration is not None:
                try:
                    self._run_migration(next_migration)
                    next_migration = self._migrations.get(self._get_current_version())
                except MigrationError:
                    self._restore_db()
                    raise
            self._logger.info("Database updated successfully")

    def _run_migration(self, migration: Migration) -> None:
        """Runs a single migration."""
        with self._lock:
            try:
                if self._get_current_version() != migration.from_version:
                    raise MigrationError(
                        f"Database is at version {self._get_current_version()}, expected {migration.from_version}"
                    )
                self._logger.debug(f"Running migration from {migration.from_version} to {migration.to_version}")
                if migration.pre_migrate:
                    self._logger.debug(f"Running {len(migration.pre_migrate)} pre-migration callbacks")
                    for callback in migration.pre_migrate:
                        callback(self._cursor, self._image_files, self._logger)
                migration.migrate(self._cursor, self._image_files, self._logger)
                self._cursor.execute("INSERT INTO migrations (version) VALUES (?);", (migration.to_version,))
                if migration.post_migrate:
                    self._logger.debug(f"Running {len(migration.post_migrate)} post-migration callbacks")
                    for callback in migration.post_migrate:
                        callback(self._cursor, self._image_files, self._logger)
                # Migration callbacks only get a cursor; they cannot commit the transaction.
                self._conn.commit()
                self._logger.debug(
                    f"Successfully migrated database from {migration.from_version} to {migration.to_version}"
                )
            except Exception as e:
                msg = f"Error migrating database from {migration.from_version} to {migration.to_version}: {e}"
                self._conn.rollback()
                self._logger.error(msg)
                raise MigrationError(msg) from e

    def _create_version_table(self) -> None:
        """Creates a version table for the database, if one does not already exist."""
        with self._lock:
            try:
                self._cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='migrations';")
                if self._cursor.fetchone() is not None:
                    return
                self._cursor.execute(
                    """--sql
                    CREATE TABLE migrations (
                        version INTEGER PRIMARY KEY,
                        migrated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW'))
                    );
                    """
                )
                self._cursor.execute("INSERT INTO migrations (version) VALUES (0);")
                self._conn.commit()
                self._logger.debug("Created migrations table")
            except sqlite3.Error as e:
                msg = f"Problem creating migrations table: {e}"
                self._logger.error(msg)
                self._conn.rollback()
                raise MigrationError(msg) from e

    def _get_current_version(self) -> int:
        """Gets the current version of the database, or 0 if the version table does not exist."""
        with self._lock:
            try:
                self._cursor.execute("SELECT MAX(version) FROM migrations;")
                version = self._cursor.fetchone()[0]
                if version is None:
                    return 0
                return version
            except sqlite3.OperationalError as e:
                if "no such table" in str(e):
                    return 0
                raise

    def _backup_db(self) -> None:
        """Backs up the databse, returning the path to the backup file."""
        if self._is_memory:
            self._logger.debug("Using memory database, skipping backup")
        # Sanity check!
        assert isinstance(self._database, Path)
        with self._lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self._database.parent / f"{self._database.stem}_{timestamp}.db"
            self._logger.info(f"Backing up database to {backup_path}")

            # Use SQLite's built in backup capabilities so we don't need to worry about locking and such.
            backup_conn = sqlite3.connect(backup_path)
            with backup_conn:
                self._conn.backup(backup_conn)
            backup_conn.close()

            # Sanity check!
            if not backup_path.is_file():
                raise MigrationError("Unable to back up database")
            self.backup_path = backup_path

    def _restore_db(
        self,
    ) -> None:
        """Restores the database from a backup file, unless the database is a memory database."""
        if self._is_memory:
            return

        with self._lock:
            self._logger.info(f"Restoring database from {self.backup_path}")
            self._conn.close()
            assert isinstance(self.backup_path, Path)
            shutil.copy2(self.backup_path, self._database)

    def _unlink_lock_file(self) -> bool:
        """Unlinks the migration lock file, returning True if it existed."""
        if self._is_memory or self._migration_lock_file_path is None:
            return False
        if self._migration_lock_file_path.is_file():
            self._migration_lock_file_path.unlink()
            return True
        return False

    def _write_migration_lock_file(self) -> None:
        """Writes a file to indicate that a migration is in progress."""
        if self._is_memory or self._migration_lock_file_path is None:
            return
        assert isinstance(self._database, Path)
        with open(self._migration_lock_file_path, "w") as f:
            f.write("1")
