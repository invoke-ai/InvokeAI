import shutil
import sqlite3
import threading
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Callable, Optional, TypeAlias

from pydantic import BaseModel, Field, model_validator

MigrateCallback: TypeAlias = Callable[[sqlite3.Cursor], None]


class MigrationError(RuntimeError):
    """Raised when a migration fails."""


class MigrationVersionError(ValueError):
    """Raised when a migration version is invalid."""


class Migration(BaseModel):
    """
    Represents a migration for a SQLite database.

    Migration callbacks will be provided an open cursor to the database. They should not commit their
    transaction; this is handled by the migrator.

    Pre- and post-migration callback may be registered with :meth:`register_pre_callback` or
    :meth:`register_post_callback`.

    If a migration has additional dependencies, it is recommended to use functools.partial to provide
    the dependencies and register the partial as the migration callback.
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

    def register_pre_callback(self, callback: MigrateCallback) -> None:
        """Registers a pre-migration callback."""
        self.pre_migrate.append(callback)

    def register_post_callback(self, callback: MigrateCallback) -> None:
        """Registers a post-migration callback."""
        self.post_migrate.append(callback)


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

    def validate_migration_chain(self) -> None:
        """
        Validates that the migrations form a single chain of migrations from version 0 to the latest version.
        Raises a MigrationError if there is a problem.
        """
        if self.count == 0:
            return
        if self.latest_version == 0:
            return
        next_migration = self.get(from_version=0)
        if next_migration is None:
            raise MigrationError("Migration chain is fragmented")
        touched_count = 1
        while next_migration is not None:
            next_migration = self.get(next_migration.to_version)
            if next_migration is not None:
                touched_count += 1
        if touched_count != self.count:
            raise MigrationError("Migration chain is fragmented")

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

    :param db_path: The path to the database to migrate, or None if using an in-memory database.
    :param conn: The connection to the database.
    :param lock: A lock to use when running migrations.
    :param logger: A logger to use for logging.
    :param log_sql: Whether to log SQL statements. Only used when the log level is set to debug.

    Migrations should be registered with :meth:`register_migration`.

    During migration, a copy of the current database is made and the migrations are run on the copy. If the migration
    is successful, the original database is backed up and the migrated database is moved to the original database's
    path. If the migration fails, the original database is left untouched and the migrated database is deleted.

    If the database is in-memory, no backup is made; the migration is run in-place.
    """

    backup_path: Optional[Path] = None

    def __init__(
        self,
        db_path: Path | None,
        conn: sqlite3.Connection,
        lock: threading.RLock,
        logger: Logger,
        log_sql: bool = False,
    ) -> None:
        self._lock = lock
        self._db_path = db_path
        self._logger = logger
        self._conn = conn
        self._log_sql = log_sql
        self._cursor = self._conn.cursor()
        self._migration_set = MigrationSet()

        # The presence of an temp database file indicates a catastrophic failure of a previous migration.
        if self._db_path and self._get_temp_db_path(self._db_path).is_file():
            self._logger.warning("Previous migration failed! Trying again...")
            self._get_temp_db_path(self._db_path).unlink()

    def register_migration(self, migration: Migration) -> None:
        """Registers a migration."""
        self._migration_set.register(migration)
        self._logger.debug(f"Registered migration {migration.from_version} -> {migration.to_version}")

    def run_migrations(self) -> bool:
        """Migrates the database to the latest version."""
        with self._lock:
            # This throws if there is a problem.
            self._migration_set.validate_migration_chain()
            self._create_migrations_table(cursor=self._cursor)

            if self._migration_set.count == 0:
                self._logger.debug("No migrations registered")
                return False

            if self._get_current_version(self._cursor) == self._migration_set.latest_version:
                self._logger.debug("Database is up to date, no migrations to run")
                return False

            self._logger.info("Database update needed")

            if self._db_path:
                # We are using a file database. Create a copy of the database to run the migrations on.
                temp_db_path = self._create_temp_db(self._db_path)
                self._logger.info(f"Copied database to {temp_db_path} for migration")
                temp_db_conn = sqlite3.connect(temp_db_path)
                # We have to re-set this because we just created a new connection.
                if self._log_sql:
                    temp_db_conn.set_trace_callback(self._logger.debug)
                temp_db_cursor = temp_db_conn.cursor()
                self._run_migrations(temp_db_cursor)
                # Close the connections, copy the original database as a backup, and move the temp database to the
                # original database's path.
                backup_db_path = self._finalize_migration(
                    temp_db_path=temp_db_path,
                    original_db_path=self._db_path,
                )
                temp_db_conn.close()
                self._conn.close()
                self._logger.info(f"Migration successful. Original DB backed up to {backup_db_path}")
            else:
                # We are using a memory database. No special backup or special handling needed.
                self._run_migrations(self._cursor)

            self._logger.info("Database updated successfully")
            return True

    def _run_migrations(self, temp_db_cursor: sqlite3.Cursor) -> None:
        """Runs all migrations in a loop."""
        next_migration = self._migration_set.get(from_version=self._get_current_version(temp_db_cursor))
        while next_migration is not None:
            self._run_migration(next_migration, temp_db_cursor)
            next_migration = self._migration_set.get(self._get_current_version(temp_db_cursor))

    def _run_migration(self, migration: Migration, temp_db_cursor: sqlite3.Cursor) -> None:
        """Runs a single migration."""
        with self._lock:
            try:
                if self._get_current_version(temp_db_cursor) != migration.from_version:
                    raise MigrationError(
                        f"Database is at version {self._get_current_version(temp_db_cursor)}, expected {migration.from_version}"
                    )
                self._logger.debug(f"Running migration from {migration.from_version} to {migration.to_version}")

                # Run pre-migration callbacks
                if migration.pre_migrate:
                    self._logger.debug(f"Running {len(migration.pre_migrate)} pre-migration callbacks")
                    for callback in migration.pre_migrate:
                        callback(temp_db_cursor)

                # Run the actual migration
                migration.migrate(temp_db_cursor)
                temp_db_cursor.execute("INSERT INTO migrations (version) VALUES (?);", (migration.to_version,))

                # Run post-migration callbacks
                if migration.post_migrate:
                    self._logger.debug(f"Running {len(migration.post_migrate)} post-migration callbacks")
                    for callback in migration.post_migrate:
                        callback(temp_db_cursor)

                # Migration callbacks only get a cursor. Commit this migration.
                temp_db_cursor.connection.commit()
                self._logger.debug(
                    f"Successfully migrated database from {migration.from_version} to {migration.to_version}"
                )
            except Exception as e:
                msg = f"Error migrating database from {migration.from_version} to {migration.to_version}: {e}"
                temp_db_cursor.connection.rollback()
                self._logger.error(msg)
                raise MigrationError(msg) from e

    def _create_migrations_table(self, cursor: sqlite3.Cursor) -> None:
        """Creates the migrations table for the database, if one does not already exist."""
        with self._lock:
            try:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='migrations';")
                if cursor.fetchone() is not None:
                    return
                cursor.execute(
                    """--sql
                    CREATE TABLE migrations (
                        version INTEGER PRIMARY KEY,
                        migrated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW'))
                    );
                    """
                )
                cursor.execute("INSERT INTO migrations (version) VALUES (0);")
                cursor.connection.commit()
                self._logger.debug("Created migrations table")
            except sqlite3.Error as e:
                msg = f"Problem creating migrations table: {e}"
                self._logger.error(msg)
                cursor.connection.rollback()
                raise MigrationError(msg) from e

    @classmethod
    def _get_current_version(cls, cursor: sqlite3.Cursor) -> int:
        """Gets the current version of the database, or 0 if the migrations table does not exist."""
        try:
            cursor.execute("SELECT MAX(version) FROM migrations;")
            version = cursor.fetchone()[0]
            if version is None:
                return 0
            return version
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                return 0
            raise

    @classmethod
    def _create_temp_db(cls, original_db_path: Path) -> Path:
        """Copies the current database to a new file for migration."""
        temp_db_path = cls._get_temp_db_path(original_db_path)
        shutil.copy2(original_db_path, temp_db_path)
        return temp_db_path

    @classmethod
    def _finalize_migration(
        cls,
        temp_db_path: Path,
        original_db_path: Path,
    ) -> Path:
        """Renames the original database as a backup and renames the migrated database to the original name."""
        backup_db_path = cls._get_backup_db_path(original_db_path)
        original_db_path.rename(backup_db_path)
        temp_db_path.rename(original_db_path)
        return backup_db_path

    @classmethod
    def _get_temp_db_path(cls, original_db_path: Path) -> Path:
        """Gets the path to the temp database."""
        return original_db_path.parent / original_db_path.name.replace(".db", ".db.temp")

    @classmethod
    def _get_backup_db_path(cls, original_db_path: Path) -> Path:
        """Gets the path to the final backup database."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return original_db_path.parent / f"{original_db_path.stem}_backup_{timestamp}.db"
