import sqlite3
from pathlib import Path
from typing import Optional

from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration, MigrationError, MigrationSet


class SQLiteMigrator:
    """
    Manages migrations for a SQLite database.

    :param db: The instanceof :class:`SqliteDatabase` to migrate.

    Migrations should be registered with :meth:`register_migration`.

    Each migration is run in a transaction. If a migration fails, the transaction is rolled back.
    """

    backup_path: Optional[Path] = None

    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db
        self._logger = db.logger
        self._migration_set = MigrationSet()

    def register_migration(self, migration: Migration) -> None:
        """Registers a migration."""
        self._migration_set.register(migration)
        self._logger.debug(f"Registered migration {migration.from_version} -> {migration.to_version}")

    def run_migrations(self) -> bool:
        """Migrates the database to the latest version."""
        with self._db.lock:
            # This throws if there is a problem.
            self._migration_set.validate_migration_chain()
            cursor = self._db.conn.cursor()
            self._create_migrations_table(cursor=cursor)

            if self._migration_set.count == 0:
                self._logger.debug("No migrations registered")
                return False

            if self._get_current_version(cursor=cursor) == self._migration_set.latest_version:
                self._logger.debug("Database is up to date, no migrations to run")
                return False

            self._logger.info("Database update needed")
            next_migration = self._migration_set.get(from_version=self._get_current_version(cursor))
            while next_migration is not None:
                self._run_migration(next_migration)
                next_migration = self._migration_set.get(self._get_current_version(cursor))
            self._logger.info("Database updated successfully")
            return True

    def _run_migration(self, migration: Migration) -> None:
        """Runs a single migration."""
        # Using sqlite3.Connection as a context manager commits a the transaction on exit, or rolls it back if an
        # exception is raised. We want to commit the transaction if the migration is successful, or roll it back if
        # there is an error.
        try:
            with self._db.lock, self._db.conn as conn:
                cursor = conn.cursor()
                if self._get_current_version(cursor) != migration.from_version:
                    raise MigrationError(
                        f"Database is at version {self._get_current_version(cursor)}, expected {migration.from_version}"
                    )
                self._logger.debug(f"Running migration from {migration.from_version} to {migration.to_version}")

                # Run pre-migration callbacks
                if migration.pre_migrate:
                    self._logger.debug(f"Running {len(migration.pre_migrate)} pre-migration callbacks")
                    for callback in migration.pre_migrate:
                        callback(cursor)

                # Run the actual migration
                migration.migrate(cursor)
                cursor.execute("INSERT INTO migrations (version) VALUES (?);", (migration.to_version,))

                # Run post-migration callbacks
                if migration.post_migrate:
                    self._logger.debug(f"Running {len(migration.post_migrate)} post-migration callbacks")
                    for callback in migration.post_migrate:
                        callback(cursor)
                self._logger.debug(
                    f"Successfully migrated database from {migration.from_version} to {migration.to_version}"
                )
        # We want to catch *any* error, mirroring the behaviour of the sqlite3 module.
        except Exception as e:
            # The connection context manager has already rolled back the migration, so we don't need to do anything.
            msg = f"Error migrating database from {migration.from_version} to {migration.to_version}: {e}"
            self._logger.error(msg)
            raise MigrationError(msg) from e

    def _create_migrations_table(self, cursor: sqlite3.Cursor) -> None:
        """Creates the migrations table for the database, if one does not already exist."""
        with self._db.lock:
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
            version: int = cursor.fetchone()[0]
            if version is None:
                return 0
            return version
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                return 0
            raise
