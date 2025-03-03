import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Optional

from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration, MigrationError, MigrationSet


class SqliteMigrator:
    """
    Manages migrations for a SQLite database.

    :param db: The instance of :class:`SqliteDatabase` to migrate.

    Migrations should be registered with :meth:`register_migration`.

    Each migration is run in a transaction. If a migration fails, the transaction is rolled back.

    Example Usage:
    ```py
    db = SqliteDatabase(db_path="my_db.db", logger=logger)
    migrator = SqliteMigrator(db=db)
    migrator.register_migration(build_migration_1())
    migrator.register_migration(build_migration_2())
    migrator.run_migrations()
    ```
    """

    backup_path: Optional[Path] = None

    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db
        self._logger = db.logger
        self._migration_set = MigrationSet()
        self._backup_path: Optional[Path] = None

    def register_migration(self, migration: Migration) -> None:
        """Registers a migration."""
        self._migration_set.register(migration)
        self._logger.debug(f"Registered migration {migration.from_version} -> {migration.to_version}")

    def run_migrations(self) -> bool:
        """Migrates the database to the latest version."""
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

        # Make a backup of the db if it needs to be updated and is a file db
        if self._db.db_path is not None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self._backup_path = self._db.db_path.parent / f"{self._db.db_path.stem}_backup_{timestamp}.db"
            self._logger.info(f"Backing up database to {str(self._backup_path)}")
            # Use SQLite to do the backup
            with closing(sqlite3.connect(self._backup_path)) as backup_conn:
                self._db.conn.backup(backup_conn)
        else:
            self._logger.info("Using in-memory database, no backup needed")

        next_migration = self._migration_set.get(from_version=self._get_current_version(cursor))
        while next_migration is not None:
            self._run_migration(next_migration)
            next_migration = self._migration_set.get(self._get_current_version(cursor))
        self._logger.info("Database updated successfully")
        return True

    def _run_migration(self, migration: Migration) -> None:
        """Runs a single migration."""
        try:
            # Using sqlite3.Connection as a context manager commits a the transaction on exit, or rolls it back if an
            # exception is raised.
            with self._db.conn as conn:
                cursor = conn.cursor()
                if self._get_current_version(cursor) != migration.from_version:
                    raise MigrationError(
                        f"Database is at version {self._get_current_version(cursor)}, expected {migration.from_version}"
                    )
                self._logger.debug(f"Running migration from {migration.from_version} to {migration.to_version}")

                # Run the actual migration
                migration.callback(cursor)

                # Update the version
                cursor.execute("INSERT INTO migrations (version) VALUES (?);", (migration.to_version,))

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
