import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration, MigrationError, MigrationSet


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

    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db
        self._logger = db.logger
        self._migration_set = MigrationSet()

        # The presence of an temp database file indicates a catastrophic failure of a previous migration.
        if self._db.db_path and self._get_temp_db_path(self._db.db_path).is_file():
            self._logger.warning("Previous migration failed! Trying again...")
            self._get_temp_db_path(self._db.db_path).unlink()

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

            if self._db.db_path:
                # We are using a file database. Create a copy of the database to run the migrations on.
                temp_db_path = self._create_temp_db(self._db.db_path)
                self._logger.info(f"Copied database to {temp_db_path} for migration")
                temp_db = SqliteDatabase(db_path=temp_db_path, logger=self._logger, verbose=self._db.verbose)
                temp_db_cursor = temp_db.conn.cursor()
                self._run_migrations(temp_db_cursor)
                # Close the connections, copy the original database as a backup, and move the temp database to the
                # original database's path.
                temp_db.close()
                self._db.close()
                backup_db_path = self._finalize_migration(
                    temp_db_path=temp_db_path,
                    original_db_path=self._db.db_path,
                )
                self._logger.info(f"Migration successful. Original DB backed up to {backup_db_path}")
            else:
                # We are using a memory database. No special backup or special handling needed.
                self._run_migrations(cursor)

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
        with self._db.lock:
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
