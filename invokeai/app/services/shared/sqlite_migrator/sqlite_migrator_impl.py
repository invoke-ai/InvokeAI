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

    Migrations should be registered with :meth:`register_migration`, either directly or via the migration loader.
    They are planned by stable migration ID dependencies and recorded in the ``applied_migrations`` table.
    Legacy numeric versions are still written for migrations that define ``to_version``.

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
        self._logger = db._logger
        self._migration_set = MigrationSet()
        self._backup_path: Optional[Path] = None

    def register_migration(self, migration: Migration) -> None:
        """Registers a migration."""
        self._migration_set.register(migration)
        self._logger.debug(f"Registered migration {migration.from_version} -> {migration.to_version}")

    def run_migrations(self) -> bool:
        """Migrates the database to the latest version."""
        # This throws if there is a problem.
        self._migration_set.validate_dependency_graph()
        cursor = self._db._conn.cursor()
        self._validate_existing_applied_migrations(cursor=cursor)
        self._create_migrations_table(cursor=cursor)
        self._validate_existing_legacy_migrations(cursor=cursor)
        self._create_applied_migrations_table(cursor=cursor)
        self._validate_existing_applied_legacy_migrations(cursor=cursor)
        self._bootstrap_applied_migrations_from_legacy_versions(cursor=cursor)

        if self._migration_set.count == 0:
            self._logger.debug("No migrations registered")
            return False

        applied_migration_ids = self._get_applied_migration_ids(cursor=cursor)
        migration_plan = self._migration_set.get_migration_plan(applied_migration_ids=applied_migration_ids)
        if len(migration_plan) == 0:
            self._logger.debug("Database is up to date, no migrations to run")
            return False

        self._logger.info("Database update needed")

        # Make a backup of the db if it needs to be updated and is a file db
        if self._db._db_path is not None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self._backup_path = self._db._db_path.parent / f"{self._db._db_path.stem}_backup_{timestamp}.db"
            self._logger.info(f"Backing up database to {str(self._backup_path)}")
            # Use SQLite to do the backup
            with closing(sqlite3.connect(self._backup_path)) as backup_conn:
                self._db._conn.backup(backup_conn)
        else:
            self._logger.info("Using in-memory database, no backup needed")

        for migration in migration_plan:
            self._run_migration(migration)
        self._logger.info("Database updated successfully")
        return True

    def _run_migration(self, migration: Migration) -> None:
        """Runs a single migration."""
        try:
            # Using sqlite3.Connection as a context manager commits a the transaction on exit, or rolls it back if an
            # exception is raised.
            with self._db._conn as conn:
                cursor = conn.cursor()
                self._create_applied_migrations_table(cursor)
                if migration.from_version is not None and self._get_current_version(cursor) != migration.from_version:
                    raise MigrationError(
                        f"Database is at version {self._get_current_version(cursor)}, expected {migration.from_version}"
                    )
                self._logger.debug(f"Running migration '{migration.id}'")

                # Run the actual migration
                migration.callback(cursor)

                if migration.to_version is not None:
                    cursor.execute("INSERT INTO migrations (version) VALUES (?);", (migration.to_version,))
                cursor.execute(
                    "INSERT INTO applied_migrations (migration_id, legacy_version) VALUES (?, ?);",
                    (migration.id, migration.to_version),
                )

                self._logger.debug(f"Successfully ran migration '{migration.id}'")
        # We want to catch *any* error, mirroring the behaviour of the sqlite3 module.
        except Exception as e:
            # The connection context manager has already rolled back the migration, so we don't need to do anything.
            msg = f"Error running migration '{migration.id}': {e}"
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

    def _create_applied_migrations_table(self, cursor: sqlite3.Cursor) -> None:
        """Creates the applied migrations table for stable migration IDs."""
        try:
            cursor.execute(
                """--sql
                CREATE TABLE IF NOT EXISTS applied_migrations (
                    migration_id TEXT PRIMARY KEY,
                    legacy_version INTEGER UNIQUE,
                    migrated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW'))
                );
                """
            )
        except sqlite3.Error as e:
            msg = f"Problem creating applied_migrations table: {e}"
            self._logger.error(msg)
            cursor.connection.rollback()
            raise MigrationError(msg) from e

    def _bootstrap_applied_migrations_from_legacy_versions(self, cursor: sqlite3.Cursor) -> None:
        """Backfills applied migration IDs from legacy numeric migration rows."""
        try:
            cursor.execute("SELECT version FROM migrations WHERE version > 0 ORDER BY version;")
            legacy_versions = [row[0] for row in cursor.fetchall()]
            registered_migration_ids = self._migration_set.migrations_by_id
            for legacy_version in legacy_versions:
                migration_id = f"migration_{legacy_version}"
                if migration_id not in registered_migration_ids:
                    cursor.connection.rollback()
                    raise MigrationError(f"Database contains unknown legacy migration version: {legacy_version}")
                cursor.execute(
                    "SELECT legacy_version FROM applied_migrations WHERE migration_id = ?;",
                    (migration_id,),
                )
                migration_row = cursor.fetchone()
                if migration_row is not None and migration_row[0] != legacy_version:
                    cursor.connection.rollback()
                    raise MigrationError(
                        "Database contains inconsistent applied migration state: "
                        f"{migration_id} is recorded with legacy version {migration_row[0]}, "
                        f"expected {legacy_version}"
                    )
                cursor.execute(
                    "SELECT migration_id FROM applied_migrations WHERE legacy_version = ?;",
                    (legacy_version,),
                )
                legacy_row = cursor.fetchone()
                if legacy_row is not None and legacy_row[0] != migration_id:
                    cursor.connection.rollback()
                    raise MigrationError(
                        "Database contains inconsistent applied migration state: "
                        f"legacy version {legacy_version} is recorded for {legacy_row[0]}, expected {migration_id}"
                    )
                cursor.execute(
                    "INSERT OR IGNORE INTO applied_migrations (migration_id, legacy_version) VALUES (?, ?);",
                    (migration_id, legacy_version),
                )
            cursor.connection.commit()
        except sqlite3.Error as e:
            msg = f"Problem bootstrapping applied migrations: {e}"
            self._logger.error(msg)
            cursor.connection.rollback()
            raise MigrationError(msg) from e

    def _validate_existing_applied_migrations(self, cursor: sqlite3.Cursor) -> None:
        """Validates existing applied migration IDs before creating or mutating migrator metadata."""
        applied_migration_ids = self._get_applied_migration_ids(cursor=cursor)
        if len(applied_migration_ids) == 0:
            return
        known_migration_ids = set(self._migration_set.migrations_by_id)
        unknown_applied_ids = applied_migration_ids - known_migration_ids
        if unknown_applied_ids:
            unknown_ids = ", ".join(sorted(unknown_applied_ids))
            raise MigrationError(f"Database contains unknown applied migration IDs: {unknown_ids}")

    def _validate_existing_legacy_migrations(self, cursor: sqlite3.Cursor) -> None:
        """Validates existing legacy migration versions before creating applied migration metadata."""
        try:
            cursor.execute("SELECT version FROM migrations WHERE version > 0 ORDER BY version;")
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                return
            raise

        registered_migration_ids = self._migration_set.migrations_by_id
        for row in cursor.fetchall():
            legacy_version = row[0]
            migration_id = f"migration_{legacy_version}"
            if migration_id not in registered_migration_ids:
                raise MigrationError(f"Database contains unknown legacy migration version: {legacy_version}")

    def _validate_existing_applied_legacy_migrations(self, cursor: sqlite3.Cursor) -> None:
        """Validates applied IDs for legacy migrations against legacy numeric rows."""
        registered_migrations = self._migration_set.migrations_by_id
        cursor.execute("SELECT migration_id, legacy_version FROM applied_migrations;")
        applied_rows = cursor.fetchall()

        cursor.execute("SELECT version FROM migrations WHERE version > 0;")
        legacy_versions = {row[0] for row in cursor.fetchall()}

        for row in applied_rows:
            migration_id = row[0]
            legacy_version = row[1]
            migration = registered_migrations[migration_id]
            if migration.to_version is None:
                continue
            if legacy_version != migration.to_version:
                raise MigrationError(
                    "Database contains inconsistent applied migration state: "
                    f"{migration_id} is recorded with legacy version {legacy_version}, "
                    f"expected {migration.to_version}"
                )
            if legacy_version not in legacy_versions:
                raise MigrationError(
                    "Database contains inconsistent applied migration state: "
                    f"{migration_id} is applied, but legacy version {legacy_version} is missing"
                )

    @classmethod
    def _get_applied_migration_ids(cls, cursor: sqlite3.Cursor) -> set[str]:
        """Gets applied stable migration IDs."""
        try:
            cursor.execute("SELECT migration_id FROM applied_migrations;")
            return {row[0] for row in cursor.fetchall()}
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                return set()
            raise

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
