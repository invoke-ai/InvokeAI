import datetime
import shutil
import sqlite3
import threading
from logging import Logger
from pathlib import Path
from typing import Callable, Optional, TypeAlias

from .sqlite_common import sqlite_memory

MigrateCallback: TypeAlias = Callable[[sqlite3.Cursor], None]


class MigrationError(Exception):
    pass


class Migration:
    def __init__(
        self,
        version: int,
        migrate: MigrateCallback,
    ) -> None:
        self.version = version
        self.migrate = migrate


class MigrationSet:
    def __init__(self, table_name: str, migrations: list[Migration]) -> None:
        self.table_name = table_name
        self.migrations = migrations


class SQLiteMigrator:
    """
    Handles SQLite database migrations.

    Migrations are registered with the `register_migration_set` method. They are applied on
    application startup with the `run_migrations` method.

    A `MigrationSet` is a set of `Migration`s for a single table. Each `Migration` has a `version`
    and `migrate` callback. The callback is provided with a `sqlite3.Cursor` and should perform the
    any migration logic. Committing, rolling back transactions and errors are handled by the migrator.

    Migrations are applied in order of version number. If the database does not have a version table
    for a given table, it is assumed to be at version 0. The migrator creates and manages the version
    tables.

    If the database is a file, it will be backed up before migrations are applied and restored if
    there are any errors.
    """

    def __init__(self, db_path: Path | str, lock: threading.RLock, logger: Logger):
        self._logger = logger
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._cursor = self._conn.cursor()
        self._lock = lock
        self._db_path = db_path
        self._migration_sets: set[MigrationSet] = set()

    def _get_version_table_name(self, table_name: str) -> str:
        """Returns the name of the version table for a given table."""
        return f"{table_name}_version"

    def _create_version_table(self, table_name: str) -> None:
        """
        Creates a version table for a given table, if it does not exist.
        Throws MigrationError if there is a problem.
        """
        version_table_name = self._get_version_table_name(table_name)
        with self._lock:
            try:
                self._cursor.execute(
                    f"""--sql
                    CREATE TABLE IF NOT EXISTS {version_table_name} (
                        version INTEGER PRIMARY KEY,
                        created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW'))
                    );
                    """
                )
                self._conn.commit()
            except sqlite3.Error as e:
                msg = f'Problem creation "{version_table_name}" table: {e}'
                self._logger.error(msg)
                self._conn.rollback()
                raise MigrationError(msg) from e

    def _get_current_version(self, table_name: str) -> Optional[int]:
        """Gets the current version of a table, or None if it doesn't exist."""
        version_table_name = self._get_version_table_name(table_name)
        try:
            self._cursor.execute(f"SELECT MAX(version) FROM {version_table_name};")
            return self._cursor.fetchone()[0]
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                return None
            raise

    def _set_version(self, table_name: str, version: int) -> None:
        """Adds a version entry to the table's version table."""
        version_table_name = self._get_version_table_name(table_name)
        self._cursor.execute(f"INSERT INTO {version_table_name} (version) VALUES (?);", (version,))

    def _run_migration(self, table_name: str, migration: Migration) -> None:
        """Runs a single migration."""
        with self._lock:
            try:
                migration.migrate(self._cursor)
                self._set_version(table_name=table_name, version=migration.version)
                self._conn.commit()
            except sqlite3.Error:
                self._conn.rollback()
                raise

    def _run_migration_set(self, migration_set: MigrationSet) -> None:
        """Runs a set of migrations for a single table."""
        with self._lock:
            table_name = migration_set.table_name
            migrations = migration_set.migrations
            self._create_version_table(table_name=table_name)
            for migration in migrations:
                current_version = self._get_current_version(table_name)
                if current_version is None or current_version < migration.version:
                    try:
                        self._logger.info(f'runing "{table_name}" migration {migration.version}')
                        self._run_migration(table_name=table_name, migration=migration)
                    except sqlite3.Error as e:
                        raise MigrationError(f'Problem runing "{table_name}" migration {migration.version}: {e}') from e

    def _backup_db(self, db_path: Path) -> Path:
        """Backs up the databse, returning the path to the backup file."""
        with self._lock:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = db_path.parent / f"{db_path.stem}_{timestamp}.db"
            self._logger.info(f"Backing up database to {backup_path}")
            backup_conn = sqlite3.connect(backup_path)
            with backup_conn:
                self._conn.backup(backup_conn)
            backup_conn.close()
            return backup_path

    def _restore_db(self, backup_path: Path) -> None:
        """Restores the database from a backup file, unless the database is a memory database."""
        if self._db_path == sqlite_memory:
            return
        with self._lock:
            self._logger.info(f"Restoring database from {backup_path}")
            self._conn.close()
            if not Path(backup_path).is_file():
                raise FileNotFoundError(f"Backup file {backup_path} does not exist")
            shutil.copy2(backup_path, self._db_path)

    def _get_is_migration_needed(self, migration_set: MigrationSet) -> bool:
        table_name = migration_set.table_name
        migrations = migration_set.migrations
        current_version = self._get_current_version(table_name)
        if current_version is None or current_version < migrations[-1].version:
            return True
        return False

    def run_migrations(self) -> None:
        """
        Applies all registered migration sets.

        If the database is a file, it will be backed up before migrations are applied and restored
        if there are any errors.
        """
        if not any(self._get_is_migration_needed(migration_set) for migration_set in self._migration_sets):
            return
        backup_path: Optional[Path] = None
        with self._lock:
            # Only make a backup if using a file database (not memory)
            if isinstance(self._db_path, Path):
                backup_path = self._backup_db(self._db_path)
            for migration_set in self._migration_sets:
                if self._get_is_migration_needed(migration_set):
                    try:
                        self._run_migration_set(migration_set)
                    except Exception as e:
                        msg = f'Problem runing "{migration_set.table_name}" migrations: {e}'
                        self._logger.error(msg)
                        if backup_path is not None:
                            self._logger.error(f" Restoring from {backup_path}")
                            self._restore_db(backup_path)
                        raise MigrationError(msg) from e
            # TODO: delete backup file?
            # if backup_path is not None:
            #     Path(backup_path).unlink()

    def register_migration_set(self, migration_set: MigrationSet) -> None:
        """Registers a migration set to be migrated on application startup."""
        self._migration_sets.add(migration_set)
