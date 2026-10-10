import logging
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import NoReturn, Optional

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import Connection, insert, inspect, select
from sqlalchemy.exc import DBAPIError

from invokeai.app.services.config.config_default import DefaultInvokeAIAppConfig
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.shared.database.copy import copy_rows
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema import metadata
from invokeai.app.services.shared.database.schema.migrator import applied_migrations, migrations
from invokeai.app.services.shared.database.session_lock import SessionLock
from invokeai.app.services.shared.sqlite_migrator.migration_loader import MigrationBuildContext, build_migrations
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import (
    Migration,
    MigrationBase,
    MigrationError,
    MigrationSet,
    PortableMigration,
    PortableMigrationContext,
)

# How long the migration of a server database waits for another process's migration of it to finish.
MIGRATION_LOCK_TIMEOUT_SECONDS = 600


class Migrator:
    """
    Brings the database up to date by running the migrations it has not had, in dependency order.

    :param database: The database to migrate.

    Migrations are registered with :meth:`register_migration`, either directly or via the migration loader.
    They are planned by stable migration ID dependencies and recorded in the ``applied_migrations`` table.
    Legacy numeric versions are still written for migrations that define ``to_version``.

    A SQLite database runs every migration, the legacy cursor migrations and the portable ones, each in a
    transaction of its own that a failure rolls back, after a backup of the database file.

    A MySQL or MariaDB database is created at the newest schema instead (see :meth:`_bootstrap`), so only the
    portable migrations added after its creation run on it. A server-wide lock keeps two processes from migrating
    the same database at once. DDL commits as it runs there, so a failed portable migration is not rolled back:
    its id is recorded only once it succeeds, and it runs again, from the start, the next time.

    Example Usage:
    ```py
    migrator = Migrator(database)
    for migration in build_migrations(migration_context):
        migrator.register_migration(migration)
    migrator.run_migrations()
    ```
    """

    backup_path: Optional[Path] = None

    def __init__(self, database: Database) -> None:
        self._database = database
        self._logger = database.logger
        self._migration_set = MigrationSet()
        self._backup_path: Optional[Path] = None

    def register_migration(self, migration: MigrationBase) -> None:
        """Registers a migration."""
        self._migration_set.register(migration)
        self._logger.debug(f"Registered migration {migration.from_version} -> {migration.to_version}")

    def run_migrations(self) -> bool:
        """Migrates the database to the latest version. Returns whether a migration ran or the database was created."""
        # This throws if there is a problem.
        self._migration_set.validate_dependency_graph()
        if self._database.dialect_name != "sqlite":
            return self._run_server_migrations()
        cursor = self._database.sqlite.conn.cursor()
        self._validate_existing_applied_migrations(cursor=cursor)
        self._create_migrations_table(cursor=cursor)
        self._validate_existing_legacy_migrations(cursor=cursor)
        if self._needs_applied_migrations_bootstrap(cursor=cursor):
            self._backup_db()
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

        self._backup_db()

        for migration in migration_plan:
            self._run_migration(migration)
        self._logger.info("Database updated successfully")
        return True

    def _backup_db(self) -> None:
        """Makes a backup of the db if it is a file db and a backup has not already been made."""
        if self._backup_path is not None:
            return
        db_path = self._database.sqlite.path
        if db_path is not None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self._backup_path = db_path.parent / f"{db_path.stem}_backup_{timestamp}.db"
            # A backup of the same second exists when the database was migrated again right away, e.g. on a restart
            # after a failed migration: keep it, and number this one.
            attempt = 1
            while self._backup_path.exists():
                self._backup_path = db_path.parent / f"{db_path.stem}_backup_{timestamp}-{attempt}.db"
                attempt += 1
            self._logger.info(f"Backing up database to {str(self._backup_path)}")
            self._database.backup(self._backup_path)
        else:
            self._logger.info("Using in-memory database, no backup needed")

    def _run_migration(self, migration: MigrationBase) -> None:
        """Runs a single migration on the SQLite database."""
        if isinstance(migration, PortableMigration):
            self._run_portable_migration(migration)
            return
        assert isinstance(migration, Migration)
        try:
            # Using sqlite3.Connection as a context manager commits a the transaction on exit, or rolls it back if an
            # exception is raised.
            with self._database.sqlite.conn as conn:
                cursor = conn.cursor()
                # Begun explicitly, because the context manager above only commits or rolls back a
                # transaction that is already open — it does not start one. The connection runs in
                # Python's legacy implicit-transaction mode, which opens a transaction before DML
                # and never before DDL, so a callback that issues nothing but `CREATE`/`DROP`/
                # `ALTER` would commit statement by statement in autocommit with nothing for the
                # rollback to undo. A migration that dropped a table and died before recreating it
                # would leave the database wedged and the migration unrecorded, so every restart
                # would re-run it and fail again. Beginning here makes atomicity a property of the
                # migrator rather than an accident of whether a given migration happens to write a
                # row.
                cursor.execute("BEGIN;")
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

    def _needs_applied_migrations_bootstrap(self, cursor: sqlite3.Cursor) -> bool:
        """Checks whether legacy numeric rows need to be written to applied_migrations."""
        cursor.execute("SELECT version FROM migrations WHERE version > 0;")
        legacy_versions = {row[0] for row in cursor.fetchall()}
        if len(legacy_versions) == 0:
            return False

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='applied_migrations';")
        if cursor.fetchone() is None:
            return True

        cursor.execute("SELECT legacy_version FROM applied_migrations WHERE legacy_version IS NOT NULL;")
        applied_legacy_versions = {row[0] for row in cursor.fetchall()}
        return not legacy_versions.issubset(applied_legacy_versions)

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

    def _run_portable_migration(self, migration: PortableMigration) -> None:
        """Runs a portable migration and records it in one transaction (on a server, its DDL commits as it runs).

        On SQLite, foreign keys are off while it runs: rebuilding a table (Alembic's batch mode) drops the old
        one, which would otherwise delete the rows of every table that references it, by cascade. The pragma
        cannot change inside a transaction, so it is set before, under the database's lock, and the foreign
        keys are checked before the transaction commits (see `_check_foreign_keys`).
        """
        self._logger.debug(f"Running migration '{migration.id}'")
        try:
            if self._database.dialect_name != "sqlite":
                self._apply_portable_migration(migration)
            else:
                sqlite = self._database.sqlite
                with sqlite.lock:
                    sqlite.conn.execute("PRAGMA foreign_keys = OFF")
                    try:
                        self._apply_portable_migration(migration)
                    finally:
                        sqlite.conn.execute("PRAGMA foreign_keys = ON")
        except Exception as e:
            msg = f"Error running migration '{migration.id}': {e}"
            self._logger.error(msg)
            raise MigrationError(msg) from e
        self._logger.debug(f"Successfully ran migration '{migration.id}'")

    def _apply_portable_migration(self, migration: PortableMigration) -> None:
        with self._database.begin(write=True) as conn:
            violations_before = self._foreign_key_violations(conn)
            if violations_before:
                self._logger.warning(
                    "The database holds rows whose foreign keys match nothing (table, referenced table): "
                    f"{dict(violations_before)}"
                )
            migration.callback(
                PortableMigrationContext(
                    conn=conn, op=Operations(MigrationContext.configure(conn)), logger=self._logger
                )
            )
            self._check_foreign_keys(conn, violations_before)
            conn.execute(insert(applied_migrations).values(migration_id=migration.id))

    def _check_foreign_keys(self, conn: Connection, before: Optional[Counter[tuple[str, str]]]) -> None:
        """Fails a SQLite migration that leaves rows whose foreign keys match nothing.

        Only rows the migration orphaned count: a database can hold orphans from before (tools that wrote with
        foreign keys off did leave some), and those must not stop every migration from then on. So the migration
        fails when there are more such rows after it than before. They are counted, not identified: a rebuilt
        table renumbers its rows, and a renamed one reports them under its new name.
        """
        if before is None:
            return
        after = self._foreign_key_violations(conn)
        if after is None:
            raise MigrationError("The migration leaves a foreign key that names no table or key")
        if after.total() > before.total():
            raise MigrationError(
                f"The migration leaves rows whose foreign keys match nothing (table, referenced table): {dict(after)}"
            )

    def _foreign_key_violations(self, conn: Connection) -> Optional[Counter[tuple[str, str]]]:
        """On SQLite, the rows whose foreign keys match nothing, per table and referenced table.

        None when SQLite cannot check (a foreign key in the database names no table or key), or on a server,
        which enforces foreign keys as rows are written.
        """
        if self._database.dialect_name != "sqlite":
            return None
        try:
            rows = conn.exec_driver_sql("PRAGMA foreign_key_check").all()
        except DBAPIError as e:
            self._logger.warning(f"The database's foreign keys could not be checked: {e}")
            return None
        return Counter((str(row[0]), str(row[2])) for row in rows)

    def has_pending_migrations(self) -> bool:
        """Whether the database lacks a registered migration, without changing it. A database without the migrator's
        records (a new one, or one from before them) lacks them all. Raises for a server database with tables but
        no records, which the app refuses as well."""
        if self._database.dialect_name != "sqlite" and self._server_has_tables():
            applied_ids = self._server_applied_migration_ids()
            return bool(self._migration_set.get_migration_plan(applied_migration_ids=applied_ids))
        with self._database.begin(write=False) as conn:
            if not inspect(conn).has_table(applied_migrations.name):
                return self._migration_set.count > 0
            applied = set(conn.execute(select(applied_migrations.c.migration_id)).scalars())
        return bool(self._migration_set.get_migration_plan(applied_migration_ids=applied))

    def _run_server_migrations(self) -> bool:
        with self._server_migration_lock() as lock:
            bootstrapped = False
            if not self._server_has_tables():
                self._bootstrap()
                bootstrapped = True
            plan = self._migration_set.get_migration_plan(applied_migration_ids=self._server_applied_migration_ids())
            portable: list[PortableMigration] = []
            for migration in plan:
                if not isinstance(migration, PortableMigration):
                    raise MigrationError(
                        f"Migration '{migration.id}' runs on SQLite only, and this {self._database.dialect_name} "
                        "database has not had it"
                    )
                portable.append(migration)
            if portable:
                self._logger.info("Database update needed")
            for migration in portable:
                lock.verify()
                self._run_portable_migration(migration)
            if portable:
                self._logger.info("Database updated successfully")
            return bootstrapped or bool(portable)

    @contextmanager
    def _server_migration_lock(self) -> Iterator["_ServerMigrationLock"]:
        """Holds a lock that one process at a time can hold for this database, on a connection of its own."""
        session_lock = SessionLock(self._database.engine, "migrate")
        try:
            lock = _ServerMigrationLock(session_lock)
            lock.acquire(self._logger)
            yield lock
        finally:
            session_lock.close()

    def _server_has_tables(self) -> bool:
        with self._database.begin(write=False) as conn:
            return bool(inspect(conn).get_table_names())

    def _server_applied_migration_ids(self) -> set[str]:
        with self._database.begin(write=False) as conn:
            applied_ids: set[str] = set()
            if inspect(conn).has_table(applied_migrations.name):
                applied_ids = set(conn.execute(select(applied_migrations.c.migration_id)).scalars())
        if not applied_ids:
            # The records are written last, so this is an interrupted creation, if not another application's tables.
            raise MigrationError(
                "The database has tables but no record of the migrations it has had: it is not an InvokeAI "
                "database, or its creation was interrupted. Use an empty database."
            )
        return applied_ids

    def _bootstrap(self) -> None:
        """Creates the newest schema in the empty server database, with the rows the migrations seed.

        The rows come from a reference: an in-memory SQLite database that the migration chain builds, in a
        temporary root, so the migrations' clean-ups of old files touch nothing real. The migrator's records go in
        last, so a creation that is interrupted leaves a database `_server_applied_migration_ids` refuses.
        """
        self._logger.info("Creating the database schema")
        # The reference's migrations report clean-ups of a root that is not the user's.
        quiet = self._logger.getChild("reference")
        quiet.setLevel(logging.WARNING)
        with tempfile.TemporaryDirectory() as root:
            # Settings from the environment or a config file would point the clean-ups at real directories.
            config = DefaultInvokeAIAppConfig()
            config._root = Path(root)
            reference = Database.open_sqlite(None, quiet)
            try:
                reference_migrator = Migrator(reference)
                context = MigrationBuildContext(app_config=config, logger=quiet, image_files=NoImageFiles())
                for migration in build_migrations(context):
                    reference_migrator.register_migration(migration)
                reference_migrator.run_migrations()

                with self._database.begin(write=True) as conn:
                    metadata.create_all(conn)
                records = [migrations, applied_migrations]
                copy_rows(reference, self._database, tables=[t for t in metadata.sorted_tables if t not in records])
                # `applied_migrations` last of all: a database without its rows is refused as incomplete.
                copy_rows(reference, self._database, tables=[migrations])
                copy_rows(reference, self._database, tables=[applied_migrations])
            finally:
                reference.dispose()


def _no_image_files(*args: object, **kwargs: object) -> NoReturn:
    raise RuntimeError("No image files were given: a migration that reads them cannot run here")


# For the migrations that read image files, where none runs or there are no images: the reference database of a
# bootstrap, and the check for pending migrations.
NoImageFiles = type(
    "NoImageFiles",
    (ImageFileStorageBase,),
    dict.fromkeys(ImageFileStorageBase.__abstractmethods__, _no_image_files),
)


_LOCK_LOST = (
    "The connection holding the migration lock was lost, so another process may be migrating the database; start again"
)


class _ServerMigrationLock:
    """The lock on migrating one server database, which one process at a time holds. It is verified before each
    migration rather than assumed: the server releases it with its connection, even one cut off while idle."""

    def __init__(self, lock: SessionLock) -> None:
        self._lock = lock

    def acquire(self, logger: logging.Logger) -> None:
        try:
            if self._lock.take(0):
                return
            logger.info("Waiting for another process to finish migrating the database")
            if self._lock.take(MIGRATION_LOCK_TIMEOUT_SECONDS):
                return
        except RuntimeError as e:
            raise MigrationError(str(e)) from e
        raise MigrationError(
            f"Another process has been migrating this database for {MIGRATION_LOCK_TIMEOUT_SECONDS} s; "
            "let it finish before starting again"
        )

    def verify(self) -> None:
        """Raises if the lock is no longer held."""
        if not self._lock.held():
            raise MigrationError(_LOCK_LOST)
