import sqlite3
import threading
from copy import deepcopy
from logging import Logger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import pytest

from invokeai.app.services.shared.sqlite.sqlite_common import sqlite_memory
from invokeai.app.services.shared.sqlite.sqlite_migrator import (
    Migration,
    MigrationError,
    MigrationVersionError,
    SQLiteMigrator,
)


@pytest.fixture
def migrator() -> SQLiteMigrator:
    conn = sqlite3.connect(sqlite_memory, check_same_thread=False)
    return SQLiteMigrator(
        conn=conn, database=sqlite_memory, lock=threading.RLock(), logger=Logger("test_sqlite_migrator")
    )


@pytest.fixture
def good_migration() -> Migration:
    return Migration(db_version=1, app_version="1.0.0", migrate=lambda cursor: None)


@pytest.fixture
def failing_migration() -> Migration:
    def failing_migration(cursor: sqlite3.Cursor) -> None:
        raise Exception("Bad migration")

    return Migration(db_version=1, app_version="1.0.0", migrate=failing_migration)


def test_register_migration(migrator: SQLiteMigrator, good_migration: Migration):
    migration = good_migration
    migrator.register_migration(migration)
    assert migration in migrator._migrations


def test_register_invalid_migration_version(migrator: SQLiteMigrator):
    with pytest.raises(MigrationError, match="Invalid migration version"):
        migrator.register_migration(Migration(db_version=0, app_version="0.0.0", migrate=lambda cursor: None))


def test_create_version_table(migrator: SQLiteMigrator):
    migrator._create_version_table()
    migrator._cursor.execute("SELECT * FROM sqlite_master WHERE type='table' AND name='version';")
    assert migrator._cursor.fetchone() is not None


def test_get_current_version(migrator: SQLiteMigrator):
    migrator._create_version_table()
    migrator._conn.commit()
    assert migrator._get_current_version() == 0  # initial version


def test_set_version(migrator: SQLiteMigrator):
    migrator._create_version_table()
    migrator._set_version(db_version=1, app_version="1.0.0")
    migrator._cursor.execute("SELECT MAX(db_version) FROM version;")
    assert migrator._cursor.fetchone()[0] == 1
    migrator._cursor.execute("SELECT app_version from version WHERE db_version = 1;")
    assert migrator._cursor.fetchone()[0] == "1.0.0"


def test_run_migration(migrator: SQLiteMigrator):
    migrator._create_version_table()

    def migration_callback(cursor: sqlite3.Cursor) -> None:
        cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY);")

    migration = Migration(db_version=1, app_version="1.0.0", migrate=migration_callback)
    migrator._run_migration(migration)
    assert migrator._get_current_version() == 1
    migrator._cursor.execute("SELECT app_version from version WHERE db_version = 1;")
    assert migrator._cursor.fetchone()[0] == "1.0.0"


def test_run_migrations(migrator: SQLiteMigrator):
    migrator._create_version_table()

    def create_migrate(i: int) -> Callable[[sqlite3.Cursor], None]:
        def migrate(cursor: sqlite3.Cursor) -> None:
            cursor.execute(f"CREATE TABLE test{i} (id INTEGER PRIMARY KEY);")

        return migrate

    migrations = [Migration(db_version=i, app_version=f"{i}.0.0", migrate=create_migrate(i)) for i in range(1, 4)]
    for migration in migrations:
        migrator.register_migration(migration)
    migrator.run_migrations()
    assert migrator._get_current_version() == 3


def test_backup_and_restore_db(migrator: SQLiteMigrator):
    with TemporaryDirectory() as tempdir:
        # must do this with a file database - we don't backup/restore for memory
        database = Path(tempdir) / "test.db"
        migrator._database = database
        migrator._cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY);")
        migrator._conn.commit()
        backup_path = migrator._backup_db(migrator._database)
        migrator._cursor.execute("DROP TABLE test;")
        migrator._conn.commit()
        migrator._restore_db(backup_path)  # this closes the connection
        # reconnect to db
        restored_conn = sqlite3.connect(database)
        restored_cursor = restored_conn.cursor()
        restored_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test';")
        assert restored_cursor.fetchone() is not None


def test_no_backup_and_restore_for_memory_db(migrator: SQLiteMigrator):
    with pytest.raises(MigrationError, match="Cannot back up memory database"):
        migrator._backup_db(sqlite_memory)


def test_failed_migration(migrator: SQLiteMigrator, failing_migration: Migration):
    migrator._create_version_table()
    with pytest.raises(MigrationError, match="Error migrating database from 0 to 1"):
        migrator._run_migration(failing_migration)
    assert migrator._get_current_version() == 0


def test_duplicate_migration_versions(migrator: SQLiteMigrator, good_migration: Migration):
    migrator._create_version_table()
    migrator.register_migration(good_migration)
    with pytest.raises(MigrationVersionError, match="already registered"):
        migrator.register_migration(deepcopy(good_migration))


def test_non_sequential_migration_registration(migrator: SQLiteMigrator):
    migrator._create_version_table()

    def create_migrate(i: int) -> Callable[[sqlite3.Cursor], None]:
        def migrate(cursor: sqlite3.Cursor) -> None:
            cursor.execute(f"CREATE TABLE test{i} (id INTEGER PRIMARY KEY);")

        return migrate

    migrations = [
        Migration(db_version=i, app_version=f"{i}.0.0", migrate=create_migrate(i)) for i in reversed(range(1, 4))
    ]
    for migration in migrations:
        migrator.register_migration(migration)
    migrator.run_migrations()
    assert migrator._get_current_version() == 3


def test_db_version_gt_last_migration(migrator: SQLiteMigrator, good_migration: Migration):
    migrator._create_version_table()
    migrator.register_migration(good_migration)
    migrator._set_version(db_version=2, app_version="2.0.0")
    with pytest.raises(MigrationError, match="greater than the latest migration version"):
        migrator.run_migrations()
    assert migrator._get_current_version() == 2


def test_idempotent_migrations(migrator: SQLiteMigrator):
    migrator._create_version_table()

    def create_test_table(cursor: sqlite3.Cursor) -> None:
        # This SQL throws if run twice
        cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY);")

    migration = Migration(db_version=1, app_version="1.0.0", migrate=create_test_table)

    migrator.register_migration(migration)
    migrator.run_migrations()
    # not throwing is sufficient
    migrator.run_migrations()
