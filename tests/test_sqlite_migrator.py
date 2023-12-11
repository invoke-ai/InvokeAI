import sqlite3
import threading
from logging import Logger
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import ValidationError

from invokeai.app.services.shared.sqlite.sqlite_common import sqlite_memory
from invokeai.app.services.shared.sqlite.sqlite_migrator import (
    MigrateCallback,
    Migration,
    MigrationError,
    MigrationSet,
    MigrationVersionError,
    SQLiteMigrator,
)


@pytest.fixture
def logger() -> Logger:
    return Logger("test_sqlite_migrator")


@pytest.fixture
def lock() -> threading.RLock:
    return threading.RLock()


@pytest.fixture
def migrator(logger: Logger, lock: threading.RLock) -> SQLiteMigrator:
    conn = sqlite3.connect(sqlite_memory, check_same_thread=False)
    return SQLiteMigrator(conn=conn, db_path=None, lock=lock, logger=logger)


@pytest.fixture
def migration_no_op() -> Migration:
    return Migration(from_version=0, to_version=1, migrate=lambda cursor: None)


@pytest.fixture
def migration_create_test_table() -> Migration:
    def migrate(cursor: sqlite3.Cursor) -> None:
        cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY);")

    return Migration(from_version=0, to_version=1, migrate=migrate)


@pytest.fixture
def failing_migration() -> Migration:
    def failing_migration(cursor: sqlite3.Cursor) -> None:
        raise Exception("Bad migration")

    return Migration(from_version=0, to_version=1, migrate=failing_migration)


@pytest.fixture
def no_op_migrate_callback() -> MigrateCallback:
    def no_op_migrate(cursor: sqlite3.Cursor) -> None:
        pass

    return no_op_migrate


@pytest.fixture
def failing_migrate_callback() -> MigrateCallback:
    def failing_migrate(cursor: sqlite3.Cursor) -> None:
        raise Exception("Bad migration")

    return failing_migrate


def create_migrate(i: int) -> MigrateCallback:
    def migrate(cursor: sqlite3.Cursor) -> None:
        cursor.execute(f"CREATE TABLE test{i} (id INTEGER PRIMARY KEY);")

    return migrate


def test_migration_to_version_gt_from_version(no_op_migrate_callback: MigrateCallback):
    with pytest.raises(ValidationError, match="greater_than_equal"):
        Migration(from_version=1, to_version=0, migrate=no_op_migrate_callback)


def test_migration_hash(no_op_migrate_callback: MigrateCallback):
    migration = Migration(from_version=0, to_version=1, migrate=no_op_migrate_callback)
    assert hash(migration) == hash((0, 1))


def test_migration_registers_pre_and_post_callbacks(no_op_migrate_callback: MigrateCallback):
    def pre_callback(cursor: sqlite3.Cursor) -> None:
        pass

    def post_callback(cursor: sqlite3.Cursor) -> None:
        pass

    migration = Migration(from_version=0, to_version=1, migrate=no_op_migrate_callback)
    migration.register_pre_callback(pre_callback)
    migration.register_post_callback(post_callback)
    assert pre_callback in migration.pre_migrate
    assert post_callback in migration.post_migrate


def test_migration_set_add_migration(migrator: SQLiteMigrator, migration_no_op: Migration):
    migration = migration_no_op
    migrator._migration_set.register(migration)
    assert migration in migrator._migration_set._migrations


def test_migration_set_may_not_register_dupes(migrator: SQLiteMigrator, no_op_migrate_callback: MigrateCallback):
    migrate_1_to_2 = Migration(from_version=1, to_version=2, migrate=no_op_migrate_callback)
    migrate_0_to_2 = Migration(from_version=0, to_version=2, migrate=no_op_migrate_callback)
    migrate_1_to_3 = Migration(from_version=1, to_version=3, migrate=no_op_migrate_callback)
    migrator._migration_set.register(migrate_1_to_2)
    with pytest.raises(MigrationVersionError, match=r"Migration to 2 already registered"):
        migrator._migration_set.register(migrate_0_to_2)
    with pytest.raises(MigrationVersionError, match=r"Migration from 1 already registered"):
        migrator._migration_set.register(migrate_1_to_3)


def test_migration_set_gets_migration(migration_no_op: Migration):
    migration_set = MigrationSet()
    migration_set.register(migration_no_op)
    assert migration_set.get(0) == migration_no_op
    assert migration_set.get(1) is None


def test_migration_set_validates_migration_path(no_op_migrate_callback: MigrateCallback):
    migration_set = MigrationSet()
    migration_set.validate_migration_chain()
    migration_set.register(Migration(from_version=0, to_version=1, migrate=no_op_migrate_callback))
    migration_set.validate_migration_chain()
    migration_set.register(Migration(from_version=1, to_version=2, migrate=no_op_migrate_callback))
    migration_set.register(Migration(from_version=2, to_version=3, migrate=no_op_migrate_callback))
    migration_set.validate_migration_chain()
    migration_set.register(Migration(from_version=4, to_version=5, migrate=no_op_migrate_callback))
    with pytest.raises(MigrationError, match="Migration chain is fragmented"):
        migration_set.validate_migration_chain()


def test_migration_set_counts_migrations(no_op_migrate_callback: MigrateCallback):
    migration_set = MigrationSet()
    assert migration_set.count == 0
    migration_set.register(Migration(from_version=0, to_version=1, migrate=no_op_migrate_callback))
    assert migration_set.count == 1
    migration_set.register(Migration(from_version=1, to_version=2, migrate=no_op_migrate_callback))
    assert migration_set.count == 2


def test_migration_set_gets_latest_version(no_op_migrate_callback: MigrateCallback):
    migration_set = MigrationSet()
    assert migration_set.latest_version == 0
    migration_set.register(Migration(from_version=1, to_version=2, migrate=no_op_migrate_callback))
    assert migration_set.latest_version == 2
    migration_set.register(Migration(from_version=0, to_version=1, migrate=no_op_migrate_callback))
    assert migration_set.latest_version == 2


def test_migrator_registers_migration(migrator: SQLiteMigrator, migration_no_op: Migration):
    migration = migration_no_op
    migrator.register_migration(migration)
    assert migration in migrator._migration_set._migrations


def test_migrator_creates_migrations_table(migrator: SQLiteMigrator):
    migrator._create_migrations_table(migrator._cursor)
    migrator._cursor.execute("SELECT * FROM sqlite_master WHERE type='table' AND name='migrations';")
    assert migrator._cursor.fetchone() is not None


def test_migrator_migration_sets_version(migrator: SQLiteMigrator, migration_no_op: Migration):
    migrator._create_migrations_table(migrator._cursor)
    migrator.register_migration(migration_no_op)
    migrator.run_migrations()
    migrator._cursor.execute("SELECT MAX(version) FROM migrations;")
    assert migrator._cursor.fetchone()[0] == 1


def test_migrator_gets_current_version(migrator: SQLiteMigrator, migration_no_op: Migration):
    assert migrator._get_current_version(migrator._cursor) == 0
    migrator._create_migrations_table(migrator._cursor)
    assert migrator._get_current_version(migrator._cursor) == 0
    migrator.register_migration(migration_no_op)
    migrator.run_migrations()
    assert migrator._get_current_version(migrator._cursor) == 1


def test_migrator_runs_single_migration(migrator: SQLiteMigrator, migration_create_test_table: Migration):
    migrator._create_migrations_table(migrator._cursor)
    migrator._run_migration(migration_create_test_table, migrator._cursor)
    assert migrator._get_current_version(migrator._cursor) == 1
    migrator._cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test';")
    assert migrator._cursor.fetchone() is not None


def test_migrator_runs_all_migrations_in_memory(
    migrator: SQLiteMigrator,
):
    migrations = [Migration(from_version=i, to_version=i + 1, migrate=create_migrate(i)) for i in range(0, 3)]
    for migration in migrations:
        migrator.register_migration(migration)
    migrator.run_migrations()
    assert migrator._get_current_version(migrator._cursor) == 3


def test_migrator_runs_all_migrations_file(logger: Logger, lock: threading.RLock):
    with TemporaryDirectory() as tempdir:
        original_db_path = Path(tempdir) / "invokeai.db"
        # The Migrator closes the database when it finishes; we cannot use a context manager.
        original_db_conn = sqlite3.connect(original_db_path)
        migrator = SQLiteMigrator(conn=original_db_conn, db_path=original_db_path, lock=lock, logger=logger)
        migrations = [Migration(from_version=i, to_version=i + 1, migrate=create_migrate(i)) for i in range(0, 3)]
        for migration in migrations:
            migrator.register_migration(migration)
        migrator.run_migrations()
        with sqlite3.connect(original_db_path) as original_db_conn:
            original_db_cursor = original_db_conn.cursor()
            assert SQLiteMigrator._get_current_version(original_db_cursor) == 3


def test_migrator_creates_temp_db():
    with TemporaryDirectory() as tempdir:
        original_db_path = Path(tempdir) / "invokeai.db"
        with sqlite3.connect(original_db_path):
            # create the db file so _create_temp_db has something to copy
            pass
        temp_db_path = SQLiteMigrator._create_temp_db(original_db_path)
        assert temp_db_path.is_file()
        assert temp_db_path == SQLiteMigrator._get_temp_db_path(original_db_path)


def test_migrator_finalizes():
    with TemporaryDirectory() as tempdir:
        original_db_path = Path(tempdir) / "invokeai.db"
        temp_db_path = SQLiteMigrator._get_temp_db_path(original_db_path)
        backup_db_path = SQLiteMigrator._get_backup_db_path(original_db_path)
        with sqlite3.connect(original_db_path) as original_db_conn, sqlite3.connect(temp_db_path) as temp_db_conn:
            original_db_cursor = original_db_conn.cursor()
            original_db_cursor.execute("CREATE TABLE original_db_test (id INTEGER PRIMARY KEY);")
            original_db_conn.commit()
            temp_db_cursor = temp_db_conn.cursor()
            temp_db_cursor.execute("CREATE TABLE temp_db_test (id INTEGER PRIMARY KEY);")
            temp_db_conn.commit()
            SQLiteMigrator._finalize_migration(
                original_db_path=original_db_path,
                temp_db_path=temp_db_path,
            )
        with sqlite3.connect(backup_db_path) as backup_db_conn, sqlite3.connect(temp_db_path) as temp_db_conn:
            backup_db_cursor = backup_db_conn.cursor()
            backup_db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='original_db_test';")
            assert backup_db_cursor.fetchone() is not None
            temp_db_cursor = temp_db_conn.cursor()
            temp_db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='temp_db_test';")
            assert temp_db_cursor.fetchone() is None


def test_migrator_makes_no_changes_on_failed_migration(
    migrator: SQLiteMigrator, migration_no_op: Migration, failing_migrate_callback: MigrateCallback
):
    migrator.register_migration(migration_no_op)
    migrator.run_migrations()
    assert migrator._get_current_version(migrator._cursor) == 1
    migrator.register_migration(Migration(from_version=1, to_version=2, migrate=failing_migrate_callback))
    with pytest.raises(MigrationError, match="Bad migration"):
        migrator.run_migrations()
    assert migrator._get_current_version(migrator._cursor) == 1


def test_idempotent_migrations(migrator: SQLiteMigrator, migration_create_test_table: Migration):
    migrator.register_migration(migration_create_test_table)
    migrator.run_migrations()
    # not throwing is sufficient
    migrator.run_migrations()
    assert migrator._get_current_version(migrator._cursor) == 1
