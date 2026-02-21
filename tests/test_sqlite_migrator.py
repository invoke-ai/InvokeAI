import sqlite3
from contextlib import closing
from logging import Logger
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import ValidationError

from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import (
    MigrateCallback,
    Migration,
    MigrationError,
    MigrationSet,
    MigrationVersionError,
)
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import (
    SqliteMigrator,
)


@pytest.fixture
def logger() -> Logger:
    return Logger("test_sqlite_migrator")


@pytest.fixture
def memory_db_conn() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")


@pytest.fixture
def memory_db_cursor(memory_db_conn: sqlite3.Connection) -> sqlite3.Cursor:
    return memory_db_conn.cursor()


@pytest.fixture
def migrator(logger: Logger) -> SqliteMigrator:
    db = SqliteDatabase(db_path=None, logger=logger, verbose=False)
    return SqliteMigrator(db=db)


@pytest.fixture
def no_op_migrate_callback() -> MigrateCallback:
    def no_op_migrate(cursor: sqlite3.Cursor, **kwargs) -> None:
        pass

    return no_op_migrate


@pytest.fixture
def migration_no_op(no_op_migrate_callback: MigrateCallback) -> Migration:
    return Migration(from_version=0, to_version=1, callback=no_op_migrate_callback)


@pytest.fixture
def migrate_callback_create_table_of_name() -> MigrateCallback:
    def migrate(cursor: sqlite3.Cursor, **kwargs) -> None:
        table_name = kwargs["table_name"]
        cursor.execute(f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY);")

    return migrate


@pytest.fixture
def migrate_callback_create_test_table() -> MigrateCallback:
    def migrate(cursor: sqlite3.Cursor, **kwargs) -> None:
        cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY);")

    return migrate


@pytest.fixture
def migration_create_test_table(migrate_callback_create_test_table: MigrateCallback) -> Migration:
    return Migration(from_version=0, to_version=1, callback=migrate_callback_create_test_table)


@pytest.fixture
def failing_migration() -> Migration:
    def failing_migration(cursor: sqlite3.Cursor, **kwargs) -> None:
        raise Exception("Bad migration")

    return Migration(from_version=0, to_version=1, callback=failing_migration)


@pytest.fixture
def failing_migrate_callback() -> MigrateCallback:
    def failing_migrate(cursor: sqlite3.Cursor, **kwargs) -> None:
        raise Exception("Bad migration")

    return failing_migrate


def create_migrate(i: int) -> MigrateCallback:
    def migrate(cursor: sqlite3.Cursor, **kwargs) -> None:
        cursor.execute(f"CREATE TABLE test{i} (id INTEGER PRIMARY KEY);")

    return migrate


def test_migration_to_version_is_one_gt_from_version(no_op_migrate_callback: MigrateCallback) -> None:
    with pytest.raises(ValidationError, match="to_version must be one greater than from_version"):
        Migration(from_version=0, to_version=2, callback=no_op_migrate_callback)
    # not raising is sufficient
    Migration(from_version=1, to_version=2, callback=no_op_migrate_callback)


def test_migration_hash(no_op_migrate_callback: MigrateCallback) -> None:
    migration = Migration(from_version=0, to_version=1, callback=no_op_migrate_callback)
    assert hash(migration) == hash((0, 1))


def test_migration_set_add_migration(migrator: SqliteMigrator, migration_no_op: Migration) -> None:
    migration = migration_no_op
    migrator._migration_set.register(migration)
    assert migration in migrator._migration_set._migrations


def test_migration_set_may_not_register_dupes(
    migrator: SqliteMigrator, no_op_migrate_callback: MigrateCallback
) -> None:
    migrate_0_to_1_a = Migration(from_version=0, to_version=1, callback=no_op_migrate_callback)
    migrate_0_to_1_b = Migration(from_version=0, to_version=1, callback=no_op_migrate_callback)
    migrator._migration_set.register(migrate_0_to_1_a)
    with pytest.raises(MigrationVersionError, match=r"Migration with from_version or to_version already registered"):
        migrator._migration_set.register(migrate_0_to_1_b)
    migrate_1_to_2_a = Migration(from_version=1, to_version=2, callback=no_op_migrate_callback)
    migrate_1_to_2_b = Migration(from_version=1, to_version=2, callback=no_op_migrate_callback)
    migrator._migration_set.register(migrate_1_to_2_a)
    with pytest.raises(MigrationVersionError, match=r"Migration with from_version or to_version already registered"):
        migrator._migration_set.register(migrate_1_to_2_b)


def test_migration_set_gets_migration(migration_no_op: Migration) -> None:
    migration_set = MigrationSet()
    migration_set.register(migration_no_op)
    assert migration_set.get(0) == migration_no_op
    assert migration_set.get(1) is None


def test_migration_set_validates_migration_chain(no_op_migrate_callback: MigrateCallback) -> None:
    migration_set = MigrationSet()
    migration_set.register(Migration(from_version=1, to_version=2, callback=no_op_migrate_callback))
    with pytest.raises(MigrationError, match="Migration chain is fragmented"):
        # no migration from 0 to 1
        migration_set.validate_migration_chain()
    migration_set.register(Migration(from_version=0, to_version=1, callback=no_op_migrate_callback))
    migration_set.validate_migration_chain()
    migration_set.register(Migration(from_version=2, to_version=3, callback=no_op_migrate_callback))
    migration_set.validate_migration_chain()
    migration_set.register(Migration(from_version=4, to_version=5, callback=no_op_migrate_callback))
    with pytest.raises(MigrationError, match="Migration chain is fragmented"):
        # no migration from 3 to 4
        migration_set.validate_migration_chain()


def test_migration_set_counts_migrations(no_op_migrate_callback: MigrateCallback) -> None:
    migration_set = MigrationSet()
    assert migration_set.count == 0
    migration_set.register(Migration(from_version=0, to_version=1, callback=no_op_migrate_callback))
    assert migration_set.count == 1
    migration_set.register(Migration(from_version=1, to_version=2, callback=no_op_migrate_callback))
    assert migration_set.count == 2


def test_migration_set_gets_latest_version(no_op_migrate_callback: MigrateCallback) -> None:
    migration_set = MigrationSet()
    assert migration_set.latest_version == 0
    migration_set.register(Migration(from_version=1, to_version=2, callback=no_op_migrate_callback))
    assert migration_set.latest_version == 2
    migration_set.register(Migration(from_version=0, to_version=1, callback=no_op_migrate_callback))
    assert migration_set.latest_version == 2


def test_migration_runs(memory_db_cursor: sqlite3.Cursor, migrate_callback_create_test_table: MigrateCallback) -> None:
    migration = Migration(
        from_version=0,
        to_version=1,
        callback=migrate_callback_create_test_table,
    )
    migration.callback(memory_db_cursor)
    memory_db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test';")
    assert memory_db_cursor.fetchone() is not None


def test_migrator_registers_migration(migrator: SqliteMigrator, migration_no_op: Migration) -> None:
    migration = migration_no_op
    migrator.register_migration(migration)
    assert migration in migrator._migration_set._migrations


def test_migrator_creates_migrations_table(migrator: SqliteMigrator) -> None:
    cursor = migrator._db._conn.cursor()
    migrator._create_migrations_table(cursor)
    cursor.execute("SELECT * FROM sqlite_master WHERE type='table' AND name='migrations';")
    assert cursor.fetchone() is not None


def test_migrator_migration_sets_version(migrator: SqliteMigrator, migration_no_op: Migration) -> None:
    cursor = migrator._db._conn.cursor()
    migrator._create_migrations_table(cursor)
    migrator.register_migration(migration_no_op)
    migrator.run_migrations()
    cursor.execute("SELECT MAX(version) FROM migrations;")
    assert cursor.fetchone()[0] == 1


def test_migrator_gets_current_version(migrator: SqliteMigrator, migration_no_op: Migration) -> None:
    cursor = migrator._db._conn.cursor()
    assert migrator._get_current_version(cursor) == 0
    migrator._create_migrations_table(cursor)
    assert migrator._get_current_version(cursor) == 0
    migrator.register_migration(migration_no_op)
    migrator.run_migrations()
    assert migrator._get_current_version(cursor) == 1


def test_migrator_runs_single_migration(migrator: SqliteMigrator, migration_create_test_table: Migration) -> None:
    cursor = migrator._db._conn.cursor()
    migrator._create_migrations_table(cursor)
    migrator._run_migration(migration_create_test_table)
    assert migrator._get_current_version(cursor) == 1
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test';")
    assert cursor.fetchone() is not None


def test_migrator_runs_all_migrations_in_memory(migrator: SqliteMigrator) -> None:
    cursor = migrator._db._conn.cursor()
    migrations = [Migration(from_version=i, to_version=i + 1, callback=create_migrate(i)) for i in range(0, 3)]
    for migration in migrations:
        migrator.register_migration(migration)
    migrator.run_migrations()
    assert migrator._get_current_version(cursor) == 3


def test_migrator_runs_all_migrations_file(logger: Logger) -> None:
    with TemporaryDirectory() as tempdir:
        original_db_path = Path(tempdir) / "invokeai.db"
        db = SqliteDatabase(db_path=original_db_path, logger=logger, verbose=False)
        migrator = SqliteMigrator(db=db)
        migrations = [Migration(from_version=i, to_version=i + 1, callback=create_migrate(i)) for i in range(0, 3)]
        for migration in migrations:
            migrator.register_migration(migration)
        migrator.run_migrations()
        with closing(sqlite3.connect(original_db_path)) as original_db_conn:
            original_db_cursor = original_db_conn.cursor()
            assert SqliteMigrator._get_current_version(original_db_cursor) == 3
        # Must manually close else we get an error on Windows
        db._conn.close()


def test_migrator_backs_up_db(logger: Logger) -> None:
    with TemporaryDirectory() as tempdir:
        original_db_path = Path(tempdir) / "invokeai.db"
        db = SqliteDatabase(db_path=original_db_path, logger=logger, verbose=False)
        # Write some data to the db to test for successful backup
        temp_cursor = db._conn.cursor()
        temp_cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY);")
        db._conn.commit()
        # Set up the migrator
        migrator = SqliteMigrator(db=db)
        migrations = [Migration(from_version=i, to_version=i + 1, callback=create_migrate(i)) for i in range(0, 3)]
        for migration in migrations:
            migrator.register_migration(migration)
        migrator.run_migrations()
        # Must manually close else we get an error on Windows
        db._conn.close()
        assert original_db_path.exists()
        # We should have a backup file when we migrated a file db
        assert migrator._backup_path
        # Check that the test table exists as a proxy for successful backup
        with closing(sqlite3.connect(migrator._backup_path)) as backup_db_conn:
            backup_db_cursor = backup_db_conn.cursor()
            backup_db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test';")
            assert backup_db_cursor.fetchone() is not None


def test_migrator_makes_no_changes_on_failed_migration(
    migrator: SqliteMigrator, migration_no_op: Migration, failing_migrate_callback: MigrateCallback
) -> None:
    cursor = migrator._db._conn.cursor()
    migrator.register_migration(migration_no_op)
    migrator.run_migrations()
    assert migrator._get_current_version(cursor) == 1
    migrator.register_migration(Migration(from_version=1, to_version=2, callback=failing_migrate_callback))
    with pytest.raises(MigrationError, match="Bad migration"):
        migrator.run_migrations()
    assert migrator._get_current_version(cursor) == 1


def test_idempotent_migrations(migrator: SqliteMigrator, migration_create_test_table: Migration) -> None:
    cursor = migrator._db._conn.cursor()
    migrator.register_migration(migration_create_test_table)
    migrator.run_migrations()
    # not throwing is sufficient
    migrator.run_migrations()
    assert migrator._get_current_version(cursor) == 1


def test_migration_27_creates_users_table(logger: Logger) -> None:
    """Test that migration 27 creates the users table and related tables."""
    from invokeai.app.services.shared.sqlite_migrator.migrations.migration_27 import Migration27Callback

    db = SqliteDatabase(db_path=None, logger=logger, verbose=False)
    cursor = db._conn.cursor()

    # Create minimal tables that migration 27 expects to exist
    cursor.execute("CREATE TABLE IF NOT EXISTS boards (board_id TEXT PRIMARY KEY);")
    cursor.execute("CREATE TABLE IF NOT EXISTS images (image_name TEXT PRIMARY KEY);")
    cursor.execute("CREATE TABLE IF NOT EXISTS workflows (workflow_id TEXT PRIMARY KEY);")
    cursor.execute("CREATE TABLE IF NOT EXISTS session_queue (item_id INTEGER PRIMARY KEY);")
    db._conn.commit()

    # Run migration callback directly (not through migrator to avoid chain validation)
    migration_callback = Migration27Callback()
    migration_callback(cursor)
    db._conn.commit()

    # Verify users table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
    assert cursor.fetchone() is not None

    # Verify user_sessions table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_sessions';")
    assert cursor.fetchone() is not None

    # Verify user_invitations table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_invitations';")
    assert cursor.fetchone() is not None

    # Verify shared_boards table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='shared_boards';")
    assert cursor.fetchone() is not None

    # Verify system user was created
    cursor.execute("SELECT user_id, email FROM users WHERE user_id='system';")
    system_user = cursor.fetchone()
    assert system_user is not None
    assert system_user[0] == "system"
    assert system_user[1] == "system@system.invokeai"

    # Verify boards table has user_id column
    cursor.execute("PRAGMA table_info(boards);")
    columns = [row[1] for row in cursor.fetchall()]
    assert "user_id" in columns
    assert "is_public" in columns

    # Verify images table has user_id column
    cursor.execute("PRAGMA table_info(images);")
    columns = [row[1] for row in cursor.fetchall()]
    assert "user_id" in columns

    # Verify workflows table has user_id and is_public columns
    cursor.execute("PRAGMA table_info(workflows);")
    columns = [row[1] for row in cursor.fetchall()]
    assert "user_id" in columns
    assert "is_public" in columns

    # Verify client_state table has the new per-user schema
    cursor.execute("PRAGMA table_info(client_state);")
    columns = [row[1] for row in cursor.fetchall()]
    assert "user_id" in columns
    assert "key" in columns
    assert "value" in columns

    # Verify app_settings table exists and contains a JWT secret
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='app_settings';")
    assert cursor.fetchone() is not None
    cursor.execute("SELECT value FROM app_settings WHERE key = 'jwt_secret';")
    jwt_row = cursor.fetchone()
    assert jwt_row is not None
    assert len(jwt_row[0]) == 64  # 32 bytes = 64 hex characters

    db._conn.close()


def test_migration_27_with_existing_client_state_data(logger: Logger) -> None:
    """Test that migration 27 correctly migrates existing data from the old client_state schema."""
    import json

    from invokeai.app.services.shared.sqlite_migrator.migrations.migration_21 import Migration21Callback
    from invokeai.app.services.shared.sqlite_migrator.migrations.migration_27 import Migration27Callback

    db = SqliteDatabase(db_path=None, logger=logger, verbose=False)
    cursor = db._conn.cursor()

    # Run migration 21 to create old-style client_state with data column
    Migration21Callback()(cursor)
    # Insert some test data
    cursor.execute(
        "INSERT INTO client_state (id, data) VALUES (1, ?);",
        (json.dumps({"galleryView": "images", "lastBoardId": "board123"}),),
    )
    db._conn.commit()

    # Run migration 27 pre-reqs
    cursor.execute("CREATE TABLE IF NOT EXISTS boards (board_id TEXT PRIMARY KEY);")
    cursor.execute("CREATE TABLE IF NOT EXISTS images (image_name TEXT PRIMARY KEY);")
    cursor.execute("CREATE TABLE IF NOT EXISTS workflows (workflow_id TEXT PRIMARY KEY);")
    cursor.execute("CREATE TABLE IF NOT EXISTS session_queue (item_id INTEGER PRIMARY KEY);")
    cursor.execute("CREATE TABLE IF NOT EXISTS style_presets (id TEXT PRIMARY KEY);")
    db._conn.commit()

    # Run migration 27
    Migration27Callback()(cursor)
    db._conn.commit()

    # Verify new client_state schema
    cursor.execute("PRAGMA table_info(client_state);")
    columns = [row[1] for row in cursor.fetchall()]
    assert "user_id" in columns
    assert "key" in columns
    assert "value" in columns
    assert "updated_at" in columns
    assert "data" not in columns

    # Verify data was migrated to 'system' user
    cursor.execute("SELECT user_id, key, value FROM client_state ORDER BY key;")
    rows = [tuple(row) for row in cursor.fetchall()]
    assert len(rows) == 2
    assert ("system", "galleryView", "images") in rows
    assert ("system", "lastBoardId", "board123") in rows

    db._conn.close()


def test_migration_27_without_client_state_data_column(logger: Logger) -> None:
    """Test that migration 27 handles old client_state table without the data column."""
    from invokeai.app.services.shared.sqlite_migrator.migrations.migration_27 import Migration27Callback

    db = SqliteDatabase(db_path=None, logger=logger, verbose=False)
    cursor = db._conn.cursor()

    # Create old client_state WITHOUT data column (simulating an older migration 21)
    cursor.execute(
        """
        CREATE TABLE client_state (
          id          INTEGER PRIMARY KEY CHECK(id = 1),
          updated_at  DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP)
        );
        """
    )
    db._conn.commit()

    # Run migration 27 pre-reqs
    cursor.execute("CREATE TABLE IF NOT EXISTS boards (board_id TEXT PRIMARY KEY);")
    cursor.execute("CREATE TABLE IF NOT EXISTS images (image_name TEXT PRIMARY KEY);")
    cursor.execute("CREATE TABLE IF NOT EXISTS workflows (workflow_id TEXT PRIMARY KEY);")
    cursor.execute("CREATE TABLE IF NOT EXISTS session_queue (item_id INTEGER PRIMARY KEY);")
    cursor.execute("CREATE TABLE IF NOT EXISTS style_presets (id TEXT PRIMARY KEY);")
    db._conn.commit()

    # Run migration 27 - should not raise even without data column
    Migration27Callback()(cursor)
    db._conn.commit()

    # Verify new client_state schema
    cursor.execute("PRAGMA table_info(client_state);")
    columns = [row[1] for row in cursor.fetchall()]
    assert "user_id" in columns
    assert "key" in columns
    assert "value" in columns
    assert "updated_at" in columns

    # No rows should be migrated (nothing to migrate)
    cursor.execute("SELECT COUNT(*) FROM client_state;")
    assert cursor.fetchone()[0] == 0

    db._conn.close()
