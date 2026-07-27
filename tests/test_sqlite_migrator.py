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


def test_legacy_migration_gets_stable_id_and_dependency(no_op_migrate_callback: MigrateCallback) -> None:
    first_migration = Migration(from_version=0, to_version=1, callback=no_op_migrate_callback)
    second_migration = Migration(from_version=1, to_version=2, callback=no_op_migrate_callback)

    assert first_migration.id == "migration_1"
    assert first_migration.depends_on is None
    assert second_migration.id == "migration_2"
    assert second_migration.depends_on == "migration_1"


def test_explicit_migration_id_and_dependency_are_preserved(no_op_migrate_callback: MigrateCallback) -> None:
    migration = Migration(
        id="2026_06_29_add_test_table",
        depends_on="migration_1",
        callback=no_op_migrate_callback,
    )

    assert migration.id == "2026_06_29_add_test_table"
    assert migration.depends_on == "migration_1"
    assert migration.from_version is None
    assert migration.to_version is None


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


def test_migration_set_may_not_register_duplicate_ids(no_op_migrate_callback: MigrateCallback) -> None:
    migration_set = MigrationSet()
    migration_set.register(Migration(id="same_id", callback=no_op_migrate_callback))

    with pytest.raises(MigrationVersionError, match="Migration with id already registered"):
        migration_set.register(Migration(id="same_id", callback=no_op_migrate_callback))


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


def test_migration_set_validates_dependency_graph(no_op_migrate_callback: MigrateCallback) -> None:
    migration_set = MigrationSet()
    migration_set.register(Migration(id="a", callback=no_op_migrate_callback))
    migration_set.register(Migration(id="b", depends_on="a", callback=no_op_migrate_callback))
    migration_set.register(Migration(id="c", depends_on="a", callback=no_op_migrate_callback))
    migration_set.register(Migration(id="d", depends_on="c", callback=no_op_migrate_callback))

    migration_set.validate_dependency_graph()


def test_migration_set_rejects_missing_dependency(no_op_migrate_callback: MigrateCallback) -> None:
    migration_set = MigrationSet()
    migration_set.register(Migration(id="a", depends_on="missing", callback=no_op_migrate_callback))

    with pytest.raises(MigrationError, match="depends on unknown migration"):
        migration_set.validate_dependency_graph()


def test_migration_set_rejects_dependency_cycle(no_op_migrate_callback: MigrateCallback) -> None:
    migration_set = MigrationSet()
    migration_set.register(Migration(id="a", depends_on="b", callback=no_op_migrate_callback))
    migration_set.register(Migration(id="b", depends_on="a", callback=no_op_migrate_callback))

    with pytest.raises(MigrationError, match="cycle"):
        migration_set.validate_dependency_graph()


def test_migration_set_rejects_legacy_migration_depending_on_graph_only_migration(
    no_op_migrate_callback: MigrateCallback,
) -> None:
    migration_set = MigrationSet()
    migration_set.register(Migration(id="2026_06_30_graph_only", callback=no_op_migrate_callback))
    migration_set.register(
        Migration(
            id="migration_2",
            depends_on="2026_06_30_graph_only",
            from_version=1,
            to_version=2,
            callback=no_op_migrate_callback,
        )
    )

    with pytest.raises(MigrationError, match="cannot depend on graph-only migration"):
        migration_set.validate_dependency_graph()


def test_migration_set_plans_branching_migrations(no_op_migrate_callback: MigrateCallback) -> None:
    migration_set = MigrationSet()
    migration_a = Migration(id="a", callback=no_op_migrate_callback)
    migration_b = Migration(id="b", depends_on="a", callback=no_op_migrate_callback)
    migration_c = Migration(id="c", depends_on="a", callback=no_op_migrate_callback)
    migration_d = Migration(id="d", depends_on="c", callback=no_op_migrate_callback)
    migration_set.register(migration_d)
    migration_set.register(migration_c)
    migration_set.register(migration_b)
    migration_set.register(migration_a)

    assert migration_set.get_migration_plan(applied_migration_ids={"a", "b"}) == [migration_c, migration_d]


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


def test_migrator_creates_applied_migrations_table(migrator: SqliteMigrator) -> None:
    cursor = migrator._db._conn.cursor()
    migrator._create_applied_migrations_table(cursor)
    cursor.execute("SELECT * FROM sqlite_master WHERE type='table' AND name='applied_migrations';")
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


def test_migrator_bootstraps_applied_migrations_from_legacy_versions(
    migrator: SqliteMigrator, no_op_migrate_callback: MigrateCallback
) -> None:
    cursor = migrator._db._conn.cursor()
    migrator._create_migrations_table(cursor)
    cursor.execute("INSERT INTO migrations (version) VALUES (1);")
    cursor.execute("INSERT INTO migrations (version) VALUES (2);")
    cursor.connection.commit()
    migrator.register_migration(Migration(from_version=0, to_version=1, callback=no_op_migrate_callback))
    migrator.register_migration(Migration(from_version=1, to_version=2, callback=no_op_migrate_callback))
    migrator.register_migration(Migration(from_version=2, to_version=3, callback=no_op_migrate_callback))

    migrator.run_migrations()

    cursor.execute("SELECT migration_id FROM applied_migrations ORDER BY migration_id;")
    assert [row[0] for row in cursor.fetchall()] == ["migration_1", "migration_2", "migration_3"]
    assert migrator._get_current_version(cursor) == 3


def test_migrator_backs_up_file_db_before_metadata_only_bootstrap(
    logger: Logger, no_op_migrate_callback: MigrateCallback
) -> None:
    with TemporaryDirectory() as tempdir:
        original_db_path = Path(tempdir) / "invokeai.db"
        db = SqliteDatabase(db_path=original_db_path, logger=logger, verbose=False)
        cursor = db._conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY);")
        cursor.execute(
            """--sql
            CREATE TABLE migrations (
                version INTEGER PRIMARY KEY,
                migrated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW'))
            );
            """
        )
        cursor.execute("INSERT INTO migrations (version) VALUES (0);")
        cursor.execute("INSERT INTO migrations (version) VALUES (1);")
        db._conn.commit()

        migrator = SqliteMigrator(db=db)
        migrator.register_migration(Migration(from_version=0, to_version=1, callback=no_op_migrate_callback))

        assert migrator.run_migrations() is False

        db._conn.close()
        assert migrator._backup_path is not None
        with closing(sqlite3.connect(migrator._backup_path)) as backup_db_conn:
            backup_db_cursor = backup_db_conn.cursor()
            backup_db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test';")
            assert backup_db_cursor.fetchone() is not None
            backup_db_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='applied_migrations';")
            assert backup_db_cursor.fetchone() is None


def test_migrator_runs_branching_graph_migrations(migrator: SqliteMigrator) -> None:
    cursor = migrator._db._conn.cursor()
    executed: list[str] = []

    def create_migration(migration_id: str, depends_on: str | None) -> Migration:
        def migrate(cursor: sqlite3.Cursor) -> None:
            executed.append(migration_id)
            cursor.execute(f"CREATE TABLE {migration_id} (id INTEGER PRIMARY KEY);")

        return Migration(id=migration_id, depends_on=depends_on, callback=migrate)

    for migration in [
        create_migration("d", "c"),
        create_migration("c", "a"),
        create_migration("b", "a"),
        create_migration("a", None),
    ]:
        migrator.register_migration(migration)

    migrator.run_migrations()

    assert executed == ["a", "b", "c", "d"]
    cursor.execute("SELECT migration_id FROM applied_migrations ORDER BY migration_id;")
    assert [row[0] for row in cursor.fetchall()] == ["a", "b", "c", "d"]


def test_migrator_rejects_unknown_applied_migration(
    migrator: SqliteMigrator, no_op_migrate_callback: MigrateCallback
) -> None:
    cursor = migrator._db._conn.cursor()
    migrator._create_migrations_table(cursor)
    migrator._create_applied_migrations_table(cursor)
    cursor.execute("INSERT INTO applied_migrations (migration_id) VALUES ('future_migration');")
    cursor.connection.commit()
    migrator.register_migration(Migration(from_version=0, to_version=1, callback=no_op_migrate_callback))

    with pytest.raises(MigrationError, match="unknown applied migration"):
        migrator.run_migrations()


def test_migrator_rejects_unknown_applied_migration_before_creating_legacy_table(
    migrator: SqliteMigrator, no_op_migrate_callback: MigrateCallback
) -> None:
    cursor = migrator._db._conn.cursor()
    migrator._create_applied_migrations_table(cursor)
    cursor.execute("INSERT INTO applied_migrations (migration_id) VALUES ('future_migration');")
    cursor.connection.commit()
    migrator.register_migration(Migration(from_version=0, to_version=1, callback=no_op_migrate_callback))

    with pytest.raises(MigrationError, match="unknown applied migration"):
        migrator.run_migrations()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='migrations';")
    assert cursor.fetchone() is None


def test_migrator_rejects_inconsistent_applied_legacy_version(
    migrator: SqliteMigrator, no_op_migrate_callback: MigrateCallback
) -> None:
    cursor = migrator._db._conn.cursor()
    migrator._create_migrations_table(cursor)
    migrator._create_applied_migrations_table(cursor)
    cursor.execute("INSERT INTO migrations (version) VALUES (1);")
    cursor.execute("INSERT INTO migrations (version) VALUES (2);")
    cursor.execute("INSERT INTO applied_migrations (migration_id, legacy_version) VALUES ('migration_1', 2);")
    cursor.connection.commit()
    callback_ran = False

    def migration_callback(cursor: sqlite3.Cursor) -> None:
        nonlocal callback_ran
        callback_ran = True

    migrator.register_migration(Migration(from_version=0, to_version=1, callback=no_op_migrate_callback))
    migrator.register_migration(Migration(from_version=1, to_version=2, callback=migration_callback))

    with pytest.raises(MigrationError, match="inconsistent applied migration state"):
        migrator.run_migrations()

    assert callback_ran is False


def test_migrator_rejects_applied_legacy_migration_missing_legacy_row(
    migrator: SqliteMigrator, no_op_migrate_callback: MigrateCallback
) -> None:
    cursor = migrator._db._conn.cursor()
    migrator._create_migrations_table(cursor)
    migrator._create_applied_migrations_table(cursor)
    cursor.execute("INSERT INTO migrations (version) VALUES (1);")
    cursor.execute("INSERT INTO applied_migrations (migration_id, legacy_version) VALUES ('migration_2', 2);")
    cursor.connection.commit()
    callback_ran = False

    def graph_migration_callback(cursor: sqlite3.Cursor) -> None:
        nonlocal callback_ran
        callback_ran = True

    migrator.register_migration(Migration(from_version=0, to_version=1, callback=no_op_migrate_callback))
    migrator.register_migration(Migration(from_version=1, to_version=2, callback=no_op_migrate_callback))
    migrator.register_migration(
        Migration(id="2026_06_30_graph_migration", depends_on="migration_2", callback=graph_migration_callback)
    )

    with pytest.raises(MigrationError, match="inconsistent applied migration state"):
        migrator.run_migrations()

    assert callback_ran is False


def test_migrator_rejects_unknown_legacy_version(
    migrator: SqliteMigrator, no_op_migrate_callback: MigrateCallback
) -> None:
    cursor = migrator._db._conn.cursor()
    migrator._create_migrations_table(cursor)
    cursor.execute("INSERT INTO migrations (version) VALUES (1);")
    cursor.execute("INSERT INTO migrations (version) VALUES (2);")
    cursor.connection.commit()
    migrator.register_migration(Migration(from_version=0, to_version=1, callback=no_op_migrate_callback))

    with pytest.raises(MigrationError, match="unknown legacy migration version"):
        migrator.run_migrations()


def test_migrator_rejects_unknown_legacy_version_before_creating_applied_table(
    migrator: SqliteMigrator, no_op_migrate_callback: MigrateCallback
) -> None:
    cursor = migrator._db._conn.cursor()
    migrator._create_migrations_table(cursor)
    cursor.execute("INSERT INTO migrations (version) VALUES (1);")
    cursor.execute("INSERT INTO migrations (version) VALUES (2);")
    cursor.connection.commit()
    migrator.register_migration(Migration(from_version=0, to_version=1, callback=no_op_migrate_callback))

    with pytest.raises(MigrationError, match="unknown legacy migration version"):
        migrator.run_migrations()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='applied_migrations';")
    assert cursor.fetchone() is None


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
    cursor.execute("SELECT migration_id FROM applied_migrations ORDER BY migration_id;")
    assert [row[0] for row in cursor.fetchall()] == ["migration_1"]


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
