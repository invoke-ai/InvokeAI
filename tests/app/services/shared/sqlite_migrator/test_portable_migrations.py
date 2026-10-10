"""Migrations on every database backend: portable migrations, and how a new database reaches the newest schema."""

import re
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional
from unittest import mock

import pytest
from pydantic import ValidationError
from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    MetaData,
    Table,
    column,
    delete,
    func,
    insert,
    inspect,
    select,
    table,
)

from invokeai.app.services.config.config_default import DefaultInvokeAIAppConfig
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.shared.database.copy import copy_rows
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.engines import MARIADB_BINARY_COLLATION, MYSQL_BINARY_COLLATION
from invokeai.app.services.shared.database.errors import ForeignKeyViolation
from invokeai.app.services.shared.database.schema import metadata
from invokeai.app.services.shared.database.schema.app_settings import app_settings
from invokeai.app.services.shared.database.schema.boards import board_images
from invokeai.app.services.shared.database.schema.migrator import applied_migrations
from invokeai.app.services.shared.database.schema.users import users
from invokeai.app.services.shared.database.types import Key
from invokeai.app.services.shared.sqlite_migrator import sqlite_migrator_impl
from invokeai.app.services.shared.sqlite_migrator.migration_loader import (
    PORTABLE_CUTOVER,
    MigrationBuildContext,
    build_migrations,
)
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import (
    Migration,
    MigrationBase,
    MigrationError,
    PortableMigration,
    PortableMigrationContext,
)
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import Migrator
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.database import external_test_db_url

server_only = pytest.mark.skipif(
    external_test_db_url() is None, reason="needs a MySQL or MariaDB server (INVOKEAI_TEST_DB_URL)"
)

# The migrations that run on SQLite only: every one up to the cutover. Their number grows only by a migration
# written before the cutover and merged after it (2026_09_30_store_embeddings_fp16), never by a new one.
SQLITE_ONLY_MIGRATIONS = 64
QUARANTINE_TABLE = "orphaned_projects_2026_08_06"
CUTOVER_SCHEMAS = Path(__file__).parent / "server_schema_at_cutover"

probe = table("portable_probe", column("id"))
parent = table("probe_parent", column("id"), column("name"))
child = table("probe_child", column("id"), column("parent_id"))


@pytest.fixture
def application_migrations(tmp_path: Path) -> list[MigrationBase]:
    # Migrations clean up legacy files under the root. No settings from the environment or a config file may
    # point them anywhere else.
    config = DefaultInvokeAIAppConfig()
    config._root = tmp_path
    context = MigrationBuildContext(
        app_config=config,
        logger=InvokeAILogger.get_logger("test_portable_migrations"),
        image_files=mock.Mock(spec=ImageFileStorageBase),
    )
    return build_migrations(context)


def _migrator(database: Database, migrations: Sequence[MigrationBase], *more: MigrationBase) -> Migrator:
    migrator = Migrator(database)
    for migration in [*migrations, *more]:
        migrator.register_migration(migration)
    return migrator


def _portable(callback: Callable[[PortableMigrationContext], None], name: str = "probe") -> PortableMigration:
    return PortableMigration(id=f"2099_01_01_{name}", depends_on=PORTABLE_CUTOVER, callback=callback)


def _applied(database: Database) -> set[str]:
    with database.begin(write=False) as conn:
        return set(conn.execute(select(applied_migrations.c.migration_id)).scalars())


def _has_table(database: Database, name: str) -> bool:
    with database.begin(write=False) as conn:
        return inspect(conn).has_table(name)


def _rows(database: Database, of: Any) -> list[tuple[Any, ...]]:
    with database.begin(write=False) as conn:
        return [tuple(row) for row in conn.execute(select(of).order_by(of.c.id))]


def _create_probe_table(context: PortableMigrationContext) -> None:
    # Idempotent, as a portable migration must be: on a server, a failed run may have created the table already.
    if not inspect(context.conn).has_table("portable_probe"):
        context.create_table("portable_probe", Column("id", Integer, primary_key=True, autoincrement=False))


def _create_parent_and_child(context: PortableMigrationContext) -> None:
    if not inspect(context.conn).has_table("probe_parent"):
        context.create_table(
            "probe_parent",
            Column("id", Integer, primary_key=True, autoincrement=False),
            Column("name", Key(), nullable=False),
        )
        context.create_table(
            "probe_child",
            Column("id", Integer, primary_key=True, autoincrement=False),
            Column("parent_id", Integer, ForeignKey("probe_parent.id", ondelete="CASCADE"), nullable=False),
        )


def test_a_portable_migration_has_no_legacy_version() -> None:
    with pytest.raises(ValidationError, match="no legacy version"):
        PortableMigration(from_version=0, to_version=1, callback=_create_probe_table)


def test_only_the_migrations_up_to_the_cutover_run_on_sqlite_only(
    application_migrations: list[MigrationBase],
) -> None:
    sqlite_only = [migration.id for migration in application_migrations if isinstance(migration, Migration)]

    # A SQLite-only migration added now, of whatever date, would be refused by every existing server database.
    assert len(sqlite_only) == SQLITE_ONLY_MIGRATIONS
    assert sqlite_only[-1] == PORTABLE_CUTOVER


def test_a_new_database_reaches_the_newest_schema_with_the_seeded_rows(
    empty_database: Database, application_migrations: list[MigrationBase]
) -> None:
    assert _migrator(empty_database, application_migrations).run_migrations()

    # The rows the migration chain seeds into a SQLite database. (On SQLite this compares the chain with itself;
    # on a server, the bootstrap with the chain.)
    reference = Database.open_sqlite(None, InvokeAILogger.get_logger("test_portable_migrations"))
    try:
        _migrator(reference, application_migrations).run_migrations()
        with reference.begin(write=False) as conn:
            expected_rows = {
                name: conn.execute(select(func.count()).select_from(each)).scalar_one()
                for name, each in metadata.tables.items()
                if name != QUARANTINE_TABLE
            }
    finally:
        reference.dispose()

    with empty_database.begin(write=False) as conn:
        tables = set(inspect(conn).get_table_names())
        rows = {
            name: conn.execute(select(func.count()).select_from(metadata.tables[name])).scalar_one()
            for name in expected_rows
        }
        secret: str = conn.execute(select(app_settings.c.value).where(app_settings.c.key == "jwt_secret")).scalar_one()
        user_ids: list[str] = list(conn.execute(select(users.c.user_id)).scalars())
    # The quarantine table is created on a server; on SQLite only where a migration had projects to keep.
    expected_tables = set(metadata.tables) - ({QUARANTINE_TABLE} if empty_database.dialect_name == "sqlite" else set())
    assert tables == expected_tables
    assert rows == expected_rows
    assert _applied(empty_database) == {migration.id for migration in application_migrations}
    assert user_ids == ["system"]
    assert len(secret) == 64

    # Up to date: a second run has nothing to do.
    assert not _migrator(empty_database, application_migrations).run_migrations()


def test_a_portable_migration_runs_once_and_is_recorded(
    empty_database: Database, application_migrations: list[MigrationBase]
) -> None:
    def add_probe(context: PortableMigrationContext) -> None:
        _create_probe_table(context)
        context.conn.execute(insert(probe).values(id=1))

    migration = _portable(add_probe)

    assert _migrator(empty_database, application_migrations, migration).run_migrations()
    # Not run again: its insert would fail on the existing row.
    assert not _migrator(empty_database, application_migrations, migration).run_migrations()

    assert _rows(empty_database, probe) == [(1,)]
    assert migration.id in _applied(empty_database)


def test_a_failed_portable_migration_is_not_recorded_and_runs_again(
    empty_database: Database, application_migrations: list[MigrationBase]
) -> None:
    def fail_halfway(context: PortableMigrationContext) -> None:
        _create_probe_table(context)
        raise RuntimeError("interrupted")

    failing = _portable(fail_halfway)
    with pytest.raises(MigrationError, match="interrupted"):
        _migrator(empty_database, application_migrations, failing).run_migrations()

    assert failing.id not in _applied(empty_database)
    # SQLite rolls the DDL back; a server commits each DDL statement as it runs, which is why a portable
    # migration checks what is there before it changes it.
    assert _has_table(empty_database, "portable_probe") is (empty_database.dialect_name != "sqlite")

    fixed = _portable(_create_probe_table)
    assert _migrator(empty_database, application_migrations, fixed).run_migrations()
    assert fixed.id in _applied(empty_database)
    assert _has_table(empty_database, "portable_probe")


def test_rebuilding_a_table_keeps_the_rows_that_reference_it(
    empty_database: Database, application_migrations: list[MigrationBase]
) -> None:
    def create_rows(context: PortableMigrationContext) -> None:
        _create_parent_and_child(context)
        context.conn.execute(insert(parent).values(id=1, name="parent"))
        context.conn.execute(insert(child).values(id=1, parent_id=1))

    def rebuild_parent(context: PortableMigrationContext) -> None:
        # On SQLite, batch mode copies the table and drops the original, which would cascade to the children.
        with context.op.batch_alter_table("probe_parent") as batch:
            batch.alter_column("name", existing_type=Key(), nullable=True)

    created = _portable(create_rows, "create_rows")
    rebuilt = PortableMigration(id="2099_01_02_rebuild", depends_on=created.id, callback=rebuild_parent)
    assert _migrator(empty_database, application_migrations, created, rebuilt).run_migrations()

    assert _rows(empty_database, child) == [(1, 1)]
    assert rebuilt.id in _applied(empty_database)

    # Foreign keys are on again: deleting the parent deletes its child, by cascade.
    with empty_database.begin(write=True) as conn:
        conn.execute(delete(parent).where(parent.c.id == 1))
    assert _rows(empty_database, child) == []


def test_a_portable_migration_that_orphans_rows_fails(
    empty_database: Database, application_migrations: list[MigrationBase]
) -> None:
    def orphan(context: PortableMigrationContext) -> None:
        _create_parent_and_child(context)
        context.conn.execute(insert(child).values(id=1, parent_id=99))

    migration = _portable(orphan)
    # A server refuses the row at once; SQLite, which runs portable migrations with foreign keys off, when the
    # migrator checks them before committing.
    with pytest.raises(MigrationError):
        _migrator(empty_database, application_migrations, migration).run_migrations()

    assert migration.id not in _applied(empty_database)
    if _has_table(empty_database, "probe_child"):
        assert _rows(empty_database, child) == []
    # Foreign keys are on again after the failure, too.
    with pytest.raises(ForeignKeyViolation):
        with empty_database.begin(write=True) as conn:
            conn.execute(insert(board_images).values(board_id="no board", image_name="no image.png"))


@pytest.mark.sqlite_only
def test_rows_orphaned_before_a_portable_migration_do_not_fail_it(
    empty_database: Database, application_migrations: list[MigrationBase]
) -> None:
    # Tools that deleted with foreign keys off left orphans in real databases; a migration did not make them.
    _migrator(empty_database, application_migrations).run_migrations()
    raw = empty_database.sqlite.conn
    raw.execute("PRAGMA foreign_keys = OFF")
    raw.execute("INSERT INTO board_images (board_id, image_name) VALUES ('deleted board', 'deleted.png')")
    raw.commit()
    raw.execute("PRAGMA foreign_keys = ON")

    migration = _portable(_create_probe_table)
    assert _migrator(empty_database, application_migrations, migration).run_migrations()

    assert migration.id in _applied(empty_database)


def test_a_database_from_a_newer_version_is_refused(
    empty_database: Database, application_migrations: list[MigrationBase]
) -> None:
    _migrator(empty_database, application_migrations).run_migrations()
    with empty_database.begin(write=True) as conn:
        conn.execute(insert(applied_migrations).values(migration_id="2999_01_01_from_a_newer_version"))

    with pytest.raises(MigrationError, match="unknown applied migration IDs"):
        _migrator(empty_database, application_migrations).run_migrations()


@server_only
def test_a_sqlite_only_migration_is_refused_on_a_server(
    empty_database: Database, application_migrations: list[MigrationBase]
) -> None:
    def cursor_callback(cursor: object) -> None:
        pass

    sqlite_only = Migration(id="2099_01_01_sqlite_only", depends_on=PORTABLE_CUTOVER, callback=cursor_callback)

    with pytest.raises(MigrationError, match="runs on SQLite only"):
        _migrator(empty_database, application_migrations, sqlite_only).run_migrations()


@server_only
def test_a_server_database_with_other_tables_is_refused(
    empty_database: Database, application_migrations: list[MigrationBase]
) -> None:
    with empty_database.begin(write=True) as conn:
        Table("orders", MetaData(), Column("id", Integer, primary_key=True)).create(conn)

    with pytest.raises(MigrationError, match="not an InvokeAI database"):
        _migrator(empty_database, application_migrations).run_migrations()


@server_only
def test_an_interrupted_creation_is_refused_rather_than_taken_for_complete(
    empty_database: Database, application_migrations: list[MigrationBase], monkeypatch: pytest.MonkeyPatch
) -> None:
    def copy_until_users(source: Database, target: Database, *, tables: Optional[Sequence[Table]] = None) -> None:
        # Copies, in the order the copy goes, what precedes the accounts, then stops as a crash would.
        wanted = set(tables if tables is not None else metadata.sorted_tables)
        before_users: list[Table] = []
        for each in metadata.sorted_tables:
            if each.name == users.name:
                break
            if each in wanted:
                before_users.append(each)
        copy_rows(source, target, tables=before_users)
        raise RuntimeError("interrupted")

    monkeypatch.setattr(sqlite_migrator_impl, "copy_rows", copy_until_users)
    with pytest.raises(RuntimeError, match="interrupted"):
        _migrator(empty_database, application_migrations).run_migrations()
    monkeypatch.undo()

    with pytest.raises(MigrationError, match="creation was interrupted"):
        _migrator(empty_database, application_migrations).run_migrations()


@server_only
def test_the_creation_reads_no_settings_from_the_environment(
    empty_database: Database,
    application_migrations: list[MigrationBase],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # A legacy core model the reference chain's clean-up deletes from the models directory it is given.
    models_dir = tmp_path / "elsewhere" / "models"
    legacy_model = models_dir / "core" / "upscaling" / "realesrgan" / "RealESRGAN_x4plus.pth"
    legacy_model.parent.mkdir(parents=True)
    legacy_model.write_bytes(b"weights")
    monkeypatch.setenv("INVOKEAI_MODELS_DIR", str(models_dir))

    assert _migrator(empty_database, application_migrations).run_migrations()

    assert legacy_model.exists()


@server_only
def test_one_process_at_a_time_creates_or_migrates_a_server_database(
    empty_database: Database, application_migrations: list[MigrationBase], monkeypatch: pytest.MonkeyPatch
) -> None:
    creating = threading.Event()
    release = threading.Event()
    outcome: list[object] = []

    def paused_copy_rows(source: Database, target: Database, *, tables: Optional[Sequence[Table]] = None) -> Any:
        creating.set()
        release.wait(60)
        return copy_rows(source, target, tables=tables)

    def create() -> None:
        try:
            outcome.append(_migrator(empty_database, application_migrations).run_migrations())
        except Exception as error:
            outcome.append(error)

    monkeypatch.setattr(sqlite_migrator_impl, "copy_rows", paused_copy_rows)
    first = threading.Thread(target=create)
    first.start()
    try:
        # The first process is creating the database: its tables exist, its rows not yet.
        while not creating.wait(0.1):
            assert first.is_alive(), outcome
        monkeypatch.setattr(sqlite_migrator_impl, "MIGRATION_LOCK_TIMEOUT_SECONDS", 1)
        with pytest.raises(MigrationError, match="Another process"):
            _migrator(empty_database, application_migrations).run_migrations()
    finally:
        release.set()
        first.join(60)

    assert outcome == [True]


@server_only
def test_portable_migrations_take_a_server_from_the_cutover_to_the_newest_schema(
    empty_database: Database, application_migrations: list[MigrationBase]
) -> None:
    # A server database as it was created at the cutover, which the migrations since must bring to what the
    # metadata creates today. Its default collation folds case, as a stock server's does, so that a table a
    # migration creates without the schema's table options shows.
    binary = MARIADB_BINARY_COLLATION if empty_database.dialect_name == "mariadb" else MYSQL_BINARY_COLLATION
    statements = _statements(CUTOVER_SCHEMAS / f"{empty_database.dialect_name}.sql")
    with empty_database.begin(write=True) as conn:
        conn.exec_driver_sql("ALTER DATABASE COLLATE utf8mb4_general_ci")
    try:
        with empty_database.begin(write=True) as conn:
            for statement in statements:
                conn.exec_driver_sql(statement)
            conn.execute(
                insert(applied_migrations),
                [
                    {"migration_id": migration.id, "legacy_version": migration.to_version}
                    for migration in application_migrations
                    if isinstance(migration, Migration)
                ],
            )
        _migrator(empty_database, application_migrations).run_migrations()
        migrated = _server_schema(empty_database)
    finally:
        with empty_database.begin(write=True) as conn:
            conn.exec_driver_sql(f"ALTER DATABASE COLLATE {binary}")

    with empty_database.begin(write=True) as conn:
        metadata.drop_all(conn)
        metadata.create_all(conn)

    assert migrated == _server_schema(empty_database)


def _statements(path: Path) -> list[str]:
    text = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("--"))
    return [statement.strip() for statement in text.split(";\n") if statement.strip()]


def _server_schema(database: Database) -> dict[str, tuple[list[str], list[str], str]]:
    """Each table as the server renders it: columns in order, keys and constraints in any order, and options."""
    schema: dict[str, tuple[list[str], list[str], str]] = {}
    with database.begin(write=False) as conn:
        for name in inspect(conn).get_table_names():
            ddl: str = conn.exec_driver_sql(f"SHOW CREATE TABLE `{name}`").one()[1]
            # MySQL names the character set of a column whose collation was given explicitly, as it is in the
            # frozen statements; the collation names it either way.
            ddl = re.sub(r"CHARACTER SET \w+ COLLATE", "COLLATE", ddl)
            lines = [line.strip().rstrip(",") for line in ddl.splitlines()]
            body = lines[1:-1]
            schema[name] = (
                [line for line in body if line.startswith("`")],
                sorted(line for line in body if not line.startswith("`")),
                re.sub(r" AUTO_INCREMENT=\d+", "", lines[-1]),
            )
    return schema
