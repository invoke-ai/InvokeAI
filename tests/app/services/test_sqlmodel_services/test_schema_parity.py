"""Schema-parity guard between the raw-SQL migrations and SQLModel.metadata (models.py).

The dual-path database strategy is:
  - existing SQLite installs build their schema from the 31 raw-SQL migrations;
  - new MySQL/Postgres backends bootstrap from ``SQLModel.metadata.create_all()`` using
    ``invokeai/app/services/shared/sqlite/models.py``.

For that to be correct, the schema the migrations produce on SQLite and the schema
``create_all()`` would produce MUST match. These tests build the schema *both* ways and
diff them. They fail the moment a migration — from this PR or any future PR — adds a
table/column that ``models.py`` does not declare (or vice-versa), which is exactly the
drift that has repeatedly broken this branch (image_subfolder, status_sequence, the
GENERATED columns). They also assert ``create_all()`` DDL compiles on every target dialect.
"""

from logging import Logger

import pytest
from sqlalchemy import inspect
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.schema import CreateTable

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite.models import SQLModel
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database

# Tables that legitimately exist only in the migrated SQLite DB and are NOT managed by
# SQLModel.metadata: the migrator owns its own `migrations` bookkeeping table, and a
# non-sqlite bootstrap creates its equivalent separately.
_DB_ONLY_TABLES_ALLOWED = {"migrations"}


@pytest.fixture
def migrated_db() -> SqliteDatabase:
    """A fresh in-memory SQLite DB with all raw-SQL migrations applied."""
    config = InvokeAIAppConfig(use_memory_db=True)
    return create_mock_sqlite_database(config=config, logger=InvokeAILogger.get_logger())


def _db_schema(db: SqliteDatabase) -> dict[str, set[str]]:
    """Reflect {table_name: {column_names}} from the migrated SQLite database."""
    inspector = inspect(db._engine)
    return {name: {c["name"] for c in inspector.get_columns(name)} for name in inspector.get_table_names()}


def test_no_migrated_tables_missing_from_models(migrated_db: SqliteDatabase) -> None:
    """Every table the migrations create must be declared in models.py, else create_all()
    will not create it on MySQL/Postgres and a SQLite->MySQL copy cannot carry it."""
    db_tables = set(_db_schema(migrated_db))
    model_tables = set(SQLModel.metadata.tables)
    missing = db_tables - model_tables - _DB_ONLY_TABLES_ALLOWED
    assert not missing, (
        "Tables present in the migrated SQLite DB but absent from models.py "
        f"(create_all() would NOT create them on MySQL/Postgres): {sorted(missing)}"
    )


def test_no_migrated_columns_missing_from_models(migrated_db: SqliteDatabase) -> None:
    """Every column the migrations create must be declared on the corresponding SQLModel
    table, else create_all() drops it on MySQL/Postgres and a data copy loses it."""
    db_schema = _db_schema(migrated_db)
    drift: dict[str, list[str]] = {}
    for table_name, table in SQLModel.metadata.tables.items():
        if table_name not in db_schema:
            continue
        missing = db_schema[table_name] - {c.name for c in table.columns}
        if missing:
            drift[table_name] = sorted(missing)
    assert not drift, (
        "Columns present in the migrated SQLite DB but NOT declared in models.py "
        "(create_all() would omit them on MySQL/Postgres; a SQLite->MySQL copy would lose them):\n"
        f"{drift}"
    )


def test_no_phantom_columns_in_models(migrated_db: SqliteDatabase) -> None:
    """Every column declared in models.py must exist in the migrated SQLite DB, else
    models.py has diverged ahead of the migrations (a column nothing creates on SQLite)."""
    db_schema = _db_schema(migrated_db)
    drift: dict[str, list[str]] = {}
    for table_name, table in SQLModel.metadata.tables.items():
        if table_name not in db_schema:
            continue
        phantom = {c.name for c in table.columns} - db_schema[table_name]
        if phantom:
            drift[table_name] = sorted(phantom)
    assert not drift, (
        "Columns declared in models.py but absent from the migrated SQLite DB "
        "(models.py is ahead of / inconsistent with the migrations):\n"
        f"{drift}"
    )


def test_create_all_bootstrap_path(tmp_path) -> None:
    """The external-backend bootstrap (db_url -> create_all, no migrations) builds a working
    schema and serves the ORM services.

    Uses a file-based SQLite URL as a stand-in for an external backend so the create_all path
    is exercised end-to-end without a running MySQL server. (models/workflow_library use the
    MySQL-only ``json_unquote`` in their generated columns, so those two tables are validated
    against real MariaDB separately.)
    """
    from unittest import mock

    from invokeai.app.services.board_records.board_records_sqlmodel import SqlModelBoardRecordStorage
    from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
    from invokeai.app.services.shared.sqlite.sqlite_util import init_db

    config = InvokeAIAppConfig(db_url=f"sqlite:///{tmp_path / 'createall.db'}")
    db = init_db(config=config, logger=InvokeAILogger.get_logger(), image_files=mock.Mock(spec=ImageFileStorageBase))

    # The external branch was taken: SQLAlchemy engine only, no raw sqlite3 connection / migrations.
    assert db._db_url is not None
    assert db._conn is None

    # The schema created by create_all() serves a real SQLModel service end-to-end.
    boards = SqlModelBoardRecordStorage(db=db)
    record = boards.save(board_name="smoke", user_id="system")
    assert boards.get(record.board_id).board_name == "smoke"

    # The dialect-aware Computed columns actually POPULATE on a create_all() backend:
    # inserting only id+config, the DB derives `base` from the config JSON via json_extract.
    import json

    from invokeai.app.services.shared.sqlite.models import ModelTable

    config = {
        "hash": "h",
        "base": "sdxl",
        "type": "main",
        "path": "/models/m1",
        "format": "diffusers",
        "name": "m1",
        "source": "s",
        "source_type": "path",
        "file_size": 123,
    }
    with db.get_session() as session:
        session.add(ModelTable(id="m1", config=json.dumps(config)))
    with db.get_readonly_session() as session:
        stored = session.get(ModelTable, "m1")
        assert stored is not None
        assert stored.base == "sdxl"
        assert stored.name == "m1"
        assert stored.file_size == 123


def test_copy_database_to_create_all_backend(tmp_path) -> None:
    """copy_database() moves rows from a migration-built SQLite DB into a create_all() backend,
    skipping DB-generated columns so the target recomputes them from the JSON on insert."""
    import json
    from unittest import mock

    from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
    from invokeai.app.services.shared.sqlite.db_copy import copy_database
    from invokeai.app.services.shared.sqlite.models import BoardTable, ModelTable
    from invokeai.app.services.shared.sqlite.sqlite_util import init_db

    logger = InvokeAILogger.get_logger()
    image_files = mock.Mock(spec=ImageFileStorageBase)

    # Source: migration-built SQLite, with one board and one model.
    source = init_db(config=InvokeAIAppConfig(use_memory_db=True), logger=logger, image_files=image_files)
    model_config = {
        "hash": "h",
        "base": "sdxl",
        "type": "main",
        "path": "/p",
        "format": "diffusers",
        "name": "m1",
        "source": "s",
        "source_type": "path",
        "file_size": 7,
    }
    with source.get_session() as s:
        s.add(BoardTable(board_id="b1", board_name="board-one", user_id="system"))
        s.add(ModelTable(id="m1", config=json.dumps(model_config)))

    # Target: create_all() backend (external path via db_url, a stand-in for MySQL/MariaDB).
    target = init_db(config=InvokeAIAppConfig(db_url=f"sqlite:///{tmp_path / 'tgt.db'}"), logger=logger, image_files=image_files)

    counts = copy_database(source, target, logger)
    assert counts["boards"] == 1
    assert counts["models"] == 1

    with target.get_readonly_session() as s:
        assert s.get(BoardTable, "b1").board_name == "board-one"
        migrated = s.get(ModelTable, "m1")
        assert migrated is not None
        # The model's generated columns were skipped on copy and recomputed by the target.
        assert migrated.base == "sdxl"
        assert migrated.file_size == 7


def test_no_phantom_tables_in_models(migrated_db: SqliteDatabase) -> None:
    """Every table declared in models.py must exist in the migrated SQLite DB, else models.py
    declares a table the migrations do not create (create_all() would build a table SQLite
    lacks, and a copy would read from a non-existent source table)."""
    db_tables = set(_db_schema(migrated_db))
    phantom = set(SQLModel.metadata.tables) - db_tables
    assert not phantom, (
        "Tables declared in models.py but absent from the migrated SQLite DB "
        f"(models.py is inconsistent with the migrations): {sorted(phantom)}"
    )


@pytest.mark.parametrize("dialect_name", ["sqlite", "mysql", "postgresql"])
def test_metadata_ddl_compiles_on_all_dialects(dialect_name: str) -> None:
    """SQLModel.metadata must emit valid CREATE TABLE DDL on every target backend, so a
    fresh create_all() install cannot abort mid-bootstrap (e.g. MySQL VARCHAR length)."""
    dialect = {"sqlite": sqlite.dialect(), "mysql": mysql.dialect(), "postgresql": postgresql.dialect()}[dialect_name]
    failures: list[str] = []
    for table in SQLModel.metadata.sorted_tables:
        try:
            str(CreateTable(table).compile(dialect=dialect))
        except Exception as e:  # noqa: BLE001 - we want to collect every failure
            failures.append(f"{table.name}: {type(e).__name__}: {e}")
    assert not failures, f"create_all() DDL fails to compile on {dialect_name}:\n" + "\n".join(failures)
