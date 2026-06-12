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
