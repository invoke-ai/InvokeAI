"""Databases on the backend under test.

Tests run against SQLite unless INVOKEAI_TEST_DB_URL names a MySQL or MariaDB server, e.g.
`mysql+pymysql://root:secret@127.0.0.1:3306`. Each xdist worker of each test run then gets schemas of
its own on that server, so neither workers nor concurrent runs (an IDE and a terminal, two worktrees) see
each other's rows. The account needs the privilege to create and drop databases.
"""

import os
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from unittest import mock

import pytest
from sqlalchemy import URL, Engine, create_engine, delete, event, insert, make_url, select

from invokeai.app.services.config.config_default import DefaultInvokeAIAppConfig
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.engines import (
    MARIADB_BINARY_COLLATION,
    MYSQL_BINARY_COLLATION,
    create_mysql_engine,
)
from invokeai.app.services.shared.database.schema import metadata
from invokeai.app.services.shared.sqlite_migrator.migration_loader import MigrationBuildContext, build_migrations
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import Migrator
from invokeai.backend.util.logging import InvokeAILogger

TEST_DB_URL_ENV = "INVOKEAI_TEST_DB_URL"
_SCHEMA_PREFIX = "invokeai_test_"
_GENERATED_SCHEMA = re.compile(r"invokeai_test_(?:[0-9a-f]{8}_)?(?:main|gw[0-9]+)(?:_app)?")


def external_test_db_url() -> Optional[str]:
    """The server named by INVOKEAI_TEST_DB_URL, or None when tests run against SQLite."""
    return os.environ.get(TEST_DB_URL_ENV) or None


def migrate_to_newest(database: Database, root: Path) -> None:
    """Runs the application's migrations: a new database is then at the newest schema, with its seeded rows.

    The migrations clean up legacy files under the root, so it must be a test's own directory; the config
    reads nothing from the environment that could point them elsewhere.
    """
    config = DefaultInvokeAIAppConfig()
    config._root = root
    migrator = Migrator(database)
    context = MigrationBuildContext(
        app_config=config,
        logger=InvokeAILogger.get_logger("test_database"),
        image_files=mock.Mock(spec=ImageFileStorageBase),
    )
    for migration in build_migrations(context):
        migrator.register_migration(migration)
    migrator.run_migrations()


@contextmanager
def _server_schema(base: URL, suffix: str) -> Iterator[URL]:
    """Creates a schema of this run and worker on the server, and drops it again."""
    run = os.environ.get("PYTEST_XDIST_TESTRUNUID", uuid.uuid4().hex)[:8]
    schema = f"{_SCHEMA_PREFIX}{run}_{os.environ.get('PYTEST_XDIST_WORKER', 'main')}{suffix}"
    # `URL.set(database=None)` keeps the component, so the server-level URL is built from parts.
    server = create_engine(
        URL.create(base.drivername, base.username, base.password, base.host, base.port, query=base.query)
    )
    # Held for the whole session: it tells other runs this schema is in use. A schema whose lock nobody
    # holds was left behind by a run that crashed, and is dropped.
    owner = server.connect()
    try:
        owner.exec_driver_sql("SELECT GET_LOCK(%s, 0)", (schema,))
        _drop_abandoned_schemas(server)
        with server.begin() as conn:
            is_mariadb = "mariadb" in str(conn.exec_driver_sql("SELECT VERSION()").scalar_one()).lower()
            # The binary NO PAD collation the layer requires, for the tables tests create.
            collation = MARIADB_BINARY_COLLATION if is_mariadb else MYSQL_BINARY_COLLATION
            conn.exec_driver_sql(f"CREATE DATABASE `{schema}` CHARACTER SET utf8mb4 COLLATE {collation}")
        yield base.set(database=schema)
    finally:
        with server.begin() as conn:
            conn.exec_driver_sql(f"DROP DATABASE IF EXISTS `{schema}`")
        owner.close()
        server.dispose()


def _drop_abandoned_schemas(server: Engine) -> None:
    with server.connect() as conn:
        schemas: list[str] = list(
            conn.exec_driver_sql(
                "SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE %s",
                (_SCHEMA_PREFIX.replace("_", "\\_") + "%",),
            ).scalars()
        )
        for schema in schemas:
            # Only names this fixture generates (and its older fixed ones), never someone's own database.
            if not _GENERATED_SCHEMA.fullmatch(schema):
                continue
            if conn.exec_driver_sql("SELECT GET_LOCK(%s, 0)", (schema,)).scalar_one() == 1:
                conn.exec_driver_sql(f"DROP DATABASE IF EXISTS `{schema}`")
                conn.exec_driver_sql("SELECT RELEASE_LOCK(%s)", (schema,))
        conn.commit()


def _open(url: URL) -> Database:
    return Database.open_url(url.render_as_string(hide_password=False), InvokeAILogger.get_logger("test_database"))


@pytest.fixture(scope="session")
def _external_test_schema() -> Iterator[Optional[URL]]:
    url = external_test_db_url()
    if url is None:
        yield None
        return
    with _server_schema(make_url(url), "") as schema:
        yield schema


@dataclass(frozen=True)
class _ApplicationSchema:
    engine: Engine
    # The rows of a new database, per table: the generated columns left out, as the database computes them.
    rows: dict[str, list[dict[str, Any]]]


@pytest.fixture(scope="session")
def _external_application_schema(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Optional[_ApplicationSchema]]:
    url = external_test_db_url()
    if url is None:
        yield None
        return
    with _server_schema(make_url(url), "_app") as schema:
        engine = create_mysql_engine(schema)
        try:
            database = Database(engine, InvokeAILogger.get_logger("test_database"))
            migrate_to_newest(database, tmp_path_factory.mktemp("migrations"))
            with database.begin(write=False) as conn:
                rows = {
                    table.name: [
                        row._asdict()
                        for row in conn.execute(
                            select(*[column for column in table.columns if column.computed is None])
                        )
                    ]
                    for table in metadata.sorted_tables
                }
            yield _ApplicationSchema(engine=engine, rows={name: each for name, each in rows.items() if each})
        finally:
            engine.dispose()


@pytest.fixture(scope="session")
def _migrated_sqlite(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Database]:
    """An in-memory SQLite database migrated once per run, which the `database` fixture copies for each test."""
    database = Database.open_sqlite(None, InvokeAILogger.get_logger("test_database"))
    try:
        migrate_to_newest(database, tmp_path_factory.mktemp("migrations"))
        yield database
    finally:
        database.dispose()


@pytest.fixture
def empty_database(_external_test_schema: Optional[URL]) -> Iterator[Database]:
    """A database without tables: SQLite in memory, or this worker's schema on the external server."""
    if _external_test_schema is None:
        database = Database.open_sqlite(None, InvokeAILogger.get_logger("test_database"))
    else:
        database = _open(_external_test_schema)
    try:
        if _external_test_schema is not None:
            # Also before the test: a teardown that failed must not hand its tables to the next test.
            _drop_all_tables(database)
        yield database
    finally:
        try:
            if _external_test_schema is not None:
                _drop_all_tables(database)
        finally:
            database.dispose()


@pytest.fixture
def database(
    _external_application_schema: Optional[_ApplicationSchema], request: pytest.FixtureRequest
) -> Iterator[Database]:
    """A database at the newest schema, holding the rows a new install starts with (the system account, the
    JWT secret, the lock rows): a copy in memory of a SQLite database migrated once per run, or this worker's
    schema on the external server, migrated once per run and reset to those rows before each test.

    On a server the ids a table generates go on counting from test to test: assert on the ids a test got back,
    not on literal ones.
    """
    logger = InvokeAILogger.get_logger("test_database")
    if _external_application_schema is None:
        migrated: Database = request.getfixturevalue("_migrated_sqlite")
        database = Database.open_sqlite(None, logger)
        try:
            migrated.sqlite.conn.backup(database.sqlite.conn)
            yield database
        finally:
            database.dispose()
        return
    # The engine, and its pool of connections, serves every test of the run.
    database = Database(_external_application_schema.engine, logger)
    with database.begin(write=True) as conn:
        for table in reversed(metadata.sorted_tables):
            conn.execute(delete(table))
        for table in metadata.sorted_tables:
            if rows := _external_application_schema.rows.get(table.name):
                conn.execute(insert(table), rows)
    yield database


def _drop_all_tables(database: Database) -> None:
    with database.engine.connect() as conn:
        tables: list[str] = list(
            conn.exec_driver_sql(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE()"
            ).scalars()
        )
        conn.exec_driver_sql("SET FOREIGN_KEY_CHECKS = 0")
        for table in tables:
            conn.exec_driver_sql(f"DROP TABLE `{table}`")
        conn.exec_driver_sql("SET FOREIGN_KEY_CHECKS = 1")
        conn.commit()


@contextmanager
def capture_statements(database: Database) -> Iterator[list[tuple[str, Any]]]:
    """The statements the database runs inside the block, each with its parameters as the driver gets them."""
    captured: list[tuple[str, Any]] = []

    def record(conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, executemany: bool) -> None:
        captured.append((statement, parameters))

    event.listen(database.engine, "before_cursor_execute", record)
    try:
        yield captured
    finally:
        event.remove(database.engine, "before_cursor_execute", record)


def explain_query_plan(database: Database, statement: str, parameters: Any) -> list[str]:
    """The details of SQLite's plan for a captured statement."""
    with database.begin(write=False) as conn:
        return [row[3] for row in conn.exec_driver_sql(f"EXPLAIN QUERY PLAN {statement}", parameters).all()]
