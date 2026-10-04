"""Databases on the backend under test.

Tests run against SQLite unless INVOKEAI_TEST_DB_URL names a MySQL or MariaDB server, e.g.
`mysql+pymysql://root:secret@127.0.0.1:3306`. Each xdist worker of each test run then gets a schema of
its own on that server, so neither workers nor concurrent runs (an IDE and a terminal, two worktrees) see
each other's rows. The account needs the privilege to create and drop databases.
"""

import os
import re
import uuid
from collections.abc import Iterator
from typing import Optional

import pytest
from sqlalchemy import URL, Engine, create_engine, make_url

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.engines import MARIADB_BINARY_COLLATION, MYSQL_BINARY_COLLATION
from invokeai.backend.util.logging import InvokeAILogger

TEST_DB_URL_ENV = "INVOKEAI_TEST_DB_URL"
_SCHEMA_PREFIX = "invokeai_test_"
_GENERATED_SCHEMA = re.compile(r"invokeai_test_(?:[0-9a-f]{8}_)?(?:main|gw[0-9]+)")


def external_test_db_url() -> Optional[str]:
    """The server named by INVOKEAI_TEST_DB_URL, or None when tests run against SQLite."""
    return os.environ.get(TEST_DB_URL_ENV) or None


@pytest.fixture(scope="session")
def _external_test_schema() -> Iterator[Optional[URL]]:
    url = external_test_db_url()
    if url is None:
        yield None
        return
    base = make_url(url)
    run = os.environ.get("PYTEST_XDIST_TESTRUNUID", uuid.uuid4().hex)[:8]
    schema = f"{_SCHEMA_PREFIX}{run}_{os.environ.get('PYTEST_XDIST_WORKER', 'main')}"
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


@pytest.fixture
def empty_database(_external_test_schema: Optional[URL]) -> Iterator[Database]:
    """A database without tables: SQLite in memory, or this worker's schema on the external server."""
    logger = InvokeAILogger.get_logger("test_database")
    if _external_test_schema is None:
        database = Database.open_sqlite(None, logger)
    else:
        database = Database.open_url(_external_test_schema.render_as_string(hide_password=False), logger)
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
