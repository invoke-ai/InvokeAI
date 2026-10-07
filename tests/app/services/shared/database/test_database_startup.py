"""Opening the database a config names, and what a server database is checked for before the app uses it."""

import logging
from pathlib import Path
from typing import Optional

import pytest
from sqlalchemy import URL, delete

from invokeai.app.services.config.config_default import DefaultInvokeAIAppConfig, InvokeAIAppConfig
from invokeai.app.services.shared.database import startup
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import DatabaseInUseError
from invokeai.app.services.shared.database.schema.migrator import applied_migrations
from invokeai.app.services.shared.database.startup import (
    DatabaseSetupError,
    init_database,
    open_database,
    open_migrated_database,
    redacted_database_url,
)
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import NoImageFiles
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.database import external_test_db_url

server_only = pytest.mark.skipif(
    external_test_db_url() is None, reason="needs a MySQL or MariaDB server (INVOKEAI_TEST_DB_URL)"
)
LOGGER = InvokeAILogger.get_logger("test_startup")


def _config(root: Path, **values: object) -> InvokeAIAppConfig:
    # Read from no file and no environment: the migrations clean up files under the root.
    config = DefaultInvokeAIAppConfig(**values)
    config._root = root
    return config


def test_a_memory_database_and_a_url_exclude_each_other(tmp_path: Path) -> None:
    config = _config(tmp_path, use_memory_db=True, db_url="mariadb+pymysql://invokeai:secret@db/invokeai")
    with pytest.raises(DatabaseSetupError, match="exclude each other"):
        open_database(config, LOGGER)


def test_a_url_is_shown_without_its_password() -> None:
    assert redacted_database_url("mariadb+pymysql://invokeai:s3cret@db/invokeai") == (
        "mariadb+pymysql://invokeai:***@db/invokeai"
    )
    assert redacted_database_url(None) is None
    assert redacted_database_url("no url at all") == "***"


@pytest.mark.parametrize(
    ("reported", "version"),
    [("8.4.3", (8, 4)), ("10.11.6-MariaDB-1:10.11.6+maria~ubu2204", (10, 11)), ("11.4.2-MariaDB-log", (11, 4))],
)
def test_server_versions_are_read_from_what_the_server_reports(reported: str, version: tuple[int, int]) -> None:
    assert startup._version_of(reported) == version


def test_a_tool_refuses_a_database_the_app_has_not_updated(tmp_path: Path) -> None:
    config = _config(tmp_path)
    database = init_database(config, LOGGER, NoImageFiles())
    with database.begin(write=True) as conn:
        conn.execute(
            delete(applied_migrations).where(applied_migrations.c.migration_id == "2026_10_07_drop_sqlite_triggers")
        )
    database.dispose()

    with pytest.raises(DatabaseSetupError, match="start InvokeAI once"):
        open_migrated_database(config, LOGGER)

    # As the app does when it starts.
    init_database(config, LOGGER, NoImageFiles()).dispose()
    open_migrated_database(config, LOGGER).dispose()


def test_a_tool_creates_no_database_where_there_is_none(tmp_path: Path) -> None:
    config = _config(tmp_path)

    with pytest.raises(DatabaseSetupError, match="no database at"):
        open_migrated_database(config, LOGGER)

    assert not config.db_path.exists()


@server_only
def test_one_process_at_a_time_holds_a_server_database(_external_test_schema: Optional[URL]) -> None:
    assert _external_test_schema is not None
    url = _external_test_schema.render_as_string(hide_password=False)
    first = Database.open_url(url, LOGGER)
    second = Database.open_url(url, LOGGER)
    try:
        first.hold_instance_lock()
        with pytest.raises(DatabaseInUseError, match="Another InvokeAI process"):
            second.hold_instance_lock()

        # The lock ends with the process that holds it, or, as here, when its database is closed.
        first.dispose()
        second.hold_instance_lock()
    finally:
        first.dispose()
        second.dispose()


@server_only
def test_a_server_without_room_for_large_projects_or_too_old_is_refused(
    empty_database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    with empty_database.begin(write=False) as conn:
        packet = int(conn.exec_driver_sql("SELECT @@max_allowed_packet").scalar_one())

    monkeypatch.setattr(startup, "MINIMUM_MAX_ALLOWED_PACKET", packet)
    monkeypatch.setitem(startup.MINIMUM_SERVER_VERSIONS, empty_database.dialect_name, (8, 0))
    startup.check_server(empty_database)

    monkeypatch.setattr(startup, "MINIMUM_MAX_ALLOWED_PACKET", packet + 1)
    with pytest.raises(DatabaseSetupError, match="max_allowed_packet"):
        startup.check_server(empty_database)

    monkeypatch.setattr(startup, "MINIMUM_MAX_ALLOWED_PACKET", packet)
    monkeypatch.setitem(startup.MINIMUM_SERVER_VERSIONS, empty_database.dialect_name, (99, 0))
    with pytest.raises(DatabaseSetupError, match="too old"):
        startup.check_server(empty_database)


@server_only
def test_a_new_server_database_beside_a_sqlite_one_is_pointed_out(
    empty_database: Database, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = _config(tmp_path, db_url="mariadb+pymysql://invokeai:secret@db/invokeai")
    config.db_path.parent.mkdir(parents=True)
    config.db_path.write_bytes(b"")

    with caplog.at_level(logging.WARNING):
        startup._warn_when_new_beside_sqlite(empty_database, config, LOGGER)

    [warning] = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert "invoke-db-copy" in warning
    assert "secret" not in warning
