import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from logging import Logger
from typing import TYPE_CHECKING
from unittest import mock

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.startup import init_database

if TYPE_CHECKING:
    from invokeai.app.services.session_queue.session_queue_default import SessionQueue


def create_mock_sqlite_database(config: InvokeAIAppConfig, logger: Logger) -> Database:
    """A SQLite database at the newest schema, as the app opens it, with image files that are not there."""
    return init_database(config=config, logger=logger, image_files=mock.Mock(spec=ImageFileStorageBase))


@contextmanager
def sqlite_cursor(database: Database) -> Iterator[sqlite3.Cursor]:
    """A raw cursor in a write transaction of a SQLite database, for tests that set up or inspect rows with SQL of
    their own; its rows are `sqlite3.Row`s. The transaction commits when the block exits normally."""
    with database.begin(write=True) as conn:
        cursor = conn.connection.driver_connection.cursor()
        cursor.row_factory = sqlite3.Row
        try:
            yield cursor
        finally:
            cursor.close()


@contextmanager
def sqlite_cursor_of(queue: "SessionQueue") -> Iterator[sqlite3.Cursor]:
    """`sqlite_cursor` on the database a session queue uses."""
    with sqlite_cursor(queue._queries._database) as cursor:
        yield cursor
