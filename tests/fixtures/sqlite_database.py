import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from logging import Logger
from typing import TYPE_CHECKING
from unittest import mock

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.shared.sqlite.sqlite_util import init_db

if TYPE_CHECKING:
    from invokeai.app.services.session_queue.session_queue_default import SessionQueue


def create_mock_sqlite_database(config: InvokeAIAppConfig, logger: Logger) -> SqliteDatabase:
    image_files = mock.Mock(spec=ImageFileStorageBase)
    db = init_db(config=config, logger=logger, image_files=image_files)
    return db


@contextmanager
def legacy_cursor_of(queue: "SessionQueue") -> Iterator[sqlite3.Cursor]:
    """Transitional, with the cursor facade: a cursor in a transaction on the SQLite database a session queue uses, for
    tests that write rows with SQL of their own."""
    with queue._queries._database.legacy_cursor() as cursor:
        yield cursor
