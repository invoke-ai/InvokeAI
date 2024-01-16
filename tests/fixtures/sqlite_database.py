from logging import Logger
from unittest import mock

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.shared.sqlite.sqlite_util import init_db


def create_mock_sqlite_database(config: InvokeAIAppConfig, logger: Logger) -> SqliteDatabase:
    image_files = mock.Mock(spec=ImageFileStorageBase)
    db = init_db(config=config, logger=logger, image_files=image_files)
    return db
