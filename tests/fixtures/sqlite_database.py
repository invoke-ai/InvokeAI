from logging import Logger
from unittest import mock

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_1 import migration_1
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2 import migration_2
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import SQLiteMigrator


def create_sqlite_database(config: InvokeAIAppConfig, logger: Logger) -> SqliteDatabase:
    db_path = None if config.use_memory_db else config.db_path
    db = SqliteDatabase(db_path=db_path, logger=logger, verbose=config.log_sql)

    image_files = mock.Mock(spec=ImageFileStorageBase)

    migrator = SQLiteMigrator(db=db)
    migration_2.provide_dependency("logger", logger)
    migration_2.provide_dependency("image_files", image_files)
    migrator.register_migration(migration_1)
    migrator.register_migration(migration_2)
    migrator.run_migrations()
    return db
