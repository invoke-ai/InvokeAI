from logging import Logger

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.sqlite_migrator.migration_loader import MigrationBuildContext, build_migrations
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import Migrator


def init_database(config: InvokeAIAppConfig, logger: Logger, image_files: ImageFileStorageBase) -> Database:
    """Opens the app's database and brings it to the newest schema.

    :param image_files: The image files service, which some migrations need.
    """
    db_path = None if config.use_memory_db else config.db_path
    database = Database.open_sqlite(db_path, logger, verbose=config.log_sql, synchronous=config.db_synchronous)

    migrator = Migrator(database)
    migration_context = MigrationBuildContext(app_config=config, logger=logger, image_files=image_files)
    for migration in build_migrations(migration_context):
        migrator.register_migration(migration)
    migrator.run_migrations()

    return database
