"""Opening the database a config names: the SQLite file in `db_dir`, or the MySQL or MariaDB database of `db_url`."""

import re
from logging import Logger
from typing import Optional

from sqlalchemy import inspect, make_url

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.sqlite_migrator.migration_loader import MigrationBuildContext, build_migrations
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import Migrator, NoImageFiles

# The oldest servers the layer is tested on and relies on (window functions, `JSON_TABLE`, `FOR SHARE`, NO PAD
# binary collations).
MINIMUM_SERVER_VERSIONS = {"mysql": (8, 4), "mariadb": (10, 11)}
# A project document is up to 32 MiB, and a statement carries it with room to spare; MariaDB's default is 16 MiB.
MINIMUM_MAX_ALLOWED_PACKET = 64 * 1024 * 1024


class DatabaseSetupError(RuntimeError):
    """The configured database cannot be used as it is set up; the message says what to change."""


def open_database(config: InvokeAIAppConfig, logger: Logger) -> Database:
    """Opens the database the config names, as it is: no checks, no migration."""
    if config.db_url:
        if config.use_memory_db:
            raise DatabaseSetupError("`use_memory_db` and `db_url` exclude each other: set one of them")
        return Database.open_url(config.db_url, logger)
    db_path = None if config.use_memory_db else config.db_path
    return Database.open_sqlite(db_path, logger, verbose=config.log_sql, synchronous=config.db_synchronous)


def init_database(config: InvokeAIAppConfig, logger: Logger, image_files: ImageFileStorageBase) -> Database:
    """Opens the app's database and brings it to the newest schema.

    A server database is checked first, and this process takes its instance lock before it migrates or anything
    else starts: one process at a time serves a database.

    :param image_files: The image files service, which some migrations need.
    """
    database = open_database(config, logger)
    try:
        if database.dialect_name != "sqlite":
            check_server(database)
            database.hold_instance_lock()
            _warn_when_new_beside_sqlite(database, config, logger)
        _migrator(database, config, logger, image_files).run_migrations()
    except BaseException:
        database.dispose()
        raise
    return database


def open_migrated_database(config: InvokeAIAppConfig, logger: Logger) -> Database:
    """Opens the database for a tool that runs beside or instead of the app: it must be at the newest schema, which
    the app brings it to when it starts."""
    if not config.db_url and not config.use_memory_db and not config.db_path.exists():
        # Opening it would create an empty database that the app, configured elsewhere, would never use.
        raise DatabaseSetupError(f"There is no database at {config.db_path}: is this the InvokeAI root you meant?")
    database = open_database(config, logger)
    try:
        if _migrator(database, config, logger, NoImageFiles()).has_pending_migrations():
            raise DatabaseSetupError(
                "The database has not been updated to this version of InvokeAI yet: start InvokeAI once, then run "
                "this again"
            )
    except BaseException:
        database.dispose()
        raise
    return database


def open_copy_target(url: str, logger: Logger) -> tuple[Database, int]:
    """Opens the server database `invoke-db-copy` fills: checked as the app checks it, held as the app holds it,
    and empty. Also its `max_allowed_packet`."""
    database = Database.open_url(url, logger)
    try:
        max_allowed_packet = check_server(database)
        database.hold_instance_lock()
        with database.begin(write=False) as conn:
            if inspect(conn).get_table_names():
                raise DatabaseSetupError(
                    "The target database is not empty: copy into a new, empty database, so nothing in it is mixed "
                    "with or overwritten by the copy"
                )
    except BaseException:
        database.dispose()
        raise
    return database, max_allowed_packet


def redacted_database_url(url: Optional[str]) -> Optional[str]:
    """The URL with its password masked, for logs and the runtime config."""
    if not url:
        return url
    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:
        # Not a URL that could be opened either; show none of it.
        return "***"


def _migrator(
    database: Database, config: InvokeAIAppConfig, logger: Logger, image_files: ImageFileStorageBase
) -> Migrator:
    migrator = Migrator(database)
    for migration in build_migrations(MigrationBuildContext(app_config=config, logger=logger, image_files=image_files)):
        migrator.register_migration(migration)
    return migrator


def check_server(database: Database) -> int:
    """Refuses a server too old, or one that takes too small a statement; its `max_allowed_packet`."""
    with database.begin(write=False) as conn:
        version_text = str(conn.exec_driver_sql("SELECT VERSION()").scalar_one())
        max_allowed_packet = int(conn.exec_driver_sql("SELECT @@max_allowed_packet").scalar_one())
    minimum = MINIMUM_SERVER_VERSIONS[database.dialect_name]
    version = _version_of(version_text)
    if version < minimum:
        product = "MariaDB" if database.dialect_name == "mariadb" else "MySQL"
        raise DatabaseSetupError(
            f"{product} {version_text} is too old: InvokeAI needs {product} {minimum[0]}.{minimum[1]} or newer"
        )
    if max_allowed_packet < MINIMUM_MAX_ALLOWED_PACKET:
        raise DatabaseSetupError(
            f"The server's max_allowed_packet is {max_allowed_packet // (1024 * 1024)} MiB; InvokeAI needs at least "
            f"{MINIMUM_MAX_ALLOWED_PACKET // (1024 * 1024)} MiB to store large projects. Set it in the server's "
            "configuration, e.g. `max_allowed_packet=64M` under [mysqld], and restart the server."
        )
    return max_allowed_packet


def _version_of(version_text: str) -> tuple[int, int]:
    match = re.match(r"(\d+)\.(\d+)", version_text)
    if match is None:
        raise DatabaseSetupError(f"The database server reports a version InvokeAI cannot read: {version_text!r}")
    return int(match.group(1)), int(match.group(2))


def _warn_when_new_beside_sqlite(database: Database, config: InvokeAIAppConfig, logger: Logger) -> None:
    with database.begin(write=False) as conn:
        empty = not inspect(conn).get_table_names()
    if empty and config.db_path.exists():
        logger.warning(
            f"The database {redacted_database_url(config.db_url)} is new, and InvokeAI starts it empty, but there is "
            f"a SQLite database at {config.db_path}. To keep its boards, images, workflows and settings, stop "
            "InvokeAI and copy it over with `invoke-db-copy` before using the new database."
        )
