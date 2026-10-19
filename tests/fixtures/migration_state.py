"""A database as it stood just before one migration, on the backend under test.

A migration's test takes the `empty_database` fixture there with `migrate_to_before`, writes rows of that time, and
then runs the migration. On SQLite the earlier migrations build the state. A server database is never migrated
from nothing (`Migrator` creates the newest schema in an empty one), so there the frozen schema of the
portable-migration cutover (`server_schema_at_cutover/`) is loaded first, with the SQLite-only migrations recorded
as applied, and the portable migrations up to the one under test run on it.
"""

from pathlib import Path
from unittest import mock

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import Connection, insert

from invokeai.app.services.config.config_default import DefaultInvokeAIAppConfig
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.migrator import applied_migrations
from invokeai.app.services.shared.sqlite_migrator.migration_loader import MigrationBuildContext, build_migrations
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration, MigrationBase
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import Migrator
from invokeai.backend.util.logging import InvokeAILogger

CUTOVER_SCHEMAS = (
    Path(__file__).parent.parent / "app" / "services" / "shared" / "sqlite_migrator" / "server_schema_at_cutover"
)


def build_application_migrations(root: Path) -> list[MigrationBase]:
    """The application's migrations. `root` must be the test's own directory: the migrations clean up legacy files
    under it, and the config reads nothing from the environment that could point them elsewhere."""
    config = DefaultInvokeAIAppConfig()
    config._root = root
    context = MigrationBuildContext(
        app_config=config,
        logger=InvokeAILogger.get_logger("test_database"),
        image_files=mock.Mock(spec=ImageFileStorageBase),
    )
    return build_migrations(context)


def run_migrations(database: Database, migrations: list[MigrationBase]) -> bool:
    """Runs those of the migrations the database has not had; whether any ran."""
    migrator = Migrator(database)
    for migration in migrations:
        migrator.register_migration(migration)
    return migrator.run_migrations()


def migrations_before(migrations: list[MigrationBase], migration_id: str) -> list[MigrationBase]:
    """Every migration except `migration_id` and those that depend on it, directly or not."""
    by_id = {migration.id: migration for migration in migrations}

    def depends_on(migration: MigrationBase) -> bool:
        current: str | None = migration.id
        while current is not None:
            if current == migration_id:
                return True
            current = by_id[current].depends_on
        return False

    return [migration for migration in migrations if not depends_on(migration)]


def migrate_to_before(database: Database, migrations: list[MigrationBase], migration_id: str) -> None:
    """Brings the empty database to the state just before `migration_id`."""
    if database.dialect_name != "sqlite":
        create_cutover_schema(database, migrations)
    assert run_migrations(database, migrations_before(migrations, migration_id))


def alembic_operations(conn: Connection) -> Operations:
    """Alembic's operations on the connection, for a test that leaves a migration's DDL half done."""
    return Operations(MigrationContext.configure(conn))


def create_cutover_schema(database: Database, migrations: list[MigrationBase]) -> None:
    """Creates, in the empty server database, the frozen schema of the cutover, with the SQLite-only migrations
    among `migrations` recorded as applied: the migrator then runs the portable ones it is given."""
    text = (CUTOVER_SCHEMAS / f"{database.dialect_name}.sql").read_text(encoding="utf-8")
    body = "\n".join(line for line in text.splitlines() if not line.startswith("--"))
    statements = [statement.strip() for statement in body.split(";\n") if statement.strip()]
    with database.begin(write=True) as conn:
        for statement in statements:
            conn.exec_driver_sql(statement)
        conn.execute(
            insert(applied_migrations),
            [
                {"migration_id": migration.id, "legacy_version": migration.to_version}
                for migration in migrations
                if isinstance(migration, Migration)
            ],
        )
