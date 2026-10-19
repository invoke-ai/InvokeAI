"""Queue items gain an empty delivery; existing items keep everything else."""

from logging import Logger
from pathlib import Path
from unittest.mock import MagicMock

from sqlalchemy import delete, inspect

from invokeai.app.services.config.config_default import DefaultInvokeAIAppConfig
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.migrator import applied_migrations
from invokeai.app.services.shared.sqlite_migrator.migration_loader import MigrationBuildContext, build_migrations
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import MigrationBase
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import Migrator
from tests.fixtures.database import migrate_to_newest

MIGRATION_ID = "2026_10_13_add_session_queue_delivery"
DELIVERY_COLUMNS = {"delivery_claimant", "delivery_claimed_at", "delivered_at"}


def _run(db: Database, migrations: list[MigrationBase]) -> bool:
    migrator = Migrator(db)
    for migration in migrations:
        migrator.register_migration(migration)
    return migrator.run_migrations()


def _columns(db: Database) -> set[str]:
    with db.begin(write=False) as conn:
        return {column["name"] for column in inspect(conn).get_columns("session_queue")}


def test_existing_items_gain_an_empty_delivery(tmp_path: Path) -> None:
    logger = Logger("test")
    config = DefaultInvokeAIAppConfig(use_memory_db=True, node_cache_size=0)
    config._root = tmp_path
    migrations = build_migrations(MigrationBuildContext(app_config=config, logger=logger, image_files=MagicMock()))
    db = Database.open_sqlite(None, logger)
    assert _run(db, [migration for migration in migrations if migration.id != MIGRATION_ID])
    assert not DELIVERY_COLUMNS & _columns(db)
    with db.begin(write=True) as conn:
        conn.exec_driver_sql(
            "INSERT INTO session_queue (queue_id, batch_id, session_id, session, status, user_id) "
            "VALUES ('default', 'batch', 'session', '{}', 'completed', 'system');"
        )

    assert _run(db, migrations)

    assert DELIVERY_COLUMNS <= _columns(db)
    with db.begin(write=False) as conn:
        row = conn.exec_driver_sql(
            "SELECT session_id, status, delivery_claimant, delivery_claimed_at, delivered_at FROM session_queue;"
        ).all()
    assert row == [("session", "completed", None, None, None)]
    db.dispose()


def test_running_again_after_an_interrupted_run_changes_nothing(database: Database, tmp_path: Path) -> None:
    """On MySQL and MariaDB a migration that failed after its DDL runs again on columns it already added."""
    with database.begin(write=True) as conn:
        conn.execute(delete(applied_migrations).where(applied_migrations.c.migration_id == MIGRATION_ID))

    migrate_to_newest(database, tmp_path)

    assert DELIVERY_COLUMNS <= _columns(database)
