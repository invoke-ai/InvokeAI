from pathlib import Path

from sqlalchemy import delete, inspect, select, text

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.locks import db_locks
from invokeai.app.services.shared.database.schema.migrator import applied_migrations
from tests.fixtures.database import migrate_to_newest

MIGRATION = "2026_10_06_add_intermediates_state"


def test_a_run_that_stopped_halfway_completes_what_it_left(database: Database, tmp_path: Path) -> None:
    # On a server each DDL statement commits as it runs: a run that failed after its first steps leaves them done,
    # the rest undone and the migration unrecorded.
    with database.begin(write=True) as conn:
        conn.execute(delete(db_locks).where(db_locks.c.name == "media_protection"))
        conn.execute(delete(applied_migrations).where(applied_migrations.c.migration_id == MIGRATION))
        on_table = "" if conn.dialect.name == "sqlite" else " ON images"
        conn.execute(text(f"DROP INDEX idx_images_intermediates_owner{on_table}"))

    migrate_to_newest(database, tmp_path)

    with database.begin(write=False) as conn:
        assert conn.execute(select(db_locks.c.name).where(db_locks.c.name == "media_protection")).first() is not None
        indexes = {index["name"]: index["column_names"] for index in inspect(conn).get_indexes("images")}
        assert "idx_images_intermediates_owner" in indexes
        recorded = select(applied_migrations.c.migration_id).where(applied_migrations.c.migration_id == MIGRATION)
        assert conn.execute(recorded).first() is not None
