from pathlib import Path

from sqlalchemy import delete, select

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.locks import db_locks
from invokeai.app.services.shared.database.schema.migrator import applied_migrations
from tests.fixtures.database import migrate_to_newest


def test_a_run_that_stopped_after_creating_the_table_adds_the_lock_rows(database: Database, tmp_path: Path) -> None:
    # On a server each DDL statement commits as it runs: a run that failed after creating the table leaves it
    # there, empty, and the migration unrecorded.
    with database.begin(write=True) as conn:
        conn.execute(delete(db_locks))
        conn.execute(delete(applied_migrations).where(applied_migrations.c.migration_id == "2026_10_04_add_db_locks"))

    migrate_to_newest(database, tmp_path)

    with database.begin(write=False) as conn:
        assert list(conn.execute(select(db_locks.c.name)).scalars()) == ["admin_accounts"]
        recorded = select(applied_migrations.c.migration_id).where(
            applied_migrations.c.migration_id == "2026_10_04_add_db_locks"
        )
        assert conn.execute(recorded).first() is not None
