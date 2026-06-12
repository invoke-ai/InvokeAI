"""Copy all data from one InvokeAI database to another, backend to backend.

Used to migrate an existing SQLite database to an external backend (MySQL/MariaDB/Postgres):
the source schema is built by the raw-SQL migrations, the target schema by
``SQLModel.metadata.create_all()`` — both produce the same logical schema (the schema-parity
guard enforces this), so rows can be copied table-by-table through SQLAlchemy.

DB-generated columns (the ``Computed`` columns on ``models`` / ``workflow_library``) are
skipped: the target derives them from the ``config`` / ``workflow`` JSON on insert, exactly
as the source did. Tables are copied in foreign-key order so parents exist before children.

The target is expected to be freshly created (empty). Existing rows would collide on primary
keys; this is a one-shot migration, not a sync.
"""

from logging import Logger

from sqlalchemy import insert, inspect, select

from invokeai.app.services.shared.sqlite.models import SQLModel
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase

_BATCH_SIZE = 1000


def copy_database(source: SqliteDatabase, target: SqliteDatabase, logger: Logger) -> dict[str, int]:
    """Copy every table from ``source`` into ``target``. Returns {table_name: rows_copied}."""
    counts: dict[str, int] = {}
    src_engine = source._engine
    tgt_engine = target._engine
    src_tables = set(inspect(src_engine).get_table_names())

    # sorted_tables is in foreign-key dependency order (parents before children).
    with src_engine.connect() as src_conn, tgt_engine.begin() as tgt_conn:
        for table in SQLModel.metadata.sorted_tables:
            if table.name not in src_tables:
                # Declared in models.py but not present in this source DB — nothing to copy.
                logger.warning(f"Skipping {table.name}: not present in source database")
                continue
            # Skip DB-generated columns — the target recomputes them on insert.
            columns = [c for c in table.columns if c.computed is None]
            result = src_conn.execution_options(stream_results=True).execute(select(*columns))

            copied = 0
            while True:
                batch = result.fetchmany(_BATCH_SIZE)
                if not batch:
                    break
                tgt_conn.execute(insert(table), [dict(row._mapping) for row in batch])
                copied += len(batch)

            counts[table.name] = copied
            if copied:
                logger.info(f"Copied {copied} rows -> {table.name}")

    total = sum(counts.values())
    logger.info(f"Database copy complete: {total} rows across {len(counts)} tables")
    return counts
