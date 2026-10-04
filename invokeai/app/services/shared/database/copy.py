"""Copying the rows of the application's tables from one database to another, whatever their backends."""

from collections.abc import Sequence
from typing import Optional

from sqlalchemy import Connection, Table, func, insert, inspect, select

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema import metadata

_BATCH_SIZE = 1000


def copy_rows(source: Database, target: Database, *, tables: Optional[Sequence[Table]] = None) -> dict[str, int]:
    """Copies every row of the application tables (by default all of them) from `source` into `target`.

    Tables are copied in foreign key order, in batches of a transaction each, and a table the source lacks is
    skipped. The target computes the generated columns itself. The ids an integer key generates next continue
    after the highest one copied, or, from a SQLite source, which records it, after the highest one the source
    ever issued: an AUTOINCREMENT table issues no id twice, across the copy too.

    :return: The number of rows copied, per table.
    """
    wanted = set(tables if tables is not None else metadata.sorted_tables)
    copied: dict[str, int] = {}
    with source.begin(write=False) as src:
        present = set(inspect(src).get_table_names())
        issued = _sqlite_sequences(src) if source.dialect_name == "sqlite" else {}
        for table in metadata.sorted_tables:
            if table not in wanted or table.name not in present:
                continue
            columns = [column for column in table.columns if column.computed is None]
            rows = src.execute(
                select(*columns).order_by(*table.primary_key.columns).execution_options(yield_per=_BATCH_SIZE)
            )
            copied[table.name] = 0
            for batch in rows.partitions():
                with target.begin(write=True) as dst:
                    dst.execute(insert(table), [row._asdict() for row in batch])
                copied[table.name] += len(batch)
            if table.autoincrement_column is not None:
                _continue_ids(target, table, issued.get(table.name, 0))
    return copied


def _sqlite_sequences(conn: Connection) -> dict[str, int]:
    """The highest id each AUTOINCREMENT table has issued, deleted rows included."""
    if conn.exec_driver_sql("SELECT 1 FROM sqlite_master WHERE name = 'sqlite_sequence'").first() is None:
        return {}
    return {row[0]: row[1] for row in conn.exec_driver_sql("SELECT name, seq FROM sqlite_sequence").all()}


def _continue_ids(target: Database, table: Table, issued: int) -> None:
    column = table.autoincrement_column
    assert column is not None
    with target.begin(write=True) as conn:
        highest = max(conn.execute(select(func.max(column))).scalar() or 0, issued)
        if target.dialect_name != "sqlite":
            name = conn.dialect.identifier_preparer.format_table(table)
            conn.exec_driver_sql(f"ALTER TABLE {name} AUTO_INCREMENT = {highest + 1}")
        elif table.dialect_options["sqlite"]["autoincrement"]:
            # Only an AUTOINCREMENT table never reuses an id; SQLite keeps its high-water mark here.
            parameters = (highest, table.name)
            if conn.exec_driver_sql("UPDATE sqlite_sequence SET seq = ? WHERE name = ?", parameters).rowcount == 0:
                conn.exec_driver_sql("INSERT INTO sqlite_sequence (seq, name) VALUES (?, ?)", parameters)
