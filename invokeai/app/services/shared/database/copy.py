"""Copying the rows of the application's tables from one database to another, whatever their backends: the
bootstrap of a server database, and `invoke-db-copy`, which moves an install's SQLite database to a server."""

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from typing import Any, NamedTuple, Optional

from sqlalchemy import Connection, Insert, LargeBinary, String, Table, cast, func, insert, inspect, select

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.dialect import insert_ignore
from invokeai.app.services.shared.database.schema import metadata
from invokeai.app.services.shared.database.schema.image_index import image_index_vocab_terms
from invokeai.app.services.shared.database.schema.migrator import applied_migrations, migrations
from invokeai.app.services.shared.database.schema.models import models
from invokeai.app.services.shared.database.types import Blob, LongText

_BATCH_SIZE = 1000
# A batch is written as one INSERT of many rows; this keeps that statement well within a server's
# max_allowed_packet, which startup requires to be 64 MiB at least.
_BATCH_BYTES = 16 * 1024 * 1024

# A row as the copy reads it: the values of the table's stored (not generated) columns, by name.
Row = dict[str, Any]
Transform = Callable[[Table, Row], Row]

# The migrator's records, written last of all: a database without them is refused as incomplete.
_RECORDS = (migrations, applied_migrations)
# Terms a server's case-insensitive collation equates, which SQLite's ASCII-only NOCASE keeps apart: the server
# keeps one of them, which is all the vocabulary needs.
_MERGED_ON_A_SERVER = (image_index_vocab_terms,)


def copy_rows(
    source: Database,
    target: Database,
    *,
    tables: Optional[Sequence[Table]] = None,
    transform: Optional[Transform] = None,
    keep_first: Sequence[Table] = (),
) -> dict[str, int]:
    """Copies every row of the application tables (by default all of them) from `source` into `target`.

    Tables are copied in foreign key order, in batches of a transaction each, and a table the source lacks is
    skipped. The target computes the generated columns itself. The ids an integer key generates next continue
    after the highest one copied, or, from a SQLite source, which records it, after the highest one the source
    ever issued: an AUTOINCREMENT table issues no id twice, across the copy too.

    :param transform: Applied to each row before it is written.
    :param keep_first: Tables where a row the target holds equal to another one is skipped, not refused.
    :return: The number of rows read, per table.
    """
    wanted = set(tables if tables is not None else metadata.sorted_tables)
    copied: dict[str, int] = {}
    with source.begin(write=False) as src:
        present = set(inspect(src).get_table_names())
        issued = _sqlite_sequences(src) if source.dialect_name == "sqlite" else {}
        for table in metadata.sorted_tables:
            if table not in wanted or table.name not in present:
                continue
            statement: Insert = insert_ignore(target.dialect_name, table) if table in keep_first else insert(table)
            copied[table.name] = 0
            for batch in _batches(src, table):
                rows = [transform(table, row) for row in batch] if transform is not None else batch
                for part in _by_size(rows):
                    with target.begin(write=True) as dst:
                        dst.execute(statement, part)
                copied[table.name] += len(batch)
            if table.autoincrement_column is not None:
                _continue_ids(target, table, issued.get(table.name, 0))
    return copied


class Problem(NamedTuple):
    """Rows the target would refuse, or that the copy changes."""

    table: str
    count: int
    what: str


def find_orphans(database: Database) -> list[Problem]:
    """The rows of a SQLite database whose foreign keys name no row: a server refuses to store them."""
    with database.begin(write=False) as conn:
        found = conn.exec_driver_sql("PRAGMA foreign_key_check").all()
    counts: dict[tuple[str, str], int] = {}
    for table, _rowid, parent, _key in found:
        counts[(table, parent)] = counts.get((table, parent), 0) + 1
    return [
        Problem(table, count, f"name a missing row of {parent}") for (table, parent), count in sorted(counts.items())
    ]


def delete_orphans(database: Database) -> int:
    """Deletes the rows of a SQLite database whose foreign keys name no row, and what the database's own rules
    delete with them (ON DELETE CASCADE); until none is left. For a snapshot only: never the user's database.

    :return: The number of rows deleted directly.
    """
    deleted = 0
    while True:
        with database.begin(write=True) as conn:
            found = conn.exec_driver_sql("PRAGMA foreign_key_check").all()
            if not found:
                return deleted
            for table, rowid, _parent, _key in {(row[0], row[1], row[2], row[3]) for row in found}:
                name = conn.dialect.identifier_preparer.quote(table)
                deleted += conn.exec_driver_sql(f"DELETE FROM {name} WHERE rowid = ?", (rowid,)).rowcount


def find_oversized(database: Database, max_allowed_packet: int) -> list[Problem]:
    """The values of a SQLite database a server cannot store: text longer than its column holds there, and values
    larger than a statement the server takes (`max_allowed_packet`)."""
    problems: list[Problem] = []
    with database.begin(write=False) as conn:
        present = set(inspect(conn).get_table_names())
        for table in metadata.sorted_tables:
            if table.name not in present:
                continue
            for column in _stored_columns(table):
                length = _length_of(column.type)
                if length is not None:
                    # A server's VARCHAR(n) counts characters, as SQLite's length() of text does.
                    too_long, what = (
                        func.length(column) > length,
                        f"hold a {column.name} longer than {length} characters",
                    )
                elif isinstance(column.type, (LongText, Blob)):
                    too_long = func.length(cast(column, LargeBinary)) > max_allowed_packet // 2
                    what = f"hold a {column.name} larger than half the server's max_allowed_packet"
                else:
                    continue
                count = conn.execute(select(func.count()).select_from(table).where(too_long)).scalar_one()
                if count:
                    problems.append(Problem(table.name, int(count), what))
    return problems


def normalize_for_a_server(table: Table, row: Row) -> Row:
    """A row as a server can store it: a model config's `file_size` a whole number of bytes. (SQLite stores what a
    JSON number holds; a server computes the integer `file_size` column from it and refuses a fraction.)"""
    if table is models:
        config = json.loads(row["config"])
        size = config.get("file_size")
        if isinstance(size, float):
            config["file_size"] = round(size)
            return {**row, "config": json.dumps(config)}
    return row


def count_normalized(database: Database) -> list[Problem]:
    """The rows `normalize_for_a_server` changes."""
    with database.begin(write=False) as conn:
        fractional = conn.execute(
            select(func.count()).where(func.json_type(models.c.config, "$.file_size") == "real")
        ).scalar_one()
    return (
        [Problem(models.name, int(fractional), "store file_size as a decimal number, written as a whole one")]
        if (fractional)
        else []
    )


def copy_database(source: Database, target: Database) -> dict[str, int]:
    """Creates the schema in the empty `target` and copies every row of `source` into it, the migrator's records
    last, so that an interrupted copy leaves a database the app refuses rather than one that looks complete."""
    with target.begin(write=True) as conn:
        metadata.create_all(conn)
    data = [table for table in metadata.sorted_tables if table not in _RECORDS]
    copied = copy_rows(source, target, tables=data, transform=normalize_for_a_server, keep_first=_MERGED_ON_A_SERVER)
    for table in _RECORDS:
        copied.update(copy_rows(source, target, tables=[table]))
    return copied


class Checksum(NamedTuple):
    rows: int
    sha256: str


def checksums(database: Database, transform: Optional[Transform] = None) -> dict[str, Checksum]:
    """The number and a checksum of the rows of each table, over their stored columns, independent of their order
    (which collations make differ between backends)."""
    result: dict[str, Checksum] = {}
    with database.begin(write=False) as conn:
        present = set(inspect(conn).get_table_names())
        for table in metadata.sorted_tables:
            if table.name not in present:
                continue
            digests: list[bytes] = []
            for batch in _batches(conn, table):
                for row in batch:
                    stored = transform(table, row) if transform is not None else row
                    digests.append(hashlib.sha256(_canonical(stored)).digest())
            digests.sort()
            combined = hashlib.sha256(b"".join(digests)).hexdigest()
            result[table.name] = Checksum(len(digests), combined)
    return result


def merged_tables() -> tuple[str, ...]:
    """The tables whose rows a server may merge (see `_MERGED_ON_A_SERVER`), so their counts may shrink."""
    return tuple(table.name for table in _MERGED_ON_A_SERVER)


def _batches(conn: Connection, table: Table) -> Iterable[list[Row]]:
    columns = _stored_columns(table)
    rows = conn.execute(select(*columns).order_by(*table.primary_key.columns).execution_options(yield_per=_BATCH_SIZE))
    for batch in rows.partitions():
        yield [row._asdict() for row in batch]


def _by_size(rows: list[Row]) -> Iterable[list[Row]]:
    part: list[Row] = []
    size = 0
    for row in rows:
        row_size = sum(len(value) for value in row.values() if isinstance(value, (str, bytes)))
        if part and size + row_size > _BATCH_BYTES:
            yield part
            part, size = [], 0
        part.append(row)
        size += row_size
    if part:
        yield part


def _stored_columns(table: Table) -> list[Any]:
    return [column for column in table.columns if column.computed is None]


def _length_of(column_type: Any) -> Optional[int]:
    impl = getattr(column_type, "impl", column_type)
    if isinstance(impl, String) and not isinstance(column_type, LongText):
        return impl.length
    return None


def _canonical(row: Row) -> bytes:
    def value(item: Any) -> Any:
        if isinstance(item, (bytes, bytearray, memoryview)):
            return {"bytes": bytes(item).hex()}
        if isinstance(item, bool):
            return int(item)
        if isinstance(item, float) and item.is_integer():
            return int(item)
        return item

    return json.dumps({name: value(item) for name, item in sorted(row.items())}, default=str).encode()


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
