"""Copying the rows of the application's tables from one database to another, whatever their backends: the
bootstrap of a server database, and `invoke-db-copy`, which moves an install's SQLite database to a server."""

import hashlib
import json
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, NamedTuple, Optional

from sqlalchemy import Connection, Insert, LargeBinary, Table, cast, func, insert, inspect, select

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.dialect import insert_ignore
from invokeai.app.services.shared.database.schema import metadata
from invokeai.app.services.shared.database.schema.image_index import image_index_vocab_terms
from invokeai.app.services.shared.database.schema.migrator import applied_migrations, migrations
from invokeai.app.services.shared.database.schema.models import models
from invokeai.app.services.shared.database.types import TIMESTAMP_LENGTH, Key, NoCaseKey, Timestamp

# Rows are read a few at a time, and written in parts of one INSERT each; a part's bytes keep that statement well
# within a server's max_allowed_packet, which startup requires to be 64 MiB at least.
_READ_ROWS = 50
_BATCH_ROWS = 1000
_BATCH_BYTES = 16 * 1024 * 1024

# A row as the copy reads it: the values of the table's stored (not generated) columns, by name.
Row = dict[str, Any]
Transform = Callable[[Table, Row], Row]

# The migrator's records, written last of all: a database without them is refused as incomplete.
_RECORDS = [migrations, applied_migrations]
_DATA = [table for table in metadata.sorted_tables if table not in _RECORDS]
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
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, int]:
    """Copies every row of the application tables (by default all of them) from `source` into `target`.

    Tables are copied in foreign key order, in batches of a transaction each, and a table the source lacks is
    skipped. The target computes the generated columns itself. The ids an integer key generates next continue
    after the highest one copied, or, from a SQLite source, which records it, after the highest one the source
    ever issued: an AUTOINCREMENT table issues no id twice, across the copy too.

    :param transform: Applied to each row before it is written.
    :param keep_first: Tables where a row the target holds equal to another one is skipped, not refused.
    :param progress: Called with each table's name before it is copied.
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
            if progress is not None:
                progress(table.name)
            statement: Insert = insert_ignore(target.dialect_name, table) if table in keep_first else insert(table)
            rows = _rows(src, table)
            if transform is not None:
                rows = (transform(table, row) for row in rows)
            copied[table.name] = 0
            for part in _parts(rows):
                with target.begin(write=True) as dst:
                    dst.execute(statement, part)
                copied[table.name] += len(part)
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
        Problem(table, count, f"with a reference to a missing row of {parent}")
        for (table, parent), count in sorted(counts.items())
    ]


def delete_orphans(database: Database) -> list[Problem]:
    """Removes what a server would refuse from a SQLite database whose foreign keys name missing rows, until none is
    left. A reference the schema clears when its row is deleted (ON DELETE SET NULL) is cleared; any other orphan row
    is deleted, with what the database's own rules delete with it (ON DELETE CASCADE). For a snapshot only: never
    the user's database.

    :return: What changed, per table.
    """
    before = _row_counts(database)
    cleared: dict[str, int] = {}
    while True:
        with database.begin(write=True) as conn:
            found = {(row[0], row[1], row[3]) for row in conn.exec_driver_sql("PRAGMA foreign_key_check").all()}
            if not found:
                break
            changed = 0
            for table, rowid, key in sorted(found):
                name = conn.dialect.identifier_preparer.quote(table)
                columns = _clearable_reference(conn, table, key)
                if columns is not None:
                    quote = conn.dialect.identifier_preparer.quote
                    assignments = ", ".join(f"{quote(column)} = NULL" for column in columns)
                    count = conn.exec_driver_sql(f"UPDATE {name} SET {assignments} WHERE rowid = ?", (rowid,)).rowcount
                    cleared[table] = cleared.get(table, 0) + count
                else:
                    count = conn.exec_driver_sql(f"DELETE FROM {name} WHERE rowid = ?", (rowid,)).rowcount
                changed += count
            if not changed:
                raise RuntimeError(f"Could not remove the rows whose foreign keys name missing rows: {sorted(found)}")
    after = _row_counts(database)
    changes = [
        Problem(table, count, "with a reference to a missing row, which was cleared")
        for table, count in cleared.items()
    ]
    changes += [
        Problem(table, count - after.get(table, 0), "left out")
        for table, count in before.items()
        if count > after.get(table, 0)
    ]
    return sorted(changes)


def find_oversized(database: Database, max_allowed_packet: int) -> list[Problem]:
    """The rows of a SQLite database a server cannot store: text longer than its column holds there (computed
    columns included, which the server computes from the copy), and rows larger than half a statement the server
    takes (`max_allowed_packet`), which leaves room for the driver's escaping."""
    problems: list[Problem] = []
    with database.begin(write=False) as conn:
        present = set(inspect(conn).get_table_names())
        for table in metadata.sorted_tables:
            if table.name not in present:
                continue
            for column in table.columns:
                length = _length_of(column.type)
                if length is None:
                    continue
                # A server's VARCHAR(n) counts characters, as SQLite's length() of text does.
                too_long = select(func.count()).select_from(table).where(func.length(column) > length)
                count = conn.execute(too_long).scalar_one()
                if count:
                    article = "an" if column.name[0] in "aeiou" else "a"
                    what = f"with {article} {column.name} longer than {length} characters"
                    problems.append(Problem(table.name, int(count), what))
            sizes = [func.coalesce(func.length(cast(column, LargeBinary)), 0) for column in _stored_columns(table)]
            too_large = (
                select(func.count()).select_from(table).where(sum(sizes[1:], sizes[0]) > max_allowed_packet // 2)
            )
            count = conn.execute(too_large).scalar_one()
            if count:
                problems.append(Problem(table.name, int(count), "larger than half the server's max_allowed_packet"))
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
        [Problem(models.name, int(fractional), "with a file_size of a fraction of a byte, rounded")]
        if (fractional)
        else []
    )


def copy_database(
    source: Database, target: Database, progress: Optional[Callable[[str], None]] = None
) -> dict[str, int]:
    """Creates the schema in the empty `target` and copies the rows of `source` into it, all but the migrator's
    records: without those, the app refuses the target as incomplete. `copy_records` adds them once `verify_copy`
    found the copy complete.

    :param progress: Called with each table's name before it is copied.
    """
    with target.begin(write=True) as conn:
        metadata.create_all(conn)
    return copy_rows(
        source,
        target,
        tables=_DATA,
        transform=normalize_for_a_server,
        keep_first=_MERGED_ON_A_SERVER,
        progress=progress,
    )


def copy_records(source: Database, target: Database) -> dict[str, int]:
    """Copies the migrator's records, which make the target a database the app takes."""
    return copy_rows(source, target, tables=_RECORDS)


def verify_copy(source: Database, target: Database, *, records: bool = False) -> list[str]:
    """Compares each table `copy_database` (or with `records`, `copy_records`) wrote with the source's, row by row;
    what differs, described. A table whose rows a server merges (see `_MERGED_ON_A_SERVER`) matches when the target
    holds only rows of the source, and one of each set of rows the server equates."""
    tables = _RECORDS if records else _DATA
    expected = _row_digests(source, tables, normalize_for_a_server)
    actual = _row_digests(target, tables)
    mismatches: list[str] = []
    for table in tables:
        want, got = expected.get(table.name, []), actual.get(table.name, [])
        if table in _MERGED_ON_A_SERVER:
            if not set(got) <= set(want) or _folded_terms(target) != _folded_terms(source):
                mismatches.append(f"{table.name}: rows or contents differ")
        elif sorted(want) != sorted(got):
            mismatches.append(f"{table.name}: rows or contents differ ({len(want)} copied, {len(got)} held)")
    return mismatches


def merged_rows(source: Database, target: Database) -> dict[str, int]:
    """How many rows fewer than the source the target holds, per table whose rows a server merges."""
    result: dict[str, int] = {}
    for table in _MERGED_ON_A_SERVER:
        fewer = _count(source, table) - _count(target, table)
        if fewer:
            result[table.name] = fewer
    return result


def _count(database: Database, table: Table) -> int:
    with database.begin(write=False) as conn:
        return int(conn.execute(select(func.count()).select_from(table)).scalar_one())


def _folded_terms(database: Database) -> set[str]:
    # The case and compatibility forms that the servers' case-insensitive collations equate.
    with database.begin(write=False) as conn:
        terms = conn.execute(select(image_index_vocab_terms.c.term)).scalars().all()
    # Ignorable code points (a soft hyphen, a zero-width space) are format characters, which they ignore.
    return {
        "".join(c for c in unicodedata.normalize("NFKC", term).casefold() if unicodedata.category(c) != "Cf")
        for term in terms
    }


def _row_digests(
    database: Database, tables: Sequence[Table], transform: Optional[Transform] = None
) -> dict[str, list[bytes]]:
    result: dict[str, list[bytes]] = {}
    with database.begin(write=False) as conn:
        present = set(inspect(conn).get_table_names())
        for table in tables:
            if table.name not in present:
                continue
            digests: list[bytes] = []
            for row in _rows(conn, table):
                stored = transform(table, row) if transform is not None else row
                digests.append(hashlib.sha256(_canonical(stored)).digest())
            result[table.name] = digests
    return result


def _row_counts(database: Database) -> dict[str, int]:
    with database.begin(write=False) as conn:
        present = set(inspect(conn).get_table_names())
        return {
            table.name: int(conn.execute(select(func.count()).select_from(table)).scalar_one())
            for table in metadata.sorted_tables
            if table.name in present
        }


def _clearable_reference(conn: Connection, table: str, key: int) -> Optional[list[str]]:
    """The columns of a SQLite foreign key that the schema sets to NULL when its row is deleted, if it does."""
    name = conn.dialect.identifier_preparer.quote(table)
    parts = [row for row in conn.exec_driver_sql(f"PRAGMA foreign_key_list({name})").all() if row[0] == key]
    if not parts or parts[0][6] != "SET NULL":
        return None
    return [row[3] for row in parts]


def _rows(conn: Connection, table: Table) -> Iterator[Row]:
    """The table's rows, a few at a time in memory: a project document alone may be 32 MiB."""
    columns = _stored_columns(table)
    statement = select(*columns).order_by(*table.primary_key.columns).execution_options(yield_per=_READ_ROWS)
    for row in conn.execute(statement):
        yield row._asdict()


def _parts(rows: Iterable[Row]) -> Iterator[list[Row]]:
    """The rows in parts of at most `_BATCH_ROWS` rows and `_BATCH_BYTES`, unless one row alone is larger."""
    part: list[Row] = []
    size = 0
    for row in rows:
        row_size = _size_of(row)
        if part and (size + row_size > _BATCH_BYTES or len(part) == _BATCH_ROWS):
            yield part
            part, size = [], 0
        part.append(row)
        size += row_size
    if part:
        yield part


def _size_of(row: Row) -> int:
    """About the bytes a row takes in a statement: its text encoded, and its binary values escaped at worst."""
    size = 0
    for value in row.values():
        if isinstance(value, str):
            size += len(value.encode())
        elif isinstance(value, (bytes, bytearray, memoryview)):
            size += 2 * len(value)
    return size


def _stored_columns(table: Table) -> list[Any]:
    return [column for column in table.columns if column.computed is None]


def _length_of(column_type: Any) -> Optional[int]:
    """The characters a server's column of this type holds, when it holds fewer than a long text."""
    if isinstance(column_type, (Key, NoCaseKey)):
        return column_type.length
    if isinstance(column_type, Timestamp):
        return TIMESTAMP_LENGTH
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
