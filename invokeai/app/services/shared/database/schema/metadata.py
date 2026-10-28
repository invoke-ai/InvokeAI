"""The metadata every table belongs to, and the rules its DDL follows on each backend.

Which type a text column gets:

- `Key` where a server must index the whole value: a primary key, a unique constraint, a foreign key, and
  indexes on identifiers and enumerated values. A server bounds it; an index there holds 768 characters in all.
- `LongText` everywhere else, unbounded. An index on free text (a name, a description) covers a prefix of it on a
  server (`LONG_TEXT_INDEX_PREFIX`), so such text has no length limit there either.

Values longer than their `Key` are refused on a server, where SQLite stores them; the lengths leave room above
what the application itself generates.

A primary key column that a migration declared without NOT NULL is `nullable=True`, so that SQLite declares it
alike; a server makes every primary key column NOT NULL regardless.

An index that differs between backends is a pair of `Index` objects with the same name, each limited to its
backends with `ddl_if`. Only `create_all()` and `Index.create()` honor that (`Table.to_metadata()` drops it), so
tables are created from this metadata, never from copies of its tables.
"""

from typing import Any

from sqlalchemy import CheckConstraint, Column, MetaData, Table, TextClause, text
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.schema import CreateColumn, SchemaItem
from sqlalchemy.sql.compiler import DDLCompiler

from invokeai.app.services.shared.database.engines import (
    MARIADB_BINARY_COLLATION,
    MYSQL_BINARY_COLLATION,
    SERVER_DIALECTS,
)
from invokeai.app.services.shared.database.types import Timestamp, now_text

metadata = MetaData(
    naming_convention={
        "pk": "pk_%(table_name)s",
        "uq": "uq_%(table_name)s_%(column_0_N_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    }
)

# Every table on a server is InnoDB and compares text byte for byte, whatever the server's defaults. Keys of up to
# 3072 bytes need the DYNAMIC row format, the default a server may have been configured away from.
_SERVER_TABLE_OPTIONS: dict[str, Any] = {
    "mysql_engine": "InnoDB",
    "mysql_charset": "utf8mb4",
    "mysql_collate": MYSQL_BINARY_COLLATION,
    "mysql_row_format": "DYNAMIC",
    "mariadb_engine": "InnoDB",
    "mariadb_charset": "utf8mb4",
    "mariadb_collate": MARIADB_BINARY_COLLATION,
    "mariadb_row_format": "DYNAMIC",
}

# `ddl_if(dialect=ON_SERVERS)`: only on the server backends. (`ddl_if` takes a tuple of dialect names, as
# documented, though its annotation names one.)
ON_SERVERS: Any = SERVER_DIALECTS

# Lengths of `Key` text other than the default 255. User ids are generated (a uuid, or 'system'); kept short so
# that keys with several text columns fit a server's index.
USER_ID_LENGTH = 64
ENUM_LENGTH = 32  # one of a fixed set of values: a status, a kind, a scope
PATH_LENGTH = 768  # a file path under a unique index, which a server can hold only whole

# An index on a LONGTEXT column covers this many leading characters on a server.
LONG_TEXT_PREFIX_LENGTH = 255
LONG_TEXT_INDEX_PREFIX: dict[str, Any] = {
    "mysql_length": LONG_TEXT_PREFIX_LENGTH,
    "mariadb_length": LONG_TEXT_PREFIX_LENGTH,
}

# The column defaults of the SQLite schema's timestamps. Servers have none: the application sets timestamps.
STRFTIME_NOW = "STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')"
CURRENT_TIMESTAMP = "CURRENT_TIMESTAMP"

# `Column.info` keys: a default rendered on SQLite only, and a generated column whose NOT NULL MariaDB gets as a
# CHECK constraint instead.
_SQLITE_DEFAULT = "invokeai_sqlite_default"
_MARIADB_NOT_NULL_CHECK = "invokeai_mariadb_not_null_check"


def default(value: str | int | float | bool) -> TextClause:
    """A constant column default.

    A string is parenthesized, as an expression default, because a server accepts a literal default for
    LONGTEXT only in that form. SQLite reports the default as the bare literal either way.
    """
    if isinstance(value, bool):
        return text("TRUE" if value else "FALSE")
    if isinstance(value, str):
        return text("('" + value.replace("'", "''") + "')")
    return text(repr(value))


def table(name: str, *items: SchemaItem, **options: Any) -> Table:
    """A table of the application schema."""
    return define_table(metadata, name, *items, **options)


def define_table(target: MetaData, name: str, *items: SchemaItem, **options: Any) -> Table:
    """A table in `target` that follows the schema's rules: the server table options, and on MariaDB a CHECK
    for each generated column that may not be NULL. Migrations create tables with it too (see
    `PortableMigrationContext.create_table`), so that a migrated server database matches a created one.
    """
    created = Table(name, target, *items, **{**_SERVER_TABLE_OPTIONS, **options})
    for column in created.columns:
        if column.computed is not None and not column.nullable:
            # MariaDB rejects NOT NULL on a generated column (see `_create_mariadb_column`).
            column.info[_MARIADB_NOT_NULL_CHECK] = True
            created.append_constraint(
                CheckConstraint(column.is_not(None), name=f"{column.name}_not_null").ddl_if(dialect="mariadb")
            )
    return created


def inserted_at(
    name: str = "created_at", *, declared: str = "DATETIME", sqlite_default: str = STRFTIME_NOW
) -> Column[str]:
    """A timestamp the application sets when it inserts the row."""
    return Column(name, Timestamp(declared), nullable=False, default=now_text, info={_SQLITE_DEFAULT: sqlite_default})


def updated_at(name: str = "updated_at", *, sqlite_default: str = STRFTIME_NOW) -> Column[str]:
    """A timestamp the application sets when it inserts the row, and when an `update()` statement changes it.

    An upsert does not set it: SQLAlchemy leaves `onupdate` out of `ON CONFLICT DO UPDATE` and `ON DUPLICATE KEY
    UPDATE`, so an upsert names it in its update values.
    """
    return Column(
        name,
        Timestamp(),
        nullable=False,
        default=now_text,
        onupdate=now_text,
        info={_SQLITE_DEFAULT: sqlite_default},
    )


@compiles(CreateColumn, "sqlite")
def _create_sqlite_column(create: CreateColumn, compiler: DDLCompiler, **kw: Any) -> str:
    spec: str = compiler.visit_create_column(create, **kw)  # type: ignore[no-untyped-call]
    sqlite_default = create.element.info.get(_SQLITE_DEFAULT)
    return spec if sqlite_default is None else f"{spec} DEFAULT ({sqlite_default})"


@compiles(CreateColumn, "mariadb")
def _create_mariadb_column(create: CreateColumn, compiler: DDLCompiler, **kw: Any) -> str:
    column = create.element
    if column.computed is None or not column.info.get(_MARIADB_NOT_NULL_CHECK):
        spec: str = compiler.visit_create_column(create, **kw)  # type: ignore[no-untyped-call]
        return spec
    # MariaDB rejects NOT NULL on a generated column, so it is left out here and `table()` adds a CHECK instead.
    return " ".join(
        (
            compiler.preparer.format_column(column),
            compiler.dialect.type_compiler_instance.process(column.type, type_expression=column),
            compiler.process(column.computed),
        )
    )
