"""Column types: each renders the declared type of the migrated SQLite schema there, and its equivalent on a server.

SQLite keeps a column's declared type as written, and the schema metadata reproduces the migrated schema
exactly, so on SQLite these types render the migrations' declarations verbatim. A server needs bounded types
where SQLite has none: an index key holds at most 3072 bytes on MySQL and MariaDB, 768 utf8mb4 characters.

| Type | SQLite | MySQL / MariaDB |
|---|---|---|
| `Key(n=255)`: text a key, a foreign key or an index covers whole | TEXT | VARCHAR(n) |
| `NoCaseKey(n=255)`: the same, compared case-insensitively | TEXT COLLATE NOCASE | VARCHAR(n), case-insensitive |
| `LongText`: any other text, JSON documents among it | TEXT | LONGTEXT |
| `Timestamp(declared)`: canonical UTC text (see `now_text`) | `declared` (DATETIME) | VARCHAR(32) |
| `BigInt` | INTEGER, so a primary key is the rowid | BIGINT |
| `Real` | REAL | DOUBLE |
| `Blob` | BLOB | LONGBLOB |

Booleans are SQLAlchemy's `Boolean`: BOOLEAN on SQLite, TINYINT(1) on a server.
"""

import functools
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, TypeVar, cast

from sqlalchemy import BigInteger, Double, Integer, LargeBinary, String, Text
from sqlalchemy.dialects.mysql import LONGBLOB, LONGTEXT
from sqlalchemy.engine import Dialect
from sqlalchemy.types import REAL, TypeDecorator, TypeEngine, UserDefinedType

from invokeai.app.services.shared.database.engines import SERVER_DIALECTS

# Case-insensitive, accent- and space-sensitive (NO PAD) Unicode collations. SQLite's NOCASE folds the case of
# ASCII letters only; these compare by the Unicode collation algorithm, so they also equate the case of other
# letters, full-width and compatibility forms ('Ｃａｔ' = 'cat', 'ﬁ' = 'fi'), and precomposed and decomposed
# accents, and they ignore ignorable code points (a soft hyphen, a zero-width space).
MYSQL_NOCASE_COLLATION = "utf8mb4_0900_as_ci"
MARIADB_NOCASE_COLLATION = "utf8mb4_uca1400_nopad_as_ci"

# 'YYYY-MM-DD HH:MM:SS.fff' needs 23 characters. Some rows hold ISO 8601 text instead, at most 32 characters:
# `datetime.isoformat()` with microseconds and a UTC offset.
TIMESTAMP_LENGTH = 32

_Load = TypeVar("_Load", bound=Callable[[Any, Dialect], TypeEngine[Any]])


def _per_backend(load: _Load) -> _Load:
    """Keeps the type that `load` gives each backend, per column type.

    SQLAlchemy asks for it again on every execution of an expanding IN, and adapts every new type it gets anew,
    which cost about 19 µs a statement.
    """

    @functools.wraps(load)
    def cached(self: Any, dialect: Dialect) -> TypeEngine[Any]:
        types: dict[tuple[type, str], TypeEngine[Any]] = self.__dict__.setdefault("_backend_types", {})
        key = (type(dialect), dialect.name)
        if key not in types:
            types[key] = load(self, dialect)
        return types[key]

    return cast(_Load, cached)


def timestamp_text(moment: datetime) -> str:
    """A UTC moment as the canonical timestamp text, 'YYYY-MM-DD HH:MM:SS.fff'.

    The format of SQLite's `STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')`, the schema's column default: timestamps
    sort and compare as text, on every backend. The application is the only clock, so a server's clock and
    time zone never enter a timestamp.
    """
    return moment.strftime("%Y-%m-%d %H:%M:%S.%f")[:23]


def now_text() -> str:
    """The current UTC time as the canonical timestamp text (see `timestamp_text`)."""
    return timestamp_text(datetime.now(timezone.utc))


class _Declared(UserDefinedType[str]):
    """A type rendered as the given declaration, verbatim; values pass through as text."""

    cache_ok = True

    def __init__(self, declaration: str) -> None:
        self.declaration = declaration

    def get_col_spec(self, **kw: Any) -> str:
        return self.declaration


class Key(TypeDecorator[str]):
    """Text that a key, a foreign key or an index covers whole: TEXT on SQLite, VARCHAR(length) on a server."""

    impl = Text
    cache_ok = True

    def __init__(self, length: int = 255) -> None:
        super().__init__()
        self.length = length

    @_per_backend
    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name in SERVER_DIALECTS:
            return dialect.type_descriptor(String(self.length))
        return dialect.type_descriptor(Text())


class NoCaseKey(TypeDecorator[str]):
    """Key text compared case-insensitively: SQLite's NOCASE, a case-insensitive collation on a server.

    PostgreSQL has no such collation by default; there it renders SQLite's, which a PostgreSQL server refuses.
    """

    impl = Text
    cache_ok = True

    def __init__(self, length: int = 255) -> None:
        super().__init__()
        self.length = length

    @_per_backend
    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name == "mysql":
            return dialect.type_descriptor(String(self.length, collation=MYSQL_NOCASE_COLLATION))
        if dialect.name == "mariadb":
            return dialect.type_descriptor(String(self.length, collation=MARIADB_NOCASE_COLLATION))
        return dialect.type_descriptor(Text(collation="NOCASE"))


class LongText(TypeDecorator[str]):
    """Text of any length that no index covers: TEXT on SQLite, LONGTEXT on a server."""

    impl = Text
    cache_ok = True

    @_per_backend
    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name in SERVER_DIALECTS:
            return dialect.type_descriptor(LONGTEXT())
        return dialect.type_descriptor(Text())


class Timestamp(TypeDecorator[str]):
    """A point in time as canonical UTC text (see `now_text`).

    :param declared: The declared type on SQLite, as the migrations wrote it: DATETIME, or TEXT for a few.
    """

    impl = String
    cache_ok = True

    def __init__(self, declared: str = "DATETIME") -> None:
        super().__init__(TIMESTAMP_LENGTH)
        self.declared = declared

    @_per_backend
    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name == "sqlite":
            return dialect.type_descriptor(_Declared(self.declared))
        return dialect.type_descriptor(String(TIMESTAMP_LENGTH))


class BigInt(TypeDecorator[int]):
    """An integer: INTEGER on SQLite, where an INTEGER primary key is the rowid; BIGINT elsewhere."""

    impl = BigInteger
    cache_ok = True

    @_per_backend
    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name == "sqlite":
            return dialect.type_descriptor(Integer())
        return dialect.type_descriptor(BigInteger())


class Real(TypeDecorator[float]):
    """A double-precision float: REAL on SQLite, DOUBLE elsewhere."""

    impl = Double
    cache_ok = True

    @_per_backend
    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name == "sqlite":
            return dialect.type_descriptor(REAL())
        return dialect.type_descriptor(Double())


class Blob(TypeDecorator[bytes]):
    """Bytes of any length: BLOB on SQLite, LONGBLOB on a server."""

    impl = LargeBinary
    cache_ok = True

    @_per_backend
    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name in SERVER_DIALECTS:
            return dialect.type_descriptor(LONGBLOB())
        return dialect.type_descriptor(LargeBinary())
