"""SQL constructs that each backend spells differently, compiled for the backend in use."""

import re
from collections.abc import Sequence
from typing import Any

from sqlalchemy import Boolean, FromClause, Insert, Join, String, Table, UniqueConstraint
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.compiler import SQLCompiler
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.functions import FunctionElement

from invokeai.app.services.shared.database.engines import SERVER_DIALECTS

_SIMPLE_JSON_PATH = re.compile(r"\$(\.[A-Za-z_][A-Za-z0-9_]*)+")
_LIKE_ESCAPE = "\\"


def _require_only_primary_key(table: Table) -> None:
    # MySQL and MariaDB resolve a conflict with any unique key, the other backends only one on the key named.
    if not table.primary_key.columns:
        raise ValueError(f"Table {table.name} has no primary key")
    if any(isinstance(constraint, UniqueConstraint) for constraint in table.constraints) or any(
        index.unique for index in table.indexes
    ):
        raise ValueError(f"Table {table.name} has a unique key besides its primary key")


def upsert(dialect_name: str, table: Table, *, update: Sequence[str]) -> Insert:
    """An INSERT into `table` that, where a row with the same primary key exists, sets that row's `update`
    columns to the values it would have inserted instead.

    Execute it with the row's values, keyed by column name. A column's `onupdate` does not apply to the update:
    list `updated_at` in `update` and pass its value. The primary key must be the table's only unique key,
    because MySQL and MariaDB update the row a conflict with any unique key finds, the others only on the key
    named.
    """
    _require_only_primary_key(table)
    if dialect_name in SERVER_DIALECTS:
        on_server = mysql.insert(table)
        return on_server.on_duplicate_key_update({name: on_server.inserted[name] for name in update})
    if dialect_name == "sqlite":
        on_sqlite = sqlite.insert(table)
        return on_sqlite.on_conflict_do_update(
            index_elements=list(table.primary_key.columns),
            set_={name: on_sqlite.excluded[name] for name in update},
        )
    if dialect_name == "postgresql":
        on_postgresql = postgresql.insert(table)
        return on_postgresql.on_conflict_do_update(
            index_elements=list(table.primary_key.columns),
            set_={name: on_postgresql.excluded[name] for name in update},
        )
    raise ValueError(f"No upsert for the {dialect_name} dialect")


def insert_ignore(dialect_name: str, table: Table) -> Insert:
    """An INSERT into `table` that leaves a row with the same primary key as it is instead of failing.

    It skips that conflict only: a NULL, a failed CHECK and a missing foreign key still raise. (SQLite's `INSERT
    OR IGNORE` would skip the first two, MySQL's `INSERT IGNORE` all three.) Its row count does not tell whether
    the row was inserted: a skipped row counts 0 on SQLite and 1 on MySQL and MariaDB, which count the rows found.
    The primary key must be the table's only unique key, as for `upsert`.
    """
    _require_only_primary_key(table)
    if dialect_name in SERVER_DIALECTS:
        # MySQL has no DO NOTHING; setting a key column to its own value changes nothing.
        key = next(iter(table.primary_key.columns))
        return mysql.insert(table).on_duplicate_key_update({key.name: key})
    if dialect_name == "sqlite":
        return sqlite.insert(table).on_conflict_do_nothing(index_elements=list(table.primary_key.columns))
    if dialect_name == "postgresql":
        return postgresql.insert(table).on_conflict_do_nothing(index_elements=list(table.primary_key.columns))
    raise ValueError(f"No insert_ignore for the {dialect_name} dialect")


class JsonValue(ColumnElement[Any]):
    """The value at a member path of a JSON document held in a text column, as SQLite's `json_extract()` gives it.

    SQL NULL for a JSON null and for a missing member, and the text of a string, alike on every backend; with
    `integer`, the whole number of an integer member (a server refuses a fraction). Booleans, arrays, objects
    and numbers written with an exponent or trailing zeros come back differently per backend (SQLite gives 1
    for true, a server 'true'), so only strings and integers are compared.

    For the expressions of generated columns. MySQL and MariaDB need the CASE: `JSON_UNQUOTE()` turns a JSON
    null into the text 'null'.
    """

    inherit_cache = False

    def __init__(self, document: str, path: str, *, integer: bool = False) -> None:
        if _SIMPLE_JSON_PATH.fullmatch(path) is None:
            raise ValueError(f"Unsupported JSON path {path!r}: only member paths such as '$.meta.category' are")
        self.document = document
        self.path = path
        self.integer = integer


@compiles(JsonValue, "sqlite")
def _compile_json_value_sqlite(element: JsonValue, compiler: SQLCompiler, **kw: Any) -> str:
    return f"json_extract({compiler.preparer.quote(element.document)}, {_literal(compiler, element.path)})"


@compiles(JsonValue, "mysql")
@compiles(JsonValue, "mariadb")
def _compile_json_value_mysql(element: JsonValue, compiler: SQLCompiler, **kw: Any) -> str:
    extracted = f"JSON_EXTRACT({compiler.preparer.quote(element.document)}, {_literal(compiler, element.path)})"
    value = f"CASE WHEN JSON_TYPE({extracted}) = 'NULL' THEN NULL ELSE JSON_UNQUOTE({extracted}) END"
    return f"CAST({value} AS SIGNED)" if element.integer else value


@compiles(JsonValue, "postgresql")
def _compile_json_value_postgresql(element: JsonValue, compiler: SQLCompiler, **kw: Any) -> str:
    members = "{" + ",".join(element.path.split(".")[1:]) + "}"
    value = f"(CAST({compiler.preparer.quote(element.document)} AS JSONB) #>> {_literal(compiler, members)})"
    return f"CAST({value} AS BIGINT)" if element.integer else value


class CaseInsensitiveLike(FunctionElement[bool]):
    """`expression LIKE pattern`, ignoring case, with backslash as the escape character in `pattern` (see
    `like_prefix`).

    SQLite's own LIKE, which ignores the case of ASCII letters, there; LOWER() on both sides elsewhere, which
    folds the case of every letter. (Lowering on SQLite would cost as much as the scan it filters.)
    """

    inherit_cache = True
    type = Boolean()
    name = "case_insensitive_like"
    # A condition as it stands: otherwise a WHERE clause compares it with 1, which keeps SQLite from turning a
    # LIKE on an index with its collation into a range of that index.
    _is_implicitly_boolean = True

    def __init__(self, expression: ColumnElement[str], pattern: ColumnElement[str]) -> None:
        super().__init__(expression, pattern)


def _escape_like(text: str) -> str:
    escaped = text.replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2)
    return escaped.replace("%", _LIKE_ESCAPE + "%").replace("_", _LIKE_ESCAPE + "_")


def like_prefix(prefix: str) -> str:
    """The `CaseInsensitiveLike` pattern of the values that start with `prefix`; `%`, `_` and `\\` in it match
    only themselves."""
    return _escape_like(prefix) + "%"


def like_contains(text: str) -> str:
    """The `CaseInsensitiveLike` pattern of the values that contain `text`; `%`, `_` and `\\` in it match only
    themselves."""
    return "%" + _escape_like(text) + "%"


@compiles(CaseInsensitiveLike, "sqlite")
def _compile_case_insensitive_like_sqlite(element: CaseInsensitiveLike, compiler: SQLCompiler, **kw: Any) -> str:
    expression, pattern = (compiler.process(clause, **kw) for clause in element.clauses)
    return f"{expression} LIKE {pattern} ESCAPE {_literal(compiler, _LIKE_ESCAPE)}"


@compiles(CaseInsensitiveLike)
def _compile_case_insensitive_like(element: CaseInsensitiveLike, compiler: SQLCompiler, **kw: Any) -> str:
    expression, pattern = (compiler.process(clause, **kw) for clause in element.clauses)
    return f"lower({expression}) LIKE lower({pattern}) ESCAPE {_literal(compiler, _LIKE_ESCAPE)}"


class CaseInsensitiveOrder(FunctionElement[str]):
    """`expression` as an ORDER BY key that ignores case.

    SQLite's NOCASE collation there, which folds the case of ASCII letters as its LOWER() does; LOWER() elsewhere,
    which folds the case of every letter. (LOWER() on SQLite is a function call per row: a fifth of the time it
    takes to order 500 models, measured.)
    """

    inherit_cache = True
    name = "case_insensitive_order"

    def __init__(self, expression: ColumnElement[str]) -> None:
        super().__init__(expression)


@compiles(CaseInsensitiveOrder, "sqlite")
def _compile_case_insensitive_order_sqlite(element: CaseInsensitiveOrder, compiler: SQLCompiler, **kw: Any) -> str:
    (expression,) = (compiler.process(clause, **kw) for clause in element.clauses)
    return f"{expression} COLLATE NOCASE"


@compiles(CaseInsensitiveOrder)
def _compile_case_insensitive_order(element: CaseInsensitiveOrder, compiler: SQLCompiler, **kw: Any) -> str:
    (expression,) = (compiler.process(clause, **kw) for clause in element.clauses)
    return f"lower({expression})"


class OrderedJoin(Join):
    """An inner join that SQLite plans with its left side as the outer loop.

    SQLite never reorders the tables of a CROSS JOIN, so it renders as `left CROSS JOIN right ON ...` there: a
    planner hint that keeps the work proportional to the left side (a board's membership, say, rather than every
    image an index would offer first). Other backends get a plain JOIN and choose the order themselves. Inner only:
    a CROSS JOIN has no outer form.
    """

    inherit_cache = True

    def __init__(self, left: FromClause, right: FromClause, onclause: ColumnElement[bool]) -> None:
        super().__init__(left, right, onclause)


@compiles(OrderedJoin, "sqlite")
def _compile_ordered_join_sqlite(element: OrderedJoin, compiler: SQLCompiler, **kw: Any) -> str:
    # `SQLCompiler.visit_join` with CROSS JOIN for JOIN. The ON clause tells SQLAlchemy's linter that the two sides
    # are joined, as it does for a plain join.
    kw.pop("asfrom", None)
    assert element.onclause is not None
    left = compiler.process(element.left, asfrom=True, **kw)
    right = compiler.process(element.right, asfrom=True, **kw)
    return f"{left} CROSS JOIN {right} ON {compiler.process(element.onclause, **kw)}"


def _literal(compiler: SQLCompiler, value: str) -> str:
    return compiler.render_literal_value(value, String())
