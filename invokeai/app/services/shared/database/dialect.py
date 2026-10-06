"""SQL constructs that each backend spells differently, compiled for the backend in use."""

import json
import re
from collections.abc import Iterable, Sequence
from typing import Any

from sqlalchemy import Boolean, FromClause, Insert, Integer, Join, String, Table, UniqueConstraint, literal_column
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.compiler import SQLCompiler
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.functions import FunctionElement

from invokeai.app.services.shared.database.engines import (
    MARIADB_BINARY_COLLATION,
    MYSQL_BINARY_COLLATION,
    SERVER_DIALECTS,
)

_SIMPLE_JSON_PATH = re.compile(r"\$(\.[A-Za-z_][A-Za-z0-9_]*)+")
_LIKE_ESCAPE = "\\"


def _require_only_primary_key(table: Table) -> None:
    # MySQL and MariaDB resolve a conflict with any unique key, the other backends only one on the key named. A
    # unique index of the primary key's own columns (some migrated SQLite tables have one) is that same key.
    if not table.primary_key.columns:
        raise ValueError(f"Table {table.name} has no primary key")
    key = set(table.primary_key.columns)
    unique_keys = [
        set(constraint.columns) for constraint in table.constraints if isinstance(constraint, UniqueConstraint)
    ]
    unique_keys += [set(index.columns) for index in table.indexes if index.unique]
    if any(columns != key for columns in unique_keys):
        raise ValueError(f"Table {table.name} has a unique key besides its primary key")


def upsert(dialect_name: str, table: Table, *, update: Sequence[str]) -> Insert:
    """An INSERT into `table` that, where a row with the same primary key exists, sets that row's `update`
    columns to the values it would have inserted instead.

    Execute it with the row's values, keyed by column name. A column's `onupdate` does not apply to the update:
    list `updated_at` in `update` and pass its value. Every unique key of the table must be on the primary key's
    columns, because MySQL and MariaDB update the row a conflict with any unique key finds, the others only on the
    key named.
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
    Every unique key must be on the primary key's columns, as for `upsert`.
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


def sql_true() -> ColumnElement[bool]:
    """TRUE as the predicates of SQLite's partial indexes spell it (`WHERE is_intermediate = TRUE`).

    SQLAlchemy renders `true()` as 1 on SQLite, and SQLite's planner uses a partial index only for a query term
    that matches its predicate as written: `is_intermediate = 1` leaves `WHERE is_intermediate = TRUE` unused.
    """
    return literal_column("TRUE", Boolean())


def fixed_limit(count: int) -> ColumnElement[int]:
    """A LIMIT written into the statement, for a count that is the same on every execution, such as a lookup's 1.

    SQLite runs a statement whose LIMIT is a bound parameter 10-20 µs slower, on every execution and with the same
    plan, than one whose LIMIT is a literal. A count the caller chooses stays a bound parameter: each literal would
    be a statement of its own.
    """
    return literal_column(str(int(count)), Integer)


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


class JsonString(FunctionElement[str]):
    """The string at a member path of a JSON document in a text column, for a query; SQL NULL when the document is
    no valid JSON, the member is missing, or it holds anything but a string.

    The same on every backend: SQLite's `json_extract()` would give 1 for true where a server gives 'true'.
    """

    inherit_cache = True
    name = "json_string"

    def __init__(self, document: ColumnElement[Any], path: str) -> None:
        if _SIMPLE_JSON_PATH.fullmatch(path) is None:
            raise ValueError(f"Unsupported JSON path {path!r}: only member paths such as '$.meta.category' are")
        super().__init__(document, literal_column(f"'{path}'"))


@compiles(JsonString, "sqlite")
def _compile_json_string_sqlite(element: JsonString, compiler: SQLCompiler, **kw: Any) -> str:
    document, path = (compiler.process(clause, **kw) for clause in element.clauses)
    # Nested, so that json_type() never reads a document json_valid() rejected: it would raise.
    member = f"CASE WHEN json_type({document}, {path}) = 'text' THEN json_extract({document}, {path}) END"
    return f"CASE WHEN json_valid({document}) THEN {member} END"


@compiles(JsonString, "mysql")
@compiles(JsonString, "mariadb")
def _compile_json_string_mysql(element: JsonString, compiler: SQLCompiler, **kw: Any) -> str:
    document, path = (compiler.process(clause, **kw) for clause in element.clauses)
    extracted = f"JSON_EXTRACT({document}, {path})"
    # JSON_TYPE() answers in utf8mb4_bin, which MySQL will not compare with a literal in the connection's collation.
    member = f"CASE WHEN JSON_TYPE({extracted}) = 'STRING' COLLATE utf8mb4_bin THEN JSON_UNQUOTE({extracted}) END"
    return f"CASE WHEN JSON_VALID({document}) THEN {member} END"


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


# The longest string a bound set holds on a server: media names are at most 255 characters.
BOUND_SET_MAX_LENGTH = 255


def bound_set(values: Iterable[str]) -> str:
    """The parameter of an `InBoundSet`: the values as one JSON array."""
    return json.dumps(sorted(set(values)))


class InBoundSet(FunctionElement[bool]):
    """`expression IN` the strings of a set bound as one parameter (`bound_set`), compared as stored, case and all.

    A set of any size is one statement and one parameter, where an IN list would bind every value and be a
    statement per length: for sets held in process memory, such as the media names of active queue items. Values
    are at most `BOUND_SET_MAX_LENGTH` characters long. SQLite reads the array with `json_each()`, MySQL and
    MariaDB with `JSON_TABLE()`, in the tables' binary collation.
    """

    inherit_cache = True
    name = "in_bound_set"
    _is_implicitly_boolean = True

    def __init__(self, expression: ColumnElement[str], values: ColumnElement[str]) -> None:
        super().__init__(expression, values)


@compiles(InBoundSet, "sqlite")
def _compile_in_bound_set_sqlite(element: InBoundSet, compiler: SQLCompiler, **kw: Any) -> str:
    expression, values = (compiler.process(clause, **kw) for clause in element.clauses)
    return f"{expression} IN (SELECT value FROM json_each({values}))"


@compiles(InBoundSet, "mysql")
@compiles(InBoundSet, "mariadb")
def _compile_in_bound_set_server(element: InBoundSet, compiler: SQLCompiler, **kw: Any) -> str:
    expression, values = (compiler.process(clause, **kw) for clause in element.clauses)
    collation = MARIADB_BINARY_COLLATION if compiler.dialect.name == "mariadb" else MYSQL_BINARY_COLLATION
    column = f"value VARCHAR({BOUND_SET_MAX_LENGTH}) CHARACTER SET utf8mb4 COLLATE {collation} PATH '$'"
    return f"{expression} IN (SELECT bound.value FROM JSON_TABLE({values}, '$[*]' COLUMNS ({column})) AS bound)"


@compiles(InBoundSet, "postgresql")
def _compile_in_bound_set_postgresql(element: InBoundSet, compiler: SQLCompiler, **kw: Any) -> str:
    expression, values = (compiler.process(clause, **kw) for clause in element.clauses)
    return f"{expression} IN (SELECT jsonb_array_elements_text(CAST({values} AS JSONB)))"


class KeysetAfter(FunctionElement[bool]):
    """`(first, second) > (first_value, second_value)`: the rows after a keyset position, in that order.

    A row value on SQLite, whose planner starts an index range at it; spelled out elsewhere, since MariaDB does
    not start a range at a row value (`first > a OR (first = a AND second > b)`).
    """

    inherit_cache = True
    name = "keyset_after"
    _is_implicitly_boolean = True

    def __init__(
        self,
        first: ColumnElement[Any],
        second: ColumnElement[Any],
        first_value: ColumnElement[Any],
        second_value: ColumnElement[Any],
    ) -> None:
        super().__init__(first, second, first_value, second_value)


@compiles(KeysetAfter, "sqlite")
def _compile_keyset_after_sqlite(element: KeysetAfter, compiler: SQLCompiler, **kw: Any) -> str:
    first, second, first_value, second_value = (compiler.process(clause, **kw) for clause in element.clauses)
    return f"({first}, {second}) > ({first_value}, {second_value})"


@compiles(KeysetAfter)
def _compile_keyset_after(element: KeysetAfter, compiler: SQLCompiler, **kw: Any) -> str:
    first, second, first_value, second_value = (compiler.process(clause, **kw) for clause in element.clauses)
    return f"({first} > {first_value} OR ({first} = {first_value} AND {second} > {second_value}))"


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
