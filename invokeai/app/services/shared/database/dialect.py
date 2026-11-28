"""SQL constructs that each backend spells differently, compiled for the backend in use."""

import functools
import json
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    FromClause,
    Insert,
    Integer,
    Join,
    String,
    Table,
    UniqueConstraint,
    bindparam,
    literal_column,
)
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.compiler import SQLCompiler
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.functions import FunctionElement
from sqlalchemy.types import TypeDecorator

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


def fold_case(text: str) -> str:
    """`text` with each character mapped to its simple lowercase (`Å` to `å`): what SQLite matches by and what the
    stored sort keys hold.

    Character by character: `ß` stays apart from `ss`, the final sigma `ς` from `σ`, and the length never changes. A
    character whose lowercase is more than one character stays as it is (`İ`).

    The stored keys (`workflow_library.name_key`, `fonts.family_key` and `fonts.label_key`) hold this: changing the
    function makes them stale, and needs a migration that writes them again.

    A server's LOWER() is not this function: it keeps some characters `fold_case` folds (MariaDB keeps `ẞ`), folds
    some it keeps (`İ`), and its tables are of another Unicode version (MariaDB's differ from Python's much more than
    MySQL's: about 740 code points against about 130 below 0x20000, measured). So a server is never given a pattern
    folded here (see `LikePattern.unfolded`); only SQLite matches by `fold_case`.
    """
    if text.isascii():
        # The same, without a call per character: the sorts of whole lists fold every name they hold.
        return text.lower()
    return "".join(lowered if len(lowered := character.lower()) == 1 else character for character in text)


# Cased characters exist in the first two planes only (`test_dialect` scans every code point to make a new Unicode
# version fail loudly).
_CASED_CODE_POINTS = 0x20000


@functools.cache
def _preimages() -> dict[str, tuple[str, ...]]:
    # The characters that fold to another one, grouped by it, minus the canonical singletons (KELVIN SIGN and
    # the like: `normalize("NFC", x) != x`). A scan of the cased planes, made on first need and kept.
    preimages: dict[str, list[str]] = {}
    for code_point in range(_CASED_CODE_POINTS):
        if 0xD800 <= code_point <= 0xDFFF:
            continue
        character = chr(code_point)
        folded = character.lower()
        if len(folded) == 1 and folded != character and unicodedata.normalize("NFC", character) == character:
            preimages.setdefault(folded, []).append(character)
    return {folded: tuple(characters) for folded, characters in preimages.items()}


def case_variants(folded: str) -> tuple[str, ...]:
    """The characters other than `folded` that fold to it and that NFC text can hold, by code point."""
    return _preimages().get(folded, ())


# The most replacement pairs a pattern has. Each is one `replace()` pass over every row SQLite reads, so a text of
# many letters would take seconds per scan (measured: 7 pairs 0.5 s, 30 pairs 2.45 s over 20 000 rows of 4 KB); the
# letters beyond them match any character on SQLite.
MAX_PAIRS = 8


@dataclass(frozen=True, slots=True)
class LikePattern:
    """What a `CaseInsensitiveLike` is matched with.

    On SQLite: `pattern`, the LIKE pattern of the folded text, and `pairs`, the characters SQLite's own LIKE does not
    already treat as equal to the ones they fold to, each as (character in the stored value, character it is
    replaced with before the match). SQLite's LIKE ignores the case of ASCII letters only, so `Å` is no `å` to it;
    the statement replaces `Å` in the stored value with `å` first, which is the same as folding the stored value.
    ASCII texts have no pairs. The pairs are:
    - every character of NFC text that folds to a letter of the text (`Å` for `å`); not the characters of NFC-unstable
      folds (the Kelvin, Ångström and Ohm signs, Greek capitals with oxia), which NFC text never holds, so a plain
      letter does not find them;
    - a character of the text itself that is one of those, so that a text holding it finds it as typed.

    Each pair is one `replace()` pass over every row, so there are at most `MAX_PAIRS`, the first characters of the
    text first. A letter left without its pair matches any one character in `pattern` (`_`): SQLite returns a
    superset of the values, never fewer.

    On a server: `unfolded`, the LIKE pattern of the text as given, which the statement lowers itself. A server's
    LOWER() keeps some characters `fold_case` folds (MariaDB keeps `ẞ`) and folds some it keeps (`İ`), so a pattern
    folded in Python could miss the very value it was taken from there. (MySQL 8.4 lowers `ẞ` to `ß`.)
    """

    pattern: str
    unfolded: str
    pairs: tuple[tuple[str, str], ...]

    @property
    def slots(self) -> int:
        """The number of replacements a statement carries for this pattern: the pairs rounded up to 0, 1, 2, 4 or
        8, so that statements share shapes."""
        return 0 if not self.pairs else 1 << (len(self.pairs) - 1).bit_length()

    def parameters(self, name: str, slots: int) -> dict[str, Optional[str]]:
        """The bound values of `CaseInsensitiveLike(column, name, slots)`: `slots` (at least `self.slots`)
        replacements, the spare ones `'' -> ''`, which change nothing."""
        return like_parameters(name, self, slots)


def like_parameters(name: str, pattern: Optional[LikePattern], slots: int) -> dict[str, Optional[str]]:
    """The bound values of `CaseInsensitiveLike(column, name, slots)` for `pattern`, or for None, which matches
    nothing."""
    pairs = pattern.pairs if pattern is not None else ()
    parameters: dict[str, Optional[str]] = {
        name: pattern.pattern if pattern is not None else None,
        f"{name}_unfolded": pattern.unfolded if pattern is not None else None,
    }
    for index in range(slots):
        replaced, replacement = pairs[index] if index < len(pairs) else ("", "")
        parameters[f"{name}_from_{index}"] = replaced
        parameters[f"{name}_to_{index}"] = replacement
    return parameters


def like_slots(*patterns: Optional[LikePattern]) -> int:
    """The replacement slots of a statement that matches all of `patterns`."""
    return max((pattern.slots for pattern in patterns if pattern is not None), default=0)


class CaseInsensitiveLike(FunctionElement[bool]):
    """`expression LIKE pattern`, ignoring case, with backslash as the escape character in the pattern (see
    `like_prefix`). `name` is the bound parameter of the pattern and `slots` the replacements of the statement's
    shape (`LikePattern.slots`); bind them with `LikePattern.parameters(name, slots)`.

    SQLite matches the pattern folded by `fold_case`. Its LIKE ignores the case of ASCII letters only, so it first
    replaces the other characters that fold to a character of the pattern in `expression`, one `replace()` pass over
    each row per slot, at most `MAX_PAIRS`; beyond them it over-matches (`LikePattern`). (Lowering on SQLite would
    cost as much as the scan it filters; a text of ASCII letters has no slots and costs what the plain LIKE did.) A
    server compares `lower(expression)` with `lower(pattern as given)`, and ignores the replacement parameters.
    """

    inherit_cache = True
    type = Boolean()
    name = "case_insensitive_like"
    # A condition as it stands: otherwise a WHERE clause compares it with 1, which keeps SQLite from turning a
    # LIKE on an index with its collation into a range of that index.
    _is_implicitly_boolean = True

    def __init__(self, expression: ColumnElement[str], name: str, slots: int) -> None:
        super().__init__(
            expression,
            bindparam(name),
            bindparam(f"{name}_unfolded"),
            *(bindparam(f"{name}_from_{index}") for index in range(slots)),
            *(bindparam(f"{name}_to_{index}") for index in range(slots)),
        )


class ContainsText(FunctionElement[bool]):
    """Whether `haystack` contains `needle` as written, case and all; NULL for a NULL needle. `instr()` on SQLite,
    MySQL and MariaDB, `strpos()` on Postgres: a substring test that needs no pattern escaping."""

    inherit_cache = True
    type = Boolean()
    name = "contains_text"
    _is_implicitly_boolean = True

    def __init__(self, haystack: ColumnElement[str], needle: ColumnElement[str]) -> None:
        super().__init__(haystack, needle)


@compiles(ContainsText)
def _compile_contains_text(element: ContainsText, compiler: SQLCompiler, **kw: Any) -> str:
    haystack, needle = (compiler.process(clause, **kw) for clause in element.clauses)
    return f"(instr({haystack}, {needle}) > 0)"


@compiles(ContainsText, "postgresql")
def _compile_contains_text_postgresql(element: ContainsText, compiler: SQLCompiler, **kw: Any) -> str:
    haystack, needle = (compiler.process(clause, **kw) for clause in element.clauses)
    return f"(strpos({haystack}, {needle}) > 0)"


def _escape_like(text: str) -> str:
    escaped = text.replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2)
    return escaped.replace("%", _LIKE_ESCAPE + "%").replace("_", _LIKE_ESCAPE + "_")


def _like_pattern(text: str, prefix: str, suffix: str) -> LikePattern:
    if text.isascii():
        # No character that NFC text holds folds to an ASCII one but the ASCII capitals, which LIKE folds itself.
        return LikePattern(prefix + _escape_like(text.lower()) + suffix, prefix + _escape_like(text) + suffix, ())
    folded = fold_case(text)
    # The characters, by the folded character, that SQLite's LIKE does not equate with it: those of NFC text that
    # fold to it, and the ones typed in the text that NFC-unstable folds make (the Kelvin sign).
    needed: dict[str, list[str]] = {}
    for character in dict.fromkeys(folded):
        variants = [variant for variant in case_variants(character) if not variant.isascii()]
        if variants:
            needed[character] = variants
    for typed in dict.fromkeys(text):
        target = fold_case(typed)
        if target != typed and not typed.isascii() and typed not in case_variants(target):
            needed.setdefault(target, []).append(typed)
    pairs: list[tuple[str, str]] = []
    uncovered: set[str] = set()
    for character in dict.fromkeys(folded):
        variants = needed.get(character, [])
        if len(pairs) + len(variants) <= MAX_PAIRS:
            pairs.extend((variant, character) for variant in variants)
        else:
            uncovered.add(character)
    # A character without its pair matches any one character, so that no value is missed.
    matching = "".join("_" if character in uncovered else _escape_like(character) for character in folded)
    return LikePattern(prefix + matching + suffix, prefix + _escape_like(text) + suffix, tuple(pairs))


def like_prefix(prefix: str) -> LikePattern:
    """The `CaseInsensitiveLike` pattern of the values that start with `prefix`; `%`, `_` and `\\` in it match
    only themselves."""
    return _like_pattern(prefix, "", "%")


def like_contains(text: str) -> LikePattern:
    """The `CaseInsensitiveLike` pattern of the values that contain `text`; `%`, `_` and `\\` in it match only
    themselves."""
    return _like_pattern(text, "%", "%")


@compiles(CaseInsensitiveLike, "sqlite")
def _compile_case_insensitive_like_sqlite(element: CaseInsensitiveLike, compiler: SQLCompiler, **kw: Any) -> str:
    # The unfolded pattern is a server's; left out, it is no parameter of the statement here.
    expression, pattern, _unfolded, *rest = element.clauses
    expression, pattern = compiler.process(expression, **kw), compiler.process(pattern, **kw)
    replacements = [compiler.process(clause, **kw) for clause in rest]
    half = len(replacements) // 2
    for replaced, replacement in zip(replacements[:half], replacements[half:], strict=False):
        expression = f"replace({expression}, {replaced}, {replacement})"
    return f"{expression} LIKE {pattern} ESCAPE {_literal(compiler, _LIKE_ESCAPE)}"


@compiles(CaseInsensitiveLike)
def _compile_case_insensitive_like(element: CaseInsensitiveLike, compiler: SQLCompiler, **kw: Any) -> str:
    expression, _folded, unfolded, *_ = element.clauses
    expression, unfolded = compiler.process(expression, **kw), compiler.process(unfolded, **kw)
    return f"lower({expression}) LIKE lower({unfolded}) ESCAPE {_literal(compiler, _LIKE_ESCAPE)}"


# The longest string a bound set holds on a server: media names are at most 255 characters.
BOUND_SET_MAX_LENGTH = 255


def bound_set(values: Iterable[str] | Iterable[int]) -> str:
    """The parameter of an `InBoundSet`: the values as one JSON array."""
    return json.dumps(sorted(set(values)))


class InBoundSet(FunctionElement[bool]):
    """`expression IN` the strings or integers of a set bound as one parameter (`bound_set`), strings compared as
    stored, case and all.

    A set of any size is one statement and one parameter, where an IN list would bind every value and be a
    statement per length: for sets held in process memory, such as the media names of active queue items, and long
    lists a caller passes. Strings are at most `BOUND_SET_MAX_LENGTH` characters long. SQLite reads the array with
    `json_each()`, MySQL and MariaDB with `JSON_TABLE()`, strings in the tables' binary collation; an integer
    expression reads integers.
    """

    inherit_cache = True
    name = "in_bound_set"
    _is_implicitly_boolean = True

    def __init__(self, expression: ColumnElement[Any], values: ColumnElement[str]) -> None:
        super().__init__(expression, values)


def _of_integers(element: InBoundSet) -> bool:
    column_type = next(iter(element.clauses)).type
    if isinstance(column_type, TypeDecorator):
        column_type = column_type.impl_instance
    return isinstance(column_type, Integer)


@compiles(InBoundSet, "sqlite")
def _compile_in_bound_set_sqlite(element: InBoundSet, compiler: SQLCompiler, **kw: Any) -> str:
    expression, values = (compiler.process(clause, **kw) for clause in element.clauses)
    return f"{expression} IN (SELECT value FROM json_each({values}))"


@compiles(InBoundSet, "mysql")
@compiles(InBoundSet, "mariadb")
def _compile_in_bound_set_server(element: InBoundSet, compiler: SQLCompiler, **kw: Any) -> str:
    expression, values = (compiler.process(clause, **kw) for clause in element.clauses)
    if _of_integers(element):
        column = "value BIGINT PATH '$'"
    else:
        collation = MARIADB_BINARY_COLLATION if compiler.dialect.name == "mariadb" else MYSQL_BINARY_COLLATION
        column = f"value VARCHAR({BOUND_SET_MAX_LENGTH}) CHARACTER SET utf8mb4 COLLATE {collation} PATH '$'"
    return f"{expression} IN (SELECT bound.value FROM JSON_TABLE({values}, '$[*]' COLUMNS ({column})) AS bound)"


@compiles(InBoundSet, "postgresql")
def _compile_in_bound_set_postgresql(element: InBoundSet, compiler: SQLCompiler, **kw: Any) -> str:
    expression, values = (compiler.process(clause, **kw) for clause in element.clauses)
    value = "CAST(value AS BIGINT)" if _of_integers(element) else "value"
    return f"{expression} IN (SELECT {value} FROM jsonb_array_elements_text(CAST({values} AS JSONB)) AS value)"


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


class Unindexed(FunctionElement[Any]):
    """`column` in a WHERE term that SQLite must not look up through an index: `+column` there, `column` elsewhere.

    SQLite keeps no table statistics, so it takes `queue_id = ?` (or `user_id = ?`) for a selective term and can
    walk an index on it through a queue's whole history, when another term (a status, a batch, an id) finds the few
    rows wanted. The unary plus keeps the term off every index.
    """

    inherit_cache = True
    name = "unindexed"

    def __init__(self, column: ColumnElement[Any]) -> None:
        super().__init__(column)
        self.type = column.type


@compiles(Unindexed, "sqlite")
def _compile_unindexed_sqlite(element: Unindexed, compiler: SQLCompiler, **kw: Any) -> str:
    (column,) = (compiler.process(clause, **kw) for clause in element.clauses)
    return f"+{column}"


@compiles(Unindexed)
def _compile_unindexed(element: Unindexed, compiler: SQLCompiler, **kw: Any) -> str:
    (column,) = (compiler.process(clause, **kw) for clause in element.clauses)
    return column


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
