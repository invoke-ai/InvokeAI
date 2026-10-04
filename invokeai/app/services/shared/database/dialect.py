"""SQL constructs that each backend spells differently, compiled for the backend in use."""

import re
from typing import Any

from sqlalchemy import String
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.compiler import SQLCompiler
from sqlalchemy.sql.elements import ColumnElement

_SIMPLE_JSON_PATH = re.compile(r"\$(\.[A-Za-z_][A-Za-z0-9_]*)+")


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


def _literal(compiler: SQLCompiler, value: str) -> str:
    return compiler.render_literal_value(value, String())
