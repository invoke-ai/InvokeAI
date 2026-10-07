"""The schema metadata describes the schema the SQLite migrations build, exactly.

Server databases are created from the metadata, and queries are compiled against it, so a difference here
would be a difference between backends: a missing index, a cascade that does not happen, a default that is
not there. Compared on SQLite, where both can be built: the migration chain on one in-memory database,
`metadata.create_all()` on another.
"""

import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterator
from typing import Any, Optional

import pytest
from sqlalchemy import create_engine

from invokeai.app.services.config.config_default import DefaultInvokeAIAppConfig
from invokeai.app.services.shared.database.schema import metadata
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_08_06_add_project_boards import (
    AddProjectBoardsMigrationCallback,
)
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database


@pytest.fixture(scope="module")
def migrated(tmp_path_factory: pytest.TempPathFactory) -> Iterator[sqlite3.Connection]:
    # Migrations clean up legacy files under the root; it must be this temporary one, whatever the environment says.
    config = DefaultInvokeAIAppConfig(use_memory_db=True)
    config._root = tmp_path_factory.mktemp("root")
    db = create_mock_sqlite_database(config, InvokeAILogger.get_logger("test_schema_parity"))
    conn = db.database.sqlite.conn
    # Created by the project boards migration only when it has projects to rescue, which a new database has not.
    AddProjectBoardsMigrationCallback(InvokeAILogger.get_logger("test_schema_parity"))._create_quarantine_table(
        conn.cursor()
    )
    conn.commit()
    yield conn
    db.database.dispose()


@pytest.fixture(scope="module")
def created() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(":memory:")
    engine = create_engine("sqlite://", creator=lambda: conn)
    metadata.create_all(engine)
    yield conn
    engine.dispose()
    conn.close()


def test_the_metadata_creates_the_migrated_schema(created: sqlite3.Connection, migrated: sqlite3.Connection) -> None:
    created_schema = _schema(created)
    migrated_schema = _schema(migrated)

    differences = [
        f"{table}.{aspect}:\n  metadata {created_schema.get(table, {}).get(aspect)!r}\n"
        f"  migrated {migrated_schema.get(table, {}).get(aspect)!r}"
        for table in sorted(created_schema.keys() | migrated_schema.keys())
        for aspect in ("table", "columns", "checks", "foreign_keys", "indexes")
        if created_schema.get(table, {}).get(aspect) != migrated_schema.get(table, {}).get(aspect)
    ]
    assert differences == []


def test_neither_schema_has_triggers(created: sqlite3.Connection, migrated: sqlite3.Connection) -> None:
    # The application sets what the triggers of the earlier migrations set; the last migrations drop them.
    assert _names(migrated, "trigger") == set()
    assert _names(created, "trigger") == set()
    assert _names(created, "view") == _names(migrated, "view") == set()


def _names(conn: sqlite3.Connection, kind: str) -> set[str]:
    return {name for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = ?", (kind,))}


def _schema(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """What SQLite reports about each table, and what only its declaration shows.

    Its declaration shows the generation expressions, CHECKs, conflict clauses, deferred constraints and
    AUTOINCREMENT.
    """
    schema: dict[str, dict[str, Any]] = {}
    for name, sql in conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall():
        definitions = _definitions(sql)
        code = _code(sql)
        generated = {
            _column_name(definition): _expression_after(definition, "GENERATED ALWAYS AS")
            for definition in definitions
            if re.search(r"\bGENERATED\s+ALWAYS\s+AS\b", definition, re.IGNORECASE)
        }
        kind, column_count, without_rowid, strict = conn.execute(
            "SELECT type, ncol, wr, strict FROM pragma_table_list WHERE schema = 'main' AND name = ?", (name,)
        ).fetchone()
        schema[name] = {
            "table": {
                "kind": kind,
                "columns": column_count,
                "without_rowid": without_rowid,
                "strict": strict,
                "autoincrement": re.search(r"\bAUTOINCREMENT\b", code, re.IGNORECASE) is not None,
                "conflict_clauses": sorted(
                    clause.upper() for clause in re.findall(r"\bON\s+CONFLICT\s+(\w+)", code, re.IGNORECASE)
                ),
                "deferred": sorted(
                    " ".join(clause.upper().split())
                    for clause in re.findall(
                        r"\b(?:NOT\s+)?DEFERRABLE(?:\s+INITIALLY\s+\w+)?", code, flags=re.IGNORECASE
                    )
                ),
            },
            "columns": [
                {
                    "name": column,
                    "type": declared_type,
                    "not_null": not_null,
                    "default": _default(default),
                    "primary_key_position": pk,
                    "hidden": hidden,
                    "collation": _collation(conn, name, column),
                    "generated": generated.get(column),
                }
                # Read whole first: `_collation` changes the schema, which an open statement over it would block.
                for _, column, declared_type, not_null, default, pk, hidden in conn.execute(
                    f"PRAGMA table_xinfo({_quote(name)})"
                ).fetchall()
            ],
            "checks": sorted(check for definition in definitions for check in _expressions_after(definition, "CHECK")),
            "foreign_keys": _foreign_keys(conn, name),
            "indexes": sorted(_indexes(conn, name), key=repr),
        }
    return schema


def _foreign_keys(conn: sqlite3.Connection, table: str) -> list[tuple[Any, ...]]:
    """Each foreign key whole: a composite key's columns in order, with its actions."""
    keys: dict[int, list[Any]] = defaultdict(list)
    for key_id, _, target, source, referenced, on_update, on_delete, match in conn.execute(
        f"PRAGMA foreign_key_list({_quote(table)})"
    ).fetchall():
        keys[key_id].append((target, source, referenced, on_update, on_delete, match))
    return sorted(
        (
            rows[0][0],
            tuple(row[1] for row in rows),
            tuple(row[2] for row in rows),
            *rows[0][3:],
        )
        for rows in keys.values()
    )


def _indexes(conn: sqlite3.Connection, table: str) -> Iterator[tuple[Any, ...]]:
    for _, index, unique, origin, partial in conn.execute(f"PRAGMA index_list({_quote(table)})").fetchall():
        keys = tuple(
            (column, descending, collation)
            for _, _, column, descending, collation, key in conn.execute(
                f"PRAGMA index_xinfo({_quote(index)})"
            ).fetchall()
            if key
        )
        sql: Optional[str] = conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (index,)).fetchone()[0]
        # SQLite reports an expression key without its expression, so the declared key list stands in for it.
        expressions = _normalized(_balanced(sql, sql.index("("))) if sql and any(k[0] is None for k in keys) else None
        predicate: Optional[str] = None
        if partial and sql:
            predicate = _normalized(re.split(r"\bWHERE\b", sql, maxsplit=1, flags=re.IGNORECASE)[1])
        # Indexes SQLite creates for a PRIMARY KEY or UNIQUE constraint are named by constraint position.
        yield (index if origin == "c" else None, unique, origin, keys, expressions, predicate)


def _collation(conn: sqlite3.Connection, table: str, column: str) -> str:
    """The column's collation, as SQLite resolves it for an index on the column alone."""
    conn.execute(f"CREATE INDEX _parity_probe ON {_quote(table)} ({_quote(column)})")
    try:
        ((collation,),) = conn.execute(
            "SELECT coll FROM pragma_index_xinfo('_parity_probe') WHERE seqno = 0"
        ).fetchall()
        return str(collation)
    finally:
        conn.execute("DROP INDEX _parity_probe")


def _default(default: Optional[str]) -> Optional[str]:
    # SQLite reads a double-quoted default that names no column as a string literal: "user" is 'user'.
    if default is not None and re.fullmatch(r'"[^"]*"', default):
        return f"'{default[1:-1]}'"
    return default


def _definitions(create_table: str) -> list[str]:
    """The column definitions and table constraints of a CREATE TABLE statement, without comments."""
    body = _balanced(create_table, create_table.index("("))
    definitions: list[str] = []
    current: list[str] = []
    depth = 0
    for token in _tokens(body):
        if token.startswith("--"):
            token = " "
        elif token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        elif token == "," and depth == 0:
            definitions.append(_normalized("".join(current)))
            current = []
            continue
        current.append(token)
    definitions.append(_normalized("".join(current)))
    return [definition for definition in definitions if definition]


def _column_name(definition: str) -> str:
    name = definition.split()[0]
    return name[1:-1] if name[0] in '"`[' else name


def _expression_after(definition: str, keyword: str) -> str:
    (expression,) = _expressions_after(definition, keyword)
    return expression


def _expressions_after(definition: str, keyword: str) -> list[str]:
    pattern = r"\b" + r"\s+".join(keyword.split()) + r"\s*\("
    return [
        _normalized(_balanced(definition, match.end() - 1))
        for match in re.finditer(pattern, _code(definition), re.IGNORECASE)
    ]


def _balanced(sql: str, open_at: int) -> str:
    """The text inside the parentheses that open at `open_at`."""
    depth = 0
    position = open_at
    for token in _tokens(sql[open_at:]):
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth == 0:
                return sql[open_at + 1 : position]
        position += len(token)
    raise ValueError(f"Unbalanced parentheses in {sql!r}")


def _tokens(sql: str) -> Iterator[str]:
    """Quoted strings and identifiers whole, comments whole, every other character on its own."""
    position = 0
    while position < len(sql):
        char = sql[position]
        if sql.startswith("--", position):
            end = sql.find("\n", position)
            end = len(sql) if end == -1 else end
            yield sql[position:end]
            position = end
            continue
        if char in "'\"`":
            end = position + 1
            while end < len(sql):
                if sql[end] == char:
                    if sql[end + 1 : end + 2] == char:
                        end += 2
                        continue
                    break
                end += 1
            yield sql[position : end + 1]
            position = end + 1
            continue
        yield char
        position += 1


def _code(sql: str) -> str:
    """The statement with comments as spaces and each string literal's text blanked, at the same positions.

    For finding keywords: only outside comments and literals are they keywords.
    """
    return "".join(
        " " * len(token)
        if token.startswith("--")
        else ("'" + " " * (len(token) - 2) + "'" if token.startswith("'") else token)
        for token in _tokens(sql)
    )


def _normalized(sql: str) -> str:
    """Comments dropped and whitespace collapsed, none just inside parentheses; string literals as written."""
    parts: list[str] = []
    for token in _tokens(sql):
        if token.isspace() or token.startswith("--"):
            if parts and parts[-1] not in (" ", "("):
                parts.append(" ")
            continue
        if token == ")" and parts and parts[-1] == " ":
            parts.pop()
        parts.append(token)
    return "".join(parts).strip()


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
