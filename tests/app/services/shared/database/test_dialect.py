"""Constructs of `dialect.py` beyond what the query modules exercise on every backend."""

import pytest
from sqlalchemy import Column, Index, Integer, MetaData, String, Table, bindparam, insert, select
from sqlalchemy.dialects import mysql, postgresql, sqlite

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.dialect import (
    CaseInsensitiveLike,
    InBoundSet,
    bound_set,
    fixed_limit,
    insert_ignore,
    upsert,
)
from invokeai.app.services.shared.database.schema.client_state import client_state
from invokeai.app.services.shared.database.schema.images import images
from invokeai.app.services.shared.database.schema.users import users


def test_an_upsert_refuses_a_table_with_a_unique_key_besides_its_primary_key() -> None:
    # MySQL would update the row whichever unique key matched; the other backends only on the primary key.
    with pytest.raises(ValueError, match="unique key besides its primary key"):
        upsert("sqlite", users, update=["display_name"])


def test_an_insert_ignore_refuses_a_table_with_a_unique_key_besides_its_primary_key() -> None:
    # MySQL would skip the row whichever unique key matched; the other backends only on the primary key.
    with pytest.raises(ValueError, match="unique key besides its primary key"):
        insert_ignore("sqlite", users)


def _keyed_table(*unique_index_columns: str) -> Table:
    table = Table("keyed", MetaData(), Column("name", String(10), primary_key=True), Column("value", Integer))
    if unique_index_columns:
        Index("keyed_unique", *(table.c[name] for name in unique_index_columns), unique=True)
    return table


def test_a_unique_index_of_the_primary_key_columns_is_that_same_key() -> None:
    # Some migrated SQLite tables index their primary key once more, uniquely (images.image_name).
    table = _keyed_table("name")

    sql = str(insert_ignore("sqlite", table).compile(dialect=sqlite.dialect()))
    upsert("sqlite", table, update=["value"])

    assert "ON CONFLICT (name) DO NOTHING" in sql


@pytest.mark.parametrize("columns", [("value",), ("name", "value")])
def test_a_unique_index_of_other_columns_is_refused(columns: tuple[str, ...]) -> None:
    # A key on more columns than the primary key's is another key too: MySQL resolves a conflict on either.
    with pytest.raises(ValueError, match="unique key besides its primary key"):
        insert_ignore("sqlite", _keyed_table(*columns))


def test_an_upsert_refuses_a_table_without_a_primary_key() -> None:
    table = Table("no_key", MetaData(), Column("value", Integer))

    with pytest.raises(ValueError, match="no primary key"):
        upsert("sqlite", table, update=["value"])


@pytest.mark.parametrize("dialect", [sqlite.dialect(), mysql.dialect()])
def test_a_case_insensitive_like_is_a_condition_as_it_stands(dialect: object) -> None:
    # Compared with 1, a LIKE keeps SQLite from searching an index with its collation by range.
    statement = select(client_state.c.key).where(CaseInsensitiveLike(client_state.c.key, bindparam("pattern")))

    sql = str(statement.compile(dialect=dialect))  # type: ignore[arg-type]

    assert "LIKE" in sql and not sql.rstrip().endswith(("= 1", "= true"))


@pytest.mark.parametrize("dialect", [sqlite.dialect(), mysql.dialect()])
def test_a_fixed_limit_is_written_into_the_statement(dialect: object) -> None:
    # Bound, it would cost SQLite 10-20 µs per execution.
    compiled = select(users.c.email).limit(fixed_limit(1)).compile(dialect=dialect)  # type: ignore[arg-type]

    assert "LIMIT 1" in str(compiled)
    assert 1 not in compiled.params.values()


def test_an_upsert_compiles_for_postgresql() -> None:
    statement = upsert("postgresql", client_state, update=["value", "updated_at"])

    sql = str(statement.compile(dialect=postgresql.dialect()))

    assert "ON CONFLICT (user_id, key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at" in sql


def test_an_insert_ignore_compiles_for_postgresql() -> None:
    statement = insert_ignore("postgresql", client_state)

    sql = str(statement.compile(dialect=postgresql.dialect()))

    assert "ON CONFLICT (user_id, key) DO NOTHING" in sql


def test_a_bound_set_matches_exactly_the_values_it_holds(database: Database) -> None:
    quoted = 'it\'s "quoted" \\ here.png'
    stored = ["a.png", "A.png", quoted, "ünïcode.png", "x" * 251 + ".png", "other.png"]
    with database.begin(write=True) as conn:
        conn.execute(
            insert(images),
            [
                {"image_name": name, "image_origin": "internal", "image_category": "general", "width": 1, "height": 1}
                for name in stored
            ],
        )
    statement = select(images.c.image_name).where(InBoundSet(images.c.image_name, bindparam("names")))

    def matching(values: list[str]) -> set[str]:
        with database.begin(write=False) as conn:
            return set(conn.execute(statement, {"names": bound_set(values)}).scalars())

    # Compared as stored: case, quotes, backslashes and letters beyond ASCII, up to the longest name.
    wanted = ["a.png", quoted, "ünïcode.png", "x" * 251 + ".png", "missing.png"]
    assert matching(wanted) == set(wanted) - {"missing.png"}
    assert matching([]) == set()
