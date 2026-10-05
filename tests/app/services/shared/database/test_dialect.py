"""Constructs of `dialect.py` beyond what the query modules exercise on every backend."""

import pytest
from sqlalchemy import Column, Integer, MetaData, Table, bindparam, select
from sqlalchemy.dialects import mysql, postgresql, sqlite

from invokeai.app.services.shared.database.dialect import CaseInsensitiveLike, insert_ignore, upsert
from invokeai.app.services.shared.database.schema.client_state import client_state
from invokeai.app.services.shared.database.schema.users import users


def test_an_upsert_refuses_a_table_with_a_unique_key_besides_its_primary_key() -> None:
    # MySQL would update the row whichever unique key matched; the other backends only on the primary key.
    with pytest.raises(ValueError, match="unique key besides its primary key"):
        upsert("sqlite", users, update=["display_name"])


def test_an_insert_ignore_refuses_a_table_with_a_unique_key_besides_its_primary_key() -> None:
    # MySQL would skip the row whichever unique key matched; the other backends only on the primary key.
    with pytest.raises(ValueError, match="unique key besides its primary key"):
        insert_ignore("sqlite", users)


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


def test_an_upsert_compiles_for_postgresql() -> None:
    statement = upsert("postgresql", client_state, update=["value", "updated_at"])

    sql = str(statement.compile(dialect=postgresql.dialect()))

    assert "ON CONFLICT (user_id, key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at" in sql


def test_an_insert_ignore_compiles_for_postgresql() -> None:
    statement = insert_ignore("postgresql", client_state)

    sql = str(statement.compile(dialect=postgresql.dialect()))

    assert "ON CONFLICT (user_id, key) DO NOTHING" in sql
