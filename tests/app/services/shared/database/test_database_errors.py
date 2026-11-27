"""Constraint violations reach callers as backend-neutral errors, whatever the database."""

import pytest
from sqlalchemy.exc import DBAPIError

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import (
    CheckViolation,
    DatabaseError,
    ForeignKeyViolation,
    NotNullViolation,
    TransactionFailedError,
    UniqueViolation,
)
from tests.fixtures.database_probe import ProbeQueries


def test_a_duplicate_unique_value_is_a_unique_violation(probe: ProbeQueries) -> None:
    probe.items.add(1, "a")

    with pytest.raises(UniqueViolation):
        probe.items.add(2, "a")


def test_a_duplicate_primary_key_is_a_unique_violation(probe: ProbeQueries) -> None:
    probe.items.add(1, "a")

    with pytest.raises(UniqueViolation):
        probe.items.add(1, "b")


def test_a_failed_check_is_a_check_violation(probe: ProbeQueries) -> None:
    with pytest.raises(CheckViolation):
        probe.items.add(1, "a", size=-1)


def test_a_missing_required_value_is_a_not_null_violation(probe: ProbeQueries) -> None:
    with pytest.raises(NotNullViolation):
        probe.items.add(1, None)


def test_foreign_keys_are_enforced_and_cascade(probe: ProbeQueries) -> None:
    with pytest.raises(ForeignKeyViolation):
        probe.items.link(1, item_id=99)

    probe.items.add(1, "a")
    probe.items.link(1, item_id=1)
    probe.items.remove(1)

    assert probe.items.link_ids() == []


class DuplicateName(Exception):
    pass


def test_a_violation_can_be_caught_inside_a_transaction_that_then_rolls_back(probe: ProbeQueries) -> None:
    # The way a service turns a violation into its own error: caught where it happens, which needs the
    # backend-neutral error there already, not only once the transaction has ended.
    probe.items.add(1, "a")

    with pytest.raises(DuplicateName):
        with probe.transaction() as q:
            q.items.add(2, "b")
            try:
                q.items.add(3, "a")
            except UniqueViolation as error:
                raise DuplicateName("a") from error

    assert probe.items.names() == ["a"]


def test_a_transaction_takes_no_further_calls_after_a_statement_failed(probe: ProbeQueries) -> None:
    probe.items.add(1, "a")

    with pytest.raises(TransactionFailedError):
        with probe.transaction() as q:
            with pytest.raises(UniqueViolation):
                q.items.add(2, "a")
            q.items.add(3, "c")

    assert probe.items.names() == ["a"]


def test_a_transaction_whose_failed_statement_was_swallowed_does_not_commit(probe: ProbeQueries) -> None:
    # Committing what came before the failure would keep half of a unit of work -- and after a deadlock on
    # MySQL, whose rollback already undid that half, only what came after it.
    probe.items.add(1, "a")

    with pytest.raises(TransactionFailedError):
        with probe.transaction() as q:
            q.items.add(2, "b")
            try:
                q.items.add(3, "a")
            except UniqueViolation:
                pass

    assert probe.items.names() == ["a"]


def test_names_compare_byte_for_byte(probe: ProbeQueries) -> None:
    # SQLite's default comparison: case and trailing spaces both make a different name. On a server this
    # holds in a database created with the binary NO PAD collation the layer requires (as the fixture's
    # is); the PAD SPACE `utf8mb4_bin` would reject the third name, a case-folding one the second.
    probe.items.add(1, "Board")
    probe.items.add(2, "board")
    probe.items.add(3, "Board ")

    assert probe.items.names() == ["Board", "board", "Board "]


def test_an_error_without_a_backend_neutral_meaning_is_not_translated(empty_database: Database) -> None:
    with pytest.raises(DBAPIError) as raised:
        with empty_database.begin(write=False) as conn:
            conn.exec_driver_sql("SELECT * FROM a_table_that_does_not_exist")

    assert not isinstance(raised.value, DatabaseError)


@pytest.mark.sqlite_only
def test_a_trigger_raising_an_error_is_not_a_foreign_key_violation(
    probe: ProbeQueries, empty_database: Database
) -> None:
    # SQLite reports both with the same error code: it enforces ON DELETE RESTRICT with a trigger program.
    with empty_database.begin(write=True) as conn:
        conn.exec_driver_sql(
            "CREATE TRIGGER refuse_names AFTER INSERT ON probe_items BEGIN SELECT RAISE(ABORT, 'refused'); END"
        )

    with pytest.raises(DBAPIError) as raised:
        probe.items.add(1, "a")

    assert not isinstance(raised.value, DatabaseError)
