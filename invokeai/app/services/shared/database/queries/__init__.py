"""Every query of the application, one module per domain.

A query module subclasses `QueryModule`. Each of its methods takes the connection right after `self` and
is decorated with `read` or `write`, which supply it, so callers never see a connection:

    board = db.queries.boards.get(board_id)  # a transaction of its own
    with db.queries.transaction() as q:  # one transaction for several calls
        q.boards.insert(board)
        q.projects.insert(project)

A query method returns plain values or DTOs, read completely before its transaction ends. It does not call
other query modules -- composition belongs to the caller's transaction -- and has no effects outside the
database, because a call that loses a race is run again from the start. It passes execution options per
statement (`conn.execute(statement, execution_options=...)`), never to the connection: on SQLite every
transaction shares one connection object, so an option set on it would stay for all later transactions.
"""

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Concatenate, Optional, ParamSpec, Protocol, Self, TypeVar

from sqlalchemy import Connection
from sqlalchemy.exc import DBAPIError

from invokeai.app.services.shared.database.errors import (
    NestedTransactionError,
    ReadOnlyTransactionError,
    TransactionFailedError,
    translate_error,
)

if TYPE_CHECKING:
    from invokeai.app.services.shared.database.database import Database

P = ParamSpec("P")
R = TypeVar("R")
M = TypeVar("M", bound="QueryModule")


class QueryScope(Protocol):
    """Where the calls of a query module run: in transactions of their own, or in a shared one."""

    def run(self, work: Callable[[Connection], R], *, write: bool) -> R: ...


class _OwnTransaction:
    """Each call opens and commits a transaction of its own, retried when it loses a race."""

    def __init__(self, database: "Database") -> None:
        self._database = database

    def run(self, work: Callable[[Connection], R], *, write: bool) -> R:
        return self._database.run(work, write=write)


class _SharedTransaction:
    """Calls run on the connection of an open transaction."""

    def __init__(self, conn: Connection, *, writable: bool, dialect_name: str) -> None:
        self._conn = conn
        self._writable = writable
        self._dialect_name = dialect_name
        self._open = True
        self.failed = False

    def run(self, work: Callable[[Connection], R], *, write: bool) -> R:
        if not self._open:
            raise RuntimeError("These queries belong to a transaction that has already ended")
        if self.failed:
            raise TransactionFailedError("A statement of this transaction failed; leave the transaction")
        if write and not self._writable:
            raise ReadOnlyTransactionError("A read-only transaction cannot write")
        try:
            return work(self._conn)
        except DBAPIError as error:
            # Backends disagree on what a failed statement leaves behind: SQLite and MySQL undo just the
            # statement (MySQL undoes a whole deadlocked transaction), Postgres refuses the rest of the
            # transaction. So none continues; and the error is translated here, not only at the end, so a
            # caller inside the transaction can catch it to raise its own.
            self.failed = True
            translated = translate_error(error, self._dialect_name)
            if translated is None:
                raise
            raise translated from error

    def close(self) -> None:
        self._open = False


class QueryModule:
    """Base of the per-domain query modules."""

    def __init__(self, scope: QueryScope) -> None:
        self._scope = scope


def read(method: Callable[Concatenate[M, Connection, P], R]) -> Callable[Concatenate[M, P], R]:
    """Marks a query method that only reads. Run on its own, it gets a read transaction (on SQLite, BEGIN)."""

    @functools.wraps(method)
    def call(self: M, /, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._scope.run(lambda conn: method(self, conn, *args, **kwargs), write=False)

    return call


def write(method: Callable[Concatenate[M, Connection, P], R]) -> Callable[Concatenate[M, P], R]:
    """Marks a query method that writes. Run on its own, it gets a write transaction (on SQLite, BEGIN IMMEDIATE)."""

    @functools.wraps(method)
    def call(self: M, /, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._scope.run(lambda conn: method(self, conn, *args, **kwargs), write=True)

    return call


class Queries:
    """Every query of the application, grouped by domain as `queries.<domain>.<method>(...)`.

    `Database.queries` runs each call in a transaction of its own. The queries yielded by `transaction()`
    share one transaction, which commits when the block exits.
    """

    def __init__(self, database: "Database", scope: Optional[QueryScope] = None) -> None:
        self._database = database
        self._scope: QueryScope = scope if scope is not None else _OwnTransaction(database)

    @contextmanager
    def transaction(self, *, read_only: bool = False) -> Iterator[Self]:
        """Runs every call on the yielded queries in one transaction, committed when the block exits normally.

        A read-only transaction sees one consistent snapshot and refuses writes. A transaction is not
        retried. After a statement failed, the transaction takes no further calls and does not commit:
        leave the block by raising (a caught `DatabaseError` can be turned into a domain error there).
        """
        if isinstance(self._scope, _SharedTransaction):
            raise NestedTransactionError("These queries already belong to a transaction")
        with self._database.begin(write=not read_only) as conn:
            scope = _SharedTransaction(conn, writable=not read_only, dialect_name=self._database.dialect_name)
            try:
                yield type(self)(self._database, scope)
                if scope.failed:
                    raise TransactionFailedError(
                        "A statement of this transaction failed and the error was not raised further; "
                        "the transaction was rolled back"
                    )
            finally:
                scope.close()
