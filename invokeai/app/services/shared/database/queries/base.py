"""What every query module is built from: `QueryModule`, the `read`, `write` and `mapped` decorators, and their scopes."""

import functools
from collections.abc import Callable
from datetime import date, timedelta
from typing import TYPE_CHECKING, Concatenate, ParamSpec, Protocol, TypeVar

from sqlalchemy import Connection
from sqlalchemy.exc import DBAPIError

from invokeai.app.services.shared.database.errors import (
    ReadOnlyTransactionError,
    TransactionFailedError,
    translate_error,
)

if TYPE_CHECKING:
    from invokeai.app.services.shared.database.database import Database

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")
M = TypeVar("M", bound="QueryModule")

# The most values one `IN (...)` list binds; longer lists are queried in chunks of this size. SQLite builds
# before 3.32 accept at most 999 parameters per statement.
IN_CHUNK = 500


class QueryScope(Protocol):
    """Where the calls of a query module run: in transactions of their own, or in a shared one."""

    def run(self, work: Callable[[Connection], R], *, write: bool) -> R: ...


class OwnTransaction:
    """Each call opens and commits a transaction of its own, retried when it loses a race."""

    def __init__(self, database: "Database") -> None:
        self._database = database

    def run(self, work: Callable[[Connection], R], *, write: bool) -> R:
        return self._database.run(work, write=write)


class SharedTransaction:
    """Calls run on the connection of an open transaction."""

    def __init__(self, conn: Connection, *, writable: bool, dialect_name: str) -> None:
        self._conn = conn
        self._writable = writable
        self._dialect_name = dialect_name
        self._open = True
        self.failed = False
        # The calls run in this transaction so far, the running one included.
        self.calls = 0

    def run(self, work: Callable[[Connection], R], *, write: bool) -> R:
        self.calls += 1
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
    """Base of the per-domain query modules.

    Build each statement once, at module level, with `bindparam()` for its values, and execute it with
    them. Building a statement costs several times what executing it does: SQLAlchemy compiles it once,
    but derives its cache key on every execution of a new statement object.
    """

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


def locking(method: Callable[Concatenate[M, Connection, P], R]) -> Callable[Concatenate[M, P], R]:
    """Marks a query method that locks rows until its transaction ends (`SELECT ... FOR UPDATE`; a SQLite transaction
    excludes every other one already). It runs as a write, and only in a `transaction()`: in a transaction of its
    own, the lock would be released as soon as it was taken."""

    @functools.wraps(method)
    def call(self: M, /, *args: P.args, **kwargs: P.kwargs) -> R:
        if not isinstance(self._scope, SharedTransaction):
            raise RuntimeError(f"{method.__qualname__} locks rows, which only a transaction() holds")
        return self._scope.run(lambda conn: method(self, conn, *args, **kwargs), write=True)

    return call


def mapped(mapper: Callable[[T], R]) -> Callable[[Callable[Concatenate[M, P], T]], Callable[Concatenate[M, P], R]]:
    """Applies `mapper` to what a `read` or `write` method returns, outside its transaction when it has one of its own.

    For building DTOs from rows: on SQLite a transaction holds the database's process-wide lock, and validating a
    DTO costs more than reading its row. Put it above `@read` or `@write`, on a method that returns its rows read
    completely (`first()`, `all()`): they outlive the transaction.
    """

    def decorate(method: Callable[Concatenate[M, P], T]) -> Callable[Concatenate[M, P], R]:
        @functools.wraps(method)
        def call(self: M, /, *args: P.args, **kwargs: P.kwargs) -> R:
            return mapper(method(self, *args, **kwargs))

        return call

    return decorate


def day_after(day: str) -> str:
    """The day after an ISO day (`YYYY-MM-DD`), as text that timestamps of that day sort before. An invalid day, or
    the last one there is, gives the empty text, which no timestamp sorts before: SQLite's DATE() gave NULL for both,
    which matched nothing."""
    try:
        parsed = date.fromisoformat(day)
        return (parsed + timedelta(days=1)).isoformat() if parsed.isoformat() == day else ""
    except (ValueError, OverflowError):
        return ""
