"""Database errors that do not depend on the backend.

The database layer raises these instead of driver exceptions, so a service can tell a duplicate key from
a missing reference without knowing which database it runs on. Errors are classified by the driver's
error code, not by message text: messages differ between backends and change between versions. The one
exception is a code SQLite shares between two causes, told apart by SQLite's fixed message.
"""

import sqlite3
from typing import Optional

from sqlalchemy.exc import DBAPIError


class DatabaseError(Exception):
    """Base class of the errors raised by the database layer."""


class IntegrityViolation(DatabaseError):
    """A constraint rejected the statement.

    Raised inside a transaction, it ends that transaction: catch it there only to raise an error of your
    own (see `TransactionFailedError`).
    """


class UniqueViolation(IntegrityViolation):
    """A primary key or unique constraint already holds the value."""


class ForeignKeyViolation(IntegrityViolation):
    """A foreign key names a missing row, or a restricting reference blocks the delete."""


class CheckViolation(IntegrityViolation):
    """A CHECK constraint rejected the row."""


class NotNullViolation(IntegrityViolation):
    """A NOT NULL column received no value."""


class TransientDatabaseError(DatabaseError):
    """Concurrent work kept the transaction from completing; it is rolled back, and may be run again."""


class ConflictError(TransientDatabaseError):
    """The transaction lost a race: a deadlock, or a snapshot that another writer invalidated.

    Running it again from the start usually succeeds; a single query call is retried automatically.
    """


class LockTimeoutError(TransientDatabaseError):
    """The transaction gave up waiting for a lock that other work held for too long.

    Not retried automatically: it has already waited, and waiting again would also hold up everything
    queued behind it.
    """


class DatabaseUnavailableError(TransientDatabaseError):
    """No connection to the database server could be made, or none came free in time. Not retried automatically.

    A connection lost during a statement is not this: the server also closes the connection of a statement larger
    than it takes, which no retry would change.
    """


class NestedTransactionError(DatabaseError):
    """A transaction was opened on a thread that already has one open on the same database.

    Nesting is refused rather than joined: an inner commit would commit the outer transaction's work
    early, and on a pooled backend the inner transaction would run on a second connection that neither
    sees the outer one's writes nor can wait for its locks. Use the outer transaction's queries instead.
    """


class DatabaseInUseError(DatabaseError):
    """Another InvokeAI process uses this server database.

    One process at a time serves a database: at startup a process cancels the queue items left running, and
    syncs its bundled workflows and style presets, which would undo or collide with another process's work.
    """


class ReadOnlyTransactionError(DatabaseError):
    """A write was attempted inside a read-only transaction."""


class TransactionFailedError(DatabaseError):
    """The transaction was used after one of its statements failed, or left without raising.

    Backends disagree on what a failed statement leaves behind -- SQLite and MySQL undo just that statement
    (MySQL a whole deadlocked transaction), Postgres refuses everything after it -- so no transaction
    continues past one: it takes no further calls and rolls back instead of committing.
    """


# Extended result codes, which Python's sqlite3 reports in `sqlite_errorcode`.
_SQLITE_CODES: dict[int, type[DatabaseError]] = {
    1555: UniqueViolation,  # SQLITE_CONSTRAINT_PRIMARYKEY
    2067: UniqueViolation,  # SQLITE_CONSTRAINT_UNIQUE
    787: ForeignKeyViolation,  # SQLITE_CONSTRAINT_FOREIGNKEY
    275: CheckViolation,  # SQLITE_CONSTRAINT_CHECK
    1299: NotNullViolation,  # SQLITE_CONSTRAINT_NOTNULL
    517: ConflictError,  # SQLITE_BUSY_SNAPSHOT: a read snapshot cannot become a write after another commit
}
# Primary result codes, matched after the extended ones; their other extended variants share the low byte.
_SQLITE_PRIMARY_CODES: dict[int, type[DatabaseError]] = {
    5: LockTimeoutError,  # SQLITE_BUSY, after the busy timeout ran out
    6: LockTimeoutError,  # SQLITE_LOCKED
}
# SQLite enforces ON DELETE RESTRICT through a trigger program, so a violation carries the code of a trigger's
# RAISE(); only SQLite's fixed foreign key message tells the two apart.
_SQLITE_CONSTRAINT_TRIGGER = 1811
_SQLITE_FOREIGN_KEY_MESSAGE = "FOREIGN KEY constraint failed"

# MySQL and MariaDB server error numbers, the first argument of the PyMySQL exception.
_MYSQL_ERRORS: dict[int, type[DatabaseError]] = {
    1062: UniqueViolation,  # ER_DUP_ENTRY
    1216: ForeignKeyViolation,  # ER_NO_REFERENCED_ROW
    1217: ForeignKeyViolation,  # ER_ROW_IS_REFERENCED
    1451: ForeignKeyViolation,  # ER_ROW_IS_REFERENCED_2
    1452: ForeignKeyViolation,  # ER_NO_REFERENCED_ROW_2
    3819: CheckViolation,  # ER_CHECK_CONSTRAINT_VIOLATED (MySQL)
    4025: CheckViolation,  # ER_CONSTRAINT_FAILED (MariaDB)
    1048: NotNullViolation,  # ER_BAD_NULL_ERROR
    1364: NotNullViolation,  # ER_NO_DEFAULT_FOR_FIELD
    1213: ConflictError,  # ER_LOCK_DEADLOCK
    1205: LockTimeoutError,  # ER_LOCK_WAIT_TIMEOUT
    2003: DatabaseUnavailableError,  # CR_CONN_HOST_ERROR
}


def translate_error(error: BaseException, dialect_name: str) -> Optional[DatabaseError]:
    """The backend-neutral error for a driver error, or None when it has no backend-neutral meaning.

    :param error: A SQLAlchemy `DBAPIError`, or the driver's own exception.
    """
    original = error.orig if isinstance(error, DBAPIError) else error
    if dialect_name == "sqlite" and isinstance(original, sqlite3.Error):
        code = getattr(original, "sqlite_errorcode", None)
        if code is None:
            return None
        if code == _SQLITE_CONSTRAINT_TRIGGER and str(original) == _SQLITE_FOREIGN_KEY_MESSAGE:
            return ForeignKeyViolation(str(original))
        neutral = _SQLITE_CODES.get(code) or _SQLITE_PRIMARY_CODES.get(code & 0xFF)
        return neutral(str(original)) if neutral is not None else None
    if dialect_name in ("mysql", "mariadb"):
        args = getattr(original, "args", ())
        if args and isinstance(args[0], int) and args[0] in _MYSQL_ERRORS:
            return _MYSQL_ERRORS[args[0]](str(original))
    return None
