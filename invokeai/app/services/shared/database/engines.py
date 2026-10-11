"""Engine construction for each supported backend."""

import sqlite3
from typing import Any

from sqlalchemy import URL, Connection, Engine, create_engine, event
from sqlalchemy.engine import ExceptionContext
from sqlalchemy.pool import AssertionPool

# The dialect names of the server backends. MariaDB must be reached through a `mariadb+` URL: SQLAlchemy names
# the dialect after the URL, and the schema's per-flavor DDL (collations, generated columns) follows that name.
SERVER_DIALECTS = ("mysql", "mariadb")

# Execution option carrying a transaction's write intent from `Database.begin()` to the MySQL begin hook.
WRITE_INTENT = "invokeai_write_intent"

# Pinned per connection so behaviour does not depend on how the server was configured. Strict mode turns
# truncation and invalid values into errors; ONLY_FULL_GROUP_BY rejects the ambiguous grouping SQLite allows.
MYSQL_SQL_MODE = (
    "STRICT_TRANS_TABLES,ONLY_FULL_GROUP_BY,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,"
    "NO_ENGINE_SUBSTITUTION"
)

# Binary, NO PAD collations: they compare byte for byte and treat trailing spaces as significant, as
# SQLite's default BINARY collation does. (`utf8mb4_bin` is PAD SPACE, so 'a ' would equal 'a'.)
MYSQL_BINARY_COLLATION = "utf8mb4_0900_bin"
MARIADB_BINARY_COLLATION = "utf8mb4_nopad_bin"

# PyMySQL's SERVER_STATUS_IN_TRANS flag: the server reports an open transaction on this connection.
_SERVER_STATUS_IN_TRANS = 0x0001


def create_sqlite_engine(conn: sqlite3.Connection) -> Engine:
    """An engine over exactly one existing SQLite connection.

    The SQLite database is one connection behind a process-wide lock, and `Database` holds the one
    SQLAlchemy connection over it; work goes through `Database.begin()`, which takes the lock and begins
    every transaction explicitly.

    The pool allows a single checkout, which that connection holds, so connecting through the engine
    directly raises. A second connection object over the same driver connection would run outside the lock,
    and closing it would roll back whatever transaction happened to be open on the shared connection.
    """
    engine = create_engine(
        "sqlite+pysqlite://", creator=lambda: conn, poolclass=AssertionPool, pool_reset_on_return=None
    )
    event.listen(engine, "handle_error", _never_disconnected)
    return engine


def _never_disconnected(context: ExceptionContext) -> None:
    # SQLAlchemy takes an exception that is not an `Exception` (KeyboardInterrupt, SystemExit) raised during
    # a statement for a lost connection and closes it. A local file cannot be disconnected, and closing the
    # only connection would end the database -- an in-memory one with all its data. The transaction around
    # the statement rolls back instead.
    context.is_disconnect = False


def create_mysql_engine(url: URL) -> Engine:
    """A pooled engine for MySQL or MariaDB.

    Reads run at the server's REPEATABLE READ, so a read that issues several statements sees one snapshot,
    as SQLite's readers do. Writers switch to READ COMMITTED, which takes no gap locks and so deadlocks far
    less under concurrent writes; their invariants are enforced by the statements themselves.

    Each checkout pings the server, so a connection the server dropped while idle is replaced instead of
    failing the first statement.
    """
    engine = create_engine(
        url,
        pool_size=10,
        max_overflow=10,
        pool_timeout=30,
        pool_pre_ping=True,
        # Below the servers' default `wait_timeout` of eight hours, with margin for proxies that cut sooner.
        pool_recycle=1800,
        # The reset hook below rolls back only when needed.
        pool_reset_on_return=None,
        isolation_level="REPEATABLE READ",
        connect_args={"charset": "utf8mb4"},
    )
    # First, so the session settings are in place before SQLAlchemy inspects the connection (it reads
    # sql_mode, e.g. ANSI_QUOTES, to decide how to quote identifiers).
    event.listen(engine, "connect", _configure_mysql_session, insert=True)
    expects_mariadb = engine.dialect.name == "mariadb"

    def set_collation(dbapi_connection: Any, connection_record: object) -> None:
        _set_mysql_collation(dbapi_connection, expects_mariadb)

    # Last, because SQLAlchemy's own connection setup sends a plain `SET NAMES`, resetting the collation.
    event.listen(engine, "connect", set_collation)
    event.listen(engine, "begin", _begin_mysql_transaction)
    event.listen(engine, "reset", _reset_mysql_connection)
    return engine


def _configure_mysql_session(dbapi_connection: Any, connection_record: object) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute(f"SET SESSION sql_mode = '{MYSQL_SQL_MODE}'")
        cursor.execute("SET SESSION time_zone = '+00:00'")
        cursor.execute("SET SESSION innodb_lock_wait_timeout = 10")
        # Metadata locks (DDL waiting for open transactions) default to waiting a year.
        cursor.execute("SET SESSION lock_wait_timeout = 60")
    finally:
        cursor.close()


def _set_mysql_collation(dbapi_connection: Any, expects_mariadb: bool) -> None:
    is_mariadb = "mariadb" in dbapi_connection.get_server_info().lower()
    if is_mariadb and not expects_mariadb:
        # (The reverse, a mariadb+ URL for MySQL, SQLAlchemy refuses itself.)
        raise ValueError(
            "The database server is MariaDB, but the URL names MySQL: use a mariadb+pymysql:// URL, so that "
            "tables are created with MariaDB's collations and column definitions"
        )
    # How literals and parameters compare; columns compare by their table's collation.
    collation = MARIADB_BINARY_COLLATION if is_mariadb else MYSQL_BINARY_COLLATION
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute(f"SET NAMES utf8mb4 COLLATE {collation}")
    finally:
        cursor.close()


def _begin_mysql_transaction(connection: Connection) -> None:
    if connection.get_execution_options().get(WRITE_INTENT, True):
        # Applies to the next transaction only, which is the one beginning now.
        connection.exec_driver_sql("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")


def _reset_mysql_connection(dbapi_connection: Any, connection_record: object, reset_state: Any) -> None:
    # The pool's own reset sends ROLLBACK on every return: one more round trip per call. A connection comes
    # back with a transaction open only after something went wrong, so roll back only then.
    if reset_state.terminate_only:
        return
    if getattr(dbapi_connection, "server_status", _SERVER_STATUS_IN_TRANS) & _SERVER_STATUS_IN_TRANS:
        dbapi_connection.rollback()
