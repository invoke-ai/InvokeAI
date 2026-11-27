"""A lock of a MySQL or MariaDB server (`GET_LOCK`), held by the session of a connection of its own."""

import hashlib
import threading
from logging import Logger
from typing import Optional

from sqlalchemy import Connection, Engine
from sqlalchemy.exc import DBAPIError

from invokeai.app.services.shared.database.errors import DatabaseInUseError


class SessionLock:
    """A server-wide lock on one database, for one purpose, held by the session of a connection that does nothing
    else. The server releases it when that session ends: when the lock is closed, when the process ends, and also
    when the connection is cut off (an idle timeout, a restart, a network failure). So a holder checks with `held()`
    rather than assuming it.
    """

    def __init__(self, engine: Engine, purpose: str) -> None:
        self._conn: Connection = engine.connect()
        try:
            self.database_name = str(self._conn.exec_driver_sql("SELECT DATABASE()").scalar_one())
            self._conn.commit()
        except BaseException:
            self.close()
            raise
        # Lock names are server-wide and at most 64 characters long.
        self.name = f"invokeai.{purpose}.{hashlib.sha1(self.database_name.encode()).hexdigest()}"

    def take(self, timeout_seconds: int) -> bool:
        """Takes the lock, waiting at most `timeout_seconds`; whether it did. Raises when the server could not try."""
        taken = self._conn.exec_driver_sql("SELECT GET_LOCK(%s, %s)", (self.name, timeout_seconds)).scalar()
        self._conn.commit()
        if taken is None:
            raise RuntimeError(f"The database server could not take the lock {self.name}")
        return taken == 1

    def held(self) -> bool:
        """Whether this lock's session still holds it. A check also keeps the session from going idle."""
        try:
            held = self._conn.exec_driver_sql("SELECT IS_USED_LOCK(%s) = CONNECTION_ID()", (self.name,)).scalar()
            self._conn.commit()
        except DBAPIError:
            return False
        return held == 1

    def set_idle_timeout(self, seconds: int) -> None:
        """Lets the server end this lock's session, and release the lock, after `seconds` without a statement: how
        long a lock outlives a host that stopped without closing its connections."""
        self._conn.exec_driver_sql(f"SET SESSION wait_timeout = {int(seconds)}")
        self._conn.commit()

    def close(self) -> None:
        """Releases the lock by ending its session. (A connection returned to the pool would keep both.)"""
        self._conn.invalidate()
        self._conn.close()


# How often a running process checks that it still holds its instance lock, which also keeps the lock's session
# from going idle.
INSTANCE_LOCK_CHECK_SECONDS = 60
# How long the server keeps the instance lock of a process whose host stopped without closing its connection.
INSTANCE_LOCK_IDLE_TIMEOUT_SECONDS = 600


class InstanceLock:
    """The lock one InvokeAI process at a time holds on a server database, for as long as it serves it.

    A thread checks it every `INSTANCE_LOCK_CHECK_SECONDS`, and takes it again when the server has ended its
    session; when another process has taken it meanwhile, it logs an error, as two processes then serve the database.
    """

    def __init__(self, engine: Engine, logger: Logger) -> None:
        self._engine = engine
        self._logger = logger
        self._guard = threading.Lock()
        self._closed = threading.Event()
        lock, refused = self._take()
        if lock is None:
            raise DatabaseInUseError(
                f"Another InvokeAI process uses the database {refused.database_name!r}. Stop it first: one process at "
                "a time serves a database. If none runs, the server still holds the lock of one that stopped without "
                f"closing its connection, and releases it within {INSTANCE_LOCK_IDLE_TIMEOUT_SECONDS // 60} minutes; "
                f"or end that connection now: SELECT IS_USED_LOCK('{refused.name}') names the one to KILL."
            )
        self._lock: Optional[SessionLock] = lock
        threading.Thread(target=self._keep, name="database-instance-lock", daemon=True).start()

    def check(self) -> None:
        """Verifies the lock, and takes it again when the server has released it."""
        with self._guard:
            if self._closed.is_set() or (self._lock is not None and self._lock.held()):
                return
            if self._lock is not None:
                self._lock.close()
                self._lock = None
                self._logger.warning(
                    "The database server ended the session holding this process's lock on the database"
                )
            try:
                lock, refused = self._take()
            except Exception as e:  # noqa: BLE001 - checked again on the next round
                self._logger.error(f"Could not take the lock on the database again: {e}")
                return
            if lock is None:
                self._logger.error(
                    f"Another InvokeAI process has taken the lock on the database {refused.database_name!r}: two "
                    "processes now serve it, and undo each other's work. Stop one of them."
                )
                return
            self._lock = lock
            self._logger.info("Took the lock on the database again")

    def close(self) -> None:
        self._closed.set()
        with self._guard:
            if self._lock is not None:
                self._lock.close()
                self._lock = None

    def _take(self) -> tuple[Optional[SessionLock], SessionLock]:
        """The lock if this process took it, and the lock it tried (closed when another process holds it)."""
        lock = SessionLock(self._engine, "instance")
        try:
            if lock.take(0):
                lock.set_idle_timeout(INSTANCE_LOCK_IDLE_TIMEOUT_SECONDS)
                return lock, lock
        except BaseException:
            lock.close()
            raise
        lock.close()
        return None, lock

    def _keep(self) -> None:
        while not self._closed.wait(INSTANCE_LOCK_CHECK_SECONDS):
            try:
                self.check()
            except Exception as e:  # noqa: BLE001 - the thread outlives one failed round
                self._logger.error(f"Could not check the lock on the database: {e}")
