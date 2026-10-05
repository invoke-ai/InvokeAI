"""Transactions racing each other, for tests of what a write does while another one is in flight.

On MySQL and MariaDB transactions run side by side: a test holds one transaction open at a chosen point and starts
another write, which must wait for it (asserted after half a second) and then sees what it committed. On SQLite a
transaction in flight blocks every other one, so these tests pass there trivially.
"""

import threading
from collections.abc import Callable

import pytest

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import ConflictError
from invokeai.app.services.shared.database.queries import Queries


@pytest.fixture
def lost_races(monkeypatch: pytest.MonkeyPatch) -> list[ConflictError]:
    """Every deadlock a retried transaction of the test loses, including those it then wins on another attempt. (A
    transaction in flight is not retried: it raises what it loses.)"""
    lost: list[ConflictError] = []
    retry_conflicts = Database.retry_conflicts

    def recording(self: Database, transaction: Callable[[], object]) -> object:
        def recorded() -> object:
            try:
                return transaction()
            except ConflictError as error:
                lost.append(error)
                raise

        return retry_conflicts(self, recorded)

    monkeypatch.setattr(Database, "retry_conflicts", recording)
    return lost


def waiting(change: Callable[[], object]) -> Callable[[], list[BaseException]]:
    """Starts `change` on another thread and asserts that half a second later it still waits, for a lock the
    caller's transaction holds. Returns a function that waits for `change` to end and returns what it raised."""
    errors: list[BaseException] = []

    def run() -> None:
        try:
            change()
        except BaseException as e:  # noqa: BLE001 - returned to the caller
            errors.append(e)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=0.5)
    assert thread.is_alive(), f"the change did not wait for the transaction in flight; it raised {errors!r}"

    def ended() -> list[BaseException]:
        thread.join(timeout=30)
        assert not thread.is_alive(), "the change still waits"
        return errors

    return ended


def while_in_flight(
    database: Database, work: Callable[[Queries], object], change: Callable[[], object]
) -> list[BaseException]:
    """Runs `change` while another transaction has done `work` and has not committed; that transaction then commits,
    and `change` goes on. Returns what `change` raised."""
    done = threading.Event()
    proceed = threading.Event()
    failures: list[BaseException] = []

    def in_flight() -> None:
        try:
            with database.queries.transaction() as q:
                work(q)
                done.set()
                proceed.wait(timeout=30)
        except BaseException as e:  # noqa: BLE001 - raised again below
            failures.append(e)
            done.set()

    holder = threading.Thread(target=in_flight)
    holder.start()
    try:
        assert done.wait(timeout=10), "the transaction in flight did not get to its work"
        if failures:
            raise failures[0]
        ended = waiting(change)
    finally:
        proceed.set()
        holder.join(timeout=10)
    if failures:
        raise failures[0]
    return ended()


def when_called(
    monkeypatch: pytest.MonkeyPatch, cls: type, method: str, change: Callable[[], object]
) -> Callable[[], list[BaseException]]:
    """Makes the first call of `cls.method` start `change` when it returns, still in its transaction, and go on once
    `change` waits for that transaction. Call it in a unit of `queries.run()`; the returned function, called after the
    unit, returns what `change` raised."""
    real = getattr(cls, method)
    first = True
    started: list[Callable[[], list[BaseException]]] = []

    def then_change(self: object, *args: object, **kwargs: object) -> object:
        nonlocal first
        result = real(self, *args, **kwargs)
        if first:
            # Cleared before the change starts: a change that calls the method itself must not start another one.
            first = False
            started.append(waiting(change))
        return result

    monkeypatch.setattr(cls, method, then_change)

    def ended() -> list[BaseException]:
        assert started, f"{cls.__name__}.{method} was not called"
        return started[0]()

    return ended
