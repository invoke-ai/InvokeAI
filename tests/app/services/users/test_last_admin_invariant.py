"""The instance must never be left with zero active administrators.

Authorization is derived from the database on every request, so removing the last active
admin is irreversible from inside the app: no authenticated path back exists. Worse, it
drops `has_admin()` to zero, which makes `GET /auth/status` report `setup_required: true`
and re-opens the **unauthenticated** `POST /auth/setup` to any caller.

The guard used to live only in the `delete_user` route, where it read `count_admins()` in
its own transaction and then wrote in another. That is a TOCTOU: two callers each observe
two admins and each remove one. It is reachable three ways —

  * two concurrent requests, now that route handlers run in a threadpool;
  * the `invoke-usermod` / `invoke-userdel` CLIs, which construct `UserService` directly
    and never reach the route guard at all;
  * a second process racing the server on the same database file.

so the invariant belongs in the service, inside the transaction that performs the write.
"""

import threading
from collections.abc import Callable
from typing import Any

import pytest
from sqlalchemy import update

from invokeai.app.services.auth.password_utils import hash_password
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.database.queries.locks import DatabaseLock
from invokeai.app.services.shared.database.queries.users import UserQueries
from invokeai.app.services.shared.database.schema.users import users as users_table
from invokeai.app.services.users import users_default
from invokeai.app.services.users.users_common import (
    SYSTEM_USER_ID,
    LastAdministratorError,
    SystemUserProtectedError,
    UserCreateRequest,
    UserUpdateRequest,
)
from invokeai.app.services.users.users_default import UserService

PASSWORD = "Sup3rSecret!pass"


@pytest.fixture
def users(database: Database) -> UserService:
    return UserService(database)


def _make(users: UserService, email: str, *, is_admin: bool) -> str:
    user = users.create(
        UserCreateRequest(email=email, display_name=email, password=PASSWORD, is_admin=is_admin),
        strict_password_checking=False,
    )
    return user.user_id


# region single caller


def test_deleting_the_last_admin_is_rejected(users: UserService) -> None:
    admin = _make(users, "admin@test.com", is_admin=True)
    _make(users, "plain@test.com", is_admin=False)

    with pytest.raises(LastAdministratorError):
        users.delete(admin)

    assert users.count_admins() == 1


def test_demoting_the_last_admin_is_rejected(users: UserService) -> None:
    """The gap this PR closes: only `delete` was ever guarded."""
    admin = _make(users, "admin@test.com", is_admin=True)

    with pytest.raises(LastAdministratorError):
        users.update(admin, UserUpdateRequest(is_admin=False), strict_password_checking=False)

    assert users.count_admins() == 1
    assert users.get(admin).is_admin is True


def test_deactivating_the_last_admin_is_rejected(users: UserService) -> None:
    """Deactivation removes an admin from `count_admins()` just as demotion does."""
    admin = _make(users, "admin@test.com", is_admin=True)

    with pytest.raises(LastAdministratorError):
        users.update(admin, UserUpdateRequest(is_active=False), strict_password_checking=False)

    assert users.count_admins() == 1
    assert users.get(admin).is_active is True


def test_the_error_is_a_value_error(users: UserService) -> None:
    """Route handlers and the CLIs already map service `ValueError` to a friendly message."""
    admin = _make(users, "admin@test.com", is_admin=True)

    with pytest.raises(ValueError):
        users.delete(admin)


# endregion

# region changes that must still be allowed


def test_renaming_the_last_admin_is_allowed(users: UserService) -> None:
    """The guard keys on the requested values, not on the target being an admin."""
    admin = _make(users, "admin@test.com", is_admin=True)

    updated = users.update(admin, UserUpdateRequest(display_name="Renamed"), strict_password_checking=False)

    assert updated.display_name == "Renamed"
    assert updated.is_admin is True


def test_password_change_for_the_last_admin_is_allowed(users: UserService) -> None:
    admin = _make(users, "admin@test.com", is_admin=True)

    users.update(admin, UserUpdateRequest(password="An0ther!Password"), strict_password_checking=False)

    assert users.authenticate("admin@test.com", "An0ther!Password") is not None


def test_demoting_one_of_two_admins_is_allowed(users: UserService) -> None:
    first = _make(users, "a1@test.com", is_admin=True)
    _make(users, "a2@test.com", is_admin=True)

    users.update(first, UserUpdateRequest(is_admin=False), strict_password_checking=False)

    assert users.count_admins() == 1


def test_deleting_an_already_inactive_admin_is_allowed(users: UserService) -> None:
    """An inactive admin is not counted, so removing them cannot reach zero."""
    active = _make(users, "active@test.com", is_admin=True)
    inactive = _make(users, "inactive@test.com", is_admin=True)
    users.update(inactive, UserUpdateRequest(is_active=False), strict_password_checking=False)
    assert users.count_admins() == 1

    users.delete(inactive)

    assert users.get(inactive) is None
    assert users.get(active) is not None


def test_deleting_a_non_admin_is_allowed(users: UserService) -> None:
    _make(users, "admin@test.com", is_admin=True)
    plain = _make(users, "plain@test.com", is_admin=False)

    users.delete(plain)

    assert users.get(plain) is None


# endregion

# region concurrency — the reason the guard moved into the transaction


def _race(target, args_a, args_b) -> list[BaseException | None]:
    """Run `target` twice concurrently, returning each call's exception (or None)."""
    results: list[BaseException | None] = [None, None]
    barrier = threading.Barrier(2)

    def run(index: int, args: tuple) -> None:
        barrier.wait()
        try:
            target(*args)
        except BaseException as e:  # noqa: BLE001 - recorded and asserted on below
            results[index] = e

    threads = [
        threading.Thread(target=run, args=(0, args_a)),
        threading.Thread(target=run, args=(1, args_b)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in threads), "a racing thread deadlocked"
    return results


def test_concurrent_deletes_cannot_remove_both_admins(users: UserService) -> None:
    """Two admins, two concurrent deletes of *different* rows. Exactly one must survive.

    With the guard in the route this failed: both callers read `count_admins() == 2`,
    both passed, and the instance was left with none.
    """
    first = _make(users, "a1@test.com", is_admin=True)
    second = _make(users, "a2@test.com", is_admin=True)

    errors = _race(users.delete, (first,), (second,))

    assert users.count_admins() == 1, "both concurrent deletes succeeded; the invariant is not atomic"
    assert sum(isinstance(e, LastAdministratorError) for e in errors) == 1


def test_concurrent_demotions_cannot_remove_both_admins(users: UserService) -> None:
    first = _make(users, "a1@test.com", is_admin=True)
    second = _make(users, "a2@test.com", is_admin=True)
    demote = UserUpdateRequest(is_admin=False)

    def update(user_id: str) -> None:
        users.update(user_id, demote, strict_password_checking=False)

    errors = _race(update, (first,), (second,))

    assert users.count_admins() == 1
    assert sum(isinstance(e, LastAdministratorError) for e in errors) == 1


def test_concurrent_delete_and_demotion_cannot_remove_both_admins(users: UserService) -> None:
    """The two paths must be mutually exclusive, not just each internally consistent."""
    first = _make(users, "a1@test.com", is_admin=True)
    second = _make(users, "a2@test.com", is_admin=True)

    def demote(user_id: str) -> None:
        users.update(user_id, UserUpdateRequest(is_admin=False), strict_password_checking=False)

    errors = _race(lambda uid: users.delete(uid) if uid == first else demote(uid), (first,), (second,))

    assert users.count_admins() == 1
    assert sum(isinstance(e, LastAdministratorError) for e in errors) == 1


def _while_another_transaction_holds_the_lock(
    database: Database, change: Callable[[], None], holders_work: Callable[[Queries], None]
) -> list[BaseException]:
    """Runs `change` while another transaction holds `ADMIN_ACCOUNTS`; that transaction then does `holders_work`
    and commits, and `change` goes on. Returns what `change` raised."""
    locked = threading.Event()
    proceed = threading.Event()

    def hold() -> None:
        with database.queries.transaction() as q:
            q.locks.acquire(DatabaseLock.ADMIN_ACCOUNTS)
            locked.set()
            proceed.wait(timeout=30)
            holders_work(q)

    errors: list[BaseException] = []

    def run_change() -> None:
        try:
            change()
        except BaseException as e:  # noqa: BLE001 - asserted on by the caller
            errors.append(e)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert locked.wait(timeout=10), "the lock was not taken"
        changing = threading.Thread(target=run_change)
        changing.start()
        changing.join(timeout=0.5)
        assert changing.is_alive(), "the change did not wait for the lock"
    finally:
        proceed.set()
        holder.join(timeout=10)
    changing.join(timeout=10)
    assert not changing.is_alive()
    return errors


@pytest.fixture
def quick_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hashing takes a large part of a second on its own; the tests below time how long a change waits."""
    monkeypatch.setattr(users_default, "hash_password", lambda password: f"hash of {password}")


@pytest.mark.parametrize("change", ["delete", "demote", "deactivate"])
def test_a_waiting_change_counts_the_admins_the_lock_holder_left(
    database: Database, users: UserService, change: str
) -> None:
    """On a server transactions run side by side: what decides is that each change takes the lock before it
    counts, so it counts what the transaction holding the lock committed. (On SQLite the held transaction blocks
    every other one anyway.)"""
    first = _make(users, "a1@test.com", is_admin=True)
    second = _make(users, "a2@test.com", is_admin=True)

    def revoke_second() -> None:
        if change == "delete":
            users.delete(second)
        else:
            revoke = UserUpdateRequest(is_admin=False) if change == "demote" else UserUpdateRequest(is_active=False)
            users.update(second, revoke, strict_password_checking=False)

    errors = _while_another_transaction_holds_the_lock(
        database, revoke_second, lambda q: q.users.update(first, is_admin=False)
    )

    assert [type(e) for e in errors] == [LastAdministratorError]
    assert users.count_admins() == 1


@pytest.mark.usefixtures("quick_hashing")
def test_a_waiting_first_admin_setup_sees_the_admin_the_lock_holder_created(
    database: Database, users: UserService
) -> None:
    def setup() -> None:
        users.create_admin(
            UserCreateRequest(email="second@test.com", password=PASSWORD), strict_password_checking=False
        )

    errors = _while_another_transaction_holds_the_lock(
        database,
        setup,
        lambda q: q.users.insert(
            user_id="first", email="first@test.com", display_name=None, password_hash="hash", is_admin=True
        ),
    )

    assert [str(e) for e in errors] == ["Admin user already exists"]
    assert users.count_admins() == 1


@pytest.mark.usefixtures("quick_hashing")
@pytest.mark.parametrize("change", ["create_admin_account", "promote", "reactivate"])
def test_first_admin_setup_waits_for_an_admin_being_added(
    database: Database, users: UserService, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """`invoke-useradd --admin` or `invoke-usermod` can add an administrator while the unauthenticated setup is
    open: the setup must not count before that change has committed."""
    if change == "promote":
        target = _make(users, "user@test.com", is_admin=False)
    elif change == "reactivate":
        target = _make(users, "admin@test.com", is_admin=True)
        with database.queries.transaction() as q:
            q.users.update(target, is_active=False)
    written = threading.Event()
    proceed = threading.Event()
    patched = "insert" if change == "create_admin_account" else "update"
    real = getattr(UserQueries, patched)

    def write_then_wait(self: UserQueries, *args: Any, **kwargs: Any) -> None:
        real(self, *args, **kwargs)
        written.set()
        assert proceed.wait(timeout=30)

    monkeypatch.setattr(UserQueries, patched, write_then_wait)

    def add_admin() -> None:
        if change == "create_admin_account":
            users.create(UserCreateRequest(email="added@test.com", password=PASSWORD, is_admin=True), False)
        else:
            added = UserUpdateRequest(is_admin=True) if change == "promote" else UserUpdateRequest(is_active=True)
            users.update(target, added, strict_password_checking=False)

    adding = threading.Thread(target=add_admin)
    adding.start()
    errors: list[BaseException] = []

    def setup() -> None:
        try:
            users.create_admin(UserCreateRequest(email="setup@test.com", password=PASSWORD), False)
        except BaseException as e:  # noqa: BLE001 - asserted on below
            errors.append(e)

    try:
        assert written.wait(timeout=10), "the admin was not added"
        setting_up = threading.Thread(target=setup)
        setting_up.start()
        setting_up.join(timeout=0.5)
        assert setting_up.is_alive(), "the setup did not wait for the lock"
    finally:
        proceed.set()
        adding.join(timeout=10)
    setting_up.join(timeout=10)

    assert [str(e) for e in errors] == ["Admin user already exists"]
    assert users.count_admins() == 1


# endregion

# region the system account


def test_the_system_user_cannot_be_promoted(users: UserService) -> None:
    """`count_admins()` counts admin rows, but the invariant that matters is "an admin who
    can log in". The system row is active and can never authenticate — it has no password —
    so promoting it would inflate the count with an unusable administrator, which is enough
    to walk the last-admin guard past the real one:

        PATCH /auth/users/system      {"is_admin": true}   -> count_admins() 1 -> 2
        PATCH /auth/users/{real}      {"is_admin": false}  -> allowed, count 2 -> 1
        login as system                                    -> 401, empty password hash

    leaving the instance with no usable administration and no authenticated way back.
    """
    admin = _make(users, "admin@test.com", is_admin=True)

    with pytest.raises(SystemUserProtectedError):
        users.update(SYSTEM_USER_ID, UserUpdateRequest(is_admin=True), strict_password_checking=False)

    assert users.count_admins() == 1

    # And with the first step refused, the second is still blocked.
    with pytest.raises(LastAdministratorError):
        users.update(admin, UserUpdateRequest(is_admin=False), strict_password_checking=False)


def test_the_system_user_cannot_be_given_a_password(users: UserService) -> None:
    """The other end of the same hole: a password turns the owner of every pre-multiuser
    board, image, and workflow into a login account."""

    with pytest.raises(SystemUserProtectedError):
        users.update(SYSTEM_USER_ID, UserUpdateRequest(password=PASSWORD), strict_password_checking=False)

    assert users.authenticate("system@system.invokeai", PASSWORD) is None


def test_the_system_user_cannot_be_deleted_or_deactivated(users: UserService) -> None:
    """The routes already refuse both, but `invoke-userdel` / `invoke-usermod` construct
    this service directly and never reach a route — the same reason the last-admin guard
    lives here."""

    with pytest.raises(SystemUserProtectedError):
        users.delete(SYSTEM_USER_ID)
    with pytest.raises(SystemUserProtectedError):
        users.update(SYSTEM_USER_ID, UserUpdateRequest(is_active=False), strict_password_checking=False)

    system = users.get(SYSTEM_USER_ID)
    assert system is not None and system.is_active is True


def test_renaming_the_system_user_is_allowed(users: UserService) -> None:
    """Not a blanket lock on the row — only the changes that would make it dangerous."""

    updated = users.update(SYSTEM_USER_ID, UserUpdateRequest(display_name="Renamed"), strict_password_checking=False)

    assert updated.display_name == "Renamed"


def test_a_system_row_carrying_a_password_still_cannot_log_in(database: Database, users: UserService) -> None:
    """The guard above only stops a password being set *from now on*.

    An instance that set one through the old `PATCH /auth/users/system` hole still carries
    a usable hash, and its email is fixed and public — so the hash is a standing login for
    the account that owns every pre-multiuser board, image, workflow, and queue item. The
    migration clears it, but migrations run once and cannot reach a row damaged afterwards
    by direct SQL, so `authenticate` refuses the account outright whatever the row holds.
    """
    with database.begin(write=True) as conn:
        conn.execute(
            update(users_table)
            .where(users_table.c.user_id == SYSTEM_USER_ID)
            .values(password_hash=hash_password(PASSWORD))
        )

    assert users.authenticate("system@system.invokeai", PASSWORD) is None


def test_refusing_the_system_account_does_not_block_other_logins(users: UserService) -> None:
    """The refusal is keyed on the user id, not on anything a real account shares."""
    _make(users, "real@test.com", is_admin=False)

    assert users.authenticate("real@test.com", PASSWORD) is not None


def test_the_system_error_is_a_value_error(users: UserService) -> None:
    """Same reason as the last-admin error: existing route and CLI handlers catch ValueError."""

    with pytest.raises(ValueError):
        users.delete(SYSTEM_USER_ID)


# endregion
