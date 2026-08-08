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
from logging import Logger

import pytest

from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.users.users_common import (
    LastAdministratorError,
    UserCreateRequest,
    UserUpdateRequest,
)
from invokeai.app.services.users.users_default import UserService

PASSWORD = "Sup3rSecret!pass"


@pytest.fixture
def db() -> SqliteDatabase:
    db = SqliteDatabase(db_path=None, logger=Logger("test_last_admin"), verbose=False)
    db._conn.execute("""
        CREATE TABLE users (
            user_id TEXT NOT NULL PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            display_name TEXT,
            password_hash TEXT NOT NULL,
            is_admin BOOLEAN NOT NULL DEFAULT FALSE,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
            updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
            last_login_at DATETIME
        );
    """)
    db._conn.commit()
    return db


@pytest.fixture
def users(db: SqliteDatabase) -> UserService:
    return UserService(db)


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


# endregion
