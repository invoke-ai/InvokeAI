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

from invokeai.app.services.auth.password_utils import hash_password
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
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
            last_login_at DATETIME,
            token_epoch INTEGER NOT NULL DEFAULT 0
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

# region the system account


def _seed_system_user(db: SqliteDatabase) -> None:
    """The row migration_27 creates: active, non-admin, and with an empty password hash."""
    db._conn.execute(
        """
        INSERT INTO users (user_id, email, display_name, password_hash, is_admin, is_active)
        VALUES ('system', 'system@system.invokeai', 'System', '', FALSE, TRUE);
        """
    )
    db._conn.commit()


def test_the_system_user_cannot_be_promoted(db: SqliteDatabase, users: UserService) -> None:
    """`count_admins()` counts admin rows, but the invariant that matters is "an admin who
    can log in". The system row is active and can never authenticate — it has no password —
    so promoting it would inflate the count with an unusable administrator, which is enough
    to walk the last-admin guard past the real one:

        PATCH /auth/users/system      {"is_admin": true}   -> count_admins() 1 -> 2
        PATCH /auth/users/{real}      {"is_admin": false}  -> allowed, count 2 -> 1
        login as system                                    -> 401, empty password hash

    leaving the instance with no usable administration and no authenticated way back.
    """
    _seed_system_user(db)
    admin = _make(users, "admin@test.com", is_admin=True)

    with pytest.raises(SystemUserProtectedError):
        users.update(SYSTEM_USER_ID, UserUpdateRequest(is_admin=True), strict_password_checking=False)

    assert users.count_admins() == 1

    # And with the first step refused, the second is still blocked.
    with pytest.raises(LastAdministratorError):
        users.update(admin, UserUpdateRequest(is_admin=False), strict_password_checking=False)


def test_the_system_user_cannot_be_given_a_password(db: SqliteDatabase, users: UserService) -> None:
    """The other end of the same hole: a password turns the owner of every pre-multiuser
    board, image, and workflow into a login account."""
    _seed_system_user(db)

    with pytest.raises(SystemUserProtectedError):
        users.update(SYSTEM_USER_ID, UserUpdateRequest(password=PASSWORD), strict_password_checking=False)

    assert users.authenticate("system@system.invokeai", PASSWORD) is None


def test_the_system_user_cannot_be_deleted_or_deactivated(db: SqliteDatabase, users: UserService) -> None:
    """The routes already refuse both, but `invoke-userdel` / `invoke-usermod` construct
    this service directly and never reach a route — the same reason the last-admin guard
    lives here."""
    _seed_system_user(db)

    with pytest.raises(SystemUserProtectedError):
        users.delete(SYSTEM_USER_ID)
    with pytest.raises(SystemUserProtectedError):
        users.update(SYSTEM_USER_ID, UserUpdateRequest(is_active=False), strict_password_checking=False)

    system = users.get(SYSTEM_USER_ID)
    assert system is not None and system.is_active is True


def test_renaming_the_system_user_is_allowed(db: SqliteDatabase, users: UserService) -> None:
    """Not a blanket lock on the row — only the changes that would make it dangerous."""
    _seed_system_user(db)

    updated = users.update(SYSTEM_USER_ID, UserUpdateRequest(display_name="Renamed"), strict_password_checking=False)

    assert updated.display_name == "Renamed"


def test_a_system_row_carrying_a_password_still_cannot_log_in(db: SqliteDatabase, users: UserService) -> None:
    """The guard above only stops a password being set *from now on*.

    An instance that set one through the old `PATCH /auth/users/system` hole still carries
    a usable hash, and its email is fixed and public — so the hash is a standing login for
    the account that owns every pre-multiuser board, image, workflow, and queue item. The
    migration clears it, but migrations run once and cannot reach a row damaged afterwards
    by direct SQL, so `authenticate` refuses the account outright whatever the row holds.
    """
    _seed_system_user(db)
    db._conn.execute(
        "UPDATE users SET password_hash = ? WHERE user_id = 'system'",
        (hash_password(PASSWORD),),
    )
    db._conn.commit()

    assert users.authenticate("system@system.invokeai", PASSWORD) is None


def test_refusing_the_system_account_does_not_block_other_logins(db: SqliteDatabase, users: UserService) -> None:
    """The refusal is keyed on the user id, not on anything a real account shares."""
    _seed_system_user(db)
    _make(users, "real@test.com", is_admin=False)

    assert users.authenticate("real@test.com", PASSWORD) is not None


def test_the_system_error_is_a_value_error(db: SqliteDatabase, users: UserService) -> None:
    """Same reason as the last-admin error: existing route and CLI handlers catch ValueError."""
    _seed_system_user(db)

    with pytest.raises(ValueError):
        users.delete(SYSTEM_USER_ID)


# endregion
