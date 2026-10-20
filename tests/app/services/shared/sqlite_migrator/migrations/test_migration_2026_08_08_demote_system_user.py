"""Tests for migration 2026_08_08_demote_system_user.

The `system` row owns everything carried over from before multiuser support. It is seeded
as a non-administrator with an empty password hash, but until the service-level guard
landed an administrator could `PATCH /auth/users/system` and both promote it and give it a
password. Preventing that from now on does not repair an instance already in that state:

- a promoted row inflates `count_admins()` with an administrator nobody can log in as,
  which is enough to walk the last-admin guard past the real one;
- a password on the row is a standing login for the owner of all pre-multiuser content,
  under a fixed, public email address.
"""

import sqlite3

import pytest

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_08_08_demote_system_user import (
    DemoteSystemUserCallback,
    build_migration,
)


def _create_users_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE users (
            user_id TEXT NOT NULL PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            display_name TEXT,
            password_hash TEXT NOT NULL,
            is_admin BOOLEAN NOT NULL DEFAULT FALSE,
            is_active BOOLEAN NOT NULL DEFAULT TRUE
        );
        """
    )


def _insert(conn: sqlite3.Connection, user_id: str, email: str, password_hash: str, is_admin: bool) -> None:
    conn.execute(
        "INSERT INTO users (user_id, email, display_name, password_hash, is_admin) VALUES (?, ?, ?, ?, ?);",
        (user_id, email, user_id, password_hash, is_admin),
    )


def _row(conn: sqlite3.Connection, user_id: str) -> tuple:
    return conn.execute(
        "SELECT is_admin, password_hash, is_active FROM users WHERE user_id = ?;", (user_id,)
    ).fetchone()


# A hash shaped like the real thing; the migration must not care what it contains.
BCRYPT_HASH = "$2b$12$abcdefghijklmnopqrstuv0123456789012345678901234567890a"


@pytest.fixture
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    _create_users_table(conn)
    return conn


class TestDemoteSystemUser:
    def test_demotes_a_promoted_system_row(self, db: sqlite3.Connection) -> None:
        _insert(db, "system", "system@system.invokeai", "", is_admin=True)

        DemoteSystemUserCallback()(db.cursor())
        db.commit()

        assert _row(db, "system")[0] == 0

    def test_clears_a_password_left_on_the_system_row(self, db: sqlite3.Connection) -> None:
        """Without this, the row remains a login for the owner of all pre-multiuser content."""
        _insert(db, "system", "system@system.invokeai", BCRYPT_HASH, is_admin=False)

        DemoteSystemUserCallback()(db.cursor())
        db.commit()

        assert _row(db, "system")[1] == ""

    def test_repairs_a_row_that_is_both_promoted_and_password_bearing(self, db: sqlite3.Connection) -> None:
        _insert(db, "system", "system@system.invokeai", BCRYPT_HASH, is_admin=True)

        DemoteSystemUserCallback()(db.cursor())
        db.commit()

        is_admin, password_hash, is_active = _row(db, "system")
        assert (is_admin, password_hash) == (0, "")
        # Deactivating it would strand the content it owns — that is not this migration's job.
        assert is_active == 1

    def test_leaves_an_undamaged_system_row_alone(self, db: sqlite3.Connection) -> None:
        _insert(db, "system", "system@system.invokeai", "", is_admin=False)

        DemoteSystemUserCallback()(db.cursor())
        db.commit()

        assert _row(db, "system") == (0, "", 1)

    def test_leaves_real_administrators_alone(self, db: sqlite3.Connection) -> None:
        """Only the `system` id is repaired; demoting a real admin would lock the instance out."""
        _insert(db, "system", "system@system.invokeai", BCRYPT_HASH, is_admin=True)
        _insert(db, "u1", "admin@example.com", BCRYPT_HASH, is_admin=True)

        DemoteSystemUserCallback()(db.cursor())
        db.commit()

        assert _row(db, "u1") == (1, BCRYPT_HASH, 1)

    def test_is_idempotent(self, db: sqlite3.Connection) -> None:
        _insert(db, "system", "system@system.invokeai", BCRYPT_HASH, is_admin=True)

        DemoteSystemUserCallback()(db.cursor())
        DemoteSystemUserCallback()(db.cursor())
        db.commit()

        assert _row(db, "system") == (0, "", 1)

    def test_no_system_row_is_not_an_error(self, db: sqlite3.Connection) -> None:
        """Databases created after migration_27 always have one, but the callback must not
        assume it — a missing row is nothing to repair."""
        DemoteSystemUserCallback()(db.cursor())
        db.commit()

        assert db.execute("SELECT COUNT(*) FROM users;").fetchone()[0] == 0

    def test_migration_metadata(self) -> None:
        migration = build_migration()

        assert migration.id == "2026_08_08_demote_system_user"
        assert migration.depends_on == "migration_27"
