"""Default SQLite implementation of user service."""

import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from uuid import uuid4

from invokeai.app.services.auth.password_utils import hash_password, validate_password_strength, verify_password
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.users.users_base import UserServiceBase
from invokeai.app.services.users.users_common import (
    LAST_ADMIN_DETAIL,
    SYSTEM_USER_ID,
    SYSTEM_USER_PROTECTED_DETAIL,
    LastAdministratorError,
    SystemUserProtectedError,
    UserCreateRequest,
    UserDTO,
    UserUpdateRequest,
)

# Bound-parameter chunk size for `get_many`. SQLite's SQLITE_MAX_VARIABLE_NUMBER is 32766
# on modern builds but 999 on older ones; stay under the smaller limit.
USER_LOOKUP_CHUNK_SIZE = 900


class UserService(UserServiceBase):
    """SQLite-based user service."""

    def __init__(self, db: SqliteDatabase):
        """Initialize user service.

        Args:
            db: SQLite database instance
        """
        self._db = db

    def create(self, user_data: UserCreateRequest, strict_password_checking: bool = True) -> UserDTO:
        """Create a new user."""
        # Validate password strength
        if strict_password_checking:
            is_valid, error_msg = validate_password_strength(user_data.password)
            if not is_valid:
                raise ValueError(error_msg)
        elif not user_data.password:
            raise ValueError("Password cannot be empty")

        # Check if email already exists
        if self.get_by_email(user_data.email) is not None:
            raise ValueError(f"User with email {user_data.email} already exists")

        user_id = str(uuid4())
        password_hash = hash_password(user_data.password)

        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    """
                    INSERT INTO users (user_id, email, display_name, password_hash, is_admin)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, user_data.email, user_data.display_name, password_hash, user_data.is_admin),
                )
            except sqlite3.IntegrityError as e:
                raise ValueError(f"Failed to create user: {e}") from e

        user = self.get(user_id)
        if user is None:
            raise RuntimeError("Failed to retrieve created user")
        return user

    def get(self, user_id: str) -> UserDTO | None:
        """Get user by ID."""
        with self._db.transaction() as cursor:
            cursor.execute(
                """
                SELECT user_id, email, display_name, is_admin, is_active, created_at, updated_at, last_login_at, token_epoch
                FROM users
                WHERE user_id = ?
                """,
                (user_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return UserDTO(
            user_id=row[0],
            email=row[1],
            display_name=row[2],
            is_admin=bool(row[3]),
            is_active=bool(row[4]),
            created_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6]),
            last_login_at=datetime.fromisoformat(row[7]) if row[7] else None,
            token_epoch=int(row[8]),
        )

    def get_many(self, user_ids: Sequence[str]) -> dict[str, UserDTO]:
        """Get users by ID, keyed by user_id. Unknown ids are absent from the result."""
        unique_ids = list(dict.fromkeys(user_ids))
        if not unique_ids:
            return {}

        users: dict[str, UserDTO] = {}
        # Chunked so a caller with many distinct owners can't exceed SQLite's bound-parameter
        # limit (999 on builds predating 3.32).
        for start in range(0, len(unique_ids), USER_LOOKUP_CHUNK_SIZE):
            chunk = unique_ids[start : start + USER_LOOKUP_CHUNK_SIZE]
            placeholders = ",".join("?" for _ in chunk)
            with self._db.transaction() as cursor:
                cursor.execute(
                    f"""
                    SELECT user_id, email, display_name, is_admin, is_active, created_at, updated_at, last_login_at
                    FROM users
                    WHERE user_id IN ({placeholders})
                    """,
                    chunk,
                )
                rows = cursor.fetchall()
            for row in rows:
                users[row[0]] = UserDTO(
                    user_id=row[0],
                    email=row[1],
                    display_name=row[2],
                    is_admin=bool(row[3]),
                    is_active=bool(row[4]),
                    created_at=datetime.fromisoformat(row[5]),
                    updated_at=datetime.fromisoformat(row[6]),
                    last_login_at=datetime.fromisoformat(row[7]) if row[7] else None,
                )
        return users

    def get_by_email(self, email: str) -> UserDTO | None:
        """Get user by email."""
        with self._db.transaction() as cursor:
            cursor.execute(
                """
                SELECT user_id, email, display_name, is_admin, is_active, created_at, updated_at, last_login_at, token_epoch
                FROM users
                WHERE email = ?
                """,
                (email,),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return UserDTO(
            user_id=row[0],
            email=row[1],
            display_name=row[2],
            is_admin=bool(row[3]),
            is_active=bool(row[4]),
            created_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6]),
            last_login_at=datetime.fromisoformat(row[7]) if row[7] else None,
            token_epoch=int(row[8]),
        )

    def update(self, user_id: str, changes: UserUpdateRequest, strict_password_checking: bool = True) -> UserDTO:
        """Update user."""
        # Check if user exists
        user = self.get(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")

        self._assert_system_user_protected(
            user_id, is_admin=changes.is_admin, is_active=changes.is_active, password=changes.password
        )

        # Validate password if provided
        if changes.password is not None:
            if strict_password_checking:
                is_valid, error_msg = validate_password_strength(changes.password)
                if not is_valid:
                    raise ValueError(error_msg)
            elif not changes.password:
                raise ValueError("Password cannot be empty")

        # Build update query dynamically based on provided fields
        updates: list[str] = []
        params: list[str | bool | int] = []

        if changes.display_name is not None:
            updates.append("display_name = ?")
            params.append(changes.display_name)

        if changes.password is not None:
            updates.append("password_hash = ?")
            params.append(hash_password(changes.password))
            # Rotating the password revokes every token issued under the old one. A JWT is
            # self-contained, so without this a stolen token survives the password change
            # meant to evict the thief — and sliding-window refresh renews it indefinitely.
            # Computed in SQL so concurrent bumps can't read-modify-write over each other.
            updates.append("token_epoch = token_epoch + 1")

        if changes.is_admin is not None:
            updates.append("is_admin = ?")
            params.append(changes.is_admin)

        if changes.is_active is not None:
            updates.append("is_active = ?")
            params.append(changes.is_active)

        if not updates:
            return user

        params.append(user_id)
        query = f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?"

        with self._db.transaction() as cursor:
            # BEGIN IMMEDIATE takes the write lock up front, so the guard below reads a count
            # no other connection can change before this transaction commits. Without it the
            # SELECT would run in autocommit and a second process (invoke-usermod) could slip
            # its own write in between. In-process callers are additionally serialized by the
            # database's shared RLock.
            cursor.execute("BEGIN IMMEDIATE")
            self._assert_not_last_admin(cursor, user_id, is_admin=changes.is_admin, is_active=changes.is_active)
            cursor.execute(query, params)

        updated_user = self.get(user_id)
        if updated_user is None:
            raise RuntimeError("Failed to retrieve updated user")
        return updated_user

    def delete(self, user_id: str) -> None:
        """Delete user."""
        user = self.get(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")

        self._assert_system_user_protected(user_id, is_deleting=True)

        with self._db.transaction() as cursor:
            # See the note in `update`: the guard and the write share one write-locked
            # transaction so the admin count cannot change between them.
            cursor.execute("BEGIN IMMEDIATE")
            self._assert_not_last_admin(cursor, user_id, is_deleting=True)
            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

    def authenticate(self, email: str, password: str) -> UserDTO | None:
        """Authenticate user credentials."""
        with self._db.transaction() as cursor:
            cursor.execute(
                """
                SELECT user_id, email, display_name, password_hash, is_admin, is_active, created_at, updated_at, last_login_at, token_epoch
                FROM users
                WHERE email = ?
                """,
                (email,),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        password_hash = row[3]
        if not verify_password(password, password_hash):
            return None

        # Update last login time
        with self._db.transaction() as cursor:
            cursor.execute(
                "UPDATE users SET last_login_at = ? WHERE user_id = ?",
                (datetime.now(timezone.utc).isoformat(), row[0]),
            )

        return UserDTO(
            user_id=row[0],
            email=row[1],
            display_name=row[2],
            is_admin=bool(row[4]),
            is_active=bool(row[5]),
            created_at=datetime.fromisoformat(row[6]),
            updated_at=datetime.fromisoformat(row[7]),
            last_login_at=datetime.now(timezone.utc),
            # password_hash occupies row[3] in this query, shifting the tail by one.
            token_epoch=int(row[9]),
        )

    def has_admin(self) -> bool:
        """Check if any admin user exists."""
        with self._db.transaction() as cursor:
            cursor.execute("SELECT COUNT(*) FROM users WHERE is_admin = TRUE AND is_active = TRUE")
            row = cursor.fetchone()
        count = row[0] if row else 0
        return bool(count > 0)

    def create_admin(self, user_data: UserCreateRequest, strict_password_checking: bool = True) -> UserDTO:
        """Create an admin user (for initial setup)."""
        if self.has_admin():
            raise ValueError("Admin user already exists")

        # Force is_admin to True
        admin_data = UserCreateRequest(
            email=user_data.email,
            display_name=user_data.display_name,
            password=user_data.password,
            is_admin=True,
        )
        return self.create(admin_data, strict_password_checking=strict_password_checking)

    def list_users(self, limit: int = 100, offset: int = 0) -> list[UserDTO]:
        """List all users."""
        with self._db.transaction() as cursor:
            cursor.execute(
                """
                SELECT user_id, email, display_name, is_admin, is_active, created_at, updated_at, last_login_at, token_epoch
                FROM users
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = cursor.fetchall()

        return [
            UserDTO(
                user_id=row[0],
                email=row[1],
                display_name=row[2],
                is_admin=bool(row[3]),
                is_active=bool(row[4]),
                created_at=datetime.fromisoformat(row[5]),
                updated_at=datetime.fromisoformat(row[6]),
                last_login_at=datetime.fromisoformat(row[7]) if row[7] else None,
                token_epoch=int(row[8]),
            )
            for row in rows
        ]

    def get_admin_email(self) -> str | None:
        """Get the email address of the first active admin user."""
        with self._db.transaction() as cursor:
            cursor.execute(
                """
                SELECT email FROM users
                WHERE is_admin = TRUE AND is_active = TRUE
                ORDER BY created_at ASC
                LIMIT 1
                """,
            )
            row = cursor.fetchone()
        return row[0] if row else None

    def count_admins(self) -> int:
        """Count active admin users."""
        with self._db.transaction() as cursor:
            cursor.execute("SELECT COUNT(*) FROM users WHERE is_admin = TRUE AND is_active = TRUE")
            row = cursor.fetchone()
        return int(row[0]) if row else 0

    def _assert_not_last_admin(
        self,
        cursor: sqlite3.Cursor,
        user_id: str,
        *,
        is_deleting: bool = False,
        is_admin: bool | None = None,
        is_active: bool | None = None,
    ) -> None:
        """Reject a change that would drop the number of active administrators to zero.

        Must be called on the cursor of the transaction that performs the write, after that
        transaction has taken its write lock — see the ``BEGIN IMMEDIATE`` in the callers.
        Reading the count in a separate transaction is what made this a TOCTOU: two callers
        could each observe two admins and each proceed to remove one.

        ``is_admin``/``is_active`` are the *requested* values, where ``None`` means "not being
        changed" — deletion is signalled separately by ``is_deleting`` rather than by both
        being ``None``, which would also describe a rename. Only a change that actually
        revokes administrator status is checked, so renaming the last admin stays allowed.
        """
        cursor.execute("SELECT is_admin, is_active FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if row is None:
            return

        # An admin who is already inactive is not counted, so removing them changes nothing.
        if not (bool(row[0]) and bool(row[1])):
            return

        if not (is_deleting or is_admin is False or is_active is False):
            return

        cursor.execute("SELECT COUNT(*) FROM users WHERE is_admin = TRUE AND is_active = TRUE")
        count_row = cursor.fetchone()
        if (int(count_row[0]) if count_row else 0) <= 1:
            raise LastAdministratorError(LAST_ADMIN_DETAIL)

    def _assert_system_user_protected(
        self,
        user_id: str,
        *,
        is_deleting: bool = False,
        is_admin: bool | None = None,
        is_active: bool | None = None,
        password: str | None = None,
    ) -> None:
        """Reject changes to the ``system`` account that no legitimate operation needs.

        The system row owns every board, image, workflow, and queue item carried over from
        before multiuser support. Deleting or deactivating it strands all of that: queued
        items are rejected at dequeue and media reads and saves raise ``PermissionError``.

        Promotion and password-setting are refused for a different reason. The system row
        is active but has an empty password hash, so it can never authenticate — yet
        ``count_admins()`` and ``has_admin()`` count any active admin row. Promoting it
        therefore inflates the administrator count with an administrator nobody can log in
        as, which is enough to satisfy the last-admin guard while the real administrator is
        demoted, leaving the instance with no usable administration and no way back in.
        Keeping the system row permanently non-admin is what makes that count mean
        "administrators who can actually log in". Giving it a password would turn the owner
        of all pre-multiuser content into a login account, which is the same hole from the
        other end.

        Lives in the service rather than only in the routes so the ``invoke-usermod`` /
        ``invoke-userdel`` CLIs, which construct :class:`UserService` directly, are covered
        too — the same reasoning that moved the last-admin invariant down here.
        """
        if user_id != SYSTEM_USER_ID:
            return
        if is_deleting or is_active is False or is_admin is True or password is not None:
            raise SystemUserProtectedError(SYSTEM_USER_PROTECTED_DETAIL)
