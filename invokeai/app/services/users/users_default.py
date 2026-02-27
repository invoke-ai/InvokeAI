"""Default SQLite implementation of user service."""

import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from invokeai.app.services.auth.password_utils import hash_password, validate_password_strength, verify_password
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.users.users_base import UserServiceBase
from invokeai.app.services.users.users_common import UserCreateRequest, UserDTO, UserUpdateRequest


class UserService(UserServiceBase):
    """SQLite-based user service."""

    def __init__(self, db: SqliteDatabase):
        """Initialize user service.

        Args:
            db: SQLite database instance
        """
        self._db = db

    def create(self, user_data: UserCreateRequest) -> UserDTO:
        """Create a new user."""
        # Validate password strength
        is_valid, error_msg = validate_password_strength(user_data.password)
        if not is_valid:
            raise ValueError(error_msg)

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
                SELECT user_id, email, display_name, is_admin, is_active, created_at, updated_at, last_login_at
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
        )

    def get_by_email(self, email: str) -> UserDTO | None:
        """Get user by email."""
        with self._db.transaction() as cursor:
            cursor.execute(
                """
                SELECT user_id, email, display_name, is_admin, is_active, created_at, updated_at, last_login_at
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
        )

    def update(self, user_id: str, changes: UserUpdateRequest) -> UserDTO:
        """Update user."""
        # Check if user exists
        user = self.get(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")

        # Validate password if provided
        if changes.password is not None:
            is_valid, error_msg = validate_password_strength(changes.password)
            if not is_valid:
                raise ValueError(error_msg)

        # Build update query dynamically based on provided fields
        updates: list[str] = []
        params: list[str | bool | int] = []

        if changes.display_name is not None:
            updates.append("display_name = ?")
            params.append(changes.display_name)

        if changes.password is not None:
            updates.append("password_hash = ?")
            params.append(hash_password(changes.password))

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

        with self._db.transaction() as cursor:
            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

    def authenticate(self, email: str, password: str) -> UserDTO | None:
        """Authenticate user credentials."""
        with self._db.transaction() as cursor:
            cursor.execute(
                """
                SELECT user_id, email, display_name, password_hash, is_admin, is_active, created_at, updated_at, last_login_at
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
        )

    def has_admin(self) -> bool:
        """Check if any admin user exists."""
        with self._db.transaction() as cursor:
            cursor.execute("SELECT COUNT(*) FROM users WHERE is_admin = TRUE AND is_active = TRUE")
            row = cursor.fetchone()
        count = row[0] if row else 0
        return bool(count > 0)

    def create_admin(self, user_data: UserCreateRequest) -> UserDTO:
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
        return self.create(admin_data)

    def list_users(self, limit: int = 100, offset: int = 0) -> list[UserDTO]:
        """List all users."""
        with self._db.transaction() as cursor:
            cursor.execute(
                """
                SELECT user_id, email, display_name, is_admin, is_active, created_at, updated_at, last_login_at
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
            )
            for row in rows
        ]
