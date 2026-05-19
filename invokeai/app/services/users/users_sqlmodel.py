"""SQLModel implementation of user service."""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import func
from sqlmodel import col, select

from invokeai.app.services.auth.password_utils import hash_password, validate_password_strength, verify_password
from invokeai.app.services.shared.sqlite.models import UserTable
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.users.users_base import UserServiceBase
from invokeai.app.services.users.users_common import UserCreateRequest, UserDTO, UserUpdateRequest


def _to_dto(row: UserTable) -> UserDTO:
    return UserDTO(
        user_id=row.user_id,
        email=row.email,
        display_name=row.display_name,
        is_admin=row.is_admin,
        is_active=row.is_active,
        created_at=row.created_at,
        updated_at=row.updated_at,
        last_login_at=row.last_login_at,
    )


class UserServiceSqlModel(UserServiceBase):
    """SQLModel-based user service."""

    def __init__(self, db: SqliteDatabase):
        self._db = db

    def create(self, user_data: UserCreateRequest, strict_password_checking: bool = True) -> UserDTO:
        if strict_password_checking:
            is_valid, error_msg = validate_password_strength(user_data.password)
            if not is_valid:
                raise ValueError(error_msg)
        elif not user_data.password:
            raise ValueError("Password cannot be empty")

        if self.get_by_email(user_data.email) is not None:
            raise ValueError(f"User with email {user_data.email} already exists")

        user_id = str(uuid4())
        password_hash = hash_password(user_data.password)

        user = UserTable(
            user_id=user_id,
            email=user_data.email,
            display_name=user_data.display_name,
            password_hash=password_hash,
            is_admin=user_data.is_admin,
        )
        with self._db.get_session() as session:
            session.add(user)

        result = self.get(user_id)
        if result is None:
            raise RuntimeError("Failed to retrieve created user")
        return result

    def get(self, user_id: str) -> UserDTO | None:
        with self._db.get_readonly_session() as session:
            row = session.get(UserTable, user_id)
            if row is None:
                return None
            return _to_dto(row)

    def get_by_email(self, email: str) -> UserDTO | None:
        with self._db.get_readonly_session() as session:
            stmt = select(UserTable).where(col(UserTable.email) == email)
            row = session.exec(stmt).first()
            if row is None:
                return None
            return _to_dto(row)

    def update(self, user_id: str, changes: UserUpdateRequest, strict_password_checking: bool = True) -> UserDTO:
        user = self.get(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")

        if changes.password is not None:
            if strict_password_checking:
                is_valid, error_msg = validate_password_strength(changes.password)
                if not is_valid:
                    raise ValueError(error_msg)
            elif not changes.password:
                raise ValueError("Password cannot be empty")

        with self._db.get_session() as session:
            row = session.get(UserTable, user_id)
            if row is None:
                raise ValueError(f"User {user_id} not found")

            if changes.display_name is not None:
                row.display_name = changes.display_name
            if changes.password is not None:
                row.password_hash = hash_password(changes.password)
            if changes.is_admin is not None:
                row.is_admin = changes.is_admin
            if changes.is_active is not None:
                row.is_active = changes.is_active

            session.add(row)

        updated_user = self.get(user_id)
        if updated_user is None:
            raise RuntimeError("Failed to retrieve updated user")
        return updated_user

    def delete(self, user_id: str) -> None:
        with self._db.get_session() as session:
            row = session.get(UserTable, user_id)
            if row is None:
                raise ValueError(f"User {user_id} not found")
            session.delete(row)

    def authenticate(self, email: str, password: str) -> UserDTO | None:
        with self._db.get_session() as session:
            stmt = select(UserTable).where(col(UserTable.email) == email)
            row = session.exec(stmt).first()
            if row is None:
                return None

            if not verify_password(password, row.password_hash):
                return None

            row.last_login_at = datetime.now(timezone.utc)
            session.add(row)

            return _to_dto(row)

    def has_admin(self) -> bool:
        with self._db.get_readonly_session() as session:
            stmt = (
                select(func.count())
                .select_from(UserTable)
                .where(
                    col(UserTable.is_admin) == True,  # noqa: E712
                    col(UserTable.is_active) == True,  # noqa: E712
                )
            )
            count = session.exec(stmt).one()
        return count > 0

    def create_admin(self, user_data: UserCreateRequest, strict_password_checking: bool = True) -> UserDTO:
        if self.has_admin():
            raise ValueError("Admin user already exists")

        admin_data = UserCreateRequest(
            email=user_data.email,
            display_name=user_data.display_name,
            password=user_data.password,
            is_admin=True,
        )
        return self.create(admin_data, strict_password_checking=strict_password_checking)

    def list_users(self, limit: int = 100, offset: int = 0) -> list[UserDTO]:
        with self._db.get_readonly_session() as session:
            stmt = select(UserTable).order_by(col(UserTable.created_at).desc()).limit(limit).offset(offset)
            rows = session.exec(stmt).all()
            return [_to_dto(r) for r in rows]

    def get_admin_email(self) -> str | None:
        with self._db.get_readonly_session() as session:
            stmt = (
                select(UserTable)
                .where(
                    col(UserTable.is_admin) == True,  # noqa: E712
                    col(UserTable.is_active) == True,  # noqa: E712
                )
                .order_by(col(UserTable.created_at).asc())
                .limit(1)
            )
            row = session.exec(stmt).first()
            return row.email if row else None

    def count_admins(self) -> int:
        with self._db.get_readonly_session() as session:
            stmt = (
                select(func.count())
                .select_from(UserTable)
                .where(
                    col(UserTable.is_admin) == True,  # noqa: E712
                    col(UserTable.is_active) == True,  # noqa: E712
                )
            )
            count = session.exec(stmt).one()
        return count
