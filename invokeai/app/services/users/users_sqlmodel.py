"""SQLModel implementation of user service."""

from invokeai.app.services.auth.password_utils import hash_password, validate_password_strength
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.users.users_base import UserServiceBase
from invokeai.app.services.users.users_common import UserCreateRequest, UserDTO, UserUpdateRequest


class UserServiceSqlModel(UserServiceBase):
    """SQLModel-based user service."""

    def __init__(self, db: SqliteDatabase):
        self._db = db
        self._q = db.queries

    def create(self, user_data: UserCreateRequest, strict_password_checking: bool = True) -> UserDTO:
        if strict_password_checking:
            is_valid, error_msg = validate_password_strength(user_data.password)
            if not is_valid:
                raise ValueError(error_msg)
        elif not user_data.password:
            raise ValueError("Password cannot be empty")

        if self.get_by_email(user_data.email) is not None:
            raise ValueError(f"User with email {user_data.email} already exists")

        user_id = self._q.users_create(
            email=user_data.email,
            display_name=user_data.display_name,
            password_hash=hash_password(user_data.password),
            is_admin=user_data.is_admin,
        )

        result = self.get(user_id)
        if result is None:
            raise RuntimeError("Failed to retrieve created user")
        return result

    def get(self, user_id: str) -> UserDTO | None:
        return self._q.users_get(user_id)

    def get_by_email(self, email: str) -> UserDTO | None:
        return self._q.users_get_by_email(email)

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

        self._q.users_apply_update(
            user_id=user_id,
            display_name=changes.display_name,
            password_hash=hash_password(changes.password) if changes.password is not None else None,
            is_admin=changes.is_admin,
            is_active=changes.is_active,
        )

        updated_user = self.get(user_id)
        if updated_user is None:
            raise RuntimeError("Failed to retrieve updated user")
        return updated_user

    def delete(self, user_id: str) -> None:
        self._q.users_delete(user_id)

    def authenticate(self, email: str, password: str) -> UserDTO | None:
        return self._q.users_authenticate(email, password)

    def has_admin(self) -> bool:
        return self._q.users_count_admins() > 0

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
        return self._q.users_list(limit=limit, offset=offset)

    def get_admin_email(self) -> str | None:
        return self._q.users_get_admin_email()

    def count_admins(self) -> int:
        return self._q.users_count_admins()
