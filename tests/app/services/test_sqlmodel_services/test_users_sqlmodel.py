"""Tests for UserServiceSqlModel."""

import pytest

from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.users.users_common import UserCreateRequest, UserUpdateRequest
from invokeai.app.services.users.users_sqlmodel import UserServiceSqlModel


@pytest.fixture
def user_service(db: SqliteDatabase) -> UserServiceSqlModel:
    return UserServiceSqlModel(db=db)


def test_create_user(user_service: UserServiceSqlModel):
    user = user_service.create(
        UserCreateRequest(email="test@example.com", display_name="Test User", password="TestPassword123")
    )
    assert user.email == "test@example.com"
    assert user.display_name == "Test User"
    assert user.is_admin is False
    assert user.is_active is True


def test_create_user_weak_password(user_service: UserServiceSqlModel):
    with pytest.raises(ValueError, match="at least 8 characters"):
        user_service.create(
            UserCreateRequest(email="test@example.com", display_name="Test", password="weak"),
            strict_password_checking=True,
        )


def test_create_user_weak_password_non_strict(user_service: UserServiceSqlModel):
    user = user_service.create(
        UserCreateRequest(email="test@example.com", display_name="Test", password="weak"),
        strict_password_checking=False,
    )
    assert user.email == "test@example.com"


def test_create_duplicate_user(user_service: UserServiceSqlModel):
    data = UserCreateRequest(email="test@example.com", display_name="Test", password="TestPassword123")
    user_service.create(data)
    with pytest.raises(ValueError, match="already exists"):
        user_service.create(data)


def test_get_user(user_service: UserServiceSqlModel):
    created = user_service.create(
        UserCreateRequest(email="test@example.com", display_name="Test", password="TestPassword123")
    )
    fetched = user_service.get(created.user_id)
    assert fetched is not None
    assert fetched.user_id == created.user_id


def test_get_nonexistent_user(user_service: UserServiceSqlModel):
    assert user_service.get("nonexistent-id") is None


def test_get_user_by_email(user_service: UserServiceSqlModel):
    user_service.create(UserCreateRequest(email="test@example.com", display_name="Test", password="TestPassword123"))
    fetched = user_service.get_by_email("test@example.com")
    assert fetched is not None
    assert fetched.email == "test@example.com"


def test_update_user(user_service: UserServiceSqlModel):
    user = user_service.create(
        UserCreateRequest(email="test@example.com", display_name="Test", password="TestPassword123")
    )
    updated = user_service.update(user.user_id, UserUpdateRequest(display_name="Updated", is_admin=True))
    assert updated.display_name == "Updated"
    assert updated.is_admin is True


def test_delete_user(user_service: UserServiceSqlModel):
    user = user_service.create(
        UserCreateRequest(email="test@example.com", display_name="Test", password="TestPassword123")
    )
    user_service.delete(user.user_id)
    assert user_service.get(user.user_id) is None


def test_authenticate_valid(user_service: UserServiceSqlModel):
    user_service.create(UserCreateRequest(email="test@example.com", display_name="Test", password="TestPassword123"))
    auth = user_service.authenticate("test@example.com", "TestPassword123")
    assert auth is not None
    assert auth.email == "test@example.com"
    assert auth.last_login_at is not None


def test_authenticate_invalid_password(user_service: UserServiceSqlModel):
    user_service.create(UserCreateRequest(email="test@example.com", display_name="Test", password="TestPassword123"))
    assert user_service.authenticate("test@example.com", "WrongPassword") is None


def test_authenticate_nonexistent(user_service: UserServiceSqlModel):
    assert user_service.authenticate("none@example.com", "TestPassword123") is None


def test_has_admin(user_service: UserServiceSqlModel):
    assert user_service.has_admin() is False
    user_service.create(
        UserCreateRequest(email="admin@example.com", display_name="Admin", password="AdminPassword123", is_admin=True)
    )
    assert user_service.has_admin() is True


def test_create_admin(user_service: UserServiceSqlModel):
    admin = user_service.create_admin(
        UserCreateRequest(email="admin@example.com", display_name="Admin", password="AdminPassword123")
    )
    assert admin.is_admin is True


def test_create_admin_when_exists(user_service: UserServiceSqlModel):
    user_service.create_admin(
        UserCreateRequest(email="admin@example.com", display_name="Admin", password="AdminPassword123")
    )
    with pytest.raises(ValueError, match="already exists"):
        user_service.create_admin(
            UserCreateRequest(email="admin2@example.com", display_name="Admin2", password="AdminPassword123")
        )


def test_list_users(user_service: UserServiceSqlModel):
    for i in range(5):
        user_service.create(
            UserCreateRequest(email=f"test{i}@example.com", display_name=f"User {i}", password="TestPassword123")
        )
    # Migration 27 creates a 'system' user, so total = 5 + 1
    assert len(user_service.list_users()) == 6
    assert len(user_service.list_users(limit=2)) == 2


def test_get_admin_email(user_service: UserServiceSqlModel):
    assert user_service.get_admin_email() is None
    user_service.create(
        UserCreateRequest(email="admin@example.com", display_name="Admin", password="AdminPassword123", is_admin=True)
    )
    assert user_service.get_admin_email() == "admin@example.com"


def test_count_admins(user_service: UserServiceSqlModel):
    assert user_service.count_admins() == 0
    user_service.create(
        UserCreateRequest(email="admin@example.com", display_name="Admin", password="AdminPassword123", is_admin=True)
    )
    assert user_service.count_admins() == 1
