"""Tests for user service."""

from logging import Logger

import pytest

from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.users.users_common import UserCreateRequest, UserUpdateRequest
from invokeai.app.services.users.users_default import UserService


@pytest.fixture
def logger() -> Logger:
    """Create a logger for testing."""
    return Logger("test_user_service")


@pytest.fixture
def db(logger: Logger) -> SqliteDatabase:
    """Create an in-memory database for testing."""
    db = SqliteDatabase(db_path=None, logger=logger, verbose=False)
    # Create users table manually for testing
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
def user_service(db: SqliteDatabase) -> UserService:
    """Create a user service for testing."""
    return UserService(db)


def test_create_user(user_service: UserService):
    """Test creating a user."""
    user_data = UserCreateRequest(
        email="test@example.com",
        display_name="Test User",
        password="TestPassword123",
        is_admin=False,
    )

    user = user_service.create(user_data)

    assert user.email == "test@example.com"
    assert user.display_name == "Test User"
    assert user.is_admin is False
    assert user.is_active is True
    assert user.user_id is not None


def test_create_user_weak_password(user_service: UserService):
    """Test creating a user with weak password."""
    user_data = UserCreateRequest(
        email="test@example.com",
        display_name="Test User",
        password="weak",
        is_admin=False,
    )

    with pytest.raises(ValueError, match="at least 8 characters"):
        user_service.create(user_data)


def test_create_duplicate_user(user_service: UserService):
    """Test creating a duplicate user."""
    user_data = UserCreateRequest(
        email="test@example.com",
        display_name="Test User",
        password="TestPassword123",
        is_admin=False,
    )

    user_service.create(user_data)

    with pytest.raises(ValueError, match="already exists"):
        user_service.create(user_data)


def test_get_user(user_service: UserService):
    """Test getting a user by ID."""
    user_data = UserCreateRequest(
        email="test@example.com",
        display_name="Test User",
        password="TestPassword123",
    )

    created_user = user_service.create(user_data)
    retrieved_user = user_service.get(created_user.user_id)

    assert retrieved_user is not None
    assert retrieved_user.user_id == created_user.user_id
    assert retrieved_user.email == created_user.email


def test_get_nonexistent_user(user_service: UserService):
    """Test getting a nonexistent user."""
    user = user_service.get("nonexistent-id")
    assert user is None


def test_get_user_by_email(user_service: UserService):
    """Test getting a user by email."""
    user_data = UserCreateRequest(
        email="test@example.com",
        display_name="Test User",
        password="TestPassword123",
    )

    created_user = user_service.create(user_data)
    retrieved_user = user_service.get_by_email("test@example.com")

    assert retrieved_user is not None
    assert retrieved_user.user_id == created_user.user_id
    assert retrieved_user.email == "test@example.com"


def test_update_user(user_service: UserService):
    """Test updating a user."""
    user_data = UserCreateRequest(
        email="test@example.com",
        display_name="Test User",
        password="TestPassword123",
    )

    user = user_service.create(user_data)

    updates = UserUpdateRequest(
        display_name="Updated Name",
        is_admin=True,
    )

    updated_user = user_service.update(user.user_id, updates)

    assert updated_user.display_name == "Updated Name"
    assert updated_user.is_admin is True


def test_delete_user(user_service: UserService):
    """Test deleting a user."""
    user_data = UserCreateRequest(
        email="test@example.com",
        display_name="Test User",
        password="TestPassword123",
    )

    user = user_service.create(user_data)
    user_service.delete(user.user_id)

    retrieved_user = user_service.get(user.user_id)
    assert retrieved_user is None


def test_authenticate_valid_credentials(user_service: UserService):
    """Test authenticating with valid credentials."""
    user_data = UserCreateRequest(
        email="test@example.com",
        display_name="Test User",
        password="TestPassword123",
    )

    user_service.create(user_data)
    authenticated_user = user_service.authenticate("test@example.com", "TestPassword123")

    assert authenticated_user is not None
    assert authenticated_user.email == "test@example.com"
    assert authenticated_user.last_login_at is not None


def test_authenticate_invalid_password(user_service: UserService):
    """Test authenticating with invalid password."""
    user_data = UserCreateRequest(
        email="test@example.com",
        display_name="Test User",
        password="TestPassword123",
    )

    user_service.create(user_data)
    authenticated_user = user_service.authenticate("test@example.com", "WrongPassword")

    assert authenticated_user is None


def test_authenticate_nonexistent_user(user_service: UserService):
    """Test authenticating nonexistent user."""
    authenticated_user = user_service.authenticate("nonexistent@example.com", "TestPassword123")
    assert authenticated_user is None


def test_has_admin(user_service: UserService):
    """Test checking if admin exists."""
    assert user_service.has_admin() is False

    user_data = UserCreateRequest(
        email="admin@example.com",
        display_name="Admin User",
        password="AdminPassword123",
        is_admin=True,
    )

    user_service.create(user_data)
    assert user_service.has_admin() is True


def test_create_admin(user_service: UserService):
    """Test creating an admin user."""
    user_data = UserCreateRequest(
        email="admin@example.com",
        display_name="Admin User",
        password="AdminPassword123",
    )

    admin = user_service.create_admin(user_data)

    assert admin.is_admin is True
    assert admin.email == "admin@example.com"


def test_create_admin_when_exists(user_service: UserService):
    """Test creating admin when one already exists."""
    user_data = UserCreateRequest(
        email="admin@example.com",
        display_name="Admin User",
        password="AdminPassword123",
    )

    user_service.create_admin(user_data)

    with pytest.raises(ValueError, match="already exists"):
        user_service.create_admin(user_data)


def test_list_users(user_service: UserService):
    """Test listing users."""
    for i in range(5):
        user_data = UserCreateRequest(
            email=f"test{i}@example.com",
            display_name=f"Test User {i}",
            password="TestPassword123",
        )
        user_service.create(user_data)

    users = user_service.list_users()
    assert len(users) == 5

    limited_users = user_service.list_users(limit=2)
    assert len(limited_users) == 2
