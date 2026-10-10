"""Tests for user service."""

import threading

import pytest
from pydantic import ValidationError
from sqlalchemy import insert, select, update

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries.base import IN_CHUNK
from invokeai.app.services.shared.database.schema.media_references import media_references
from invokeai.app.services.shared.database.schema.users import users as users_table
from invokeai.app.services.users import users_default
from invokeai.app.services.users.users_common import (
    MAX_EMAIL_LENGTH,
    SYSTEM_USER_ID,
    UserCreateRequest,
    UserUpdateRequest,
)
from invokeai.app.services.users.users_default import UserService


@pytest.fixture
def user_service(database: Database) -> UserService:
    """Create a user service for testing."""
    return UserService(database)


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
    """Test creating a user with weak password fails when strict checking is enabled."""
    user_data = UserCreateRequest(
        email="test@example.com",
        display_name="Test User",
        password="weak",
        is_admin=False,
    )

    with pytest.raises(ValueError, match="at least 8 characters"):
        user_service.create(user_data, strict_password_checking=True)


def test_create_user_weak_password_non_strict(user_service: UserService):
    """Test creating a user with weak password succeeds when strict checking is disabled."""
    user_data = UserCreateRequest(
        email="weakpass@example.com",
        display_name="Test User",
        password="weak",
        is_admin=False,
    )

    user = user_service.create(user_data, strict_password_checking=False)
    assert user.email == "weakpass@example.com"


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


def test_delete_user_drops_references_only_of_documents_that_cascade(user_service: UserService, database: Database):
    leaving = user_service.create(UserCreateRequest(email="leaving@example.com", password="TestPassword123")).user_id
    staying = user_service.create(UserCreateRequest(email="staying@example.com", password="TestPassword123")).user_id
    kinds = ("project", "client_state", "workflow", "quarantined_project")
    with database.begin(write=True) as conn:
        conn.execute(
            insert(media_references),
            [
                {"owner_kind": kind, "user_id": user, "owner_id": "owner", "media_kind": "image", "media_name": "a.png"}
                for kind in kinds
                for user in (leaving, staying)
            ],
        )

    user_service.delete(leaving)

    with database.begin(write=False) as conn:
        remaining = {
            (user_id, kind)
            for user_id, kind in conn.execute(select(media_references.c.user_id, media_references.c.owner_kind))
        }
    assert remaining == {(leaving, "quarantined_project"), (leaving, "workflow")} | {(staying, kind) for kind in kinds}


def test_an_account_created_meanwhile_with_the_same_email_is_refused(
    user_service: UserService, database: Database, monkeypatch: pytest.MonkeyPatch
):
    """The email is checked before the password is hashed, which takes a while; the unique key decides."""
    real_hash_password = users_default.hash_password

    def hash_while_someone_else_signs_up(password: str) -> str:
        with database.begin(write=True) as conn:
            conn.execute(insert(users_table).values(user_id="first", email="taken@example.com", password_hash="hash"))
        return real_hash_password(password)

    monkeypatch.setattr(users_default, "hash_password", hash_while_someone_else_signs_up)

    with pytest.raises(ValueError, match="Failed to create user"):
        user_service.create(UserCreateRequest(email="taken@example.com", password="TestPassword123"))

    assert [user.user_id for user in user_service.list_users() if user.email == "taken@example.com"] == ["first"]


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


def test_a_login_is_recorded_on_that_account_only(user_service: UserService):
    user_service.create(UserCreateRequest(email="first@example.com", password="TestPassword123"))
    second = user_service.create(UserCreateRequest(email="second@example.com", password="TestPassword123"))

    logged_in = user_service.authenticate("first@example.com", "TestPassword123")

    assert logged_in is not None
    first = user_service.get(logged_in.user_id)
    assert first is not None and first.last_login_at == logged_in.last_login_at
    unchanged = user_service.get(second.user_id)
    assert unchanged is not None and unchanged.last_login_at is None


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
    user_service.create_admin(
        UserCreateRequest(
            email="admin@example.com",
            display_name="Admin User",
            password="AdminPassword123",
        )
    )

    # A *different* email, so this exercises the admin guard rather than the unique-email one.
    with pytest.raises(ValueError, match="Admin user already exists"):
        user_service.create_admin(
            UserCreateRequest(
                email="second-admin@example.com",
                display_name="Second Admin",
                password="AdminPassword123",
            )
        )


def test_concurrent_create_admin_creates_exactly_one_admin(user_service: UserService, monkeypatch):
    """Two concurrent `POST /auth/setup` requests must not both create an administrator.

    The route is unauthenticated during the first-run window and runs in the threadpool, so the
    two requests really do interleave. Checking has_admin() in its own transaction before the
    INSERT leaves a window in which both callers see no admin and both create one; the loser then
    holds a persistent admin account instead of getting the intended 400.

    The barrier models that interleaving deterministically: it releases both threads only once
    both have passed every step preceding the write.
    """
    barrier = threading.Barrier(2, timeout=30)
    real_hash_password = users_default.hash_password

    def synchronized_hash_password(password: str) -> str:
        barrier.wait()
        return real_hash_password(password)

    monkeypatch.setattr(users_default, "hash_password", synchronized_hash_password)

    results: dict[int, object] = {}

    def attempt(index: int) -> None:
        try:
            results[index] = user_service.create_admin(
                UserCreateRequest(
                    email=f"admin{index}@example.com",
                    display_name=f"Admin {index}",
                    password="AdminPassword123",
                )
            )
        except ValueError as exc:
            results[index] = exc

    threads = [threading.Thread(target=attempt, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()

    created = [result for result in results.values() if not isinstance(result, ValueError)]
    rejected = [result for result in results.values() if isinstance(result, ValueError)]

    assert len(created) == 1, f"Both requests created an administrator: {results}"
    assert len(rejected) == 1
    assert "Admin user already exists" in str(rejected[0])
    assert user_service.count_admins() == 1


def test_list_users(user_service: UserService):
    """Test listing users."""
    for i in range(5):
        user_data = UserCreateRequest(
            email=f"test{i}@example.com",
            display_name=f"Test User {i}",
            password="TestPassword123",
        )
        user_service.create(user_data)

    # Every database starts with the system account.
    users = user_service.list_users()
    assert SYSTEM_USER_ID in {user.user_id for user in users}
    assert len(users) == 6

    limited_users = user_service.list_users(limit=2)
    assert len(limited_users) == 2


def test_list_users_lists_newest_first(user_service: UserService, database: Database):
    created = {
        user_service.create(UserCreateRequest(email=f"user{day}@example.com", password="TestPassword123")).user_id: day
        for day in (2, 3, 1)
    }
    with database.begin(write=True) as conn:
        for user_id, day in created.items():
            conn.execute(
                update(users_table)
                .where(users_table.c.user_id == user_id)
                .values(created_at=f"2099-01-0{day} 00:00:00.000")
            )

    listed = [user.email for user in user_service.list_users(limit=3)]

    assert listed == ["user3@example.com", "user2@example.com", "user1@example.com"]


def test_get_many_returns_users_keyed_by_id(user_service: UserService):
    """Batch lookup: dedups input, keys by user_id, and omits unknown ids."""
    created = [
        user_service.create(
            UserCreateRequest(
                email=f"batch{index}@example.com",
                display_name=f"Batch User {index}",
                password="TestPassword123",
            )
        )
        for index in range(3)
    ]

    requested = [created[0].user_id, created[1].user_id, created[0].user_id, "does-not-exist"]
    users = user_service.get_many(requested)

    assert set(users) == {created[0].user_id, created[1].user_id}
    assert users[created[0].user_id].email == "batch0@example.com"
    assert users[created[1].user_id].display_name == "Batch User 1"


def test_get_many_agrees_with_get_after_a_password_rotation(user_service: UserService):
    """Every projection that yields a `UserDTO` must select the same columns.

    `get_many` selected all of them but `token_epoch`, so the DTO fell back to the model
    default of 0 — the value a never-rotated account has, which makes a revoked epoch
    indistinguishable from a fresh one in any caller that reads it from a bulk lookup.
    """
    user = user_service.create(
        UserCreateRequest(email="rotated@example.com", display_name="Rotated", password="TestPassword123")
    )
    user_service.update(user.user_id, UserUpdateRequest(password="DifferentPassword456"))

    single = user_service.get(user.user_id)
    assert single is not None
    assert single.token_epoch > 0, "precondition: rotating the password advances the epoch"
    assert user_service.get_many([user.user_id])[user.user_id] == single


def test_get_many_with_no_ids_returns_empty(user_service: UserService):
    assert user_service.get_many([]) == {}


def test_get_many_chunks_beyond_sqlite_parameter_limit(user_service: UserService):
    """More ids than SQLite's bound-parameter limit must not raise."""
    user = user_service.create(
        UserCreateRequest(email="chunked@example.com", display_name="Chunked", password="TestPassword123")
    )
    ids = [f"missing-{index}" for index in range(IN_CHUNK * 2 + 5)] + [user.user_id]

    users = user_service.get_many(ids)

    assert set(users) == {user.user_id}


def test_an_address_at_a_special_use_domain_is_accepted_up_to_the_longest_address() -> None:
    domain = "@studio.local"
    longest = "a" * (MAX_EMAIL_LENGTH - len(domain)) + domain

    assert UserCreateRequest(email=longest, password="TestPassword123").email == longest
    with pytest.raises(ValidationError, match=f"at most {MAX_EMAIL_LENGTH} characters"):
        UserCreateRequest(email="a" + longest, password="TestPassword123")
