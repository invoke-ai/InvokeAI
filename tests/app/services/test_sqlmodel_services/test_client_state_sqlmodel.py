"""Tests for ClientStatePersistenceSqlModel."""

import pytest

from invokeai.app.services.client_state_persistence.client_state_persistence_sqlmodel import (
    ClientStatePersistenceSqlModel,
)
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.users.users_sqlmodel import UserServiceSqlModel


@pytest.fixture
def users(db: SqliteDatabase) -> UserServiceSqlModel:
    return UserServiceSqlModel(db=db)


@pytest.fixture
def user_id(users: UserServiceSqlModel) -> str:
    user = users.create(
        UserCreateRequest(email="test@test.com", display_name="Test", password="TestPassword123"),
    )
    return user.user_id


@pytest.fixture
def client_state(db: SqliteDatabase) -> ClientStatePersistenceSqlModel:
    return ClientStatePersistenceSqlModel(db=db)


def test_get_nonexistent(client_state: ClientStatePersistenceSqlModel, user_id: str):
    assert client_state.get_by_key(user_id, "nonexistent") is None


def test_set_and_get(client_state: ClientStatePersistenceSqlModel, user_id: str):
    client_state.set_by_key(user_id, "theme", "dark")
    assert client_state.get_by_key(user_id, "theme") == "dark"


def test_set_overwrites(client_state: ClientStatePersistenceSqlModel, user_id: str):
    client_state.set_by_key(user_id, "key", "val1")
    client_state.set_by_key(user_id, "key", "val2")
    assert client_state.get_by_key(user_id, "key") == "val2"


def test_delete_user_state(client_state: ClientStatePersistenceSqlModel, user_id: str):
    client_state.set_by_key(user_id, "key1", "val1")
    client_state.set_by_key(user_id, "key2", "val2")
    client_state.delete(user_id)
    assert client_state.get_by_key(user_id, "key1") is None
    assert client_state.get_by_key(user_id, "key2") is None


def test_user_isolation(client_state: ClientStatePersistenceSqlModel, users: UserServiceSqlModel):
    user1 = users.create(UserCreateRequest(email="u1@test.com", display_name="U1", password="TestPassword123"))
    user2 = users.create(UserCreateRequest(email="u2@test.com", display_name="U2", password="TestPassword123"))
    client_state.set_by_key(user1.user_id, "key", "user1_val")
    client_state.set_by_key(user2.user_id, "key", "user2_val")
    assert client_state.get_by_key(user1.user_id, "key") == "user1_val"
    assert client_state.get_by_key(user2.user_id, "key") == "user2_val"
