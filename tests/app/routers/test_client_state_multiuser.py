"""Tests for multiuser client state functionality."""

from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.users.users_common import UserCreateRequest


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class MockApiDependencies(ApiDependencies):
    """Mock API dependencies for testing."""

    invoker: Invoker

    def __init__(self, invoker: Invoker) -> None:
        self.invoker = invoker


def setup_test_user(
    mock_invoker: Invoker, email: str, display_name: str, password: str = "TestPass123", is_admin: bool = False
) -> str:
    """Helper to create a test user and return user_id."""
    user_service = mock_invoker.services.users
    user_data = UserCreateRequest(
        email=email,
        display_name=display_name,
        password=password,
        is_admin=is_admin,
    )
    user = user_service.create(user_data)
    return user.user_id


def get_user_token(client: TestClient, email: str, password: str = "TestPass123") -> str:
    """Helper to login and get a user token."""
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": email,
            "password": password,
            "remember_me": False,
        },
    )
    assert response.status_code == 200
    return response.json()["token"]


@pytest.fixture
def admin_token(monkeypatch: Any, mock_invoker: Invoker, client: TestClient):
    """Get an admin token for testing."""
    # Enable multiuser mode for auth endpoints
    mock_invoker.services.configuration.multiuser = True

    # Mock ApiDependencies for auth and client_state routers
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", MockApiDependencies(mock_invoker))
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", MockApiDependencies(mock_invoker))
    monkeypatch.setattr("invokeai.app.api.routers.client_state.ApiDependencies", MockApiDependencies(mock_invoker))

    # Create admin user
    setup_test_user(mock_invoker, "admin@test.com", "Admin User", is_admin=True)

    return get_user_token(client, "admin@test.com")


@pytest.fixture
def user1_token(monkeypatch: Any, mock_invoker: Invoker, client: TestClient, admin_token: str):
    """Get a token for test user 1."""
    # Create a regular user
    setup_test_user(mock_invoker, "user1@test.com", "User One", is_admin=False)

    return get_user_token(client, "user1@test.com")


@pytest.fixture
def user2_token(monkeypatch: Any, mock_invoker: Invoker, client: TestClient, admin_token: str):
    """Get a token for test user 2."""
    # Create another regular user
    setup_test_user(mock_invoker, "user2@test.com", "User Two", is_admin=False)

    return get_user_token(client, "user2@test.com")


def test_get_client_state_without_auth_uses_system_user(client: TestClient, monkeypatch, mock_invoker: Invoker):
    """Test that getting client state without authentication uses the system user."""
    # Mock ApiDependencies
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", MockApiDependencies(mock_invoker))
    monkeypatch.setattr("invokeai.app.api.routers.client_state.ApiDependencies", MockApiDependencies(mock_invoker))

    # Set a value for the system user directly
    mock_invoker.services.client_state_persistence.set_by_key("system", "test_key", "system_value")

    # Get without authentication - should return system user's value
    response = client.get("/api/v1/client_state/default/get_by_key?key=test_key")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == "system_value"


def test_set_client_state_without_auth_uses_system_user(client: TestClient, monkeypatch, mock_invoker: Invoker):
    """Test that setting client state without authentication uses the system user."""
    # Mock ApiDependencies
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", MockApiDependencies(mock_invoker))
    monkeypatch.setattr("invokeai.app.api.routers.client_state.ApiDependencies", MockApiDependencies(mock_invoker))

    # Set without authentication - should set for system user
    response = client.post(
        "/api/v1/client_state/default/set_by_key?key=test_key",
        json="unauthenticated_value",
    )
    assert response.status_code == status.HTTP_200_OK

    # Verify it was set for system user
    value = mock_invoker.services.client_state_persistence.get_by_key("system", "test_key")
    assert value == "unauthenticated_value"


def test_delete_client_state_without_auth_uses_system_user(client: TestClient, monkeypatch, mock_invoker: Invoker):
    """Test that deleting client state without authentication uses the system user."""
    # Mock ApiDependencies
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", MockApiDependencies(mock_invoker))
    monkeypatch.setattr("invokeai.app.api.routers.client_state.ApiDependencies", MockApiDependencies(mock_invoker))

    # Set a value for system user
    mock_invoker.services.client_state_persistence.set_by_key("system", "test_key", "system_value")

    # Delete without authentication - should delete system user's data
    response = client.post("/api/v1/client_state/default/delete")
    assert response.status_code == status.HTTP_200_OK

    # Verify it was deleted for system user
    value = mock_invoker.services.client_state_persistence.get_by_key("system", "test_key")
    assert value is None


def test_set_and_get_client_state(client: TestClient, admin_token: str):
    """Test that authenticated users can set and get their client state."""
    # Set a value
    set_response = client.post(
        "/api/v1/client_state/default/set_by_key?key=test_key",
        json="test_value",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert set_response.status_code == status.HTTP_200_OK
    assert set_response.json() == "test_value"

    # Get the value back
    get_response = client.get(
        "/api/v1/client_state/default/get_by_key?key=test_key",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert get_response.status_code == status.HTTP_200_OK
    assert get_response.json() == "test_value"


def test_client_state_isolation_between_users(client: TestClient, user1_token: str, user2_token: str):
    """Test that client state is isolated between different users."""
    # User 1 sets a value
    user1_set_response = client.post(
        "/api/v1/client_state/default/set_by_key?key=shared_key",
        json="user1_value",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert user1_set_response.status_code == status.HTTP_200_OK

    # User 2 sets a different value for the same key
    user2_set_response = client.post(
        "/api/v1/client_state/default/set_by_key?key=shared_key",
        json="user2_value",
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert user2_set_response.status_code == status.HTTP_200_OK

    # User 1 should still see their own value
    user1_get_response = client.get(
        "/api/v1/client_state/default/get_by_key?key=shared_key",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert user1_get_response.status_code == status.HTTP_200_OK
    assert user1_get_response.json() == "user1_value"

    # User 2 should see their own value
    user2_get_response = client.get(
        "/api/v1/client_state/default/get_by_key?key=shared_key",
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert user2_get_response.status_code == status.HTTP_200_OK
    assert user2_get_response.json() == "user2_value"


def test_get_nonexistent_key_returns_null(client: TestClient, admin_token: str):
    """Test that getting a nonexistent key returns null."""
    response = client.get(
        "/api/v1/client_state/default/get_by_key?key=nonexistent_key",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == status.HTTP_200_OK
    assert response.json() is None


def test_delete_client_state(client: TestClient, admin_token: str):
    """Test that users can delete their own client state."""
    # Set some values
    client.post(
        "/api/v1/client_state/default/set_by_key?key=key1",
        json="value1",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    client.post(
        "/api/v1/client_state/default/set_by_key?key=key2",
        json="value2",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    # Verify values exist
    get_response = client.get(
        "/api/v1/client_state/default/get_by_key?key=key1",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert get_response.json() == "value1"

    # Delete all client state
    delete_response = client.post(
        "/api/v1/client_state/default/delete",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert delete_response.status_code == status.HTTP_200_OK

    # Verify values are gone
    get_response = client.get(
        "/api/v1/client_state/default/get_by_key?key=key1",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert get_response.json() is None

    get_response = client.get(
        "/api/v1/client_state/default/get_by_key?key=key2",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert get_response.json() is None


def test_update_existing_key(client: TestClient, admin_token: str):
    """Test that updating an existing key works correctly."""
    # Set initial value
    client.post(
        "/api/v1/client_state/default/set_by_key?key=update_key",
        json="initial_value",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    # Update the value
    update_response = client.post(
        "/api/v1/client_state/default/set_by_key?key=update_key",
        json="updated_value",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert update_response.status_code == status.HTTP_200_OK

    # Verify the updated value
    get_response = client.get(
        "/api/v1/client_state/default/get_by_key?key=update_key",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert get_response.status_code == status.HTTP_200_OK
    assert get_response.json() == "updated_value"


def test_complex_json_values(client: TestClient, admin_token: str):
    """Test that complex JSON values can be stored and retrieved."""
    import json

    complex_dict = {"params": {"model": "test-model", "steps": 50}, "prompt": "a beautiful landscape"}
    complex_value = json.dumps(complex_dict)

    # Set complex value
    set_response = client.post(
        "/api/v1/client_state/default/set_by_key?key=complex_key",
        json=complex_value,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert set_response.status_code == status.HTTP_200_OK

    # Get it back
    get_response = client.get(
        "/api/v1/client_state/default/get_by_key?key=complex_key",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert get_response.status_code == status.HTTP_200_OK
    assert get_response.json() == complex_value
