"""Integration tests for authentication router endpoints."""

import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.users.users_common import UserCreateRequest


@pytest.fixture(autouse=True, scope="module")
def client(invokeai_root_dir: Path) -> TestClient:
    """Create a test client for the FastAPI app."""
    os.environ["INVOKEAI_ROOT"] = invokeai_root_dir.as_posix()
    return TestClient(app)


class MockApiDependencies(ApiDependencies):
    """Mock API dependencies for testing."""

    invoker: Invoker

    def __init__(self, invoker) -> None:
        self.invoker = invoker


def setup_test_user(mock_invoker: Invoker, email: str = "test@example.com", password: str = "TestPass123") -> str:
    """Helper to create a test user and return user_id."""
    user_service = mock_invoker.services.users
    user_data = UserCreateRequest(
        email=email,
        display_name="Test User",
        password=password,
        is_admin=False,
    )
    user = user_service.create(user_data)
    return user.user_id


def setup_test_admin(mock_invoker: Invoker, email: str = "admin@example.com", password: str = "AdminPass123") -> str:
    """Helper to create a test admin user and return user_id."""
    user_service = mock_invoker.services.users
    user_data = UserCreateRequest(
        email=email,
        display_name="Admin User",
        password=password,
        is_admin=True,
    )
    user = user_service.create(user_data)
    return user.user_id


def test_login_success(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    """Test successful login with valid credentials."""
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", MockApiDependencies(mock_invoker))

    # Create a test user
    setup_test_user(mock_invoker, "test@example.com", "TestPass123")

    # Attempt login
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "test@example.com",
            "password": "TestPass123",
            "remember_me": False,
        },
    )

    assert response.status_code == 200
    json_response = response.json()
    assert "token" in json_response
    assert "user" in json_response
    assert "expires_in" in json_response
    assert json_response["user"]["email"] == "test@example.com"
    assert json_response["user"]["is_admin"] is False


def test_login_with_remember_me(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    """Test login with remember_me flag sets longer expiration."""
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", MockApiDependencies(mock_invoker))

    setup_test_user(mock_invoker, "test2@example.com", "TestPass123")

    # Login with remember_me=True
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "test2@example.com",
            "password": "TestPass123",
            "remember_me": True,
        },
    )

    assert response.status_code == 200
    json_response = response.json()
    # Remember me should give 7 days = 604800 seconds
    assert json_response["expires_in"] == 604800


def test_login_invalid_password(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    """Test login fails with invalid password."""
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", MockApiDependencies(mock_invoker))

    setup_test_user(mock_invoker, "test3@example.com", "TestPass123")

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "test3@example.com",
            "password": "WrongPassword",
            "remember_me": False,
        },
    )

    assert response.status_code == 401
    assert "Incorrect email or password" in response.json()["detail"]


def test_login_nonexistent_user(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    """Test login fails with nonexistent user."""
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", MockApiDependencies(mock_invoker))

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "nonexistent@example.com",
            "password": "TestPass123",
            "remember_me": False,
        },
    )

    assert response.status_code == 401
    assert "Incorrect email or password" in response.json()["detail"]


def test_login_inactive_user(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    """Test login fails with inactive user."""
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", MockApiDependencies(mock_invoker))

    user_id = setup_test_user(mock_invoker, "inactive@example.com", "TestPass123")

    # Deactivate the user
    user_service = mock_invoker.services.users
    from invokeai.app.services.users.users_common import UserUpdateRequest

    user_service.update(user_id, UserUpdateRequest(is_active=False))

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "inactive@example.com",
            "password": "TestPass123",
            "remember_me": False,
        },
    )

    assert response.status_code == 403
    assert "disabled" in response.json()["detail"]


def test_logout(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    """Test logout endpoint."""
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", MockApiDependencies(mock_invoker))
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", MockApiDependencies(mock_invoker))

    setup_test_user(mock_invoker, "test4@example.com", "TestPass123")

    # Login first to get token
    login_response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "test4@example.com",
            "password": "TestPass123",
            "remember_me": False,
        },
    )
    token = login_response.json()["token"]

    # Logout with token
    response = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json()["success"] is True


def test_logout_without_token(client: TestClient) -> None:
    """Test logout fails without authentication token."""
    response = client.post("/api/v1/auth/logout")

    assert response.status_code == 401


def test_get_current_user_info(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    """Test getting current user info with valid token."""
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", MockApiDependencies(mock_invoker))
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", MockApiDependencies(mock_invoker))

    setup_test_user(mock_invoker, "test5@example.com", "TestPass123")

    # Login to get token
    login_response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "test5@example.com",
            "password": "TestPass123",
            "remember_me": False,
        },
    )
    token = login_response.json()["token"]

    # Get user info
    response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["email"] == "test5@example.com"
    assert json_response["display_name"] == "Test User"
    assert json_response["is_admin"] is False


def test_get_current_user_info_without_token(client: TestClient) -> None:
    """Test getting user info fails without token."""
    response = client.get("/api/v1/auth/me")

    assert response.status_code == 401


def test_get_current_user_info_invalid_token(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    """Test getting user info fails with invalid token."""
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", MockApiDependencies(mock_invoker))

    response = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer invalid_token"})

    assert response.status_code == 401


def test_setup_admin_first_time(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    """Test setting up first admin user."""
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", MockApiDependencies(mock_invoker))

    response = client.post(
        "/api/v1/auth/setup",
        json={
            "email": "admin@example.com",
            "display_name": "Admin User",
            "password": "AdminPass123",
        },
    )

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["success"] is True
    assert json_response["user"]["email"] == "admin@example.com"
    assert json_response["user"]["is_admin"] is True


def test_setup_admin_already_exists(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    """Test setup fails when admin already exists."""
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", MockApiDependencies(mock_invoker))

    # Create first admin
    setup_test_admin(mock_invoker, "admin1@example.com", "AdminPass123")

    # Try to setup another admin
    response = client.post(
        "/api/v1/auth/setup",
        json={
            "email": "admin2@example.com",
            "display_name": "Second Admin",
            "password": "AdminPass123",
        },
    )

    assert response.status_code == 400
    assert "already configured" in response.json()["detail"]


def test_setup_admin_weak_password(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    """Test setup fails with weak password."""
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", MockApiDependencies(mock_invoker))

    response = client.post(
        "/api/v1/auth/setup",
        json={
            "email": "admin3@example.com",
            "display_name": "Admin User",
            "password": "weak",
        },
    )

    assert response.status_code == 400
    assert "Password" in response.json()["detail"]


def test_admin_user_token_has_admin_flag(monkeypatch: Any, mock_invoker: Invoker, client: TestClient) -> None:
    """Test that admin user login returns token with admin flag."""
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", MockApiDependencies(mock_invoker))

    setup_test_admin(mock_invoker, "admin4@example.com", "AdminPass123")

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "admin4@example.com",
            "password": "AdminPass123",
            "remember_me": False,
        },
    )

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["user"]["is_admin"] is True
