"""Tests for multiuser boards functionality."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.users.users_common import UserCreateRequest


class MockApiDependencies(ApiDependencies):
    """Mock API dependencies for testing."""

    invoker: Invoker

    def __init__(self, invoker: Invoker) -> None:
        self.invoker = invoker


@pytest.fixture
def setup_jwt_secret():
    """Initialize JWT secret for token generation."""
    from invokeai.app.services.auth.token_service import set_jwt_secret

    # Use a test secret key
    set_jwt_secret("test-secret-key-for-unit-tests-only-do-not-use-in-production")


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


def setup_test_user(
    mock_invoker: Invoker,
    email: str,
    display_name: str,
    password: str = "TestPass123",
    is_admin: bool = False,
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
        json={"email": email, "password": password, "remember_me": False},
    )
    assert response.status_code == 200
    return response.json()["token"]


@pytest.fixture
def enable_multiuser_for_tests(monkeypatch: Any, mock_invoker: Invoker):
    """Enable multiuser mode and patch ApiDependencies for all relevant routers."""
    mock_invoker.services.configuration.multiuser = True
    # Provide a mock board_images service so delete/image_names endpoints don't 500
    mock_board_images = MagicMock()
    mock_board_images.get_all_board_image_names_for_board.return_value = []
    mock_invoker.services.board_images = mock_board_images

    mock_deps = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.boards.ApiDependencies", mock_deps)
    yield


@pytest.fixture
def admin_token(setup_jwt_secret: None, enable_multiuser_for_tests: Any, mock_invoker: Invoker, client: TestClient):
    """Create an admin user and return a login token."""
    setup_test_user(mock_invoker, "admin@test.com", "Test Admin", is_admin=True)
    return get_user_token(client, "admin@test.com")


@pytest.fixture
def user1_token(enable_multiuser_for_tests: Any, mock_invoker: Invoker, client: TestClient, admin_token: str):
    """Create a regular user and return a login token."""
    setup_test_user(mock_invoker, "user1@test.com", "User One", is_admin=False)
    return get_user_token(client, "user1@test.com")


@pytest.fixture
def user2_token(enable_multiuser_for_tests: Any, mock_invoker: Invoker, client: TestClient, admin_token: str):
    """Create a second regular user and return a login token."""
    setup_test_user(mock_invoker, "user2@test.com", "User Two", is_admin=False)
    return get_user_token(client, "user2@test.com")


# ---------------------------------------------------------------------------
# Basic auth requirement tests
# ---------------------------------------------------------------------------


def test_create_board_requires_auth(enable_multiuser_for_tests: Any, client: TestClient):
    """Test that creating a board requires authentication."""
    response = client.post("/api/v1/boards/?board_name=Test+Board")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_list_boards_requires_auth(enable_multiuser_for_tests: Any, client: TestClient):
    """Test that listing boards requires authentication."""
    response = client.get("/api/v1/boards/?all=true")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_get_board_requires_auth(enable_multiuser_for_tests: Any, client: TestClient):
    """Test that getting a board requires authentication."""
    response = client.get("/api/v1/boards/some-board-id")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_update_board_requires_auth(enable_multiuser_for_tests: Any, client: TestClient):
    """Test that updating a board requires authentication."""
    response = client.patch("/api/v1/boards/some-board-id", json={"board_name": "New Name"})
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_delete_board_requires_auth(enable_multiuser_for_tests: Any, client: TestClient):
    """Test that deleting a board requires authentication."""
    response = client.delete("/api/v1/boards/some-board-id")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_list_board_image_names_requires_auth(enable_multiuser_for_tests: Any, client: TestClient):
    """Test that listing board image names requires authentication."""
    response = client.get("/api/v1/boards/some-board-id/image_names")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


# ---------------------------------------------------------------------------
# Basic create / list tests
# ---------------------------------------------------------------------------


def test_create_board_with_auth(client: TestClient, admin_token: str):
    """Test that authenticated users can create boards."""
    response = client.post(
        "/api/v1/boards/?board_name=My+Test+Board",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["board_name"] == "My Test Board"
    assert "board_id" in data


def test_list_boards_with_auth(client: TestClient, admin_token: str):
    """Test that authenticated users can list their boards."""
    # First create a board
    client.post(
        "/api/v1/boards/?board_name=Listed+Board",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    # Now list boards
    response = client.get(
        "/api/v1/boards/?all=true",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == status.HTTP_200_OK
    boards = response.json()
    assert isinstance(boards, list)
    board_names = [b["board_name"] for b in boards]
    assert "Listed Board" in board_names


def test_user_boards_are_isolated(client: TestClient, admin_token: str, user1_token: str):
    """Test that boards are isolated between users."""
    # Admin creates a board
    admin_response = client.post(
        "/api/v1/boards/?board_name=Admin+Board",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert admin_response.status_code == status.HTTP_201_CREATED

    # Admin can see their own board
    list_response = client.get(
        "/api/v1/boards/?all=true",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_response.status_code == status.HTTP_200_OK
    boards = list_response.json()
    board_names = [b["board_name"] for b in boards]
    assert "Admin Board" in board_names

    # user1 should not see admin's board in their own listing
    user1_list = client.get(
        "/api/v1/boards/?all=true",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert user1_list.status_code == status.HTTP_200_OK
    user1_board_names = [b["board_name"] for b in user1_list.json()]
    assert "Admin Board" not in user1_board_names


# ---------------------------------------------------------------------------
# Ownership enforcement: get_board
# ---------------------------------------------------------------------------


def test_get_board_owner_succeeds(client: TestClient, user1_token: str):
    """Test that the board owner can retrieve their own board."""
    create = client.post(
        "/api/v1/boards/?board_name=User1+Board",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert create.status_code == status.HTTP_201_CREATED
    board_id = create.json()["board_id"]

    response = client.get(
        f"/api/v1/boards/{board_id}",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["board_id"] == board_id


def test_get_board_other_user_forbidden(client: TestClient, user1_token: str, user2_token: str):
    """Test that a non-owner cannot retrieve another user's board."""
    create = client.post(
        "/api/v1/boards/?board_name=User1+Private+Board",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert create.status_code == status.HTTP_201_CREATED
    board_id = create.json()["board_id"]

    response = client.get(
        f"/api/v1/boards/{board_id}",
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_get_board_admin_can_access_any_board(client: TestClient, admin_token: str, user1_token: str):
    """Test that an admin can retrieve any user's board."""
    create = client.post(
        "/api/v1/boards/?board_name=User1+Board+For+Admin",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert create.status_code == status.HTTP_201_CREATED
    board_id = create.json()["board_id"]

    response = client.get(
        f"/api/v1/boards/{board_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == status.HTTP_200_OK


# ---------------------------------------------------------------------------
# Ownership enforcement: update_board
# ---------------------------------------------------------------------------


def test_update_board_owner_succeeds(client: TestClient, user1_token: str):
    """Test that the board owner can update their own board."""
    create = client.post(
        "/api/v1/boards/?board_name=Original+Name",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert create.status_code == status.HTTP_201_CREATED
    board_id = create.json()["board_id"]

    response = client.patch(
        f"/api/v1/boards/{board_id}",
        json={"board_name": "Updated Name"},
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["board_name"] == "Updated Name"


def test_update_board_other_user_forbidden(client: TestClient, user1_token: str, user2_token: str):
    """Test that a non-owner cannot update another user's board."""
    create = client.post(
        "/api/v1/boards/?board_name=User1+Board+To+Update",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert create.status_code == status.HTTP_201_CREATED
    board_id = create.json()["board_id"]

    response = client.patch(
        f"/api/v1/boards/{board_id}",
        json={"board_name": "Hijacked Name"},
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_update_board_admin_can_update_any_board(client: TestClient, admin_token: str, user1_token: str):
    """Test that an admin can update any user's board."""
    create = client.post(
        "/api/v1/boards/?board_name=User1+Board+Admin+Update",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert create.status_code == status.HTTP_201_CREATED
    board_id = create.json()["board_id"]

    response = client.patch(
        f"/api/v1/boards/{board_id}",
        json={"board_name": "Admin Updated Name"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["board_name"] == "Admin Updated Name"


# ---------------------------------------------------------------------------
# Ownership enforcement: delete_board
# ---------------------------------------------------------------------------


def test_delete_board_owner_succeeds(client: TestClient, user1_token: str):
    """Test that the board owner can delete their own board."""
    create = client.post(
        "/api/v1/boards/?board_name=Board+To+Delete",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert create.status_code == status.HTTP_201_CREATED
    board_id = create.json()["board_id"]

    response = client.delete(
        f"/api/v1/boards/{board_id}",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["board_id"] == board_id


def test_delete_board_other_user_forbidden(client: TestClient, user1_token: str, user2_token: str):
    """Test that a non-owner cannot delete another user's board."""
    create = client.post(
        "/api/v1/boards/?board_name=User1+Board+To+Delete",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert create.status_code == status.HTTP_201_CREATED
    board_id = create.json()["board_id"]

    response = client.delete(
        f"/api/v1/boards/{board_id}",
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_delete_board_admin_can_delete_any_board(client: TestClient, admin_token: str, user1_token: str):
    """Test that an admin can delete any user's board."""
    create = client.post(
        "/api/v1/boards/?board_name=User1+Board+Admin+Delete",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert create.status_code == status.HTTP_201_CREATED
    board_id = create.json()["board_id"]

    response = client.delete(
        f"/api/v1/boards/{board_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == status.HTTP_200_OK


# ---------------------------------------------------------------------------
# Ownership enforcement: list_all_board_image_names
# ---------------------------------------------------------------------------


def test_list_board_image_names_owner_succeeds(client: TestClient, user1_token: str):
    """Test that the board owner can list image names for their board."""
    create = client.post(
        "/api/v1/boards/?board_name=User1+Images+Board",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert create.status_code == status.HTTP_201_CREATED
    board_id = create.json()["board_id"]

    response = client.get(
        f"/api/v1/boards/{board_id}/image_names",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert response.status_code == status.HTTP_200_OK
    assert isinstance(response.json(), list)


def test_list_board_image_names_other_user_forbidden(client: TestClient, user1_token: str, user2_token: str):
    """Test that a non-owner cannot list image names for another user's board."""
    create = client.post(
        "/api/v1/boards/?board_name=User1+Private+Images+Board",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert create.status_code == status.HTTP_201_CREATED
    board_id = create.json()["board_id"]

    response = client.get(
        f"/api/v1/boards/{board_id}/image_names",
        headers={"Authorization": f"Bearer {user2_token}"},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_list_board_image_names_admin_can_access_any_board(client: TestClient, admin_token: str, user1_token: str):
    """Test that an admin can list image names for any user's board."""
    create = client.post(
        "/api/v1/boards/?board_name=User1+Board+Admin+Images",
        headers={"Authorization": f"Bearer {user1_token}"},
    )
    assert create.status_code == status.HTTP_201_CREATED
    board_id = create.json()["board_id"]

    response = client.get(
        f"/api/v1/boards/{board_id}/image_names",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == status.HTTP_200_OK


def test_list_board_image_names_none_board_no_auth_check(enable_multiuser_for_tests: Any, client: TestClient):
    """Test that listing image names for the 'none' board requires auth but no ownership check."""
    # The 'none' board is the uncategorized images board — no ownership check needed,
    # but auth is still required in multiuser mode.
    response = client.get("/api/v1/boards/none/image_names")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


# ---------------------------------------------------------------------------
# Misc tests
# ---------------------------------------------------------------------------


def test_enqueue_batch_requires_auth(enable_multiuser_for_tests: Any, client: TestClient):
    """Test that enqueuing a batch requires authentication."""
    response = client.post(
        "/api/v1/queue/default/enqueue_batch",
        json={
            "batch": {
                "batch_id": "test-batch",
                "data": [],
                "graph": {"nodes": {}, "edges": []},
            },
            "prepend": False,
        },
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
