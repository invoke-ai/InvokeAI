"""Integration tests for multi-user data isolation.

Tests to ensure users can only access their own data and cannot access
other users' data unless explicitly shared.
"""

import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.board_access.board_access_common import BoardViewer
from invokeai.app.services.board_records.board_records_common import BoardRecordOrderBy
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import SQLiteDirection
from invokeai.app.services.users.users_common import UserCreateRequest


@pytest.fixture(autouse=True, scope="module")
def client(invokeai_root_dir: Path) -> TestClient:
    """Create a test client for the FastAPI app."""
    os.environ["INVOKEAI_ROOT"] = invokeai_root_dir.as_posix()
    return TestClient(app)


@pytest.fixture(autouse=True)
def enable_multiuser_for_auth_tests(mock_invoker: Invoker) -> None:
    """Enable multiuser mode for auth tests.

    Auth tests need multiuser mode enabled since the login/setup endpoints
    return 403 when multiuser is disabled.
    """
    mock_invoker.services.configuration.multiuser = True


class MockApiDependencies(ApiDependencies):
    """Mock API dependencies for testing."""

    invoker: Invoker

    def __init__(self, invoker) -> None:
        self.invoker = invoker


def create_user_and_login(
    mock_invoker: Invoker, client: TestClient, monkeypatch: Any, email: str, password: str, is_admin: bool = False
) -> tuple[str, str]:
    """Helper to create a user, login, and return (user_id, token)."""
    monkeypatch.setattr("invokeai.app.api.routers.auth.ApiDependencies", MockApiDependencies(mock_invoker))
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", MockApiDependencies(mock_invoker))

    user_service = mock_invoker.services.users
    user_data = UserCreateRequest(
        email=email,
        display_name=f"User {email}",
        password=password,
        is_admin=is_admin,
    )
    user = user_service.create(user_data)

    # Login to get token
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": email,
            "password": password,
            "remember_me": False,
        },
    )

    assert response.status_code == 200
    token = response.json()["token"]

    return user.user_id, token


class TestBoardDataIsolation:
    """Tests for board data isolation between users."""

    def test_user_can_only_see_own_boards(self, monkeypatch: Any, mock_invoker: Invoker, client: TestClient):
        """Test that users can only see their own boards."""
        monkeypatch.setattr("invokeai.app.api.routers.boards.ApiDependencies", MockApiDependencies(mock_invoker))

        # Create two users
        user1_id, user1_token = create_user_and_login(
            mock_invoker, client, monkeypatch, "user1@example.com", "TestPass123"
        )
        user2_id, user2_token = create_user_and_login(
            mock_invoker, client, monkeypatch, "user2@example.com", "TestPass123"
        )

        # Create board for user1
        board_service = mock_invoker.services.boards
        user1_board = board_service.create(
            board_name="User 1 Board", user_id=user1_id, viewer=BoardViewer(user_id=user1_id, is_admin=False)
        )

        # Create board for user2
        user2_board = board_service.create(
            board_name="User 2 Board", user_id=user2_id, viewer=BoardViewer(user_id=user2_id, is_admin=False)
        )

        # User1 should only see their board
        user1_boards = board_service.get_many(
            user_id=user1_id,
            is_admin=False,
            order_by=BoardRecordOrderBy.CreatedAt,
            direction=SQLiteDirection.Ascending,
        )

        user1_board_ids = [b.board_id for b in user1_boards.items]
        assert user1_board.board_id in user1_board_ids
        assert user2_board.board_id not in user1_board_ids

        # User2 should only see their board
        user2_boards = board_service.get_many(
            user_id=user2_id,
            is_admin=False,
            order_by=BoardRecordOrderBy.CreatedAt,
            direction=SQLiteDirection.Ascending,
        )

        user2_board_ids = [b.board_id for b in user2_boards.items]
        assert user2_board.board_id in user2_board_ids
        assert user1_board.board_id not in user2_board_ids


class TestAdminAuthorization:
    """Tests for admin-only functionality."""

    def test_regular_user_cannot_create_admin(self, mock_invoker: Invoker):
        """Test that regular users cannot create admin accounts."""
        user_service = mock_invoker.services.users

        # Create first admin
        admin_data = UserCreateRequest(
            email="admin@example.com", display_name="Admin", password="AdminPass123", is_admin=True
        )
        user_service.create(admin_data)

        # Try to create another admin (should fail)
        with pytest.raises(ValueError, match="already exists"):
            another_admin_data = UserCreateRequest(
                email="another@example.com", display_name="Another Admin", password="AdminPass123"
            )
            user_service.create_admin(another_admin_data)


class TestDataIntegrity:
    """Tests for data integrity in multi-user scenarios."""

    def test_a_deleted_users_boards_stay_for_admins(self, mock_invoker: Invoker):
        """Deleting an account takes its projects, but its boards and their media stay, for admins, in the Library."""
        user_service = mock_invoker.services.users
        board_service = mock_invoker.services.boards
        user = user_service.create(
            UserCreateRequest(email="deleteme@example.com", display_name="Delete Me", password="TestPass123")
        )
        project = mock_invoker.services.project_records.create(user.user_id, "Project", {})
        board = board_service.create(
            board_name="My Board",
            user_id=user.user_id,
            project_id=project.project_id,
            viewer=BoardViewer(user_id=user.user_id, is_admin=False),
        )

        user_service.delete(user.user_id)

        kept = board_service.get_dto(board.board_id, viewer=BoardViewer(user_id="admin", is_admin=True))
        assert (kept.board_name, kept.project_id) == ("My Board", None)

    def test_concurrent_user_operations_maintain_isolation(self, mock_invoker: Invoker):
        """Test that concurrent operations from different users maintain data isolation.

        This is a basic test - comprehensive concurrency testing would require
        multiple threads/processes and more complex scenarios.
        """
        user_service = mock_invoker.services.users
        board_service = mock_invoker.services.boards

        # Create two users
        user1_data = UserCreateRequest(
            email="user1@example.com", display_name="User 1", password="TestPass123", is_admin=False
        )
        user1 = user_service.create(user1_data)

        user2_data = UserCreateRequest(
            email="user2@example.com", display_name="User 2", password="TestPass123", is_admin=False
        )
        user2 = user_service.create(user2_data)

        # Both users create boards
        user1_board = board_service.create(
            board_name="User 1 Board", user_id=user1.user_id, viewer=BoardViewer(user_id=user1.user_id, is_admin=False)
        )
        user2_board = board_service.create(
            board_name="User 2 Board", user_id=user2.user_id, viewer=BoardViewer(user_id=user2.user_id, is_admin=False)
        )

        # Verify isolation is maintained
        user1_boards = board_service.get_many(
            user_id=user1.user_id,
            is_admin=False,
            order_by=BoardRecordOrderBy.CreatedAt,
            direction=SQLiteDirection.Ascending,
        )
        user2_boards = board_service.get_many(
            user_id=user2.user_id,
            is_admin=False,
            order_by=BoardRecordOrderBy.CreatedAt,
            direction=SQLiteDirection.Ascending,
        )

        user1_board_ids = [b.board_id for b in user1_boards.items]
        user2_board_ids = [b.board_id for b in user2_boards.items]

        # Each user should only see their own board
        assert user1_board.board_id in user1_board_ids
        assert user2_board.board_id not in user1_board_ids

        assert user2_board.board_id in user2_board_ids
        assert user1_board.board_id not in user2_board_ids
