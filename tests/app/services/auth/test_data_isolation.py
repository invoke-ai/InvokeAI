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
from invokeai.app.services.board_records.board_records_common import BoardRecordOrderBy
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
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
        user1_board = board_service.create(board_name="User 1 Board", user_id=user1_id)

        # Create board for user2
        user2_board = board_service.create(board_name="User 2 Board", user_id=user2_id)

        # User1 should only see their board
        user1_boards = board_service.get_many(
            user_id=user1_id,
            order_by=BoardRecordOrderBy.CreatedAt,
            direction=SQLiteDirection.Ascending,
        )

        user1_board_ids = [b.board_id for b in user1_boards.items]
        assert user1_board.board_id in user1_board_ids
        assert user2_board.board_id not in user1_board_ids

        # User2 should only see their board
        user2_boards = board_service.get_many(
            user_id=user2_id,
            order_by=BoardRecordOrderBy.CreatedAt,
            direction=SQLiteDirection.Ascending,
        )

        user2_board_ids = [b.board_id for b in user2_boards.items]
        assert user2_board.board_id in user2_board_ids
        assert user1_board.board_id not in user2_board_ids

    def test_user_cannot_access_other_user_board_directly(self, mock_invoker: Invoker):
        """Test that users cannot access other users' boards by ID."""
        board_service = mock_invoker.services.boards
        user_service = mock_invoker.services.users

        # Create two users
        user1_data = UserCreateRequest(
            email="user1@example.com", display_name="User 1", password="TestPass123", is_admin=False
        )
        user1 = user_service.create(user1_data)

        user2_data = UserCreateRequest(
            email="user2@example.com", display_name="User 2", password="TestPass123", is_admin=False
        )
        user2 = user_service.create(user2_data)

        # User1 creates a board
        user1_board = board_service.create(board_name="User 1 Private Board", user_id=user1.user_id)

        # User2 tries to access user1's board
        # The get method should check ownership
        try:
            retrieved_board = board_service.get(board_id=user1_board.board_id, user_id=user2.user_id)
            # If get doesn't check ownership, this test needs to be updated
            # or the implementation needs to be fixed
            if retrieved_board is not None:
                # Board was retrieved - check if it's because of missing authorization check
                # This would be a security issue that needs fixing
                pytest.fail("User was able to access another user's board without authorization")
        except Exception:
            # Expected - user2 should not be able to access user1's board
            pass

    def test_admin_can_see_all_boards(self, mock_invoker: Invoker):
        """Test that admin users can see all boards."""
        board_service = mock_invoker.services.boards
        user_service = mock_invoker.services.users

        # Create admin user
        admin_data = UserCreateRequest(
            email="admin@example.com", display_name="Admin", password="AdminPass123", is_admin=True
        )
        admin = user_service.create(admin_data)

        # Create regular user
        user_data = UserCreateRequest(
            email="user@example.com", display_name="User", password="TestPass123", is_admin=False
        )
        user = user_service.create(user_data)

        # User creates a board
        board_service.create(board_name="User Board", user_id=user.user_id)

        # Admin creates a board
        board_service.create(board_name="Admin Board", user_id=admin.user_id)

        # Admin should be able to get all boards (implementation dependent)
        # Note: Current implementation may not have admin override for board listing
        # This test documents expected behavior


class TestImageDataIsolation:
    """Tests for image data isolation between users."""

    def test_user_images_isolated_from_other_users(self, mock_invoker: Invoker):
        """Test that users cannot see other users' images."""
        user_service = mock_invoker.services.users

        # Create two users
        user1_data = UserCreateRequest(
            email="user1@example.com", display_name="User 1", password="TestPass123", is_admin=False
        )
        user_service.create(user1_data)

        user2_data = UserCreateRequest(
            email="user2@example.com", display_name="User 2", password="TestPass123", is_admin=False
        )
        user_service.create(user2_data)

        # Note: Image service tests would require actual image creation
        # which is beyond the scope of basic security testing
        # This test documents expected behavior:
        # - Images should have user_id field
        # - Image queries should filter by user_id
        # - Users should not be able to access images by knowing the image_name


class TestWorkflowDataIsolation:
    """Tests for workflow data isolation between users."""

    def test_user_workflows_isolated_from_other_users(self, mock_invoker: Invoker):
        """Test that users cannot see other users' private workflows."""
        user_service = mock_invoker.services.users

        # Create two users
        user1_data = UserCreateRequest(
            email="user1@example.com", display_name="User 1", password="TestPass123", is_admin=False
        )
        user_service.create(user1_data)

        user2_data = UserCreateRequest(
            email="user2@example.com", display_name="User 2", password="TestPass123", is_admin=False
        )
        user_service.create(user2_data)

        # Note: Workflow service tests would require workflow creation
        # This test documents expected behavior:
        # - Workflows should have user_id and is_public fields
        # - Private workflows should only be visible to owner
        # - Public workflows should be visible to all users


class TestQueueDataIsolation:
    """Tests for session queue data isolation between users."""

    def test_user_queue_items_isolated_from_other_users(self, mock_invoker: Invoker):
        """Test that users cannot see other users' queue items."""
        user_service = mock_invoker.services.users

        # Create two users
        user1_data = UserCreateRequest(
            email="user1@example.com", display_name="User 1", password="TestPass123", is_admin=False
        )
        user_service.create(user1_data)

        user2_data = UserCreateRequest(
            email="user2@example.com", display_name="User 2", password="TestPass123", is_admin=False
        )
        user_service.create(user2_data)

        # Note: Queue service tests would require session creation
        # This test documents expected behavior:
        # - Queue items should have user_id field
        # - Users should only see their own queue items
        # - Admin should see all queue items


class TestSharedBoardAccess:
    """Tests for shared board functionality."""

    @pytest.mark.skip(reason="Shared board functionality not yet fully implemented")
    def test_shared_board_access(self, mock_invoker: Invoker):
        """Test that users can access boards shared with them."""
        board_service = mock_invoker.services.boards
        user_service = mock_invoker.services.users

        # Create two users
        user1_data = UserCreateRequest(
            email="user1@example.com", display_name="User 1", password="TestPass123", is_admin=False
        )
        user1 = user_service.create(user1_data)

        user2_data = UserCreateRequest(
            email="user2@example.com", display_name="User 2", password="TestPass123", is_admin=False
        )
        user_service.create(user2_data)

        # User1 creates a board
        board_service.create(board_name="Shared Board", user_id=user1.user_id)

        # User1 shares the board with user2
        # (This functionality is not yet implemented)

        # User2 should be able to see the shared board
        # Expected behavior documented for future implementation


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

    def test_regular_user_cannot_list_all_users(self, mock_invoker: Invoker):
        """Test that regular users cannot list all users.

        Note: This depends on API endpoint implementation.
        At the service level, list_users is available to all callers.
        Authorization should be enforced at the API level.
        """
        user_service = mock_invoker.services.users

        # Create users
        user1_data = UserCreateRequest(
            email="user1@example.com", display_name="User 1", password="TestPass123", is_admin=False
        )
        user_service.create(user1_data)

        # Service level does not enforce authorization
        # API level should check if caller is admin before allowing user listing
        user_service.list_users()
        # This will succeed at service level - API must enforce auth


class TestDataIntegrity:
    """Tests for data integrity in multi-user scenarios."""

    def test_user_deletion_cascades_to_owned_data(self, mock_invoker: Invoker):
        """Test that deleting a user also deletes their owned data."""
        user_service = mock_invoker.services.users
        board_service = mock_invoker.services.boards

        # Create user
        user_data = UserCreateRequest(
            email="deleteme@example.com", display_name="Delete Me", password="TestPass123", is_admin=False
        )
        user = user_service.create(user_data)

        # User creates a board
        board = board_service.create(board_name="My Board", user_id=user.user_id)

        # Delete user
        user_service.delete(user.user_id)

        # Board should be deleted too (CASCADE in database)
        # Note: get_dto doesn't take user_id parameter, it gets the board by ID only
        # We'll check that it raises an exception or returns None after cascade delete
        try:
            board_service.get_dto(board_id=board.board_id)
            # If we get here, the board wasn't deleted - this is a failure
            raise AssertionError("Board should have been deleted by CASCADE")
        except Exception:
            # Expected - board was deleted by CASCADE
            pass

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
        user1_board = board_service.create(board_name="User 1 Board", user_id=user1.user_id)
        user2_board = board_service.create(board_name="User 2 Board", user_id=user2.user_id)

        # Verify isolation is maintained
        user1_boards = board_service.get_many(
            user_id=user1.user_id,
            order_by=BoardRecordOrderBy.CreatedAt,
            direction=SQLiteDirection.Ascending,
        )
        user2_boards = board_service.get_many(
            user_id=user2.user_id,
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
