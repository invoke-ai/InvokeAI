"""Abstract base class for user service."""

from abc import ABC, abstractmethod

from invokeai.app.services.users.users_common import UserCreateRequest, UserDTO, UserUpdateRequest


class UserServiceBase(ABC):
    """High-level service for user management."""

    @abstractmethod
    def create(self, user_data: UserCreateRequest) -> UserDTO:
        """Create a new user.

        Args:
            user_data: User creation data

        Returns:
            The created user

        Raises:
            ValueError: If email already exists or password is weak
        """
        pass

    @abstractmethod
    def get(self, user_id: str) -> UserDTO | None:
        """Get user by ID.

        Args:
            user_id: The user ID

        Returns:
            UserDTO if found, None otherwise
        """
        pass

    @abstractmethod
    def get_by_email(self, email: str) -> UserDTO | None:
        """Get user by email.

        Args:
            email: The email address

        Returns:
            UserDTO if found, None otherwise
        """
        pass

    @abstractmethod
    def update(self, user_id: str, changes: UserUpdateRequest) -> UserDTO:
        """Update user.

        Args:
            user_id: The user ID
            changes: Fields to update

        Returns:
            The updated user

        Raises:
            ValueError: If user not found or password is weak
        """
        pass

    @abstractmethod
    def delete(self, user_id: str) -> None:
        """Delete user.

        Args:
            user_id: The user ID

        Raises:
            ValueError: If user not found
        """
        pass

    @abstractmethod
    def authenticate(self, email: str, password: str) -> UserDTO | None:
        """Authenticate user credentials.

        Args:
            email: User email
            password: User password

        Returns:
            UserDTO if authentication successful, None otherwise
        """
        pass

    @abstractmethod
    def has_admin(self) -> bool:
        """Check if any admin user exists.

        Returns:
            True if at least one admin user exists, False otherwise
        """
        pass

    @abstractmethod
    def create_admin(self, user_data: UserCreateRequest) -> UserDTO:
        """Create an admin user (for initial setup).

        Args:
            user_data: User creation data

        Returns:
            The created admin user

        Raises:
            ValueError: If admin already exists or password is weak
        """
        pass

    @abstractmethod
    def list_users(self, limit: int = 100, offset: int = 0) -> list[UserDTO]:
        """List all users.

        Args:
            limit: Maximum number of users to return
            offset: Number of users to skip

        Returns:
            List of users
        """
        pass
