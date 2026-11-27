"""Service for managing application-level settings stored in the database."""

from typing import Optional

from invokeai.app.services.shared.database.database import Database


class AppSettingsService:
    """Service for accessing application-level settings from the database.

    This service provides a simple key-value store for application-level configuration
    that needs to be persisted across restarts, such as JWT secrets.
    """

    def __init__(self, database: Database) -> None:
        self._queries = database.queries

    def get(self, key: str) -> Optional[str]:
        """Get a setting value by key.

        Args:
            key: The setting key

        Returns:
            The setting value if found, None otherwise
        """
        return self._queries.app_settings.get(key)

    def get_jwt_secret(self) -> str:
        """Get the JWT secret key from the database.

        Returns:
            The JWT secret key

        Raises:
            RuntimeError: If the JWT secret is not found in the database
        """
        secret = self.get("jwt_secret")
        if secret is None:
            raise RuntimeError(
                "JWT secret not found in database. This should have been created during database migration. "
                "Please ensure database migrations have been run successfully."
            )
        return secret
