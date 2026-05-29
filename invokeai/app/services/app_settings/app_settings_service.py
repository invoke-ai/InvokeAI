"""Service for managing application-level settings stored in the database."""

from typing import Optional

from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class AppSettingsService:
    """Service for accessing application-level settings from the database.

    This service provides a simple key-value store for application-level configuration
    that needs to be persisted across restarts, such as JWT secrets.
    """

    def __init__(self, db: SqliteDatabase) -> None:
        """Initialize the app settings service.

        Args:
            db: The SQLite database instance
        """
        self._db = db

    def get(self, key: str) -> Optional[str]:
        """Get a setting value by key.

        Args:
            key: The setting key

        Returns:
            The setting value if found, None otherwise
        """
        try:
            with self._db.transaction() as cursor:
                cursor.execute("SELECT value FROM app_settings WHERE key = ?;", (key,))
                row = cursor.fetchone()
                return row[0] if row else None
        except Exception:
            return None

    def set(self, key: str, value: str) -> None:
        """Set a setting value.

        Args:
            key: The setting key
            value: The setting value
        """
        with self._db.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO app_settings (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW');
                """,
                (key, value),
            )

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
