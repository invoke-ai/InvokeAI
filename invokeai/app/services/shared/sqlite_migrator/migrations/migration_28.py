"""Migration 28: Add app_settings table for storing JWT secret and other app-level settings.

This migration adds the app_settings table to securely store application-level configuration:
- Creates app_settings table with key-value storage
- Generates a random cryptographically secure JWT secret key
- Stores the JWT secret in the database for token signing/verification
"""

import secrets
import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration28Callback:
    """Migration to add app_settings table and JWT secret."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._create_app_settings_table(cursor)
        self._generate_jwt_secret(cursor)

    def _create_app_settings_table(self, cursor: sqlite3.Cursor) -> None:
        """Create app_settings table for storing application-level configuration."""
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT NOT NULL PRIMARY KEY,
                value TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW'))
            );
            """
        )

        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS tg_app_settings_updated_at
            AFTER UPDATE ON app_settings
            FOR EACH ROW
            BEGIN
                UPDATE app_settings SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                WHERE key = OLD.key;
            END;
            """
        )

    def _generate_jwt_secret(self, cursor: sqlite3.Cursor) -> None:
        """Generate and store a cryptographically secure JWT secret key.

        The secret is a 64-character hexadecimal string (256 bits of entropy),
        which is suitable for HS256 JWT signing.
        """
        # Check if JWT secret already exists
        cursor.execute("SELECT value FROM app_settings WHERE key = 'jwt_secret';")
        existing_secret = cursor.fetchone()

        if existing_secret is None:
            # Generate a new cryptographically secure secret (256 bits)
            jwt_secret = secrets.token_hex(32)  # 32 bytes = 256 bits = 64 hex characters

            # Store in database
            cursor.execute(
                "INSERT INTO app_settings (key, value) VALUES ('jwt_secret', ?);",
                (jwt_secret,),
            )


def build_migration_28() -> Migration:
    """Builds the migration object for migrating from version 27 to version 28.

    This migration adds the app_settings table and generates a JWT secret for token signing.
    """
    return Migration(
        from_version=27,
        to_version=28,
        callback=Migration28Callback(),
    )
