"""Migration 25: Add multi-user support.

This migration adds the database schema for multi-user support, including:
- users table for user accounts
- user_sessions table for session management
- user_invitations table for invitation system
- shared_boards table for board sharing
- Adding user_id columns to existing tables for data ownership
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration25Callback:
    """Migration to add multi-user support."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._create_users_table(cursor)
        self._create_user_sessions_table(cursor)
        self._create_user_invitations_table(cursor)
        self._create_shared_boards_table(cursor)
        self._update_boards_table(cursor)
        self._update_images_table(cursor)
        self._update_workflows_table(cursor)
        self._update_session_queue_table(cursor)
        self._update_style_presets_table(cursor)
        self._create_system_user(cursor)

    def _create_users_table(self, cursor: sqlite3.Cursor) -> None:
        """Create users table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT NOT NULL PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                display_name TEXT,
                password_hash TEXT NOT NULL,
                is_admin BOOLEAN NOT NULL DEFAULT FALSE,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                last_login_at DATETIME
            );
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_is_admin ON users(is_admin);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);")

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS tg_users_updated_at
            AFTER UPDATE ON users FOR EACH ROW
            BEGIN
                UPDATE users SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                WHERE user_id = old.user_id;
            END;
        """)

    def _create_user_sessions_table(self, cursor: sqlite3.Cursor) -> None:
        """Create user_sessions table for session management."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT NOT NULL PRIMARY KEY,
                user_id TEXT NOT NULL,
                token_hash TEXT NOT NULL,
                expires_at DATETIME NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                last_activity_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_token_hash ON user_sessions(token_hash);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);")

    def _create_user_invitations_table(self, cursor: sqlite3.Cursor) -> None:
        """Create user_invitations table for invitation system."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_invitations (
                invitation_id TEXT NOT NULL PRIMARY KEY,
                email TEXT NOT NULL,
                invited_by TEXT NOT NULL,
                invitation_code TEXT NOT NULL UNIQUE,
                is_admin BOOLEAN NOT NULL DEFAULT FALSE,
                expires_at DATETIME NOT NULL,
                used_at DATETIME,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                FOREIGN KEY (invited_by) REFERENCES users(user_id) ON DELETE CASCADE
            );
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_invitations_email ON user_invitations(email);")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_invitations_invitation_code ON user_invitations(invitation_code);"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_invitations_expires_at ON user_invitations(expires_at);")

    def _create_shared_boards_table(self, cursor: sqlite3.Cursor) -> None:
        """Create shared_boards table for board sharing."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shared_boards (
                board_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                can_edit BOOLEAN NOT NULL DEFAULT FALSE,
                shared_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                PRIMARY KEY (board_id, user_id),
                FOREIGN KEY (board_id) REFERENCES boards(board_id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_shared_boards_user_id ON shared_boards(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_shared_boards_board_id ON shared_boards(board_id);")

    def _update_boards_table(self, cursor: sqlite3.Cursor) -> None:
        """Add user_id and is_public columns to boards table."""
        # Check if boards table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='boards';")
        if cursor.fetchone() is None:
            return

        # Check if user_id column exists
        cursor.execute("PRAGMA table_info(boards);")
        columns = [row[1] for row in cursor.fetchall()]

        if "user_id" not in columns:
            cursor.execute("ALTER TABLE boards ADD COLUMN user_id TEXT DEFAULT 'system';")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_boards_user_id ON boards(user_id);")

        if "is_public" not in columns:
            cursor.execute("ALTER TABLE boards ADD COLUMN is_public BOOLEAN NOT NULL DEFAULT FALSE;")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_boards_is_public ON boards(is_public);")

    def _update_images_table(self, cursor: sqlite3.Cursor) -> None:
        """Add user_id column to images table."""
        # Check if images table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images';")
        if cursor.fetchone() is None:
            return

        cursor.execute("PRAGMA table_info(images);")
        columns = [row[1] for row in cursor.fetchall()]

        if "user_id" not in columns:
            cursor.execute("ALTER TABLE images ADD COLUMN user_id TEXT DEFAULT 'system';")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_user_id ON images(user_id);")

    def _update_workflows_table(self, cursor: sqlite3.Cursor) -> None:
        """Add user_id and is_public columns to workflows table."""
        # Check if workflows table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='workflows';")
        if cursor.fetchone() is None:
            return

        cursor.execute("PRAGMA table_info(workflows);")
        columns = [row[1] for row in cursor.fetchall()]

        if "user_id" not in columns:
            cursor.execute("ALTER TABLE workflows ADD COLUMN user_id TEXT DEFAULT 'system';")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_workflows_user_id ON workflows(user_id);")

        if "is_public" not in columns:
            cursor.execute("ALTER TABLE workflows ADD COLUMN is_public BOOLEAN NOT NULL DEFAULT FALSE;")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_workflows_is_public ON workflows(is_public);")

    def _update_session_queue_table(self, cursor: sqlite3.Cursor) -> None:
        """Add user_id column to session_queue table."""
        # Check if session_queue table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='session_queue';")
        if cursor.fetchone() is None:
            return

        cursor.execute("PRAGMA table_info(session_queue);")
        columns = [row[1] for row in cursor.fetchall()]

        if "user_id" not in columns:
            cursor.execute("ALTER TABLE session_queue ADD COLUMN user_id TEXT DEFAULT 'system';")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_queue_user_id ON session_queue(user_id);")

    def _update_style_presets_table(self, cursor: sqlite3.Cursor) -> None:
        """Add user_id and is_public columns to style_presets table."""
        # Check if style_presets table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='style_presets';")
        if cursor.fetchone() is None:
            return

        cursor.execute("PRAGMA table_info(style_presets);")
        columns = [row[1] for row in cursor.fetchall()]

        if "user_id" not in columns:
            cursor.execute("ALTER TABLE style_presets ADD COLUMN user_id TEXT DEFAULT 'system';")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_style_presets_user_id ON style_presets(user_id);")

        if "is_public" not in columns:
            cursor.execute("ALTER TABLE style_presets ADD COLUMN is_public BOOLEAN NOT NULL DEFAULT FALSE;")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_style_presets_is_public ON style_presets(is_public);")

    def _create_system_user(self, cursor: sqlite3.Cursor) -> None:
        """Create system user for backward compatibility.

        The system user is NOT an admin - it's just used to own existing data
        from before multi-user support was added. Real admin users should be
        created through the /auth/setup endpoint.
        """
        cursor.execute("""
            INSERT OR IGNORE INTO users (user_id, email, display_name, password_hash, is_admin, is_active)
            VALUES ('system', 'system@system.invokeai', 'System', '', FALSE, TRUE);
        """)


def build_migration_25() -> Migration:
    """Builds the migration object for migrating from version 24 to version 25.

    This migration adds multi-user support to the database schema.
    """
    return Migration(
        from_version=24,
        to_version=25,
        callback=Migration25Callback(),
    )
