"""Add a token revocation epoch to the users table.

A JWT proves identity and is otherwise self-contained: nothing in the database can
make an already-issued token stop verifying. Authorization fields are re-derived
from the user record on every request, so demotion and deactivation take effect
immediately — but a *password change* had no way to invalidate anything. A stolen
token therefore outlived the password rotation meant to evict the thief, and the
sliding-window middleware renewed it indefinitely while the account stayed active.

``token_epoch`` closes that: it is stamped into every minted token and compared
against the user record on every authenticated request. Bumping it invalidates all
tokens issued before the bump, which is the general "revoke everything issued so
far" primitive the auth layer was missing.

Existing rows and existing tokens both start at 0, so upgrading does not log anyone
out — only an actual bump revokes.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class AddUserTokenEpochCallback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("PRAGMA table_info(users);")
        existing_columns = {row[1] for row in cursor.fetchall()}
        if "token_epoch" not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN token_epoch INTEGER NOT NULL DEFAULT 0;")


def build_migration() -> Migration:
    """Add ``users.token_epoch``.

    Depends on migration_27, which creates the users table.
    """
    return Migration(
        id="2026_07_31_add_user_token_epoch",
        depends_on="migration_27",
        callback=AddUserTokenEpochCallback(),
    )
