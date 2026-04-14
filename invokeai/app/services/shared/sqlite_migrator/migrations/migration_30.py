"""Migration 30: Make preexisting system-owned workflows public.

Migration 28 added user_id and is_public columns to workflow_library, but
assigned preexisting workflows user_id='system' with is_public=FALSE. This
caused those workflows to disappear from users' libraries because the query
filter scopes by user_id and excludes non-public workflows owned by other
users. This migration fixes the issue by marking all system-owned workflows
as public so they are visible to all users.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration30Callback:
    """Migration to make system-owned workflows publicly visible."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._make_system_workflows_public(cursor)

    def _make_system_workflows_public(self, cursor: sqlite3.Cursor) -> None:
        """Set is_public=TRUE for all workflows owned by the 'system' user."""
        cursor.execute("UPDATE workflow_library SET is_public = TRUE WHERE user_id = 'system';")


def build_migration_30() -> Migration:
    """Builds the migration object for migrating from version 29 to version 30.

    This migration marks all preexisting system-owned workflows as public
    so they remain visible to all users after the multiuser migration.
    """
    return Migration(
        from_version=29,
        to_version=30,
        callback=Migration30Callback(),
    )
