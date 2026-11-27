"""Drop the SQLite triggers: the application sets what they set, on every backend.

The triggers stamped `updated_at` on update, `started_at` and `completed_at` on queue status changes, and counted
`session_revision` when a queue item's session changed. MySQL and MariaDB databases never had them, so the
application sets those columns itself; with both writing on SQLite, the triggers only overwrote the application's
values, and in a format of their own for boards and client state (`CURRENT_TIMESTAMP`, without milliseconds).

One behaviour changes with them: `session_revision` now counts every write of a session, as it does on servers,
rather than only those that change its text. Its one reader compares it for equality.
"""

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import (
    PortableMigration,
    PortableMigrationContext,
)

# Literal rather than read from the database: a migration keeps meaning what it meant when it was written.
_TRIGGERS = (
    "models_updated_at",
    "style_presets",
    "tg_app_settings_updated_at",
    "tg_board_images_updated_at",
    "tg_board_videos_updated_at",
    "tg_boards_updated_at",
    "tg_client_state_updated_at",
    "tg_image_projections_updated_at",
    "tg_image_subfolder_move_jobs_updated_at",
    "tg_images_updated_at",
    "tg_projects_updated_at",
    "tg_session_queue_completed_at",
    "tg_session_queue_session_revision",
    "tg_session_queue_started_at",
    "tg_session_queue_updated_at",
    "tg_system_prompts_updated_at",
    "tg_users_updated_at",
    "tg_videos_updated_at",
    "tg_wildcards_updated_at",
    "tg_workflow_library_updated_at",
)


def _drop_sqlite_triggers(context: PortableMigrationContext) -> None:
    if context.conn.dialect.name != "sqlite":
        return
    for name in _TRIGGERS:
        context.conn.exec_driver_sql(f'DROP TRIGGER IF EXISTS "{name}"')


def build_migration() -> PortableMigration:
    return PortableMigration(
        id="2026_10_07_drop_sqlite_triggers",
        depends_on="2026_10_07_add_session_queue_admission_lock",
        callback=_drop_sqlite_triggers,
    )
