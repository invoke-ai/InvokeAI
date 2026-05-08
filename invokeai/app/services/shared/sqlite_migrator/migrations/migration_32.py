import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration32Callback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS image_subfolder_move_jobs (
                id INTEGER PRIMARY KEY,
                state TEXT NOT NULL CHECK (
                    state IN ('planned', 'moving', 'moved', 'committed', 'error')
                ),
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                error_message TEXT
            );
            """
        )
        cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS image_subfolder_move_items (
                job_id INTEGER NOT NULL REFERENCES image_subfolder_move_jobs(id),
                image_name TEXT NOT NULL REFERENCES images(image_name),
                old_subfolder TEXT NOT NULL,
                new_subfolder TEXT NOT NULL,
                old_path TEXT,
                new_path TEXT,
                old_thumbnail_path TEXT,
                new_thumbnail_path TEXT,
                state TEXT NOT NULL CHECK (
                    state IN ('planned', 'moved', 'committed', 'error')
                ),
                error_message TEXT,
                PRIMARY KEY (job_id, image_name)
            );
            """
        )
        cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_image_subfolder_move_items_job_state
            ON image_subfolder_move_items(job_id, state);
            """
        )
        cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_image_subfolder_move_items_image_name
            ON image_subfolder_move_items(image_name);
            """
        )
        cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_image_subfolder_move_jobs_updated_at
            AFTER UPDATE
            ON image_subfolder_move_jobs FOR EACH ROW
            BEGIN
                UPDATE image_subfolder_move_jobs
                SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                WHERE id = old.id;
            END;
            """
        )


def build_migration_32() -> Migration:
    return Migration(
        from_version=31,
        to_version=32,
        callback=Migration32Callback(),
    )
