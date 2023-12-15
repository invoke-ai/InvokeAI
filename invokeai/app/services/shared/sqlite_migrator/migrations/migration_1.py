import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration1Callback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        """Migration callback for database version 1."""

        self._create_board_images(cursor)
        self._create_boards(cursor)
        self._create_images(cursor)
        self._create_model_config(cursor)
        self._create_session_queue(cursor)
        self._create_workflow_images(cursor)
        self._create_workflows(cursor)

    def _create_board_images(self, cursor: sqlite3.Cursor) -> None:
        """Creates the `board_images` table, indices and triggers."""
        tables = [
            """--sql
            CREATE TABLE IF NOT EXISTS board_images (
                board_id TEXT NOT NULL,
                image_name TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Soft delete, currently unused
                deleted_at DATETIME,
                -- enforce one-to-many relationship between boards and images using PK
                -- (we can extend this to many-to-many later)
                PRIMARY KEY (image_name),
                FOREIGN KEY (board_id) REFERENCES boards (board_id) ON DELETE CASCADE,
                FOREIGN KEY (image_name) REFERENCES images (image_name) ON DELETE CASCADE
            );
            """
        ]

        indices = [
            "CREATE INDEX IF NOT EXISTS idx_board_images_board_id ON board_images (board_id);",
            "CREATE INDEX IF NOT EXISTS idx_board_images_board_id_created_at ON board_images (board_id, created_at);",
        ]

        triggers = [
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_board_images_updated_at
            AFTER UPDATE
            ON board_images FOR EACH ROW
            BEGIN
                UPDATE board_images SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE board_id = old.board_id AND image_name = old.image_name;
            END;
            """
        ]

        for stmt in tables + indices + triggers:
            cursor.execute(stmt)

    def _create_boards(self, cursor: sqlite3.Cursor) -> None:
        """Creates the `boards` table, indices and triggers."""
        tables = [
            """--sql
            CREATE TABLE IF NOT EXISTS boards (
                board_id TEXT NOT NULL PRIMARY KEY,
                board_name TEXT NOT NULL,
                cover_image_name TEXT,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Soft delete, currently unused
                deleted_at DATETIME,
                FOREIGN KEY (cover_image_name) REFERENCES images (image_name) ON DELETE SET NULL
            );
            """
        ]

        indices = ["CREATE INDEX IF NOT EXISTS idx_boards_created_at ON boards (created_at);"]

        triggers = [
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_boards_updated_at
            AFTER UPDATE
            ON boards FOR EACH ROW
            BEGIN
                UPDATE boards SET updated_at = current_timestamp
                    WHERE board_id = old.board_id;
            END;
            """
        ]

        for stmt in tables + indices + triggers:
            cursor.execute(stmt)

    def _create_images(self, cursor: sqlite3.Cursor) -> None:
        """Creates the `images` table, indices and triggers. Adds the `starred` column."""

        tables = [
            """--sql
            CREATE TABLE IF NOT EXISTS images (
                image_name TEXT NOT NULL PRIMARY KEY,
                -- This is an enum in python, unrestricted string here for flexibility
                image_origin TEXT NOT NULL,
                -- This is an enum in python, unrestricted string here for flexibility
                image_category TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                session_id TEXT,
                node_id TEXT,
                metadata TEXT,
                is_intermediate BOOLEAN DEFAULT FALSE,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Soft delete, currently unused
                deleted_at DATETIME
            );
            """
        ]

        indices = [
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_images_image_name ON images(image_name);",
            "CREATE INDEX IF NOT EXISTS idx_images_image_origin ON images(image_origin);",
            "CREATE INDEX IF NOT EXISTS idx_images_image_category ON images(image_category);",
            "CREATE INDEX IF NOT EXISTS idx_images_created_at ON images(created_at);",
        ]

        triggers = [
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_images_updated_at
            AFTER UPDATE
            ON images FOR EACH ROW
            BEGIN
                UPDATE images SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE image_name = old.image_name;
            END;
            """
        ]

        # Add the 'starred' column to `images` if it doesn't exist
        cursor.execute("PRAGMA table_info(images)")
        columns = [column[1] for column in cursor.fetchall()]

        if "starred" not in columns:
            tables.append("ALTER TABLE images ADD COLUMN starred BOOLEAN DEFAULT FALSE;")
            indices.append("CREATE INDEX IF NOT EXISTS idx_images_starred ON images(starred);")

        for stmt in tables + indices + triggers:
            cursor.execute(stmt)

    def _create_model_config(self, cursor: sqlite3.Cursor) -> None:
        """Creates the `model_config` table, `model_manager_metadata` table, indices and triggers."""

        tables = [
            """--sql
            CREATE TABLE IF NOT EXISTS model_config (
                id TEXT NOT NULL PRIMARY KEY,
                -- The next 3 fields are enums in python, unrestricted string here
                base TEXT NOT NULL,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                original_hash TEXT, -- could be null
                -- Serialized JSON representation of the whole config object,
                -- which will contain additional fields from subclasses
                config TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- unique constraint on combo of name, base and type
                UNIQUE(name, base, type)
            );
            """,
            """--sql
            CREATE TABLE IF NOT EXISTS model_manager_metadata (
                metadata_key TEXT NOT NULL PRIMARY KEY,
                metadata_value TEXT NOT NULL
            );
            """,
        ]

        # Add trigger for `updated_at`.
        triggers = [
            """--sql
            CREATE TRIGGER IF NOT EXISTS model_config_updated_at
            AFTER UPDATE
            ON model_config FOR EACH ROW
            BEGIN
                UPDATE model_config SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE id = old.id;
            END;
            """
        ]

        # Add indexes for searchable fields
        indices = [
            "CREATE INDEX IF NOT EXISTS base_index ON model_config(base);",
            "CREATE INDEX IF NOT EXISTS type_index ON model_config(type);",
            "CREATE INDEX IF NOT EXISTS name_index ON model_config(name);",
            "CREATE UNIQUE INDEX IF NOT EXISTS path_index ON model_config(path);",
        ]

        for stmt in tables + indices + triggers:
            cursor.execute(stmt)

    def _create_session_queue(self, cursor: sqlite3.Cursor) -> None:
        tables = [
            """--sql
            CREATE TABLE IF NOT EXISTS session_queue (
                item_id INTEGER PRIMARY KEY AUTOINCREMENT, -- used for ordering, cursor pagination
                batch_id TEXT NOT NULL, -- identifier of the batch this queue item belongs to
                queue_id TEXT NOT NULL, -- identifier of the queue this queue item belongs to
                session_id TEXT NOT NULL UNIQUE, -- duplicated data from the session column, for ease of access
                field_values TEXT, -- NULL if no values are associated with this queue item
                session TEXT NOT NULL, -- the session to be executed
                status TEXT NOT NULL DEFAULT 'pending', -- the status of the queue item, one of 'pending', 'in_progress', 'completed', 'failed', 'canceled'
                priority INTEGER NOT NULL DEFAULT 0, -- the priority, higher is more important
                error TEXT, -- any errors associated with this queue item
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')), -- updated via trigger
                started_at DATETIME, -- updated via trigger
                completed_at DATETIME -- updated via trigger, completed items are cleaned up on application startup
                -- Ideally this is a FK, but graph_executions uses INSERT OR REPLACE, and REPLACE triggers the ON DELETE CASCADE...
                -- FOREIGN KEY (session_id) REFERENCES graph_executions (id) ON DELETE CASCADE
            );
            """
        ]

        indices = [
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_session_queue_item_id ON session_queue(item_id);",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_session_queue_session_id ON session_queue(session_id);",
            "CREATE INDEX IF NOT EXISTS idx_session_queue_batch_id ON session_queue(batch_id);",
            "CREATE INDEX IF NOT EXISTS idx_session_queue_created_priority ON session_queue(priority);",
            "CREATE INDEX IF NOT EXISTS idx_session_queue_created_status ON session_queue(status);",
        ]

        triggers = [
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_session_queue_completed_at
            AFTER UPDATE OF status ON session_queue
            FOR EACH ROW
            WHEN
            NEW.status = 'completed'
            OR NEW.status = 'failed'
            OR NEW.status = 'canceled'
            BEGIN
            UPDATE session_queue
            SET completed_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
            WHERE item_id = NEW.item_id;
            END;
            """,
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_session_queue_started_at
            AFTER UPDATE OF status ON session_queue
            FOR EACH ROW
            WHEN
            NEW.status = 'in_progress'
            BEGIN
            UPDATE session_queue
            SET started_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
            WHERE item_id = NEW.item_id;
            END;
            """,
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_session_queue_updated_at
            AFTER UPDATE
            ON session_queue FOR EACH ROW
            BEGIN
                UPDATE session_queue
                SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                WHERE item_id = old.item_id;
            END;
            """,
        ]

        for stmt in tables + indices + triggers:
            cursor.execute(stmt)

    def _create_workflow_images(self, cursor: sqlite3.Cursor) -> None:
        tables = [
            """--sql
            CREATE TABLE IF NOT EXISTS workflow_images (
                workflow_id TEXT NOT NULL,
                image_name TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Soft delete, currently unused
                deleted_at DATETIME,
                -- enforce one-to-many relationship between workflows and images using PK
                -- (we can extend this to many-to-many later)
                PRIMARY KEY (image_name),
                FOREIGN KEY (workflow_id) REFERENCES workflows (workflow_id) ON DELETE CASCADE,
                FOREIGN KEY (image_name) REFERENCES images (image_name) ON DELETE CASCADE
            );
            """
        ]

        indices = [
            "CREATE INDEX IF NOT EXISTS idx_workflow_images_workflow_id ON workflow_images (workflow_id);",
            "CREATE INDEX IF NOT EXISTS idx_workflow_images_workflow_id_created_at ON workflow_images (workflow_id, created_at);",
        ]

        triggers = [
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_workflow_images_updated_at
            AFTER UPDATE
            ON workflow_images FOR EACH ROW
            BEGIN
                UPDATE workflow_images SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE workflow_id = old.workflow_id AND image_name = old.image_name;
            END;
            """
        ]

        for stmt in tables + indices + triggers:
            cursor.execute(stmt)

    def _create_workflows(self, cursor: sqlite3.Cursor) -> None:
        tables = [
            """--sql
            CREATE TABLE IF NOT EXISTS workflows (
                workflow TEXT NOT NULL,
                workflow_id TEXT GENERATED ALWAYS AS (json_extract(workflow, '$.id')) VIRTUAL NOT NULL UNIQUE, -- gets implicit index
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')) -- updated via trigger
            );
            """
        ]

        triggers = [
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_workflows_updated_at
            AFTER UPDATE
            ON workflows FOR EACH ROW
            BEGIN
                UPDATE workflows
                SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                WHERE workflow_id = old.workflow_id;
            END;
            """
        ]

        for stmt in tables + triggers:
            cursor.execute(stmt)


def build_migration_1() -> Migration:
    """
    Builds the migration from database version 0 (init) to 1.

    This migration represents the state of the database circa InvokeAI v3.4.0, which was the last
    version to not use migrations to manage the database.

    As such, this migration does include some ALTER statements, and the SQL statements are written
    to be idempotent.

    - Create `board_images` junction table
    - Create `boards` table
    - Create `images` table, add `starred` column
    - Create `model_config` table
    - Create `session_queue` table
    - Create `workflow_images` junction table
    - Create `workflows` table
    """

    migration_1 = Migration(
        from_version=0,
        to_version=1,
        callback=Migration1Callback(),
    )

    return migration_1
