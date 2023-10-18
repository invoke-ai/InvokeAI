import sqlite3
import threading
from typing import Optional, cast

from invokeai.app.services.shared.sqlite import SqliteDatabase
from invokeai.app.services.workflow_image_records.workflow_image_records_base import WorkflowImageRecordsStorageBase


class SqliteWorkflowImageRecordsStorage(WorkflowImageRecordsStorageBase):
    """SQLite implementation of WorkflowImageRecordsStorageBase."""

    _conn: sqlite3.Connection
    _cursor: sqlite3.Cursor
    _lock: threading.RLock

    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._lock = db.lock
        self._conn = db.conn
        self._cursor = self._conn.cursor()

        try:
            self._lock.acquire()
            self._create_tables()
            self._conn.commit()
        finally:
            self._lock.release()

    def _create_tables(self) -> None:
        # Create the `workflow_images` junction table.
        self._cursor.execute(
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
        )

        # Add index for workflow id
        self._cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_workflow_images_workflow_id ON workflow_images (workflow_id);
            """
        )

        # Add index for workflow id, sorted by created_at
        self._cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_workflow_images_workflow_id_created_at ON workflow_images (workflow_id, created_at);
            """
        )

        # Add trigger for `updated_at`.
        self._cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_workflow_images_updated_at
            AFTER UPDATE
            ON workflow_images FOR EACH ROW
            BEGIN
                UPDATE workflow_images SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE workflow_id = old.workflow_id AND image_name = old.image_name;
            END;
            """
        )

    def create(
        self,
        workflow_id: str,
        image_name: str,
    ) -> None:
        """Creates a workflow-image record."""
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                INSERT INTO workflow_images (workflow_id, image_name)
                VALUES (?, ?)
                ON CONFLICT (image_name) DO UPDATE SET workflow_id = ?;
                """,
                (workflow_id, image_name, workflow_id),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise e
        finally:
            self._lock.release()

    def get_workflow_for_image(
        self,
        image_name: str,
    ) -> Optional[str]:
        """Gets an image's workflow id, if it has one."""
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT workflow_id
                FROM workflow_images
                WHERE image_name = ?;
                """,
                (image_name,),
            )
            result = self._cursor.fetchone()
            if result is None:
                return None
            return cast(str, result[0])
        except sqlite3.Error as e:
            self._conn.rollback()
            raise e
        finally:
            self._lock.release()
