from typing import Optional

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.workflow_records.workflow_records_base import WorkflowRecordsStorageBase
from invokeai.app.services.workflow_records.workflow_records_common import (
    Workflow,
    WorkflowCategory,
    WorkflowNotFoundError,
    WorkflowRecordDTO,
    WorkflowRecordListItemDTO,
    WorkflowRecordListItemDTOValidator,
    WorkflowRecordOrderBy,
    WorkflowValidator,
    WorkflowWithoutID,
)


class SqliteWorkflowRecordsStorage(WorkflowRecordsStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._lock = db.lock
        self._conn = db.conn
        self._cursor = self._conn.cursor()
        self._create_tables()

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def get(self, workflow_id: str) -> WorkflowRecordDTO:
        """Gets a workflow by ID. Updates the opened_at column."""
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                UPDATE workflow_library
                SET opened_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                WHERE workflow_id = ?;
                """,
                (workflow_id,),
            )
            self._conn.commit()
            self._cursor.execute(
                """--sql
                SELECT workflow_id, workflow, name, created_at, updated_at, opened_at
                FROM workflow_library
                WHERE workflow_id = ?;
                """,
                (workflow_id,),
            )
            row = self._cursor.fetchone()
            if row is None:
                raise WorkflowNotFoundError(f"Workflow with id {workflow_id} not found")
            return WorkflowRecordDTO.from_dict(dict(row))
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()

    def create(self, workflow: WorkflowWithoutID) -> WorkflowRecordDTO:
        try:
            # Only user workflows may be created by this method
            assert workflow.meta.category is WorkflowCategory.User
            workflow_with_id = WorkflowValidator.validate_python(workflow.model_dump())
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                INSERT OR IGNORE INTO workflow_library (
                    workflow_id,
                    workflow
                )
                VALUES (?, ?);
                """,
                (workflow_with_id.id, workflow_with_id.model_dump_json()),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return self.get(workflow_with_id.id)

    def update(self, workflow: Workflow) -> WorkflowRecordDTO:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                UPDATE workflow_library
                SET workflow = ?
                WHERE workflow_id = ? AND category = 'user';
                """,
                (workflow.model_dump_json(), workflow.id),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return self.get(workflow.id)

    def delete(self, workflow_id: str) -> None:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                DELETE from workflow_library
                WHERE workflow_id = ? AND category = 'user';
                """,
                (workflow_id,),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return None

    def get_many(
        self,
        page: int,
        per_page: int,
        order_by: WorkflowRecordOrderBy,
        direction: SQLiteDirection,
        category: WorkflowCategory,
        filter_text: Optional[str] = None,
    ) -> PaginatedResults[WorkflowRecordListItemDTO]:
        try:
            self._lock.acquire()
            # sanitize!
            assert order_by in WorkflowRecordOrderBy
            assert direction in SQLiteDirection
            assert category in WorkflowCategory
            count_query = "SELECT COUNT(*) FROM workflow_library WHERE category = ?"
            main_query = """
                SELECT
                    workflow_id,
                    category,
                    name,
                    description,
                    created_at,
                    updated_at,
                    opened_at
                FROM workflow_library
                WHERE category = ?
                """
            main_params = [category.value]
            count_params = [category.value]
            stripped_filter_name = filter_text.strip() if filter_text else None
            if stripped_filter_name:
                filter_string = "%" + stripped_filter_name + "%"
                main_query += " AND name LIKE ? OR description LIKE ? "
                count_query += " AND name LIKE ? OR description LIKE ?;"
                main_params.extend([filter_string, filter_string])
                count_params.extend([filter_string, filter_string])

            main_query += f" ORDER BY {order_by.value} {direction.value} LIMIT ? OFFSET ?;"
            main_params.extend([per_page, page * per_page])
            self._cursor.execute(main_query, main_params)
            rows = self._cursor.fetchall()
            workflows = [WorkflowRecordListItemDTOValidator.validate_python(dict(row)) for row in rows]

            self._cursor.execute(count_query, count_params)
            total = self._cursor.fetchone()[0]
            pages = int(total / per_page) + 1

            return PaginatedResults(
                items=workflows,
                page=page,
                per_page=per_page,
                pages=pages,
                total=total,
            )
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()

    def _create_system_workflow(self, workflow: Workflow) -> None:
        try:
            self._lock.acquire()
            # Only system workflows may be managed by this method
            assert workflow.meta.category is WorkflowCategory.System
            self._cursor.execute(
                """--sql
                INSERT OR REPLACE INTO workflow_library (
                    workflow_id,
                    workflow
                )
                VALUES (?, ?);
                """,
                (workflow.id, workflow.model_dump_json()),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()

    def _update_system_workflow(self, workflow: Workflow) -> None:
        try:
            self._lock.acquire()
            # Only system workflows may be managed by this method
            assert workflow.meta.category is WorkflowCategory.System
            self._cursor.execute(
                """--sql
                UPDATE workflow_library
                SET workflow = ?
                WHERE workflow_id = ? AND category = 'system';
                """,
                (workflow.model_dump_json(), workflow.id),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()

    def _delete_system_workflow(self, workflow_id: str) -> None:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                DELETE FROM workflow_library
                WHERE workflow_id = ? AND category = 'system';
                """,
                (workflow_id,),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()

    def _get_all_system_workflows(self) -> list[Workflow]:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT workflow FROM workflow_library
                WHERE category = 'system';
                """
            )
            rows = self._cursor.fetchall()
            return [WorkflowValidator.validate_json(dict(row)["workflow"]) for row in rows]
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()

    def _create_tables(self) -> None:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                CREATE TABLE IF NOT EXISTS workflow_library (
                    workflow_id TEXT NOT NULL PRIMARY KEY,
                    workflow TEXT NOT NULL,
                    created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                    -- updated via trigger
                    updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                    -- updated manually when retrieving workflow
                    opened_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                    -- Generated columns, needed for indexing and searching
                    category TEXT GENERATED ALWAYS as (json_extract(workflow, '$.meta.category')) VIRTUAL NOT NULL,
                    name TEXT GENERATED ALWAYS as (json_extract(workflow, '$.name')) VIRTUAL NOT NULL,
                    description TEXT GENERATED ALWAYS as (json_extract(workflow, '$.description')) VIRTUAL NOT NULL
                );
                """
            )

            self._cursor.execute(
                """--sql
                CREATE TRIGGER IF NOT EXISTS tg_workflow_library_updated_at
                AFTER UPDATE
                ON workflow_library FOR EACH ROW
                BEGIN
                    UPDATE workflow_library
                    SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE workflow_id = old.workflow_id;
                END;
                """
            )

            self._cursor.execute(
                """--sql
                CREATE INDEX IF NOT EXISTS idx_workflow_library_created_at ON workflow_library(created_at);
                """
            )
            self._cursor.execute(
                """--sql
                CREATE INDEX IF NOT EXISTS idx_workflow_library_updated_at ON workflow_library(updated_at);
                """
            )
            self._cursor.execute(
                """--sql
                CREATE INDEX IF NOT EXISTS idx_workflow_library_opened_at ON workflow_library(opened_at);
                """
            )
            self._cursor.execute(
                """--sql
                CREATE INDEX IF NOT EXISTS idx_workflow_library_category ON workflow_library(category);
                """
            )
            self._cursor.execute(
                """--sql
                CREATE INDEX IF NOT EXISTS idx_workflow_library_name ON workflow_library(name);
                """
            )
            self._cursor.execute(
                """--sql
                CREATE INDEX IF NOT EXISTS idx_workflow_library_description ON workflow_library(description);
                """
            )

            # We do not need the original `workflows` table or `workflow_images` junction table.
            self._cursor.execute(
                """--sql
                DROP TABLE IF EXISTS workflow_images;
                """
            )
            self._cursor.execute(
                """--sql
                DROP TABLE IF EXISTS workflows;
                """
            )

            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
