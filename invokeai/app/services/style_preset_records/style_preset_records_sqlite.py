from pathlib import Path
from typing import Optional

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.style_preset_records.style_preset_records_base import StylePresetRecordsStorageBase
from invokeai.app.services.style_preset_records.style_preset_records_common import (
    StylePresetNotFoundError,
    StylePresetRecordDTO,
    StylePresetWithoutId,
)
from invokeai.app.services.workflow_records.workflow_records_common import (
    WorkflowRecordOrderBy,
)
from invokeai.app.util.misc import uuid_string


class SqliteStylePresetRecordsStorage(StylePresetRecordsStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._lock = db.lock
        self._conn = db.conn
        self._cursor = self._conn.cursor()

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def get(self, id: str) -> StylePresetRecordDTO:
        """Gets a style preset by ID."""
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT *
                FROM style_presets
                WHERE id = ?;
                """,
                (id,),
            )
            row = self._cursor.fetchone()
            if row is None:
                raise StylePresetNotFoundError(f"Style preset with id {id} not found")
            return StylePresetRecordDTO.from_dict(dict(row))
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()

    def create(self, style_preset: StylePresetWithoutId) -> StylePresetRecordDTO:
        id = uuid_string()
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                INSERT OR IGNORE INTO style_presets (
                    id,
                    name,
                    preset_data,
                )
                VALUES (?, ?, ?);
                """,
                (
                    id,
                    style_preset.name,
                    style_preset.preset_data,
                ),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return self.get(id)

    def update(self, id: str, changes: StylePresetWithoutId) -> StylePresetRecordDTO:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                UPDATE style_presets
                SET preset_data = ?
                WHERE id = ? ;
                """,
                (
                    changes.preset_data,
                    id,
                ),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return self.get(id)

    def delete(self, id: str) -> None:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                DELETE from style_presets
                WHERE id = ? ;
                """,
                (id,),
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
        query: Optional[str] = None,
    ) -> PaginatedResults[StylePresetRecordDTO]:
        try:
            self._lock.acquire()
            # sanitize!
            assert order_by in WorkflowRecordOrderBy
            assert direction in SQLiteDirection
            count_query = "SELECT COUNT(*) FROM style_presets"
            main_query = """
                SELECT
                    *
                FROM style_presets
                """
            main_params: list[int | str] = []
            count_params: list[int | str] = []
            stripped_query = query.strip() if query else None
            if stripped_query:
                wildcard_query = "%" + stripped_query + "%"
                main_query += " AND name LIKE ? OR description LIKE ? "
                count_query += " AND name LIKE ? OR description LIKE ?;"
                main_params.extend([wildcard_query, wildcard_query])
                count_params.extend([wildcard_query, wildcard_query])

            main_query += f" ORDER BY {order_by.value} {direction.value} LIMIT ? OFFSET ?;"
            main_params.extend([per_page, page * per_page])
            self._cursor.execute(main_query, main_params)
            rows = self._cursor.fetchall()
            style_presets = [StylePresetRecordDTO.from_dict(dict(row)) for row in rows]

            self._cursor.execute(count_query, count_params)
            total = self._cursor.fetchone()[0]
            pages = total // per_page + (total % per_page > 0)

            return PaginatedResults(
                items=style_presets,
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
