
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.style_preset_records.style_preset_records_base import StylePresetRecordsStorageBase
from invokeai.app.services.style_preset_records.style_preset_records_common import (
    StylePresetChanges,
    StylePresetNotFoundError,
    StylePresetRecordDTO,
    StylePresetWithoutId,
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
                    preset_data
                )
                VALUES (?, ?, ?);
                """,
                (
                    id,
                    style_preset.name,
                    style_preset.preset_data.model_dump_json(),
                ),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
        return self.get(id)

    def update(self, id: str, changes: StylePresetChanges) -> StylePresetRecordDTO:
        try:
            self._lock.acquire()
            # Change the name of a style preset
            if changes.name is not None:
                self._cursor.execute(
                    """--sql
                    UPDATE style_presets
                    SET name = ?
                    WHERE id = ?;
                    """,
                    (changes.name, id),
                )

            # Change the preset data for a style preset
            if changes.preset_data is not None:
                self._cursor.execute(
                    """--sql
                    UPDATE style_presets
                    SET preset_data = ?
                    WHERE id = ?;
                    """,
                    (changes.preset_data.model_dump_json(), id),
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
    ) -> list[StylePresetRecordDTO]:
        try:
            self._lock.acquire()
            main_query = """
                SELECT
                    *
                FROM style_presets
                ORDER BY name ASC
                """

            self._cursor.execute(main_query)
            rows = self._cursor.fetchall()
            style_presets = [StylePresetRecordDTO.from_dict(dict(row)) for row in rows]

            return style_presets
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._lock.release()
