import json
from pathlib import Path

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.style_preset_records.style_preset_records_base import StylePresetRecordsStorageBase
from invokeai.app.services.style_preset_records.style_preset_records_common import (
    PresetType,
    StylePresetChanges,
    StylePresetNotFoundError,
    StylePresetRecordDTO,
    StylePresetWithoutId,
)
from invokeai.app.util.misc import uuid_string

# System user id used for default / shipped presets and for legacy rows pre-dating
# the per-user ownership columns added in migration 27.
SYSTEM_USER_ID = "system"


class SqliteStylePresetRecordsStorage(StylePresetRecordsStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        self._sync_default_style_presets()

    def get(self, style_preset_id: str) -> StylePresetRecordDTO:
        """Gets a style preset by ID."""
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT *
                FROM style_presets
                WHERE id = ?;
                """,
                (style_preset_id,),
            )
            row = cursor.fetchone()
        if row is None:
            raise StylePresetNotFoundError(f"Style preset with id {style_preset_id} not found")
        return StylePresetRecordDTO.from_dict(dict(row))

    def create(self, style_preset: StylePresetWithoutId, user_id: str) -> StylePresetRecordDTO:
        style_preset_id = uuid_string()
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                INSERT OR IGNORE INTO style_presets (
                    id,
                    name,
                    preset_data,
                    type,
                    user_id,
                    is_public
                )
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    style_preset_id,
                    style_preset.name,
                    style_preset.preset_data.model_dump_json(),
                    style_preset.type,
                    user_id,
                    1 if style_preset.is_public else 0,
                ),
            )
        return self.get(style_preset_id)

    def create_many(self, style_presets: list[StylePresetWithoutId], user_id: str) -> None:
        with self._db.transaction() as cursor:
            for style_preset in style_presets:
                style_preset_id = uuid_string()
                cursor.execute(
                    """--sql
                    INSERT OR IGNORE INTO style_presets (
                        id,
                        name,
                        preset_data,
                        type,
                        user_id,
                        is_public
                    )
                    VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (
                        style_preset_id,
                        style_preset.name,
                        style_preset.preset_data.model_dump_json(),
                        style_preset.type,
                        user_id,
                        1 if style_preset.is_public else 0,
                    ),
                )

        return None

    def update(self, style_preset_id: str, changes: StylePresetChanges) -> StylePresetRecordDTO:
        with self._db.transaction() as cursor:
            if changes.name is not None:
                cursor.execute(
                    """--sql
                    UPDATE style_presets
                    SET name = ?
                    WHERE id = ?;
                    """,
                    (changes.name, style_preset_id),
                )

            if changes.preset_data is not None:
                cursor.execute(
                    """--sql
                    UPDATE style_presets
                    SET preset_data = ?
                    WHERE id = ?;
                    """,
                    (changes.preset_data.model_dump_json(), style_preset_id),
                )

            if changes.is_public is not None:
                cursor.execute(
                    """--sql
                    UPDATE style_presets
                    SET is_public = ?
                    WHERE id = ?;
                    """,
                    (1 if changes.is_public else 0, style_preset_id),
                )

        return self.get(style_preset_id)

    def delete(self, style_preset_id: str) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                DELETE from style_presets
                WHERE id = ?;
                """,
                (style_preset_id,),
            )
        return None

    def get_many(
        self,
        type: PresetType | None = None,
        user_id: str | None = None,
        is_admin: bool = False,
    ) -> list[StylePresetRecordDTO]:
        clauses: list[str] = []
        params: list[object] = []

        if not is_admin:
            # Visible to non-admin: own + default + public.
            visibility = "(type = 'default' OR is_public = 1"
            if user_id is not None:
                visibility += " OR user_id = ?"
                params.append(user_id)
            visibility += ")"
            clauses.append(visibility)

        if type is not None:
            clauses.append("type = ?")
            params.append(type)

        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        query = f"SELECT * FROM style_presets {where}ORDER BY LOWER(name) ASC"

        with self._db.transaction() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        return [StylePresetRecordDTO.from_dict(dict(row)) for row in rows]

    def _sync_default_style_presets(self) -> None:
        """Syncs default style presets to the database. Internal use only."""
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                DELETE FROM style_presets
                WHERE type = "default";
                """
            )
        with open(Path(__file__).parent / Path("default_style_presets.json"), "r") as file:
            presets = json.load(file)
            for preset in presets:
                style_preset = StylePresetWithoutId.model_validate(preset)
                self.create(style_preset, user_id=SYSTEM_USER_ID)
