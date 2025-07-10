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

    def create(self, style_preset: StylePresetWithoutId) -> StylePresetRecordDTO:
        style_preset_id = uuid_string()
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                INSERT OR IGNORE INTO style_presets (
                    id,
                    name,
                    preset_data,
                    type
                )
                VALUES (?, ?, ?, ?);
                """,
                (
                    style_preset_id,
                    style_preset.name,
                    style_preset.preset_data.model_dump_json(),
                    style_preset.type,
                ),
            )
        return self.get(style_preset_id)

    def create_many(self, style_presets: list[StylePresetWithoutId]) -> None:
        style_preset_ids = []
        with self._db.transaction() as cursor:
            for style_preset in style_presets:
                style_preset_id = uuid_string()
                style_preset_ids.append(style_preset_id)
                cursor.execute(
                    """--sql
                    INSERT OR IGNORE INTO style_presets (
                        id,
                        name,
                        preset_data,
                        type
                    )
                    VALUES (?, ?, ?, ?);
                    """,
                    (
                        style_preset_id,
                        style_preset.name,
                        style_preset.preset_data.model_dump_json(),
                        style_preset.type,
                    ),
                )

        return None

    def update(self, style_preset_id: str, changes: StylePresetChanges) -> StylePresetRecordDTO:
        with self._db.transaction() as cursor:
            # Change the name of a style preset
            if changes.name is not None:
                cursor.execute(
                    """--sql
                    UPDATE style_presets
                    SET name = ?
                    WHERE id = ?;
                    """,
                    (changes.name, style_preset_id),
                )

            # Change the preset data for a style preset
            if changes.preset_data is not None:
                cursor.execute(
                    """--sql
                    UPDATE style_presets
                    SET preset_data = ?
                    WHERE id = ?;
                    """,
                    (changes.preset_data.model_dump_json(), style_preset_id),
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

    def get_many(self, type: PresetType | None = None) -> list[StylePresetRecordDTO]:
        with self._db.transaction() as cursor:
            main_query = """
                SELECT
                    *
                FROM style_presets
                """

            if type is not None:
                main_query += "WHERE type = ? "

            main_query += "ORDER BY LOWER(name) ASC"

            if type is not None:
                cursor.execute(main_query, (type,))
            else:
                cursor.execute(main_query)

            rows = cursor.fetchall()
        style_presets = [StylePresetRecordDTO.from_dict(dict(row)) for row in rows]

        return style_presets

    def _sync_default_style_presets(self) -> None:
        """Syncs default style presets to the database. Internal use only."""
        with self._db.transaction() as cursor:
            # First delete all existing default style presets
            cursor.execute(
                """--sql
                DELETE FROM style_presets
                WHERE type = "default";
                """
            )
        # Next, parse and create the default style presets
        with open(Path(__file__).parent / Path("default_style_presets.json"), "r") as file:
            presets = json.load(file)
            for preset in presets:
                style_preset = StylePresetWithoutId.model_validate(preset)
                self.create(style_preset)
