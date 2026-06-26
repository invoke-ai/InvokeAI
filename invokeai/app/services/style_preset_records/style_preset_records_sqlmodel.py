import json
from pathlib import Path

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.style_preset_records.style_preset_records_base import StylePresetRecordsStorageBase
from invokeai.app.services.style_preset_records.style_preset_records_common import (
    PresetType,
    StylePresetChanges,
    StylePresetRecordDTO,
    StylePresetWithoutId,
)


class SqlModelStylePresetRecordsStorage(StylePresetRecordsStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db
        self._q = db.queries

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        self._sync_default_style_presets()

    def get(self, style_preset_id: str) -> StylePresetRecordDTO:
        return self._q.style_presets_get(style_preset_id)

    def create(self, style_preset: StylePresetWithoutId, user_id: str = "system") -> StylePresetRecordDTO:
        return self._q.style_presets_create(style_preset, user_id)

    def create_many(self, style_presets: list[StylePresetWithoutId], user_id: str = "system") -> None:
        self._q.style_presets_create_many(style_presets, user_id)

    def update(self, style_preset_id: str, changes: StylePresetChanges) -> StylePresetRecordDTO:
        return self._q.style_presets_update(style_preset_id, changes)

    def delete(self, style_preset_id: str) -> None:
        self._q.style_presets_delete(style_preset_id)

    def get_many(
        self,
        type: PresetType | None = None,
        user_id: str | None = None,
        is_admin: bool = False,
    ) -> list[StylePresetRecordDTO]:
        return self._q.style_presets_get_many(type=type, user_id=user_id, is_admin=is_admin)

    def _sync_default_style_presets(self) -> None:
        """Syncs default style presets to the database."""
        # Delete existing defaults, then re-create them from file
        self._q.style_presets_delete_defaults()

        with open(Path(__file__).parent / Path("default_style_presets.json"), "r") as file:
            presets = json.load(file)
            for preset in presets:
                style_preset = StylePresetWithoutId.model_validate(preset)
                self.create(style_preset)
