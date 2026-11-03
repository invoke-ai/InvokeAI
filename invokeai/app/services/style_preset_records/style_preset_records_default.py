import json
from pathlib import Path
from typing import Optional

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import Queries
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


def _found(preset: Optional[StylePresetRecordDTO], style_preset_id: str) -> StylePresetRecordDTO:
    if preset is None:
        raise StylePresetNotFoundError(f"Style preset with id {style_preset_id} not found")
    return preset


class StylePresetRecordsStorage(StylePresetRecordsStorageBase):
    def __init__(self, database: Database) -> None:
        super().__init__()
        self._queries = database.queries

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        self._sync_default_style_presets()

    def get(self, style_preset_id: str) -> StylePresetRecordDTO:
        """Gets a style preset by ID."""
        return _found(self._queries.style_presets.get(style_preset_id), style_preset_id)

    def create(self, style_preset: StylePresetWithoutId, user_id: str) -> StylePresetRecordDTO:
        style_preset_id = uuid_string()
        self._queries.style_presets.insert({style_preset_id: style_preset}, user_id)
        # The record is what was stored, so it is not read back.
        return StylePresetRecordDTO(**style_preset.model_dump(), id=style_preset_id, user_id=user_id)

    def create_many(self, style_presets: list[StylePresetWithoutId], user_id: str) -> None:
        self._queries.style_presets.insert({uuid_string(): style_preset for style_preset in style_presets}, user_id)

    def update(self, style_preset_id: str, changes: StylePresetChanges) -> StylePresetRecordDTO:
        def apply(q: Queries) -> Optional[StylePresetRecordDTO]:
            q.style_presets.update(style_preset_id, changes)
            return q.style_presets.get(style_preset_id)

        return _found(self._queries.run(apply), style_preset_id)

    def delete(self, style_preset_id: str) -> None:
        self._queries.style_presets.delete(style_preset_id)

    def get_many(
        self,
        type: PresetType | None = None,
        user_id: str | None = None,
        is_admin: bool = False,
    ) -> list[StylePresetRecordDTO]:
        # Visible to non-admin: own + default + public.
        return self._queries.style_presets.all(type=type, user_id=user_id, is_admin=is_admin)

    def _sync_default_style_presets(self) -> None:
        """Syncs default style presets to the database. Internal use only."""
        with open(Path(__file__).parent / Path("default_style_presets.json"), "r") as file:
            bundled = {uuid_string(): StylePresetWithoutId.model_validate(preset) for preset in json.load(file)}

        def sync(q: Queries) -> None:
            q.style_presets.delete_bundled()
            q.style_presets.insert(bundled, SYSTEM_USER_ID)

        self._queries.run(sync)
