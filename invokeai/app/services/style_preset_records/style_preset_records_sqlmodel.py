import json
from pathlib import Path

from sqlmodel import col, select

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.sqlite.models import StylePresetTable
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


def _to_dto(row: StylePresetTable) -> StylePresetRecordDTO:
    return StylePresetRecordDTO.from_dict(
        {
            "id": row.id,
            "name": row.name,
            "preset_data": row.preset_data,
            "type": row.type,
            "created_at": str(row.created_at),
            "updated_at": str(row.updated_at),
        }
    )


class SqlModelStylePresetRecordsStorage(StylePresetRecordsStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        self._sync_default_style_presets()

    def get(self, style_preset_id: str) -> StylePresetRecordDTO:
        with self._db.get_readonly_session() as session:
            row = session.get(StylePresetTable, style_preset_id)
            if row is None:
                raise StylePresetNotFoundError(f"Style preset with id {style_preset_id} not found")
            return _to_dto(row)

    def create(self, style_preset: StylePresetWithoutId) -> StylePresetRecordDTO:
        style_preset_id = uuid_string()
        row = StylePresetTable(
            id=style_preset_id,
            name=style_preset.name,
            preset_data=style_preset.preset_data.model_dump_json(),
            type=style_preset.type,
        )
        with self._db.get_session() as session:
            session.add(row)
        return self.get(style_preset_id)

    def create_many(self, style_presets: list[StylePresetWithoutId]) -> None:
        with self._db.get_session() as session:
            for style_preset in style_presets:
                row = StylePresetTable(
                    id=uuid_string(),
                    name=style_preset.name,
                    preset_data=style_preset.preset_data.model_dump_json(),
                    type=style_preset.type,
                )
                session.add(row)

    def update(self, style_preset_id: str, changes: StylePresetChanges) -> StylePresetRecordDTO:
        with self._db.get_session() as session:
            row = session.get(StylePresetTable, style_preset_id)
            if row is None:
                raise StylePresetNotFoundError(f"Style preset with id {style_preset_id} not found")

            if changes.name is not None:
                row.name = changes.name
            if changes.preset_data is not None:
                row.preset_data = changes.preset_data.model_dump_json()

            session.add(row)
        return self.get(style_preset_id)

    def delete(self, style_preset_id: str) -> None:
        with self._db.get_session() as session:
            row = session.get(StylePresetTable, style_preset_id)
            if row is not None:
                session.delete(row)

    def get_many(self, type: PresetType | None = None) -> list[StylePresetRecordDTO]:
        with self._db.get_readonly_session() as session:
            stmt = select(StylePresetTable)
            if type is not None:
                stmt = stmt.where(col(StylePresetTable.type) == type)
            stmt = stmt.order_by(col(StylePresetTable.name).asc())
            rows = session.exec(stmt).all()
            return [_to_dto(r) for r in rows]

    def _sync_default_style_presets(self) -> None:
        """Syncs default style presets to the database."""
        # Delete existing defaults
        with self._db.get_session() as session:
            stmt = select(StylePresetTable).where(col(StylePresetTable.type) == "default")
            rows = session.exec(stmt).all()
            for row in rows:
                session.delete(row)

        # Re-create from file
        with open(Path(__file__).parent / Path("default_style_presets.json"), "r") as file:
            presets = json.load(file)
            for preset in presets:
                style_preset = StylePresetWithoutId.model_validate(preset)
                self.create(style_preset)
