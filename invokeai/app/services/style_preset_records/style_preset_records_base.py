from abc import ABC, abstractmethod

from invokeai.app.services.style_preset_records.style_preset_records_common import (
    PresetType,
    StylePresetChanges,
    StylePresetRecordDTO,
    StylePresetWithoutId,
)


class StylePresetRecordsStorageBase(ABC):
    """Base class for style preset storage services."""

    @abstractmethod
    def get(self, style_preset_id: str) -> StylePresetRecordDTO:
        """Get style preset by id. Authorization is the caller's responsibility."""
        pass

    @abstractmethod
    def create(self, style_preset: StylePresetWithoutId, user_id: str) -> StylePresetRecordDTO:
        """Creates a style preset owned by user_id."""
        pass

    @abstractmethod
    def create_many(self, style_presets: list[StylePresetWithoutId], user_id: str) -> None:
        """Creates many style presets owned by user_id."""
        pass

    @abstractmethod
    def update(self, style_preset_id: str, changes: StylePresetChanges) -> StylePresetRecordDTO:
        """Updates a style preset. Authorization is the caller's responsibility."""
        pass

    @abstractmethod
    def delete(self, style_preset_id: str) -> None:
        """Deletes a style preset. Authorization is the caller's responsibility."""
        pass

    @abstractmethod
    def get_many(
        self,
        type: PresetType | None = None,
        user_id: str | None = None,
        is_admin: bool = False,
    ) -> list[StylePresetRecordDTO]:
        """Gets style presets visible to user_id.

        Visibility rules:
        - is_admin=True: all presets.
        - Else: presets owned by user_id, plus all `default` presets, plus any public preset.
        - If user_id is None and is_admin is False: only `default` and public presets.
        """
        pass
