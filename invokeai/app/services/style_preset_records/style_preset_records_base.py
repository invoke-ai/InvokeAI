from abc import ABC, abstractmethod
from typing import Optional

from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.style_preset_records.style_preset_records_common import (
    StylePresetChanges,
    StylePresetRecordDTO,
    StylePresetWithoutId,
    StylePresetRecordOrderBy,
)


class StylePresetRecordsStorageBase(ABC):
    """Base class for style preset storage services."""

    @abstractmethod
    def get(self, id: str) -> StylePresetRecordDTO:
        """Get style preset by id."""
        pass

    @abstractmethod
    def create(self, style_preset: StylePresetWithoutId) -> StylePresetRecordDTO:
        """Creates a style preset."""
        pass

    @abstractmethod
    def update(self, id: str, changes: StylePresetChanges) -> StylePresetRecordDTO:
        """Updates a style preset."""
        pass

    @abstractmethod
    def delete(self, id: str) -> None:
        """Deletes a style preset."""
        pass

    @abstractmethod
    def get_many(
        self,
        page: int,
        per_page: int,
        order_by: StylePresetRecordOrderBy,
        direction: SQLiteDirection,
        query: Optional[str],
    ) -> PaginatedResults[StylePresetRecordDTO]:
        """Gets many workflows."""
        pass
