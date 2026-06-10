from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from invokeai.app.services.canvas_project_records.canvas_project_records_common import (
    CanvasProjectNamesResult,
    CanvasProjectRecord,
    CanvasProjectRecordChanges,
)
from invokeai.app.services.image_records.image_records_common import ResourceOrigin
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection


class CanvasProjectRecordStorageBase(ABC):
    """Low-level service responsible for interfacing with the canvas project record store."""

    @abstractmethod
    def get(self, project_name: str) -> CanvasProjectRecord:
        """Gets a canvas project record."""
        pass

    @abstractmethod
    def update(self, project_name: str, changes: CanvasProjectRecordChanges) -> None:
        """Updates a canvas project record."""
        pass

    @abstractmethod
    def get_many(
        self,
        offset: int = 0,
        limit: int = 10,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        project_origin: Optional[ResourceOrigin] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> OffsetPaginatedResults[CanvasProjectRecord]:
        """Gets a page of canvas project records."""
        pass

    @abstractmethod
    def delete(self, project_name: str) -> None:
        """Deletes a canvas project record."""
        pass

    @abstractmethod
    def delete_many(self, project_names: list[str]) -> None:
        """Deletes many canvas project records."""
        pass

    @abstractmethod
    def save(
        self,
        project_name: str,
        project_origin: ResourceOrigin,
        name: str,
        app_version: str,
        width: int,
        height: int,
        image_count: int,
        has_thumbnail: bool,
        is_intermediate: Optional[bool] = False,
        starred: Optional[bool] = False,
        user_id: Optional[str] = None,
        project_subfolder: str = "",
    ) -> datetime:
        """Saves a canvas project record."""
        pass

    @abstractmethod
    def set_has_thumbnail(self, project_name: str, has_thumbnail: bool) -> None:
        """Updates the has_thumbnail flag for a project."""
        pass

    @abstractmethod
    def update_file_metadata(
        self,
        project_name: str,
        width: int,
        height: int,
        image_count: int,
        has_thumbnail: bool,
        app_version: str,
    ) -> None:
        """Updates the fields that change when a project's ZIP is replaced in place."""
        pass

    @abstractmethod
    def get_user_id(self, project_name: str) -> Optional[str]:
        """Gets the user_id of the project owner. Returns None if project not found."""
        pass

    @abstractmethod
    def get_project_names(
        self,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        project_origin: Optional[ResourceOrigin] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> CanvasProjectNamesResult:
        """Gets ordered list of project names with metadata for optimistic updates."""
        pass
