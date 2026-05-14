from abc import ABC, abstractmethod
from typing import Callable, Optional

from invokeai.app.services.canvas_project_records.canvas_project_records_common import (
    CanvasProjectNamesResult,
    CanvasProjectRecord,
    CanvasProjectRecordChanges,
)
from invokeai.app.services.canvas_projects.canvas_projects_common import CanvasProjectDTO
from invokeai.app.services.image_records.image_records_common import ResourceOrigin
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection


class CanvasProjectServiceABC(ABC):
    """High-level service for canvas project (.invk) management."""

    _on_changed_callbacks: list[Callable[[CanvasProjectDTO], None]]
    _on_deleted_callbacks: list[Callable[[str], None]]

    def __init__(self) -> None:
        self._on_changed_callbacks = []
        self._on_deleted_callbacks = []

    def on_changed(self, on_changed: Callable[[CanvasProjectDTO], None]) -> None:
        """Register a callback for when a canvas project is changed."""
        self._on_changed_callbacks.append(on_changed)

    def on_deleted(self, on_deleted: Callable[[str], None]) -> None:
        """Register a callback for when a canvas project is deleted."""
        self._on_deleted_callbacks.append(on_deleted)

    def _on_changed(self, item: CanvasProjectDTO) -> None:
        for callback in self._on_changed_callbacks:
            callback(item)

    def _on_deleted(self, item_id: str) -> None:
        for callback in self._on_deleted_callbacks:
            callback(item_id)

    @abstractmethod
    def create(
        self,
        zip_bytes: bytes,
        name: str,
        app_version: str,
        width: int,
        height: int,
        image_count: int,
        thumbnail_bytes: Optional[bytes] = None,
        project_origin: ResourceOrigin = ResourceOrigin.INTERNAL,
        board_id: Optional[str] = None,
        is_intermediate: Optional[bool] = False,
        starred: Optional[bool] = False,
        user_id: Optional[str] = None,
    ) -> CanvasProjectDTO:
        """Creates a canvas project record + writes the ZIP (and optional thumbnail) to disk."""
        pass

    @abstractmethod
    def update(self, project_name: str, changes: CanvasProjectRecordChanges) -> CanvasProjectDTO:
        """Updates a canvas project."""
        pass

    @abstractmethod
    def replace_file(
        self,
        project_name: str,
        zip_bytes: bytes,
        width: int,
        height: int,
        image_count: int,
        app_version: str,
        thumbnail_bytes: Optional[bytes] = None,
    ) -> CanvasProjectDTO:
        """Replaces the on-disk ZIP and thumbnail for an existing project. Keeps project_name,
        board assignment, starred state, ownership. Updates dimensions / image count / app version
        / has_thumbnail."""
        pass

    @abstractmethod
    def get_record(self, project_name: str) -> CanvasProjectRecord:
        """Gets a canvas project record."""
        pass

    @abstractmethod
    def get_dto(self, project_name: str) -> CanvasProjectDTO:
        """Gets a canvas project DTO."""
        pass

    @abstractmethod
    def get_path(self, project_name: str, thumbnail: bool = False) -> str:
        """Gets a canvas project's on-disk path."""
        pass

    @abstractmethod
    def get_url(self, project_name: str, thumbnail: bool = False) -> str:
        """Gets a canvas project's URL."""
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
    ) -> OffsetPaginatedResults[CanvasProjectDTO]:
        """Gets a paginated list of canvas project DTOs."""
        pass

    @abstractmethod
    def delete(self, project_name: str) -> None:
        """Deletes a canvas project (record + files)."""
        pass

    @abstractmethod
    def delete_projects_on_board(self, board_id: str) -> None:
        """Deletes all canvas projects on a board."""
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
        """Gets ordered list of canvas project names."""
        pass
