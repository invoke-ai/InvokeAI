from typing import Optional

from pydantic import BaseModel, Field

from invokeai.app.services.canvas_project_records.canvas_project_records_common import CanvasProjectRecord
from invokeai.app.util.model_exclude_null import BaseModelExcludeNull


class CanvasProjectUrlsDTO(BaseModelExcludeNull):
    """The URLs for a canvas project and its thumbnail."""

    project_name: str = Field(description="The unique name of the canvas project.")
    project_url: str = Field(description="The URL of the canvas project ZIP file (.invk).")
    thumbnail_url: Optional[str] = Field(
        default=None, description="The URL of the canvas project's preview thumbnail (WebP), if any."
    )


class CanvasProjectDTO(CanvasProjectRecord, CanvasProjectUrlsDTO):
    """Deserialized canvas project record, enriched for the frontend."""

    board_id: Optional[str] = Field(
        default=None, description="The id of the board the canvas project belongs to, if one exists."
    )


def canvas_project_record_to_dto(
    project_record: CanvasProjectRecord,
    project_url: str,
    thumbnail_url: Optional[str],
    board_id: Optional[str],
) -> CanvasProjectDTO:
    """Converts a canvas project record to a canvas project DTO."""
    return CanvasProjectDTO(
        **project_record.model_dump(),
        project_url=project_url,
        thumbnail_url=thumbnail_url,
        board_id=board_id,
    )


class CanvasProjectResultWithAffectedBoards(BaseModel):
    affected_boards: list[str] = Field(description="The ids of boards affected by the operation")


class DeleteCanvasProjectsResult(CanvasProjectResultWithAffectedBoards):
    deleted_projects: list[str] = Field(description="The names of the canvas projects that were deleted")


class StarredCanvasProjectsResult(CanvasProjectResultWithAffectedBoards):
    starred_projects: list[str] = Field(description="The names of the canvas projects that were starred")


class UnstarredCanvasProjectsResult(CanvasProjectResultWithAffectedBoards):
    unstarred_projects: list[str] = Field(description="The names of the canvas projects that were unstarred")


class AddCanvasProjectsToBoardResult(CanvasProjectResultWithAffectedBoards):
    added_projects: list[str] = Field(description="The project names that were added to the board")


class RemoveCanvasProjectsFromBoardResult(CanvasProjectResultWithAffectedBoards):
    removed_projects: list[str] = Field(description="The project names that were removed from their board")
