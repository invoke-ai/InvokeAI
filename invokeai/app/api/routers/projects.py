from typing import Any

from fastapi import Body, HTTPException, Path, status
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.project_records.project_records_common import (
    ProjectBoardNotFoundError,
    ProjectBoardSnapshotDTO,
    ProjectBoardTooLargeError,
    ProjectBoardUnavailableError,
    ProjectRecordConflictError,
    ProjectRecordDTO,
    ProjectRecordExistsError,
    ProjectRecordNotFoundError,
    ProjectSummaryDTO,
)

projects_router = APIRouter(prefix="/v1/projects", tags=["projects"])


class ProjectCreateRequest(BaseModel):
    """Request body for creating a project."""

    project_id: str | None = Field(
        default=None, description="Client-generated project id (e.g. for imports); generated when omitted"
    )
    board_id: str | None = Field(
        default=None,
        description=(
            "An existing unclaimed private board for the project to adopt, renamed to match. Omit to create"
            " one. Restoring a project uploads its media into such a board first, so that creating the"
            " project is the single commit point for an import."
        ),
    )
    name: str = Field(description="The project's display name")
    data: dict[str, Any] = Field(description="The opaque client-owned project document")


class ProjectUpdateRequest(BaseModel):
    """Request body for saving a project with optimistic concurrency."""

    name: str = Field(description="The project's display name")
    data: dict[str, Any] = Field(description="The opaque client-owned project document")
    expected_revision: int = Field(description="The revision this save is based on; mismatch returns 409")


@projects_router.get("/", operation_id="list_projects", response_model=list[ProjectSummaryDTO])
def list_projects(current_user: CurrentUserOrDefault) -> list[ProjectSummaryDTO]:
    """Lists the current user's projects as lightweight summaries (no documents)."""
    return ApiDependencies.invoker.services.project_records.list(current_user.user_id)


@projects_router.post(
    "/", operation_id="create_project", response_model=ProjectRecordDTO, status_code=status.HTTP_201_CREATED
)
def create_project(
    current_user: CurrentUserOrDefault,
    request: ProjectCreateRequest = Body(description="The project to create"),
) -> ProjectRecordDTO:
    """Creates a project, and the private board it owns, for the current user."""
    try:
        return ApiDependencies.invoker.services.project_records.create(
            user_id=current_user.user_id,
            name=request.name,
            data=request.data,
            project_id=request.project_id,
            board_id=request.board_id,
        )
    except (ProjectRecordExistsError, ProjectBoardUnavailableError) as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ProjectBoardNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@projects_router.get("/{project_id}", operation_id="get_project", response_model=ProjectRecordDTO)
def get_project(
    current_user: CurrentUserOrDefault,
    project_id: str = Path(description="The id of the project to get"),
) -> ProjectRecordDTO:
    """Gets one of the current user's projects, including its document."""
    try:
        return ApiDependencies.invoker.services.project_records.get(current_user.user_id, project_id)
    except ProjectRecordNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@projects_router.put("/{project_id}", operation_id="update_project", response_model=ProjectRecordDTO)
def update_project(
    current_user: CurrentUserOrDefault,
    project_id: str = Path(description="The id of the project to save"),
    request: ProjectUpdateRequest = Body(description="The project document and the revision it is based on"),
) -> ProjectRecordDTO:
    """Saves a project. Returns 409 with the current revision when the save is based on a stale revision."""
    try:
        return ApiDependencies.invoker.services.project_records.update(
            user_id=current_user.user_id,
            project_id=project_id,
            expected_revision=request.expected_revision,
            name=request.name,
            data=request.data,
        )
    except ProjectRecordNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ProjectRecordConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": str(e), "current_revision": e.current_revision},
        )


@projects_router.get(
    "/{project_id}/board-snapshot",
    operation_id="get_project_board_snapshot",
    response_model=ProjectBoardSnapshotDTO,
)
def get_project_board_snapshot(
    current_user: CurrentUserOrDefault,
    project_id: str = Path(description="The id of the project whose board to enumerate"),
) -> ProjectBoardSnapshotDTO:
    """Lists everything on the project's board that the gallery would show.

    Intermediates and the canvas's private `other` category are excluded. Unpaginated: the caller
    that needs this — exporting a project — has to hold the whole list anyway. It is still bounded,
    because the answer is built entirely in memory and any client with a project id can ask for it;
    a board past the ceiling is one an export could not have packed either, so it is refused as a
    413 rather than paged.
    """
    try:
        return ApiDependencies.invoker.services.project_records.get_board_snapshot(current_user.user_id, project_id)
    except ProjectRecordNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ProjectBoardTooLargeError as e:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(e))


@projects_router.delete("/{project_id}", operation_id="delete_project", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(
    current_user: CurrentUserOrDefault,
    project_id: str = Path(description="The id of the project to delete"),
) -> None:
    """Deletes one of the current user's projects, and the board it owns, in one transaction.

    Idempotent. The media survives: deleting the board drops its memberships, so the images and
    videos on it return to Uncategorized, exactly as they would if the board were deleted without
    `include_images`. There is deliberately no option to take them with it — a project is a
    workspace, and emptying someone's gallery is not what deleting one should be able to mean.
    Nothing is reported back for the same reason: nothing was destroyed to report.
    """
    ApiDependencies.invoker.services.project_records.delete(current_user.user_id, project_id)
