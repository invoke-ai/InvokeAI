from typing import Any

from fastapi import Body, HTTPException, Path, status
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.project_records.project_records_common import (
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
    name: str = Field(description="The project's display name")
    data: dict[str, Any] = Field(description="The opaque client-owned project document")


class ProjectUpdateRequest(BaseModel):
    """Request body for saving a project with optimistic concurrency."""

    name: str = Field(description="The project's display name")
    data: dict[str, Any] = Field(description="The opaque client-owned project document")
    expected_revision: int = Field(description="The revision this save is based on; mismatch returns 409")


@projects_router.get("/", operation_id="list_projects", response_model=list[ProjectSummaryDTO])
async def list_projects(current_user: CurrentUserOrDefault) -> list[ProjectSummaryDTO]:
    """Lists the current user's projects as lightweight summaries (no documents)."""
    return ApiDependencies.invoker.services.project_records.list(current_user.user_id)


@projects_router.post(
    "/", operation_id="create_project", response_model=ProjectRecordDTO, status_code=status.HTTP_201_CREATED
)
async def create_project(
    current_user: CurrentUserOrDefault,
    request: ProjectCreateRequest = Body(description="The project to create"),
) -> ProjectRecordDTO:
    """Creates a project for the current user."""
    try:
        return ApiDependencies.invoker.services.project_records.create(
            user_id=current_user.user_id,
            name=request.name,
            data=request.data,
            project_id=request.project_id,
        )
    except ProjectRecordExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@projects_router.get("/{project_id}", operation_id="get_project", response_model=ProjectRecordDTO)
async def get_project(
    current_user: CurrentUserOrDefault,
    project_id: str = Path(description="The id of the project to get"),
) -> ProjectRecordDTO:
    """Gets one of the current user's projects, including its document."""
    try:
        return ApiDependencies.invoker.services.project_records.get(current_user.user_id, project_id)
    except ProjectRecordNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@projects_router.put("/{project_id}", operation_id="update_project", response_model=ProjectRecordDTO)
async def update_project(
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


@projects_router.delete("/{project_id}", operation_id="delete_project", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    current_user: CurrentUserOrDefault,
    project_id: str = Path(description="The id of the project to delete"),
) -> None:
    """Deletes one of the current user's projects. Idempotent."""
    ApiDependencies.invoker.services.project_records.delete(current_user.user_id, project_id)
