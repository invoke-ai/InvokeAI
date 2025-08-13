from fastapi import  Body, HTTPException
from fastapi.routing import APIRouter
from invokeai.app.services.resources.resources_common import DeleteResourcesResult, ResourceIdentifier, ResourceType, StarredResourcesResult, UnstarredResourcesResult
from invokeai.app.services.video_records.video_records_common import VideoRecordChanges
from pydantic import BaseModel, Field

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.image_records.image_records_common import (
    ImageRecordChanges,
)

# routes that act on both images and videos, possibly together
resources_router = APIRouter(prefix="/v1/resources", tags=["resources"])

@resources_router.post("/delete", operation_id="delete_resources_from_list", response_model=DeleteResourcesResult)
async def delete_resources_from_list(
    resources: list[ResourceIdentifier] = Body(description="The list of resources to delete", embed=True),
) -> DeleteResourcesResult:
    try:
        deleted_resources: set[ResourceIdentifier] = set()
        affected_boards: set[str] = set()
        for resource in resources:
            if resource.resource_type == ResourceType.IMAGE:
                try:
                    image_dto = ApiDependencies.invoker.services.images.get_dto(resource.resource_id)
                    board_id = image_dto.board_id or "none"
                    ApiDependencies.invoker.services.images.delete(resource.resource_id)
                    deleted_resources.add(resource)
                    affected_boards.add(board_id)
                except Exception:
                    pass
            elif resource.resource_type == ResourceType.VIDEO:
                try:
                    video_dto = ApiDependencies.invoker.services.videos.get_dto(resource.resource_id)
                    board_id = video_dto.board_id or "none"
                    ApiDependencies.invoker.services.videos.delete(resource.resource_id)
                    deleted_resources.add(resource)
                    affected_boards.add(board_id)
                except Exception:
                    pass
        return DeleteResourcesResult(
            deleted_resources=list(deleted_resources),
            affected_boards=list(affected_boards),
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to delete images")


@resources_router.delete("/uncategorized", operation_id="delete_uncategorized_resources", response_model=DeleteResourcesResult)
async def delete_uncategorized_resources() -> DeleteResourcesResult:
    """Deletes all resources that are uncategorized"""

    resources = ApiDependencies.invoker.services.board_resources.get_all_board_resource_ids_for_board(
        board_id="none"
    )

    try:
        deleted_resources: set[ResourceIdentifier] = set()
        affected_boards: set[str] = set()
        for resource in resources:
            if resource.resource_type == ResourceType.IMAGE:
                try:
                    ApiDependencies.invoker.services.images.delete(resource.resource_id)
                    deleted_resources.add(resource)
                    affected_boards.add("none")
                except Exception:
                    pass
            elif resource.resource_type == ResourceType.VIDEO:
                try:
                    ApiDependencies.invoker.services.videos.delete(resource.resource_id)
                    deleted_resources.add(resource)
                    affected_boards.add("none")
                except Exception:
                    pass
        return DeleteResourcesResult(
            deleted_resources=list(deleted_resources),
            affected_boards=list(affected_boards),
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to delete images")


class ResourcesUpdatedFromListResult(BaseModel):
    updated_resource: list[ResourceIdentifier] = Field(description="The resource ids that were updated")


@resources_router.post("/star", operation_id="star_resources_in_list", response_model=StarredResourcesResult)
async def star_resources_in_list(
    resources: list[ResourceIdentifier] = Body(description="The list of resources to star", embed=True),
) -> StarredResourcesResult:
    try:
        starred_resources: set[ResourceIdentifier] = set()
        affected_boards: set[str] = set()
        for resource in resources:
            if resource.resource_type == ResourceType.IMAGE:
                try:
                    updated_resource_dto = ApiDependencies.invoker.services.images.update(
                        resource.resource_id, changes=ImageRecordChanges(starred=True)
                    )
                    starred_resources.add(resource)
                    affected_boards.add(updated_resource_dto.board_id or "none")
                except Exception:
                    pass
            elif resource.resource_type == ResourceType.VIDEO:
                try:
                    updated_resource_dto = ApiDependencies.invoker.services.videos.update(
                        resource.resource_id, changes=VideoRecordChanges(starred=True)
                    )
                    starred_resources.add(resource)
                    affected_boards.add(updated_resource_dto.board_id or "none")
                except Exception:
                    pass
        return StarredResourcesResult(
            starred_resources=list(starred_resources),
            affected_boards=list(affected_boards),
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to star images")


@resources_router.post("/unstar", operation_id="unstar_resources_in_list", response_model=UnstarredResourcesResult)
async def unstar_resources_in_list(
    resources: list[ResourceIdentifier] = Body(description="The list of resources to unstar", embed=True),
) -> UnstarredResourcesResult:
    try:
        unstarred_resources: set[ResourceIdentifier] = set()
        affected_boards: set[str] = set()
        for resource in resources:
            if resource.resource_type == ResourceType.IMAGE:
                try:
                    updated_resource_dto = ApiDependencies.invoker.services.images.update(
                        resource.resource_id, changes=ImageRecordChanges(starred=False)
                    )
                    unstarred_resources.add(resource)
                    affected_boards.add(updated_resource_dto.board_id or "none")
                except Exception:
                    pass
            elif resource.resource_type == ResourceType.VIDEO:
                try:
                    updated_resource_dto = ApiDependencies.invoker.services.videos.update(
                        resource.resource_id, changes=VideoRecordChanges(starred=False)
                    )
                    unstarred_resources.add(resource)
                    affected_boards.add(updated_resource_dto.board_id or "none")
                except Exception:
                    pass
        return UnstarredResourcesResult(
            unstarred_resources=list(unstarred_resources),
            affected_boards=list(affected_boards),
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to unstar images")

