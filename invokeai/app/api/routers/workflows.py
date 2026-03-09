import io
import traceback
from typing import Optional

from fastapi import APIRouter, Body, File, HTTPException, Path, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.workflow_records.workflow_records_common import (
    Workflow,
    WorkflowCategory,
    WorkflowNotFoundError,
    WorkflowRecordDTO,
    WorkflowRecordListItemWithThumbnailDTO,
    WorkflowRecordOrderBy,
    WorkflowRecordWithThumbnailDTO,
    WorkflowWithoutID,
)
from invokeai.app.services.workflow_thumbnails.workflow_thumbnails_common import WorkflowThumbnailFileNotFoundException

IMAGE_MAX_AGE = 31536000
workflows_router = APIRouter(prefix="/v1/workflows", tags=["workflows"])


@workflows_router.get(
    "/i/{workflow_id}",
    operation_id="get_workflow",
    responses={
        200: {"model": WorkflowRecordWithThumbnailDTO},
    },
)
async def get_workflow(
    current_user: CurrentUserOrDefault,
    workflow_id: str = Path(description="The workflow to get"),
) -> WorkflowRecordWithThumbnailDTO:
    """Gets a workflow"""
    try:
        workflow = ApiDependencies.invoker.services.workflow_records.get(workflow_id)
    except WorkflowNotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")

    config = ApiDependencies.invoker.services.configuration
    if config.multiuser:
        is_default = workflow.workflow.meta.category is WorkflowCategory.Default
        is_owner = workflow.user_id == current_user.user_id
        if not (is_default or is_owner or workflow.is_public or current_user.is_admin):
            raise HTTPException(status_code=403, detail="Not authorized to access this workflow")

    thumbnail_url = ApiDependencies.invoker.services.workflow_thumbnails.get_url(workflow_id)
    return WorkflowRecordWithThumbnailDTO(thumbnail_url=thumbnail_url, **workflow.model_dump())


@workflows_router.patch(
    "/i/{workflow_id}",
    operation_id="update_workflow",
    responses={
        200: {"model": WorkflowRecordDTO},
    },
)
async def update_workflow(
    current_user: CurrentUserOrDefault,
    workflow: Workflow = Body(description="The updated workflow", embed=True),
) -> WorkflowRecordDTO:
    """Updates a workflow"""
    config = ApiDependencies.invoker.services.configuration
    if config.multiuser:
        try:
            existing = ApiDependencies.invoker.services.workflow_records.get(workflow.id)
        except WorkflowNotFoundError:
            raise HTTPException(status_code=404, detail="Workflow not found")
        if not current_user.is_admin and existing.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Not authorized to update this workflow")
    return ApiDependencies.invoker.services.workflow_records.update(workflow=workflow)


@workflows_router.delete(
    "/i/{workflow_id}",
    operation_id="delete_workflow",
)
async def delete_workflow(
    current_user: CurrentUserOrDefault,
    workflow_id: str = Path(description="The workflow to delete"),
) -> None:
    """Deletes a workflow"""
    config = ApiDependencies.invoker.services.configuration
    if config.multiuser:
        try:
            existing = ApiDependencies.invoker.services.workflow_records.get(workflow_id)
        except WorkflowNotFoundError:
            raise HTTPException(status_code=404, detail="Workflow not found")
        if not current_user.is_admin and existing.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Not authorized to delete this workflow")
    try:
        ApiDependencies.invoker.services.workflow_thumbnails.delete(workflow_id)
    except WorkflowThumbnailFileNotFoundException:
        # It's OK if the workflow has no thumbnail file. We can still delete the workflow.
        pass
    ApiDependencies.invoker.services.workflow_records.delete(workflow_id)


@workflows_router.post(
    "/",
    operation_id="create_workflow",
    responses={
        200: {"model": WorkflowRecordDTO},
    },
)
async def create_workflow(
    current_user: CurrentUserOrDefault,
    workflow: WorkflowWithoutID = Body(description="The workflow to create", embed=True),
) -> WorkflowRecordDTO:
    """Creates a workflow"""
    return ApiDependencies.invoker.services.workflow_records.create(workflow=workflow, user_id=current_user.user_id)


@workflows_router.get(
    "/",
    operation_id="list_workflows",
    responses={
        200: {"model": PaginatedResults[WorkflowRecordListItemWithThumbnailDTO]},
    },
)
async def list_workflows(
    current_user: CurrentUserOrDefault,
    page: int = Query(default=0, description="The page to get"),
    per_page: Optional[int] = Query(default=None, description="The number of workflows per page"),
    order_by: WorkflowRecordOrderBy = Query(
        default=WorkflowRecordOrderBy.Name, description="The attribute to order by"
    ),
    direction: SQLiteDirection = Query(default=SQLiteDirection.Ascending, description="The direction to order by"),
    categories: Optional[list[WorkflowCategory]] = Query(default=None, description="The categories of workflow to get"),
    tags: Optional[list[str]] = Query(default=None, description="The tags of workflow to get"),
    query: Optional[str] = Query(default=None, description="The text to query by (matches name and description)"),
    has_been_opened: Optional[bool] = Query(default=None, description="Whether to include/exclude recent workflows"),
    is_public: Optional[bool] = Query(default=None, description="Filter by public/shared status"),
) -> PaginatedResults[WorkflowRecordListItemWithThumbnailDTO]:
    """Gets a page of workflows"""
    config = ApiDependencies.invoker.services.configuration

    # In multiuser mode, scope user-category workflows to the current user unless fetching shared workflows
    user_id_filter: Optional[str] = None
    if config.multiuser:
        # Only filter 'user' category results by user_id when not explicitly listing public workflows
        has_user_category = not categories or WorkflowCategory.User in categories
        if has_user_category and is_public is not True:
            user_id_filter = current_user.user_id

    workflows_with_thumbnails: list[WorkflowRecordListItemWithThumbnailDTO] = []
    workflows = ApiDependencies.invoker.services.workflow_records.get_many(
        order_by=order_by,
        direction=direction,
        page=page,
        per_page=per_page,
        query=query,
        categories=categories,
        tags=tags,
        has_been_opened=has_been_opened,
        user_id=user_id_filter,
        is_public=is_public,
    )
    for workflow in workflows.items:
        workflows_with_thumbnails.append(
            WorkflowRecordListItemWithThumbnailDTO(
                thumbnail_url=ApiDependencies.invoker.services.workflow_thumbnails.get_url(workflow.workflow_id),
                **workflow.model_dump(),
            )
        )
    return PaginatedResults[WorkflowRecordListItemWithThumbnailDTO](
        items=workflows_with_thumbnails,
        total=workflows.total,
        page=workflows.page,
        pages=workflows.pages,
        per_page=workflows.per_page,
    )


@workflows_router.put(
    "/i/{workflow_id}/thumbnail",
    operation_id="set_workflow_thumbnail",
    responses={
        200: {"model": WorkflowRecordDTO},
    },
)
async def set_workflow_thumbnail(
    current_user: CurrentUserOrDefault,
    workflow_id: str = Path(description="The workflow to update"),
    image: UploadFile = File(description="The image file to upload"),
):
    """Sets a workflow's thumbnail image"""
    try:
        existing = ApiDependencies.invoker.services.workflow_records.get(workflow_id)
    except WorkflowNotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")

    config = ApiDependencies.invoker.services.configuration
    if config.multiuser and not current_user.is_admin and existing.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to update this workflow")

    if not image.content_type or not image.content_type.startswith("image"):
        raise HTTPException(status_code=415, detail="Not an image")

    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))

    except Exception:
        ApiDependencies.invoker.services.logger.error(traceback.format_exc())
        raise HTTPException(status_code=415, detail="Failed to read image")

    try:
        ApiDependencies.invoker.services.workflow_thumbnails.save(workflow_id, pil_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@workflows_router.delete(
    "/i/{workflow_id}/thumbnail",
    operation_id="delete_workflow_thumbnail",
    responses={
        200: {"model": WorkflowRecordDTO},
    },
)
async def delete_workflow_thumbnail(
    current_user: CurrentUserOrDefault,
    workflow_id: str = Path(description="The workflow to update"),
):
    """Removes a workflow's thumbnail image"""
    try:
        existing = ApiDependencies.invoker.services.workflow_records.get(workflow_id)
    except WorkflowNotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")

    config = ApiDependencies.invoker.services.configuration
    if config.multiuser and not current_user.is_admin and existing.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to update this workflow")

    try:
        ApiDependencies.invoker.services.workflow_thumbnails.delete(workflow_id)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@workflows_router.get(
    "/i/{workflow_id}/thumbnail",
    operation_id="get_workflow_thumbnail",
    responses={
        200: {
            "description": "The workflow thumbnail was fetched successfully",
        },
        400: {"description": "Bad request"},
        404: {"description": "The workflow thumbnail could not be found"},
    },
    status_code=200,
)
async def get_workflow_thumbnail(
    workflow_id: str = Path(description="The id of the workflow thumbnail to get"),
) -> FileResponse:
    """Gets a workflow's thumbnail image"""

    try:
        path = ApiDependencies.invoker.services.workflow_thumbnails.get_path(workflow_id)

        response = FileResponse(
            path,
            media_type="image/png",
            filename=workflow_id + ".png",
            content_disposition_type="inline",
        )
        response.headers["Cache-Control"] = f"max-age={IMAGE_MAX_AGE}"
        return response
    except Exception:
        raise HTTPException(status_code=404)


@workflows_router.patch(
    "/i/{workflow_id}/is_public",
    operation_id="update_workflow_is_public",
    responses={
        200: {"model": WorkflowRecordDTO},
    },
)
async def update_workflow_is_public(
    current_user: CurrentUserOrDefault,
    workflow_id: str = Path(description="The workflow to update"),
    is_public: bool = Body(description="Whether the workflow should be shared publicly", embed=True),
) -> WorkflowRecordDTO:
    """Updates whether a workflow is shared publicly"""
    try:
        existing = ApiDependencies.invoker.services.workflow_records.get(workflow_id)
    except WorkflowNotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")

    config = ApiDependencies.invoker.services.configuration
    if config.multiuser and not current_user.is_admin and existing.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to update this workflow")

    return ApiDependencies.invoker.services.workflow_records.update_is_public(
        workflow_id=workflow_id, is_public=is_public
    )


@workflows_router.get("/tags", operation_id="get_all_tags")
async def get_all_tags(
    current_user: CurrentUserOrDefault,
    categories: Optional[list[WorkflowCategory]] = Query(default=None, description="The categories to include"),
    is_public: Optional[bool] = Query(default=None, description="Filter by public/shared status"),
) -> list[str]:
    """Gets all unique tags from workflows"""
    config = ApiDependencies.invoker.services.configuration
    user_id_filter: Optional[str] = None
    if config.multiuser:
        has_user_category = not categories or WorkflowCategory.User in categories
        if has_user_category and is_public is not True:
            user_id_filter = current_user.user_id

    return ApiDependencies.invoker.services.workflow_records.get_all_tags(
        categories=categories, user_id=user_id_filter, is_public=is_public
    )


@workflows_router.get("/counts_by_tag", operation_id="get_counts_by_tag")
async def get_counts_by_tag(
    current_user: CurrentUserOrDefault,
    tags: list[str] = Query(description="The tags to get counts for"),
    categories: Optional[list[WorkflowCategory]] = Query(default=None, description="The categories to include"),
    has_been_opened: Optional[bool] = Query(default=None, description="Whether to include/exclude recent workflows"),
    is_public: Optional[bool] = Query(default=None, description="Filter by public/shared status"),
) -> dict[str, int]:
    """Counts workflows by tag"""
    config = ApiDependencies.invoker.services.configuration
    user_id_filter: Optional[str] = None
    if config.multiuser:
        has_user_category = not categories or WorkflowCategory.User in categories
        if has_user_category and is_public is not True:
            user_id_filter = current_user.user_id

    return ApiDependencies.invoker.services.workflow_records.counts_by_tag(
        tags=tags, categories=categories, has_been_opened=has_been_opened, user_id=user_id_filter, is_public=is_public
    )


@workflows_router.get("/counts_by_category", operation_id="counts_by_category")
async def counts_by_category(
    current_user: CurrentUserOrDefault,
    categories: list[WorkflowCategory] = Query(description="The categories to include"),
    has_been_opened: Optional[bool] = Query(default=None, description="Whether to include/exclude recent workflows"),
    is_public: Optional[bool] = Query(default=None, description="Filter by public/shared status"),
) -> dict[str, int]:
    """Counts workflows by category"""
    config = ApiDependencies.invoker.services.configuration
    user_id_filter: Optional[str] = None
    if config.multiuser:
        has_user_category = WorkflowCategory.User in categories
        if has_user_category and is_public is not True:
            user_id_filter = current_user.user_id

    return ApiDependencies.invoker.services.workflow_records.counts_by_category(
        categories=categories, has_been_opened=has_been_opened, user_id=user_id_filter, is_public=is_public
    )


@workflows_router.put(
    "/i/{workflow_id}/opened_at",
    operation_id="update_opened_at",
)
async def update_opened_at(
    workflow_id: str = Path(description="The workflow to update"),
) -> None:
    """Updates the opened_at field of a workflow"""
    ApiDependencies.invoker.services.workflow_records.update_opened_at(workflow_id)
