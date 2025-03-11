import io
import traceback
from typing import Optional

from fastapi import APIRouter, Body, File, HTTPException, Path, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

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
    workflow_id: str = Path(description="The workflow to get"),
) -> WorkflowRecordWithThumbnailDTO:
    """Gets a workflow"""
    try:
        thumbnail_url = ApiDependencies.invoker.services.workflow_thumbnails.get_url(workflow_id)
        workflow = ApiDependencies.invoker.services.workflow_records.get(workflow_id)
        return WorkflowRecordWithThumbnailDTO(thumbnail_url=thumbnail_url, **workflow.model_dump())
    except WorkflowNotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")


@workflows_router.patch(
    "/i/{workflow_id}",
    operation_id="update_workflow",
    responses={
        200: {"model": WorkflowRecordDTO},
    },
)
async def update_workflow(
    workflow: Workflow = Body(description="The updated workflow", embed=True),
) -> WorkflowRecordDTO:
    """Updates a workflow"""
    return ApiDependencies.invoker.services.workflow_records.update(workflow=workflow)


@workflows_router.delete(
    "/i/{workflow_id}",
    operation_id="delete_workflow",
)
async def delete_workflow(
    workflow_id: str = Path(description="The workflow to delete"),
) -> None:
    """Deletes a workflow"""
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
    workflow: WorkflowWithoutID = Body(description="The workflow to create", embed=True),
) -> WorkflowRecordDTO:
    """Creates a workflow"""
    return ApiDependencies.invoker.services.workflow_records.create(workflow=workflow)


@workflows_router.get(
    "/",
    operation_id="list_workflows",
    responses={
        200: {"model": PaginatedResults[WorkflowRecordListItemWithThumbnailDTO]},
    },
)
async def list_workflows(
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
) -> PaginatedResults[WorkflowRecordListItemWithThumbnailDTO]:
    """Gets a page of workflows"""
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
    workflow_id: str = Path(description="The workflow to update"),
    image: UploadFile = File(description="The image file to upload"),
):
    """Sets a workflow's thumbnail image"""
    try:
        ApiDependencies.invoker.services.workflow_records.get(workflow_id)
    except WorkflowNotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")

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
    workflow_id: str = Path(description="The workflow to update"),
):
    """Removes a workflow's thumbnail image"""
    try:
        ApiDependencies.invoker.services.workflow_records.get(workflow_id)
    except WorkflowNotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")

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


@workflows_router.get("/counts_by_tag", operation_id="get_counts_by_tag")
async def get_counts_by_tag(
    tags: list[str] = Query(description="The tags to get counts for"),
    categories: Optional[list[WorkflowCategory]] = Query(default=None, description="The categories to include"),
    has_been_opened: Optional[bool] = Query(default=None, description="Whether to include/exclude recent workflows"),
) -> dict[str, int]:
    """Counts workflows by tag"""

    return ApiDependencies.invoker.services.workflow_records.counts_by_tag(
        tags=tags, categories=categories, has_been_opened=has_been_opened
    )


@workflows_router.get("/counts_by_category", operation_id="counts_by_category")
async def counts_by_category(
    categories: list[WorkflowCategory] = Query(description="The categories to include"),
    has_been_opened: Optional[bool] = Query(default=None, description="Whether to include/exclude recent workflows"),
) -> dict[str, int]:
    """Counts workflows by category"""

    return ApiDependencies.invoker.services.workflow_records.counts_by_category(
        categories=categories, has_been_opened=has_been_opened
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
