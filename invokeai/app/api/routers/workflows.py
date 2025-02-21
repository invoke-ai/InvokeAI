import io
import traceback
from typing import Optional
import json

from fastapi import APIRouter, Body, File, HTTPException, Path, Query, UploadFile, Form
from fastapi.responses import FileResponse
from PIL import Image

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers.model_manager import IMAGE_MAX_AGE
from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.workflow_records.workflow_records_common import (
    WorkflowValidator,
    WorkflowCategory,
    WorkflowNotFoundError,
    WorkflowRecordDTO,
    WorkflowRecordListItemDTO,
    WorkflowRecordOrderBy,
    WorkflowWithoutIDValidator,
)
from invokeai.app.services.workflow_thumbnails.workflow_thumbnails_common import (
    WorkflowThumbnailFileNotFoundException,
)

workflows_router = APIRouter(prefix="/v1/workflows", tags=["workflows"])


@workflows_router.get(
    "/i/{workflow_id}",
    operation_id="get_workflow",
    responses={
        200: {"model": WorkflowRecordDTO},
    },
)
async def get_workflow(
    workflow_id: str = Path(description="The workflow to get"),
) -> WorkflowRecordDTO:
    """Gets a workflow"""
    try:
        thumbnail_url = ApiDependencies.invoker.services.workflow_thumbnails.get_url(workflow_id)
        workflow = ApiDependencies.invoker.services.workflow_records.get(workflow_id)
        workflow.thumbnail_url = thumbnail_url
        return workflow
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
    workflow: str = Form(description="The updated workflow"),
    image: Optional[UploadFile] = File(description="The image file to upload", default=None),
) -> WorkflowRecordDTO:
    """Updates a workflow"""

    # parsed_data = json.loads(workflow)
    validated_workflow = WorkflowValidator.validate_json(workflow)

    if image is not None:
        if not image.content_type or not image.content_type.startswith("image"):
            raise HTTPException(status_code=415, detail="Not an image")

        contents = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(contents))

        except Exception:
            ApiDependencies.invoker.services.logger.error(traceback.format_exc())
            raise HTTPException(status_code=415, detail="Failed to read image")

        try:
            ApiDependencies.invoker.services.workflow_thumbnails.save(
                workflow_id=validated_workflow.id, image=pil_image
            )
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
    else:
        try:
            ApiDependencies.invoker.services.workflow_thumbnails.delete(workflow_id=validated_workflow.id)
        except WorkflowThumbnailFileNotFoundException:
            pass

    updated_workflow = ApiDependencies.invoker.services.workflow_records.update(workflow=validated_workflow)
    thumbnail_url = ApiDependencies.invoker.services.workflow_thumbnails.get_url(validated_workflow.id)
    updated_workflow.thumbnail_url = thumbnail_url
    return updated_workflow


@workflows_router.delete(
    "/i/{workflow_id}",
    operation_id="delete_workflow",
)
async def delete_workflow(
    workflow_id: str = Path(description="The workflow to delete"),
) -> None:
    """Deletes a workflow"""
    ApiDependencies.invoker.services.workflow_records.delete(workflow_id)
    ApiDependencies.invoker.services.workflow_thumbnails.delete(workflow_id)


@workflows_router.post(
    "/",
    operation_id="create_workflow",
    responses={
        200: {"model": WorkflowRecordDTO},
    },
)
async def create_workflow(
    workflow: str = Form(description="The workflow to create"),
    image: Optional[UploadFile] = File(description="The image file to upload", default=None),
) -> WorkflowRecordDTO:
    """Creates a workflow"""

    # parsed_data = json.loads(workflow)
    validated_workflow = WorkflowWithoutIDValidator.validate_json(workflow)

    new_workflow = ApiDependencies.invoker.services.workflow_records.create(workflow=validated_workflow)

    if image is not None:
        if not image.content_type or not image.content_type.startswith("image"):
            raise HTTPException(status_code=415, detail="Not an image")

        contents = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(contents))
        except Exception:
            ApiDependencies.invoker.services.logger.error(traceback.format_exc())
            raise HTTPException(status_code=415, detail="Failed to read image")

        try:
            ApiDependencies.invoker.services.workflow_thumbnails.save(
                workflow_id=new_workflow.workflow_id, image=pil_image
            )
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    thumbnail_url = ApiDependencies.invoker.services.workflow_thumbnails.get_url(new_workflow.workflow_id)
    new_workflow.thumbnail_url = thumbnail_url
    return new_workflow


@workflows_router.get(
    "/",
    operation_id="list_workflows",
    responses={
        200: {"model": PaginatedResults[WorkflowRecordListItemDTO]},
    },
)
async def list_workflows(
    page: int = Query(default=0, description="The page to get"),
    per_page: Optional[int] = Query(default=None, description="The number of workflows per page"),
    order_by: WorkflowRecordOrderBy = Query(
        default=WorkflowRecordOrderBy.Name, description="The attribute to order by"
    ),
    direction: SQLiteDirection = Query(default=SQLiteDirection.Ascending, description="The direction to order by"),
    category: WorkflowCategory = Query(default=WorkflowCategory.User, description="The category of workflow to get"),
    query: Optional[str] = Query(default=None, description="The text to query by (matches name and description)"),
) -> PaginatedResults[WorkflowRecordListItemDTO]:
    """Gets a page of workflows"""
    workflows = ApiDependencies.invoker.services.workflow_records.get_many(
        order_by=order_by, direction=direction, page=page, per_page=per_page, query=query, category=category
    )
    for workflow in workflows.items:
        workflow.thumbnail_url = ApiDependencies.invoker.services.workflow_thumbnails.get_url(workflow.workflow_id)
    return workflows


@workflows_router.get(
    "i/{workflow_id}/thumbnail",
    operation_id="get_workflow_thumbnail",
    responses={
        200: {"description": "Thumbnail retrieved successfully"},
        404: {"description": "Thumbnail not found"},
    },
)
async def get_workflow_thumbnail(
    workflow_id: str,
) -> FileResponse:
    """Gets the thumbnail for a workflow"""
    try:
        path = ApiDependencies.invoker.services.workflow_thumbnails.get_path(workflow_id)
        response = FileResponse(
            path,
            media_type="image/png",
            filename=f"{workflow_id}.png",
            content_disposition_type="inline",
        )
        response.headers["Cache-Control"] = f"max-age={IMAGE_MAX_AGE}"
        return response
    except WorkflowThumbnailFileNotFoundException:
        raise HTTPException(status_code=404, detail="Thumbnail not found")
