import io
import traceback
from typing import Optional

from fastapi import BackgroundTasks, Body, HTTPException, Path, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter
from PIL import Image
from pydantic import BaseModel, Field

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.extract_metadata_from_image import extract_metadata_from_image
from invokeai.app.invocations.fields import MetadataField
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ImageRecordChanges,
    ResourceOrigin,
)
from invokeai.app.services.images.images_common import ImageDTO, ImageUrlsDTO
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection

images_router = APIRouter(prefix="/v1/images", tags=["images"])


# images are immutable; set a high max-age
IMAGE_MAX_AGE = 31536000


@images_router.post(
    "/upload",
    operation_id="upload_image",
    responses={
        201: {"description": "The image was uploaded successfully"},
        415: {"description": "Image upload failed"},
    },
    status_code=201,
    response_model=ImageDTO,
)
async def upload_image(
    file: UploadFile,
    request: Request,
    response: Response,
    image_category: ImageCategory = Query(description="The category of the image"),
    is_intermediate: bool = Query(description="Whether this is an intermediate image"),
    board_id: Optional[str] = Query(default=None, description="The board to add this image to, if any"),
    session_id: Optional[str] = Query(default=None, description="The session ID associated with this upload, if any"),
    crop_visible: Optional[bool] = Query(default=False, description="Whether to crop the image"),
    metadata: Optional[str] = Body(
        default=None,
        description="The metadata to associate with the image, must be a stringified JSON dict",
        embed=True,
    ),
) -> ImageDTO:
    """Uploads an image"""
    if not file.content_type or not file.content_type.startswith("image"):
        raise HTTPException(status_code=415, detail="Not an image")

    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
        if crop_visible:
            bbox = pil_image.getbbox()
            pil_image = pil_image.crop(bbox)
    except Exception:
        ApiDependencies.invoker.services.logger.error(traceback.format_exc())
        raise HTTPException(status_code=415, detail="Failed to read image")

    extracted_metadata = extract_metadata_from_image(
        pil_image=pil_image,
        invokeai_metadata_override=metadata,
        invokeai_workflow_override=None,
        invokeai_graph_override=None,
        logger=ApiDependencies.invoker.services.logger,
    )

    try:
        image_dto = ApiDependencies.invoker.services.images.create(
            image=pil_image,
            image_origin=ResourceOrigin.EXTERNAL,
            image_category=image_category,
            session_id=session_id,
            board_id=board_id,
            metadata=extracted_metadata.invokeai_metadata,
            workflow=extracted_metadata.invokeai_workflow,
            graph=extracted_metadata.invokeai_graph,
            is_intermediate=is_intermediate,
        )

        response.status_code = 201
        response.headers["Location"] = image_dto.image_url

        return image_dto
    except Exception:
        ApiDependencies.invoker.services.logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to create image")


@images_router.delete("/i/{image_name}", operation_id="delete_image")
async def delete_image(
    image_name: str = Path(description="The name of the image to delete"),
) -> None:
    """Deletes an image"""

    try:
        ApiDependencies.invoker.services.images.delete(image_name)
    except Exception:
        # TODO: Does this need any exception handling at all?
        pass


@images_router.delete("/intermediates", operation_id="clear_intermediates")
async def clear_intermediates() -> int:
    """Clears all intermediates"""

    try:
        count_deleted = ApiDependencies.invoker.services.images.delete_intermediates()
        return count_deleted
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to clear intermediates")
        pass


@images_router.get("/intermediates", operation_id="get_intermediates_count")
async def get_intermediates_count() -> int:
    """Gets the count of intermediate images"""

    try:
        return ApiDependencies.invoker.services.images.get_intermediates_count()
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get intermediates")
        pass


@images_router.patch(
    "/i/{image_name}",
    operation_id="update_image",
    response_model=ImageDTO,
)
async def update_image(
    image_name: str = Path(description="The name of the image to update"),
    image_changes: ImageRecordChanges = Body(description="The changes to apply to the image"),
) -> ImageDTO:
    """Updates an image"""

    try:
        return ApiDependencies.invoker.services.images.update(image_name, image_changes)
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to update image")


@images_router.get(
    "/i/{image_name}",
    operation_id="get_image_dto",
    response_model=ImageDTO,
)
async def get_image_dto(
    image_name: str = Path(description="The name of image to get"),
) -> ImageDTO:
    """Gets an image's DTO"""

    try:
        return ApiDependencies.invoker.services.images.get_dto(image_name)
    except Exception:
        raise HTTPException(status_code=404)


@images_router.get(
    "/i/{image_name}/metadata",
    operation_id="get_image_metadata",
    response_model=Optional[MetadataField],
)
async def get_image_metadata(
    image_name: str = Path(description="The name of image to get"),
) -> Optional[MetadataField]:
    """Gets an image's metadata"""

    try:
        return ApiDependencies.invoker.services.images.get_metadata(image_name)
    except Exception:
        raise HTTPException(status_code=404)


class WorkflowAndGraphResponse(BaseModel):
    workflow: Optional[str] = Field(description="The workflow used to generate the image, as stringified JSON")
    graph: Optional[str] = Field(description="The graph used to generate the image, as stringified JSON")


@images_router.get(
    "/i/{image_name}/workflow", operation_id="get_image_workflow", response_model=WorkflowAndGraphResponse
)
async def get_image_workflow(
    image_name: str = Path(description="The name of image whose workflow to get"),
) -> WorkflowAndGraphResponse:
    try:
        workflow = ApiDependencies.invoker.services.images.get_workflow(image_name)
        graph = ApiDependencies.invoker.services.images.get_graph(image_name)
        return WorkflowAndGraphResponse(workflow=workflow, graph=graph)
    except Exception:
        raise HTTPException(status_code=404)


@images_router.get(
    "/i/{image_name}/full",
    operation_id="get_image_full",
    response_class=Response,
    responses={
        200: {
            "description": "Return the full-resolution image",
            "content": {"image/png": {}},
        },
        404: {"description": "Image not found"},
    },
)
@images_router.head(
    "/i/{image_name}/full",
    operation_id="get_image_full_head",
    response_class=Response,
    responses={
        200: {
            "description": "Return the full-resolution image",
            "content": {"image/png": {}},
        },
        404: {"description": "Image not found"},
    },
)
async def get_image_full(
    image_name: str = Path(description="The name of full-resolution image file to get"),
) -> Response:
    """Gets a full-resolution image file"""

    try:
        path = ApiDependencies.invoker.services.images.get_path(image_name)
        with open(path, "rb") as f:
            content = f.read()
        response = Response(content, media_type="image/png")
        response.headers["Cache-Control"] = f"max-age={IMAGE_MAX_AGE}"
        response.headers["Content-Disposition"] = f'inline; filename="{image_name}"'
        return response
    except Exception:
        raise HTTPException(status_code=404)


@images_router.get(
    "/i/{image_name}/thumbnail",
    operation_id="get_image_thumbnail",
    response_class=Response,
    responses={
        200: {
            "description": "Return the image thumbnail",
            "content": {"image/webp": {}},
        },
        404: {"description": "Image not found"},
    },
)
async def get_image_thumbnail(
    image_name: str = Path(description="The name of thumbnail image file to get"),
) -> Response:
    """Gets a thumbnail image file"""

    try:
        path = ApiDependencies.invoker.services.images.get_path(image_name, thumbnail=True)
        with open(path, "rb") as f:
            content = f.read()
        response = Response(content, media_type="image/webp")
        response.headers["Cache-Control"] = f"max-age={IMAGE_MAX_AGE}"
        return response
    except Exception:
        raise HTTPException(status_code=404)


@images_router.get(
    "/i/{image_name}/urls",
    operation_id="get_image_urls",
    response_model=ImageUrlsDTO,
)
async def get_image_urls(
    image_name: str = Path(description="The name of the image whose URL to get"),
) -> ImageUrlsDTO:
    """Gets an image and thumbnail URL"""

    try:
        image_url = ApiDependencies.invoker.services.images.get_url(image_name)
        thumbnail_url = ApiDependencies.invoker.services.images.get_url(image_name, thumbnail=True)
        return ImageUrlsDTO(
            image_name=image_name,
            image_url=image_url,
            thumbnail_url=thumbnail_url,
        )
    except Exception:
        raise HTTPException(status_code=404)


@images_router.get(
    "/",
    operation_id="list_image_dtos",
    response_model=OffsetPaginatedResults[ImageDTO],
)
async def list_image_dtos(
    image_origin: Optional[ResourceOrigin] = Query(default=None, description="The origin of images to list."),
    categories: Optional[list[ImageCategory]] = Query(default=None, description="The categories of image to include."),
    is_intermediate: Optional[bool] = Query(default=None, description="Whether to list intermediate images."),
    board_id: Optional[str] = Query(
        default=None,
        description="The board id to filter by. Use 'none' to find images without a board.",
    ),
    offset: int = Query(default=0, description="The page offset"),
    limit: int = Query(default=10, description="The number of images per page"),
    order_dir: SQLiteDirection = Query(default=SQLiteDirection.Descending, description="The order of sort"),
    starred_first: bool = Query(default=True, description="Whether to sort by starred images first"),
    search_term: Optional[str] = Query(default=None, description="The term to search for"),
) -> OffsetPaginatedResults[ImageDTO]:
    """Gets a list of image DTOs"""

    image_dtos = ApiDependencies.invoker.services.images.get_many(
        offset, limit, starred_first, order_dir, image_origin, categories, is_intermediate, board_id, search_term
    )

    return image_dtos


class DeleteImagesFromListResult(BaseModel):
    deleted_images: list[str]


@images_router.post("/delete", operation_id="delete_images_from_list", response_model=DeleteImagesFromListResult)
async def delete_images_from_list(
    image_names: list[str] = Body(description="The list of names of images to delete", embed=True),
) -> DeleteImagesFromListResult:
    try:
        deleted_images: list[str] = []
        for image_name in image_names:
            try:
                ApiDependencies.invoker.services.images.delete(image_name)
                deleted_images.append(image_name)
            except Exception:
                pass
        return DeleteImagesFromListResult(deleted_images=deleted_images)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to delete images")


class ImagesUpdatedFromListResult(BaseModel):
    updated_image_names: list[str] = Field(description="The image names that were updated")


@images_router.post("/star", operation_id="star_images_in_list", response_model=ImagesUpdatedFromListResult)
async def star_images_in_list(
    image_names: list[str] = Body(description="The list of names of images to star", embed=True),
) -> ImagesUpdatedFromListResult:
    try:
        updated_image_names: list[str] = []
        for image_name in image_names:
            try:
                ApiDependencies.invoker.services.images.update(image_name, changes=ImageRecordChanges(starred=True))
                updated_image_names.append(image_name)
            except Exception:
                pass
        return ImagesUpdatedFromListResult(updated_image_names=updated_image_names)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to star images")


@images_router.post("/unstar", operation_id="unstar_images_in_list", response_model=ImagesUpdatedFromListResult)
async def unstar_images_in_list(
    image_names: list[str] = Body(description="The list of names of images to unstar", embed=True),
) -> ImagesUpdatedFromListResult:
    try:
        updated_image_names: list[str] = []
        for image_name in image_names:
            try:
                ApiDependencies.invoker.services.images.update(image_name, changes=ImageRecordChanges(starred=False))
                updated_image_names.append(image_name)
            except Exception:
                pass
        return ImagesUpdatedFromListResult(updated_image_names=updated_image_names)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to unstar images")


class ImagesDownloaded(BaseModel):
    response: Optional[str] = Field(
        default=None, description="The message to display to the user when images begin downloading"
    )
    bulk_download_item_name: Optional[str] = Field(
        default=None, description="The name of the bulk download item for which events will be emitted"
    )


@images_router.post(
    "/download", operation_id="download_images_from_list", response_model=ImagesDownloaded, status_code=202
)
async def download_images_from_list(
    background_tasks: BackgroundTasks,
    image_names: Optional[list[str]] = Body(
        default=None, description="The list of names of images to download", embed=True
    ),
    board_id: Optional[str] = Body(
        default=None, description="The board from which image should be downloaded", embed=True
    ),
) -> ImagesDownloaded:
    if (image_names is None or len(image_names) == 0) and board_id is None:
        raise HTTPException(status_code=400, detail="No images or board id specified.")
    bulk_download_item_id: str = ApiDependencies.invoker.services.bulk_download.generate_item_id(board_id)

    background_tasks.add_task(
        ApiDependencies.invoker.services.bulk_download.handler,
        image_names,
        board_id,
        bulk_download_item_id,
    )
    return ImagesDownloaded(bulk_download_item_name=bulk_download_item_id + ".zip")


@images_router.api_route(
    "/download/{bulk_download_item_name}",
    methods=["GET"],
    operation_id="get_bulk_download_item",
    response_class=Response,
    responses={
        200: {
            "description": "Return the complete bulk download item",
            "content": {"application/zip": {}},
        },
        404: {"description": "Image not found"},
    },
)
async def get_bulk_download_item(
    background_tasks: BackgroundTasks,
    bulk_download_item_name: str = Path(description="The bulk_download_item_name of the bulk download item to get"),
) -> FileResponse:
    """Gets a bulk download zip file"""
    try:
        path = ApiDependencies.invoker.services.bulk_download.get_path(bulk_download_item_name)

        response = FileResponse(
            path,
            media_type="application/zip",
            filename=bulk_download_item_name,
            content_disposition_type="inline",
        )
        response.headers["Cache-Control"] = f"max-age={IMAGE_MAX_AGE}"
        background_tasks.add_task(ApiDependencies.invoker.services.bulk_download.delete, bulk_download_item_name)
        return response
    except Exception:
        raise HTTPException(status_code=404)
