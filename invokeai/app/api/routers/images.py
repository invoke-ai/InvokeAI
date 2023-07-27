import io
from typing import Optional

from fastapi import Body, HTTPException, Path, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter
from PIL import Image

from invokeai.app.invocations.metadata import ImageMetadata
from invokeai.app.models.image import ImageCategory, ResourceOrigin
from invokeai.app.services.image_record_storage import OffsetPaginatedResults
from invokeai.app.services.item_storage import PaginatedResults
from invokeai.app.services.models.image_record import (
    ImageDTO,
    ImageRecordChanges,
    ImageUrlsDTO,
)

from ..dependencies import ApiDependencies

images_router = APIRouter(prefix="/v1/images", tags=["images"])

# images are immutable; set a high max-age
IMAGE_MAX_AGE = 31536000


@images_router.post(
    "/",
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
) -> ImageDTO:
    """Uploads an image"""
    if not file.content_type.startswith("image"):
        raise HTTPException(status_code=415, detail="Not an image")

    contents = await file.read()

    try:
        pil_image = Image.open(io.BytesIO(contents))
        if crop_visible:
            bbox = pil_image.getbbox()
            pil_image = pil_image.crop(bbox)
    except:
        # Error opening the image
        raise HTTPException(status_code=415, detail="Failed to read image")

    try:
        image_dto = ApiDependencies.invoker.services.images.create(
            image=pil_image,
            image_origin=ResourceOrigin.EXTERNAL,
            image_category=image_category,
            session_id=session_id,
            board_id=board_id,
            is_intermediate=is_intermediate,
        )

        response.status_code = 201
        response.headers["Location"] = image_dto.image_url

        return image_dto
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create image")


@images_router.delete("/{image_name}", operation_id="delete_image")
async def delete_image(
    image_name: str = Path(description="The name of the image to delete"),
) -> None:
    """Deletes an image"""

    try:
        ApiDependencies.invoker.services.images.delete(image_name)
    except Exception as e:
        # TODO: Does this need any exception handling at all?
        pass


@images_router.post("/clear-intermediates", operation_id="clear_intermediates")
async def clear_intermediates() -> int:
    """Clears all intermediates"""

    try:
        count_deleted = ApiDependencies.invoker.services.images.delete_intermediates()
        return count_deleted
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to clear intermediates")
        pass


@images_router.patch(
    "/{image_name}",
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
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to update image")


@images_router.get(
    "/{image_name}",
    operation_id="get_image_dto",
    response_model=ImageDTO,
)
async def get_image_dto(
    image_name: str = Path(description="The name of image to get"),
) -> ImageDTO:
    """Gets an image's DTO"""

    try:
        return ApiDependencies.invoker.services.images.get_dto(image_name)
    except Exception as e:
        raise HTTPException(status_code=404)


@images_router.get(
    "/{image_name}/metadata",
    operation_id="get_image_metadata",
    response_model=ImageMetadata,
)
async def get_image_metadata(
    image_name: str = Path(description="The name of image to get"),
) -> ImageMetadata:
    """Gets an image's metadata"""

    try:
        return ApiDependencies.invoker.services.images.get_metadata(image_name)
    except Exception as e:
        raise HTTPException(status_code=404)


@images_router.get(
    "/{image_name}/full",
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
async def get_image_full(
    image_name: str = Path(description="The name of full-resolution image file to get"),
) -> FileResponse:
    """Gets a full-resolution image file"""

    try:
        path = ApiDependencies.invoker.services.images.get_path(image_name)

        if not ApiDependencies.invoker.services.images.validate_path(path):
            raise HTTPException(status_code=404)

        response = FileResponse(
            path,
            media_type="image/png",
            filename=image_name,
            content_disposition_type="inline",
        )
        response.headers["Cache-Control"] = f"max-age={IMAGE_MAX_AGE}"
        return response
    except Exception as e:
        raise HTTPException(status_code=404)


@images_router.get(
    "/{image_name}/thumbnail",
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
) -> FileResponse:
    """Gets a thumbnail image file"""

    try:
        path = ApiDependencies.invoker.services.images.get_path(image_name, thumbnail=True)
        if not ApiDependencies.invoker.services.images.validate_path(path):
            raise HTTPException(status_code=404)

        response = FileResponse(path, media_type="image/webp", content_disposition_type="inline")
        response.headers["Cache-Control"] = f"max-age={IMAGE_MAX_AGE}"
        return response
    except Exception as e:
        raise HTTPException(status_code=404)


@images_router.get(
    "/{image_name}/urls",
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
    except Exception as e:
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
) -> OffsetPaginatedResults[ImageDTO]:
    """Gets a list of image DTOs"""

    image_dtos = ApiDependencies.invoker.services.images.get_many(
        offset,
        limit,
        image_origin,
        categories,
        is_intermediate,
        board_id,
    )

    return image_dtos
