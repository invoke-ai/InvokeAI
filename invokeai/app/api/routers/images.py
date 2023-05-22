import io
from fastapi import HTTPException, Path, Query, Request, Response, UploadFile
from fastapi.routing import APIRouter
from fastapi.responses import FileResponse
from PIL import Image
from invokeai.app.models.image import (
    ImageCategory,
    ImageType,
)
from invokeai.app.services.models.image_record import ImageDTO, ImageUrlsDTO
from invokeai.app.services.item_storage import PaginatedResults

from ..dependencies import ApiDependencies

images_router = APIRouter(prefix="/v1/images", tags=["images"])


@images_router.post(
    "/",
    operation_id="upload_image",
    responses={
        201: {"description": "The image was uploaded successfully"},
        415: {"description": "Image upload failed"},
    },
    status_code=201,
)
async def upload_image(
    file: UploadFile,
    image_type: ImageType,
    request: Request,
    response: Response,
    image_category: ImageCategory = ImageCategory.IMAGE,
) -> ImageDTO:
    """Uploads an image"""
    if not file.content_type.startswith("image"):
        raise HTTPException(status_code=415, detail="Not an image")

    contents = await file.read()

    try:
        pil_image = Image.open(io.BytesIO(contents))
    except:
        # Error opening the image
        raise HTTPException(status_code=415, detail="Failed to read image")

    try:
        image_dto = ApiDependencies.invoker.services.images_new.create(
            pil_image,
            image_type,
            image_category,
        )

        response.status_code = 201
        response.headers["Location"] = image_dto.image_url

        return image_dto
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create image")


@images_router.delete("/{image_type}/{image_name}", operation_id="delete_image")
async def delete_image(
    image_type: ImageType = Query(description="The type of image to delete"),
    image_name: str = Path(description="The name of the image to delete"),
) -> None:
    """Deletes an image"""

    try:
        ApiDependencies.invoker.services.images_new.delete(image_type, image_name)
    except Exception as e:
        # TODO: Does this need any exception handling at all?
        pass


@images_router.get(
    "/{image_type}/{image_name}/record",
    operation_id="get_image_record",
    response_model=ImageDTO,
)
async def get_image_record(
    image_type: ImageType = Path(description="The type of the image record to get"),
    image_name: str = Path(description="The id of the image record to get"),
) -> ImageDTO:
    """Gets an image record by id"""

    try:
        return ApiDependencies.invoker.services.images_new.get_dto(
            image_type, image_name
        )
    except Exception as e:
        raise HTTPException(status_code=404)


@images_router.get("/{image_type}/{image_name}/image", operation_id="get_image")
async def get_image(
    image_type: ImageType = Path(description="The type of the image to get"),
    image_name: str = Path(description="The id of the image to get"),
) -> FileResponse:
    """Gets an image"""

    try:
        path = ApiDependencies.invoker.services.images_new.get_path(
            image_type, image_name
        )

        return FileResponse(path)
    except Exception as e:
        raise HTTPException(status_code=404)


@images_router.get("/{image_type}/{image_name}/thumbnail", operation_id="get_thumbnail")
async def get_thumbnail(
    image_type: ImageType = Path(
        description="The type of the image whose thumbnail to get"
    ),
    image_name: str = Path(description="The id of the image whose thumbnail to get"),
) -> FileResponse:
    """Gets a thumbnail"""

    try:
        path = ApiDependencies.invoker.services.images_new.get_path(
            image_type, image_name, thumbnail=True
        )

        return FileResponse(path)
    except Exception as e:
        raise HTTPException(status_code=404)


@images_router.get(
    "/{image_type}/{image_name}/urls",
    operation_id="get_image_urls",
    response_model=ImageUrlsDTO,
)
async def get_image_urls(
    image_type: ImageType = Path(description="The type of the image whose URL to get"),
    image_name: str = Path(description="The id of the image whose URL to get"),
) -> ImageUrlsDTO:
    """Gets an image and thumbnail URL"""

    try:
        image_url = ApiDependencies.invoker.services.images_new.get_url(
            image_type, image_name
        )
        thumbnail_url = ApiDependencies.invoker.services.images_new.get_url(
            image_type, image_name, thumbnail=True
        )
        return ImageUrlsDTO(
            image_type=image_type,
            image_name=image_name,
            image_url=image_url,
            thumbnail_url=thumbnail_url,
        )
    except Exception as e:
        raise HTTPException(status_code=404)


@images_router.get(
    "/",
    operation_id="list_image_records",
    response_model=PaginatedResults[ImageDTO],
)
async def list_image_records(
    image_type: ImageType = Query(description="The type of image records to get"),
    image_category: ImageCategory = Query(
        description="The kind of image records to get"
    ),
    page: int = Query(default=0, description="The page of image records to get"),
    per_page: int = Query(
        default=10, description="The number of image records per page"
    ),
) -> PaginatedResults[ImageDTO]:
    """Gets a list of image records by type and category"""

    image_dtos = ApiDependencies.invoker.services.images_new.get_many(
        image_type,
        image_category,
        page,
        per_page,
    )

    return image_dtos
