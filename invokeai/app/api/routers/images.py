import io
import uuid
from fastapi import HTTPException, Path, Query, Request, Response, UploadFile
from fastapi.routing import APIRouter
from PIL import Image
from invokeai.app.models.image import (
    ImageCategory,
    ImageType,
)
from invokeai.app.services.image_record_storage import ImageRecordStorageBase
from invokeai.app.services.image_file_storage import ImageFileStorageBase
from invokeai.app.services.models.image_record import ImageRecord
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
) -> ImageRecord:
    """Uploads an image"""
    if not file.content_type.startswith("image"):
        raise HTTPException(status_code=415, detail="Not an image")

    contents = await file.read()

    try:
        img = Image.open(io.BytesIO(contents))
    except:
        # Error opening the image
        raise HTTPException(status_code=415, detail="Failed to read image")

    try:
        image_record = ApiDependencies.invoker.services.images_new.create(
            image=img,
            image_type=image_type,
            image_category=image_category,
        )

        response.status_code = 201
        response.headers["Location"] = image_record.image_url

        return image_record
    except Exception as e:
        raise HTTPException(status_code=500)



@images_router.delete("/{image_type}/{image_name}", operation_id="delete_image")
async def delete_image_record(
    image_type: ImageType = Query(description="The type of image to delete"),
    image_name: str = Path(description="The name of the image to delete"),
) -> None:
    """Deletes an image record"""

    try:
        ApiDependencies.invoker.services.images_new.delete(
            image_type=image_type, image_name=image_name
        )
    except Exception as e:
        # TODO: Does this need any exception handling at all?
        pass
