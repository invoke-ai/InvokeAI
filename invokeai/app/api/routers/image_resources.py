from fastapi import HTTPException, Path, Query
from fastapi.routing import APIRouter
from invokeai.app.models.image import (
    ImageCategory,
    ImageType,
)
from invokeai.app.services.image_db import ImageRecordServiceBase
from invokeai.app.services.image_storage import ImageStorageBase
from invokeai.app.services.models.image_record import ImageRecord
from invokeai.app.services.item_storage import PaginatedResults

from ..dependencies import ApiDependencies

image_records_router = APIRouter(prefix="/v1/records/images", tags=["records"])


@image_records_router.get("/{image_type}/{image_name}", operation_id="get_image_record")
async def get_image_record(
    image_type: ImageType = Path(description="The type of the image record to get"),
    image_name: str = Path(description="The id of the image record to get"),
) -> ImageRecord:
    """Gets an image record by id"""

    try:
        return ApiDependencies.invoker.services.images_new.get_record(
            image_type=image_type, image_name=image_name
        )
    except ImageRecordServiceBase.ImageRecordNotFoundException:
        raise HTTPException(status_code=404)


@image_records_router.get(
    "/",
    operation_id="list_image_records",
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
) -> PaginatedResults[ImageRecord]:
    """Gets a list of image records by type and category"""

    images = ApiDependencies.invoker.services.images_new.get_many(
        image_type=image_type,
        image_category=image_category,
        page=page,
        per_page=per_page,
    )

    return images


@image_records_router.delete("/{image_type}/{image_name}", operation_id="delete_image")
async def delete_image_record(
    image_type: ImageType = Query(description="The type of image records to get"),
    image_name: str = Path(description="The name of the image to delete"),
) -> None:
    """Deletes an image record"""

    try:
        ApiDependencies.invoker.services.images_new.delete(
            image_type=image_type, image_name=image_name
        )
    except ImageStorageBase.ImageFileDeleteException:
        # TODO: log this
        pass
    except ImageRecordServiceBase.ImageRecordDeleteException:
        # TODO: log this
        pass
