from fastapi import HTTPException, Path, Query
from fastapi.routing import APIRouter
from invokeai.app.models.image import (
    ImageCategory,
    ImageType,
)
from invokeai.app.services.database.images.models import ImageEntity
from invokeai.app.services.item_storage import PaginatedResults

from ..dependencies import ApiDependencies

image_resources_router = APIRouter(prefix="/v1/resources/images", tags=["resources"])


@image_resources_router.get("/{image_id}", operation_id="get_image_resource")
async def get_image_resource(
    image_id: str = Path(description="The id of the image resource to get"),
) -> ImageEntity:
    """Gets a resource (eg image or tensor)"""

    image = ApiDependencies.invoker.services.images_db.get(id=image_id)

    if image is None:
        raise HTTPException(status_code=404)

    urls = ApiDependencies.invoker.services.urls.get_image_urls(image_id=image_id)

    image.image_url = urls.image_url
    image.thumbnail_url = urls.thumbnail_url

    return image


@image_resources_router.get(
    "/",
    operation_id="list_image_resources",
)
async def list_image_resources(
    image_type: ImageType = Query(description="The origin of image resources to get"),
    image_category: ImageCategory = Query(
        description="The kind of image resources to get"
    ),
    page: int = Query(default=0, description="The page of image resources to get"),
    per_page: int = Query(
        default=10, description="The number of image resources per page"
    ),
) -> PaginatedResults[ImageEntity]:
    """Gets a list of image resources"""

    images = ApiDependencies.invoker.services.images_db.get_many(
        image_type=image_type,
        image_category=image_category,
        page=page,
        per_page=per_page,
    )

    for i in images.items:
        urls = ApiDependencies.invoker.services.urls.get_image_urls(image_id=i.id)

        i.image_url = urls.image_url
        i.thumbnail_url = urls.thumbnail_url

    return images


@image_resources_router.delete("/{image_id}", operation_id="delete_image_resource")
async def delete_image_resource(
    image_id: str = Path(description="The id of the image resource to delete"),
) -> None:
    """Deletes an image resource"""

    ApiDependencies.invoker.services.images_db.delete(id=image_id)
