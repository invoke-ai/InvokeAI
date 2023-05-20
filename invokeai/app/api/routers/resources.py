from fastapi import HTTPException, Path, Query
from fastapi.routing import APIRouter
from invokeai.app.models.resources import ImageKind, ResourceOrigin
from invokeai.app.services.database.images.models import ImageEntity
from invokeai.app.services.item_storage import PaginatedResults

from ..dependencies import ApiDependencies

resources_router = APIRouter(prefix="/v1/resources", tags=["resources"])


@resources_router.get("/images/{image_id}", operation_id="get_image_resource")
async def get_image_resource(
    image_id: str = Path(description="The id of the image resource to get"),
) -> ImageEntity:
    """Gets an image resource"""

    image = ApiDependencies.invoker.services.images_db.get(id=image_id)

    if image is None:
        raise HTTPException(status_code=404)

    return image


@resources_router.get(
    "/images",
    operation_id="list_images",
)
async def list_images(
    origin: ResourceOrigin = Query(description="The origin of image resources to get"),
    image_kind: ImageKind = Query(description="The kind of image resources to get"),
    page: int = Query(default=0, description="The page of image resources to get"),
    per_page: int = Query(
        default=10, description="The number of image resources per page"
    ),
) -> PaginatedResults[ImageEntity]:
    """Gets a list of image resources"""

    result = ApiDependencies.invoker.services.images_db.get_many(
        image_kind=image_kind, origin=origin, page=page, per_page=per_page
    )

    return result


@resources_router.delete("/images/{image_id}", operation_id="delete_image")
async def delete_image(
    image_id: str = Path(description="The id of the image resource to get"),
) -> None:
    """Deletes an image resource"""

    ApiDependencies.invoker.services.images_db.delete(id=image_id)
