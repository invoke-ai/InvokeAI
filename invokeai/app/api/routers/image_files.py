# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team
from fastapi import HTTPException, Path
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter
from invokeai.app.models.image import ImageType

from ..dependencies import ApiDependencies

image_files_router = APIRouter(prefix="/v1/files/images", tags=["images", "files"])


@image_files_router.get("/{image_type}/{image_name}", operation_id="get_image")
async def get_image(
    image_type: ImageType = Path(description="The type of the image to get"),
    image_name: str = Path(description="The id of the image to get"),
) -> FileResponse:
    """Gets an image"""

    try:
        path = ApiDependencies.invoker.services.images_new.get_path(
            image_type=image_type, image_name=image_name
        )

        return FileResponse(path)
    except Exception as e:
        raise HTTPException(status_code=404)


@image_files_router.get(
    "/{image_type}/{image_name}/thumbnail", operation_id="get_thumbnail"
)
async def get_thumbnail(
    image_type: ImageType = Path(
        description="The type of the image whose thumbnail to get"
    ),
    image_name: str = Path(description="The id of the image whose thumbnail to get"),
) -> FileResponse:
    """Gets a thumbnail"""

    try:
        path = ApiDependencies.invoker.services.images_new.get_path(
            image_type=image_type, image_name=image_name, thumbnail=True
        )

        return FileResponse(path)
    except Exception as e:
        raise HTTPException(status_code=404)
