# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from fastapi import Path
from fastapi.routing import APIRouter
from fastapi.responses import FileResponse
from ...services.image_storage import ImageType
from ..dependencies import ApiDependencies

images_router = APIRouter(
    prefix = '/v1/images',
    tags = ['images']
)


@images_router.get('/{image_type}/{image_name}',
    operation_id = 'get_image'
    )
async def get_image(
    image_type: ImageType = Path(description = "The type of image to get"),
    image_name: str = Path(description = "The name of the image to get")
):
    """Gets a result"""
    # TODO: This is not really secure at all. At least make sure only output results are served
    filename = ApiDependencies.invoker.services.images.get_path(image_type, image_name)
    return FileResponse(filename)
