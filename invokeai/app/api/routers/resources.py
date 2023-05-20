# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)
import io
from datetime import datetime, timezone
import json
import os
from typing import Any, Union
import uuid

from fastapi import Body, HTTPException, Path, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.routing import APIRouter
from PIL import Image
from invokeai.app.api.models.images import (
    ImageResponse,
    ImageResponseMetadata,
)
from invokeai.app.models.resources import ImageKind, ResourceOrigin
from invokeai.app.services.database.images.models import GeneratedImageEntity, UploadedImageEntity
from invokeai.app.services.item_storage import PaginatedResults

from ...services.image_storage import ImageType
from ..dependencies import ApiDependencies

resources_router = APIRouter(prefix="/v1/images2", tags=["resources"])


@resources_router.get("/metadata", operation_id="get_image_metadata")
async def get_image_metadata(
    id: str = Query(description="The name of the image to get"),
) -> Union[GeneratedImageEntity, UploadedImageEntity]:
    """Gets an image"""

    image = ApiDependencies.invoker.services.images_db.get(id)

    if image is None:
        raise HTTPException(status_code=404)
    
    return image 


# @resources_router.delete("/{image_type}/{image_name}", operation_id="delete_image")
# async def delete_image(
#     image_type: ImageType = Path(description="The type of image to delete"),
#     image_name: str = Path(description="The name of the image to delete"),
# ) -> None:
#     """Deletes an image and its thumbnail"""

#     ApiDependencies.invoker.services.images.delete(
#         image_type=image_type, image_name=image_name
#     )


# @resources_router.get(
#     "/{thumbnail_type}/thumbnails/{thumbnail_name}", operation_id="get_thumbnail"
# )
# async def get_thumbnail(
#     thumbnail_type: ImageType = Path(description="The type of thumbnail to get"),
#     thumbnail_name: str = Path(description="The name of the thumbnail to get"),
# ) -> FileResponse | Response:
#     """Gets a thumbnail"""

#     path = ApiDependencies.invoker.services.images.get_path(
#         image_type=thumbnail_type, image_name=thumbnail_name, is_thumbnail=True
#     )

#     if ApiDependencies.invoker.services.images.validate_path(path):
#         return FileResponse(path)
#     else:
#         raise HTTPException(status_code=404)


# @resources_router.post(
#     "/uploads/",
#     operation_id="upload_image",
#     responses={
#         201: {
#             "description": "The image was uploaded successfully",
#             "model": ImageResponse,
#         },
#         415: {"description": "Image upload failed"},
#     },
#     status_code=201,
# )
# async def upload_image(
#     file: UploadFile, image_type: ImageType, request: Request, response: Response
# ) -> ImageResponse:
#     if not file.content_type.startswith("image"):
#         raise HTTPException(status_code=415, detail="Not an image")

#     contents = await file.read()

#     try:
#         img = Image.open(io.BytesIO(contents))
#     except:
#         # Error opening the image
#         raise HTTPException(status_code=415, detail="Failed to read image")

#     filename = f"{uuid.uuid4()}_{str(int(datetime.now(timezone.utc).timestamp()))}.png"

#     saved_image = ApiDependencies.invoker.services.images.save(
#         image_type, filename, img
#     )

#     invokeai_metadata = ApiDependencies.invoker.services.metadata.get_metadata(img)

#     image_url = ApiDependencies.invoker.services.images.get_uri(
#         image_type, saved_image.image_name
#     )

#     thumbnail_url = ApiDependencies.invoker.services.images.get_uri(
#         image_type, saved_image.image_name, True
#     )

#     res = ImageResponse(
#         image_type=image_type,
#         image_name=saved_image.image_name,
#         image_url=image_url,
#         thumbnail_url=thumbnail_url,
#         metadata=ImageResponseMetadata(
#             created=saved_image.created,
#             width=img.width,
#             height=img.height,
#             invokeai=invokeai_metadata,
#         ),
#     )

#     response.status_code = 201
#     response.headers["Location"] = image_url

#     return res


# @resources_router.get(
#     "/",
#     operation_id="list_images",
#     responses={200: {"model": PaginatedResults[ImageResponse]}},
# )
# async def list_images(
#     image_type: ImageType = Query(
#         default=ImageType.RESULT, description="The type of images to get"
#     ),
#     page: int = Query(default=0, description="The page of images to get"),
#     per_page: int = Query(default=10, description="The number of images per page"),
# ) -> PaginatedResults[ImageResponse]:
#     """Gets a list of images"""
#     result = ApiDependencies.invoker.services.images.list(image_type, page, per_page)
#     return result
