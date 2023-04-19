# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)
import io
from datetime import datetime, timezone
import json
import os
import uuid

from fastapi import HTTPException, Path, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.routing import APIRouter
from PIL import Image
from invokeai.app.api.models.images import ImageResponse
from invokeai.app.modules.metadata import ImageMetadata, InvokeAIMetadata
from invokeai.app.services.item_storage import PaginatedResults

from ...services.image_storage import ImageType
from ..dependencies import ApiDependencies

images_router = APIRouter(prefix="/v1/images", tags=["images"])


@images_router.get("/{image_type}/{image_name}", operation_id="get_image")
async def get_image(
    image_type: ImageType = Path(description="The type of image to get"),
    image_name: str = Path(description="The name of the image to get"),
) -> FileResponse | Response:
    """Gets a result"""

    path = ApiDependencies.invoker.services.images.get_path(
        image_type=image_type, image_name=image_name
    )

    if ApiDependencies.invoker.services.images.validate_path(path):
        return FileResponse(path)
    else:
        raise HTTPException(status_code=404)


@images_router.get(
    "/{image_type}/thumbnails/{image_name}", operation_id="get_thumbnail"
)
async def get_thumbnail(
    image_type: ImageType = Path(description="The type of image to get"),
    image_name: str = Path(description="The name of the image to get"),
) -> FileResponse | Response:
    """Gets a thumbnail"""

    path = ApiDependencies.invoker.services.images.get_path(
        image_type=image_type, image_name=image_name, is_thumbnail=True
    )

    if ApiDependencies.invoker.services.images.validate_path(path):
        return FileResponse(path)
    else:
        raise HTTPException(status_code=404)


@images_router.post(
    "/uploads/",
    operation_id="upload_image",
    responses={
        201: {
            "description": "The image was uploaded successfully",
            "model": ImageResponse,
        },
        415: {"description": "Image upload failed"},
    },
    status_code=201,
)
async def upload_image(
    file: UploadFile, request: Request, response: Response
) -> ImageResponse:
    if not file.content_type.startswith("image"):
        raise HTTPException(status_code=415, detail="Not an image")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))
    except:
        # Error opening the image
        raise HTTPException(status_code=415, detail="Failed to read image")

    filename = f"{uuid.uuid4()}_{str(int(datetime.now(timezone.utc).timestamp()))}.png"
    (image_path, thumbnail_path, ctime) = ApiDependencies.invoker.services.images.save(
        ImageType.UPLOAD, filename, img
    )

    # TODO: handle old `sd-metadata` style metadata
    invokeai_metadata = img.info.get("invokeai", None)

    if invokeai_metadata is not None:
        invokeai_metadata = InvokeAIMetadata(**json.loads(invokeai_metadata))

    # TODO: should creation of this object should happen elsewhere?
    res = ImageResponse(
        image_type=ImageType.UPLOAD,
        image_name=filename,
        image_url=f"api/v1/images/{ImageType.UPLOAD.value}/{filename}",
        thumbnail_url=f"api/v1/images/{ImageType.UPLOAD.value}/thumbnails/{os.path.splitext(filename)[0]}.webp",
        metadata=ImageMetadata(
            created=ctime,
            width=img.width,
            height=img.height,
            mode=img.mode,
            invokeai=invokeai_metadata,
        ),
    )

    response.status_code = 201
    response.headers["Location"] = request.url_for(
        "get_image", image_type=ImageType.UPLOAD.value, image_name=filename
    )

    return res


@images_router.get(
    "/",
    operation_id="list_images",
    responses={200: {"model": PaginatedResults[ImageResponse]}},
)
async def list_images(
    image_type: ImageType = Query(
        default=ImageType.RESULT, description="The type of images to get"
    ),
    page: int = Query(default=0, description="The page of images to get"),
    per_page: int = Query(default=10, description="The number of images per page"),
) -> PaginatedResults[ImageResponse]:
    """Gets a list of images"""
    result = ApiDependencies.invoker.services.images.list(image_type, page, per_page)
    return result
