# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from datetime import datetime, timezone
from io import BytesIO

from fastapi import Path, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.routing import APIRouter
from PIL import Image

from ...services.image_storage import ImageType
from ..dependencies import ApiDependencies

images_router = APIRouter(prefix="/v1/images", tags=["images"])


@images_router.get("/{image_type}/{image_name}", operation_id="get_image")
async def get_image(
    image_type: ImageType = Path(description="The type of image to get"),
    image_name: str = Path(description="The name of the image to get"),
):
    """Gets a result"""
    image = ApiDependencies.invoker.services.images.get(image_type, image_name)

    # return the image from memory
    resp = BytesIO()
    image.save(resp, "PNG")
    resp.seek(0)

    return StreamingResponse(resp, media_type="image/png")


@images_router.post(
    "/uploads/",
    operation_id="upload_image",
    responses={
        201: {"description": "The image was uploaded successfully"},
        404: {"description": "Session not found"},
    },
)
async def upload_image(file: UploadFile, request: Request):
    if not file.content_type.startswith("image"):
        return Response(status_code=415)

    contents = await file.read()
    try:
        im = Image.open(contents)
    except:
        # Error opening the image
        return Response(status_code=415)

    filename = f"{str(int(datetime.now(timezone.utc).timestamp()))}.png"
    ApiDependencies.invoker.services.images.save(ImageType.UPLOAD, filename, im)

    return Response(
        status_code=201,
        headers={
            "Location": request.url_for(
                "get_image", image_type=ImageType.UPLOAD, image_name=filename
            )
        },
    )
