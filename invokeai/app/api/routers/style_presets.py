import io
import traceback
from typing import Optional

from fastapi import APIRouter, Body, File, Form, HTTPException, Path, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers.model_manager import IMAGE_MAX_AGE
from invokeai.app.services.style_preset_images.style_preset_images_common import StylePresetImageFileNotFoundException
from invokeai.app.services.style_preset_records.style_preset_records_common import (
    PresetData,
    StylePresetChanges,
    StylePresetNotFoundError,
    StylePresetRecordWithImage,
    StylePresetWithoutId,
)

style_presets_router = APIRouter(prefix="/v1/style_presets", tags=["style_presets"])


@style_presets_router.get(
    "/i/{style_preset_id}",
    operation_id="get_style_preset",
    responses={
        200: {"model": StylePresetRecordWithImage},
    },
)
async def get_style_preset(
    style_preset_id: str = Path(description="The style preset to get"),
) -> StylePresetRecordWithImage:
    """Gets a style preset"""
    try:
        image = ApiDependencies.invoker.services.style_preset_images_service.get_url(style_preset_id)
        style_preset = ApiDependencies.invoker.services.style_preset_records.get(style_preset_id)
        return StylePresetRecordWithImage(image=image, **style_preset.model_dump())
    except StylePresetNotFoundError:
        raise HTTPException(status_code=404, detail="Style preset not found")


@style_presets_router.patch(
    "/i/{style_preset_id}",
    operation_id="update_style_preset",
    responses={
        200: {"model": StylePresetRecordWithImage},
    },
)
async def update_style_preset(
    style_preset_id: str = Path(description="The id of the style preset to update"),
    name: str = Form(description="The name of the style preset to create"),
    positive_prompt: str = Form(description="The positive prompt of the style preset"),
    negative_prompt: str = Form(description="The negative prompt of the style preset"),
    image: Optional[UploadFile] = File(description="The image file to upload", default=None),
) -> StylePresetRecordWithImage:
    """Updates a style preset"""
    if image is not None:
        if not image.content_type or not image.content_type.startswith("image"):
            raise HTTPException(status_code=415, detail="Not an image")

        contents = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(contents))

        except Exception:
            ApiDependencies.invoker.services.logger.error(traceback.format_exc())
            raise HTTPException(status_code=415, detail="Failed to read image")

        try:
            ApiDependencies.invoker.services.style_preset_images_service.save(pil_image, style_preset_id)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
    else:
        try:
            ApiDependencies.invoker.services.style_preset_images_service.delete(style_preset_id)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    preset_data = PresetData(positive_prompt=positive_prompt, negative_prompt=negative_prompt)
    changes = StylePresetChanges(name=name, preset_data=preset_data)

    style_preset_image = ApiDependencies.invoker.services.style_preset_images_service.get_url(style_preset_id)
    style_preset = ApiDependencies.invoker.services.style_preset_records.update(id=style_preset_id, changes=changes)
    return StylePresetRecordWithImage(image=style_preset_image, **style_preset.model_dump())


@style_presets_router.delete(
    "/i/{style_preset_id}",
    operation_id="delete_style_preset",
)
async def delete_style_preset(
    style_preset_id: str = Path(description="The style preset to delete"),
) -> None:
    """Deletes a style preset"""
    try:
        ApiDependencies.invoker.services.style_preset_images_service.delete(style_preset_id)
    except StylePresetImageFileNotFoundException:
        pass

    ApiDependencies.invoker.services.style_preset_records.delete(style_preset_id)


@style_presets_router.post(
    "/",
    operation_id="create_style_preset",
    responses={
        200: {"model": StylePresetRecordWithImage},
    },
)
async def create_style_preset(
    name: str = Form(description="The name of the style preset to create"),
    positive_prompt: str = Form(description="The positive prompt of the style preset"),
    negative_prompt: str = Form(description="The negative prompt of the style preset"),
    image: Optional[UploadFile] = File(description="The image file to upload", default=None),
) -> StylePresetRecordWithImage:
    """Creates a style preset"""
    preset_data = PresetData(positive_prompt=positive_prompt, negative_prompt=negative_prompt)
    style_preset = StylePresetWithoutId(name=name, preset_data=preset_data)
    new_style_preset = ApiDependencies.invoker.services.style_preset_records.create(style_preset=style_preset)

    if image is not None:
        if not image.content_type or not image.content_type.startswith("image"):
            raise HTTPException(status_code=415, detail="Not an image")

        contents = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(contents))

        except Exception:
            ApiDependencies.invoker.services.logger.error(traceback.format_exc())
            raise HTTPException(status_code=415, detail="Failed to read image")

        try:
            ApiDependencies.invoker.services.style_preset_images_service.save(pil_image, new_style_preset.id)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    preset_image = ApiDependencies.invoker.services.style_preset_images_service.get_url(new_style_preset.id)
    return StylePresetRecordWithImage(image=preset_image, **new_style_preset.model_dump())


@style_presets_router.get(
    "/",
    operation_id="list_style_presets",
    responses={
        200: {"model": list[StylePresetRecordWithImage]},
    },
)
async def list_style_presets() -> list[StylePresetRecordWithImage]:
    """Gets a page of style presets"""
    style_presets_with_image: list[StylePresetRecordWithImage] = []
    style_presets = ApiDependencies.invoker.services.style_preset_records.get_many()
    for preset in style_presets:
        image = ApiDependencies.invoker.services.style_preset_images_service.get_url(preset.id)
        style_preset_with_image = StylePresetRecordWithImage(image=image, **preset.model_dump())
        style_presets_with_image.append(style_preset_with_image)

    return style_presets_with_image


@style_presets_router.get(
    "/i/{style_preset_id}/image",
    operation_id="get_style_preset_image",
    responses={
        200: {
            "description": "The style preset image was fetched successfully",
        },
        400: {"description": "Bad request"},
        404: {"description": "The style preset image could not be found"},
    },
    status_code=200,
)
async def get_style_preset_image(
    style_preset_id: str = Path(description="The id of the style preset image to get"),
) -> FileResponse:
    """Gets an image file that previews the model"""

    try:
        path = ApiDependencies.invoker.services.style_preset_images_service.get_path(style_preset_id)

        response = FileResponse(
            path,
            media_type="image/png",
            filename=style_preset_id + ".png",
            content_disposition_type="inline",
        )
        response.headers["Cache-Control"] = f"max-age={IMAGE_MAX_AGE}"
        return response
    except Exception:
        raise HTTPException(status_code=404)
