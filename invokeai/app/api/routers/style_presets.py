import csv
import io
import json
import traceback
from typing import Optional

import pydantic
from fastapi import APIRouter, File, Form, HTTPException, Path, Response, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers.model_manager import IMAGE_MAX_AGE
from invokeai.app.services.style_preset_images.style_preset_images_common import StylePresetImageFileNotFoundException
from invokeai.app.services.style_preset_records.style_preset_records_common import (
    InvalidPresetImportDataError,
    PresetData,
    PresetType,
    StylePresetChanges,
    StylePresetNotFoundError,
    StylePresetRecordWithImage,
    StylePresetWithoutId,
    UnsupportedFileTypeError,
    parse_presets_from_file,
)


class StylePresetFormData(BaseModel):
    name: str = Field(description="Preset name")
    positive_prompt: str = Field(description="Positive prompt")
    negative_prompt: str = Field(description="Negative prompt")
    type: PresetType = Field(description="Preset type")


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
        image = ApiDependencies.invoker.services.style_preset_image_files.get_url(style_preset_id)
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
    image: Optional[UploadFile] = File(description="The image file to upload", default=None),
    style_preset_id: str = Path(description="The id of the style preset to update"),
    data: str = Form(description="The data of the style preset to update"),
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
            ApiDependencies.invoker.services.style_preset_image_files.save(style_preset_id, pil_image)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
    else:
        try:
            ApiDependencies.invoker.services.style_preset_image_files.delete(style_preset_id)
        except StylePresetImageFileNotFoundException:
            pass

    try:
        parsed_data = json.loads(data)
        validated_data = StylePresetFormData(**parsed_data)

        name = validated_data.name
        type = validated_data.type
        positive_prompt = validated_data.positive_prompt
        negative_prompt = validated_data.negative_prompt

    except pydantic.ValidationError:
        raise HTTPException(status_code=400, detail="Invalid preset data")

    preset_data = PresetData(positive_prompt=positive_prompt, negative_prompt=negative_prompt)
    changes = StylePresetChanges(name=name, preset_data=preset_data, type=type)

    style_preset_image = ApiDependencies.invoker.services.style_preset_image_files.get_url(style_preset_id)
    style_preset = ApiDependencies.invoker.services.style_preset_records.update(
        style_preset_id=style_preset_id, changes=changes
    )
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
        ApiDependencies.invoker.services.style_preset_image_files.delete(style_preset_id)
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
    image: Optional[UploadFile] = File(description="The image file to upload", default=None),
    data: str = Form(description="The data of the style preset to create"),
) -> StylePresetRecordWithImage:
    """Creates a style preset"""

    try:
        parsed_data = json.loads(data)
        validated_data = StylePresetFormData(**parsed_data)

        name = validated_data.name
        type = validated_data.type
        positive_prompt = validated_data.positive_prompt
        negative_prompt = validated_data.negative_prompt

    except pydantic.ValidationError:
        raise HTTPException(status_code=400, detail="Invalid preset data")

    preset_data = PresetData(positive_prompt=positive_prompt, negative_prompt=negative_prompt)
    style_preset = StylePresetWithoutId(name=name, preset_data=preset_data, type=type)
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
            ApiDependencies.invoker.services.style_preset_image_files.save(new_style_preset.id, pil_image)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    preset_image = ApiDependencies.invoker.services.style_preset_image_files.get_url(new_style_preset.id)
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
        image = ApiDependencies.invoker.services.style_preset_image_files.get_url(preset.id)
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
        path = ApiDependencies.invoker.services.style_preset_image_files.get_path(style_preset_id)

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


@style_presets_router.get(
    "/export",
    operation_id="export_style_presets",
    responses={200: {"content": {"text/csv": {}}, "description": "A CSV file with the requested data."}},
    status_code=200,
)
async def export_style_presets():
    # Create an in-memory stream to store the CSV data
    output = io.StringIO()
    writer = csv.writer(output)

    # Write the header
    writer.writerow(["name", "prompt", "negative_prompt"])

    style_presets = ApiDependencies.invoker.services.style_preset_records.get_many(type=PresetType.User)

    for preset in style_presets:
        writer.writerow([preset.name, preset.preset_data.positive_prompt, preset.preset_data.negative_prompt])

    csv_data = output.getvalue()
    output.close()

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=prompt_templates.csv"},
    )


@style_presets_router.post(
    "/import",
    operation_id="import_style_presets",
)
async def import_style_presets(file: UploadFile = File(description="The file to import")):
    try:
        style_presets = await parse_presets_from_file(file)
        ApiDependencies.invoker.services.style_preset_records.create_many(style_presets)
    except InvalidPresetImportDataError as e:
        ApiDependencies.invoker.services.logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))
    except UnsupportedFileTypeError as e:
        ApiDependencies.invoker.services.logger.error(traceback.format_exc())
        raise HTTPException(status_code=415, detail=str(e))
