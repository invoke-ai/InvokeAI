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

from invokeai.app.api.auth_dependencies import AdminUserOrDefault, CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers.model_manager import IMAGE_MAX_AGE
from invokeai.app.services.auth.token_service import TokenData
from invokeai.app.services.style_preset_images.style_preset_images_common import StylePresetImageFileNotFoundException
from invokeai.app.services.style_preset_records.style_preset_records_common import (
    InvalidPresetImportDataError,
    PresetData,
    PresetType,
    StylePresetChanges,
    StylePresetNotFoundError,
    StylePresetRecordDTO,
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
    is_public: bool = Field(default=False, description="Whether the preset is visible to other users")


style_presets_router = APIRouter(prefix="/v1/style_presets", tags=["style_presets"])


def _assert_preset_read(record: StylePresetRecordDTO, current_user: TokenData) -> None:
    """Allow read access if admin, owner, default preset, or public preset."""
    if current_user.is_admin:
        return
    if record.type == PresetType.Default:
        return
    if record.is_public:
        return
    if record.user_id == current_user.user_id:
        return
    raise HTTPException(status_code=403, detail="Not authorized to access this style preset")


def _assert_preset_write(record: StylePresetRecordDTO, current_user: TokenData) -> None:
    """Allow write access only for admin or owner. Defaults are immutable for non-admins."""
    if current_user.is_admin:
        return
    if record.type == PresetType.Default:
        raise HTTPException(status_code=403, detail="Default style presets cannot be modified")
    if record.user_id == current_user.user_id:
        return
    raise HTTPException(status_code=403, detail="Not authorized to modify this style preset")


def _load_record_or_404(style_preset_id: str) -> StylePresetRecordDTO:
    try:
        return ApiDependencies.invoker.services.style_preset_records.get(style_preset_id)
    except StylePresetNotFoundError:
        raise HTTPException(status_code=404, detail="Style preset not found")


@style_presets_router.get(
    "/i/{style_preset_id}",
    operation_id="get_style_preset",
    responses={
        200: {"model": StylePresetRecordWithImage},
    },
)
async def get_style_preset(
    current_user: CurrentUserOrDefault,
    style_preset_id: str = Path(description="The style preset to get"),
) -> StylePresetRecordWithImage:
    """Gets a style preset"""
    record = _load_record_or_404(style_preset_id)
    _assert_preset_read(record, current_user)
    image = ApiDependencies.invoker.services.style_preset_image_files.get_url(style_preset_id)
    return StylePresetRecordWithImage(image=image, **record.model_dump())


@style_presets_router.patch(
    "/i/{style_preset_id}",
    operation_id="update_style_preset",
    responses={
        200: {"model": StylePresetRecordWithImage},
    },
)
async def update_style_preset(
    current_user: CurrentUserOrDefault,
    image: Optional[UploadFile] = File(description="The image file to upload", default=None),
    style_preset_id: str = Path(description="The id of the style preset to update"),
    data: str = Form(description="The data of the style preset to update"),
) -> StylePresetRecordWithImage:
    """Updates a style preset"""
    # Validate the data payload BEFORE any image-state mutation so a malformed
    # request can't leave the preset image partially updated.
    try:
        parsed_data = json.loads(data)
        validated_data = StylePresetFormData(**parsed_data)

        name = validated_data.name
        type = validated_data.type
        positive_prompt = validated_data.positive_prompt
        negative_prompt = validated_data.negative_prompt
        is_public = validated_data.is_public

    except (json.JSONDecodeError, pydantic.ValidationError):
        raise HTTPException(status_code=400, detail="Invalid preset data")

    record = _load_record_or_404(style_preset_id)
    _assert_preset_write(record, current_user)

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

    preset_data = PresetData(positive_prompt=positive_prompt, negative_prompt=negative_prompt)
    changes = StylePresetChanges(name=name, preset_data=preset_data, type=type, is_public=is_public)

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
    current_user: CurrentUserOrDefault,
    style_preset_id: str = Path(description="The style preset to delete"),
) -> None:
    """Deletes a style preset"""
    record = _load_record_or_404(style_preset_id)
    _assert_preset_write(record, current_user)

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
    current_user: CurrentUserOrDefault,
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
        is_public = validated_data.is_public

    except (json.JSONDecodeError, pydantic.ValidationError):
        raise HTTPException(status_code=400, detail="Invalid preset data")

    # Only admins may create default-typed presets — they're the shipped catalog.
    if type == PresetType.Default and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Only admins can create default presets")

    preset_data = PresetData(positive_prompt=positive_prompt, negative_prompt=negative_prompt)
    style_preset = StylePresetWithoutId(name=name, preset_data=preset_data, type=type, is_public=is_public)
    new_style_preset = ApiDependencies.invoker.services.style_preset_records.create(
        style_preset=style_preset, user_id=current_user.user_id
    )

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
async def list_style_presets(current_user: CurrentUserOrDefault) -> list[StylePresetRecordWithImage]:
    """Gets the style presets visible to the current user."""
    style_presets_with_image: list[StylePresetRecordWithImage] = []
    style_presets = ApiDependencies.invoker.services.style_preset_records.get_many(
        user_id=current_user.user_id,
        is_admin=current_user.is_admin,
    )
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
    current_user: CurrentUserOrDefault,
    style_preset_id: str = Path(description="The id of the style preset image to get"),
) -> FileResponse:
    """Gets an image file that previews the model"""
    record = _load_record_or_404(style_preset_id)
    _assert_preset_read(record, current_user)

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
async def export_style_presets(current_user: AdminUserOrDefault):
    # Admin-only export covers every user preset.
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["name", "prompt", "negative_prompt"])

    style_presets = ApiDependencies.invoker.services.style_preset_records.get_many(
        type=PresetType.User,
        user_id=current_user.user_id,
        is_admin=True,
    )

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
async def import_style_presets(
    current_user: AdminUserOrDefault,
    file: UploadFile = File(description="The file to import"),
):
    try:
        style_presets = await parse_presets_from_file(file)
        ApiDependencies.invoker.services.style_preset_records.create_many(style_presets, user_id=current_user.user_id)
    except InvalidPresetImportDataError as e:
        ApiDependencies.invoker.services.logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))
    except UnsupportedFileTypeError as e:
        ApiDependencies.invoker.services.logger.error(traceback.format_exc())
        raise HTTPException(status_code=415, detail=str(e))
