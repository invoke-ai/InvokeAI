from fastapi import APIRouter, Body, HTTPException, Path

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.style_preset_records.style_preset_records_common import (
    StylePresetChanges,
    StylePresetNotFoundError,
    StylePresetRecordDTO,
    StylePresetWithoutId,
)

style_presets_router = APIRouter(prefix="/v1/style_presets", tags=["style_presets"])


@style_presets_router.get(
    "/i/{style_preset_id}",
    operation_id="get_style_preset",
    responses={
        200: {"model": StylePresetRecordDTO},
    },
)
async def get_style_preset(
    style_preset_id: str = Path(description="The style preset to get"),
) -> StylePresetRecordDTO:
    """Gets a style preset"""
    try:
        return ApiDependencies.invoker.services.style_preset_records.get(style_preset_id)
    except StylePresetNotFoundError:
        raise HTTPException(status_code=404, detail="Style preset not found")


@style_presets_router.patch(
    "/i/{style_preset_id}",
    operation_id="update_style_preset",
    responses={
        200: {"model": StylePresetRecordDTO},
    },
)
async def update_style_preset(
    style_preset_id: str = Path(description="The id of the style preset to update"),
    changes: StylePresetChanges = Body(description="The updated style preset", embed=True),
) -> StylePresetRecordDTO:
    """Updates a style preset"""
    return ApiDependencies.invoker.services.style_preset_records.update(id=style_preset_id, changes=changes)


@style_presets_router.delete(
    "/i/{style_preset_id}",
    operation_id="delete_style_preset",
)
async def delete_style_preset(
    style_preset_id: str = Path(description="The style preset to delete"),
) -> None:
    """Deletes a style preset"""
    ApiDependencies.invoker.services.style_preset_records.delete(style_preset_id)


@style_presets_router.post(
    "/",
    operation_id="create_style_preset",
    responses={
        200: {"model": StylePresetRecordDTO},
    },
)
async def create_style_preset(
    style_preset: StylePresetWithoutId = Body(description="The style preset to create", embed=True),
) -> StylePresetRecordDTO:
    """Creates a style preset"""
    return ApiDependencies.invoker.services.style_preset_records.create(style_preset=style_preset)


@style_presets_router.get(
    "/",
    operation_id="list_style_presets",
    responses={
        200: {"model": list[StylePresetRecordDTO]},
    },
)
async def list_style_presets() -> list[StylePresetRecordDTO]:
    """Gets a page of style presets"""
    return ApiDependencies.invoker.services.style_preset_records.get_many()
