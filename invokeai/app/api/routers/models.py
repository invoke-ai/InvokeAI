# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654), 2023 Kent Keirsey (https://github.com/hipsterusername), 2023 Lincoln D. Stein


import pathlib
from typing import List, Literal, Optional, Union

from fastapi import Body, Path, Query, Response
from fastapi.routing import APIRouter
from pydantic import BaseModel, parse_obj_as
from starlette.exceptions import HTTPException

from invokeai.backend import BaseModelType, ModelType
from invokeai.backend.model_manager import (
    OPENAPI_MODEL_CONFIGS,
    DuplicateModelException,
    InvalidModelException,
    ModelConfigBase,
    ModelInstallJob,
    SchedulerPredictionType,
    UnknownModelException,
)
from invokeai.backend.model_manager.download import DownloadJobStatus
from invokeai.backend.model_manager.merge import MergeInterpolationMethod

from ..dependencies import ApiDependencies

models_router = APIRouter(prefix="/v1/models", tags=["models"])

# NOTE: The generic configuration classes defined in invokeai.backend.model_manager.config
# such as "MainCheckpointConfig" are repackaged by code original written by Stalker
# into base-specific classes such as `abc.StableDiffusion1ModelCheckpointConfig`
# This is the reason for the calls to dict() followed by pydantic.parse_obj_as()
UpdateModelResponse = Union[tuple(OPENAPI_MODEL_CONFIGS)]
ImportModelResponse = Union[tuple(OPENAPI_MODEL_CONFIGS)]
ConvertModelResponse = Union[tuple(OPENAPI_MODEL_CONFIGS)]
MergeModelResponse = Union[tuple(OPENAPI_MODEL_CONFIGS)]
ImportModelAttributes = Union[tuple(OPENAPI_MODEL_CONFIGS)]


class ModelsList(BaseModel):
    models: list[Union[tuple(OPENAPI_MODEL_CONFIGS)]]


class ModelImportStatus(BaseModel):
    """Return information about a background installation job."""

    job_id: int
    source: str
    priority: int
    status: DownloadJobStatus


@models_router.get(
    "/",
    operation_id="list_models",
    responses={200: {"model": ModelsList}},
)
async def list_models(
    base_models: Optional[List[BaseModelType]] = Query(default=None, description="Base models to include"),
    model_type: Optional[ModelType] = Query(default=None, description="The type of model to get"),
) -> ModelsList:
    """Get a list of models."""
    manager = ApiDependencies.invoker.services.model_manager
    if base_models and len(base_models) > 0:
        models_raw = list()
        for base_model in base_models:
            models_raw.extend([x.dict() for x in manager.list_models(base_model=base_model, model_type=model_type)])
    else:
        models_raw = [x.dict() for x in manager.list_models(model_type=model_type)]
    models = parse_obj_as(ModelsList, {"models": models_raw})
    return models


@models_router.patch(
    "/{key}",
    operation_id="update_model",
    responses={
        200: {"description": "The model was updated successfully"},
        400: {"description": "Bad request"},
        404: {"description": "The model could not be found"},
        409: {"description": "There is already a model corresponding to the new name"},
    },
    status_code=200,
    response_model=UpdateModelResponse,
)
async def update_model(
    key: str = Path(description="Unique key of model"),
    info: Union[tuple(OPENAPI_MODEL_CONFIGS)] = Body(description="Model configuration"),
) -> UpdateModelResponse:
    """Update model contents with a new config. If the model name or base fields are changed, then the model is renamed."""
    logger = ApiDependencies.invoker.services.logger

    try:
        info_dict = info.dict()
        info_dict = {x: info_dict[x] if info_dict[x] else None for x in info_dict.keys()}
        new_config = ApiDependencies.invoker.services.model_manager.update_model(key, new_config=info_dict)
        model_response = parse_obj_as(UpdateModelResponse, new_config.dict())
    except UnknownModelException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(str(e))
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=400, detail=str(e))

    return model_response


@models_router.post(
    "/import",
    operation_id="import_model",
    responses={
        201: {"description": "The model imported successfully"},
        404: {"description": "The model could not be found"},
        415: {"description": "Unrecognized file/folder format"},
        424: {"description": "The model appeared to import successfully, but could not be found in the model manager"},
        409: {"description": "There is already a model corresponding to this path or repo_id"},
    },
    status_code=201,
    response_model=ModelImportStatus,
)
async def import_model(
    location: str = Body(description="A model path, repo_id or URL to import"),
    prediction_type: Optional[Literal["v_prediction", "epsilon", "sample"]] = Body(
        description="Prediction type for SDv2 checkpoint files", default="v_prediction"
    ),
) -> ModelImportStatus:
    """
    Add a model using its local path, repo_id, or remote URL.

    Model characteristics will be probed and configured automatically.
    The return object is a ModelInstallJob job ID. The work will be
    performed in the background. Listen on the event bus for a series of
    `model_event` events with an `id` matching the returned job id to get
    the progress, completion status, errors, and information on the
    model that was installed.
    """

    logger = ApiDependencies.invoker.services.logger
    try:
        result = ApiDependencies.invoker.services.model_manager.install_model(
            location, model_attributes={"prediction_type": SchedulerPredictionType(prediction_type)}
        )
        return ModelImportStatus(job_id=result.id, source=result.source, priority=result.priority, status=result.status)
    except UnknownModelException as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidModelException as e:
        logger.error(str(e))
        raise HTTPException(status_code=415)
    except ValueError as e:
        logger.error(str(e))
        raise HTTPException(status_code=409, detail=str(e))


@models_router.post(
    "/add",
    operation_id="add_model",
    responses={
        201: {"description": "The model added successfully"},
        404: {"description": "The model could not be found"},
        424: {"description": "The model appeared to add successfully, but could not be found in the model manager"},
        409: {"description": "There is already a model corresponding to this path or repo_id"},
    },
    status_code=201,
    response_model=ImportModelResponse,
)
async def add_model(
    info: Union[tuple(OPENAPI_MODEL_CONFIGS)] = Body(description="Model configuration"),
) -> ImportModelResponse:
    """Add a model using the configuration information appropriate for its type. Only local models can be added by path"""

    logger = ApiDependencies.invoker.services.logger

    try:
        ApiDependencies.invoker.services.model_manager.add_model(
            info.model_name, info.base_model, info.model_type, model_attributes=info.dict()
        )
        logger.info(f"Successfully added {info.model_name}")
        model_raw = ApiDependencies.invoker.services.model_manager.list_model(
            model_name=info.model_name, base_model=info.base_model, model_type=info.model_type
        )
        return parse_obj_as(ImportModelResponse, model_raw)
    except UnknownModelException as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(str(e))
        raise HTTPException(status_code=409, detail=str(e))


@models_router.delete(
    "/{key}",
    operation_id="del_model",
    responses={204: {"description": "Model deleted successfully"}, 404: {"description": "Model not found"}},
    status_code=204,
    response_model=None,
)
async def delete_model(
    key: str = Path(description="Unique key of model to remove from model registry."),
    delete_files: Optional[bool] = Query(description="Delete underlying files and directories as well.", default=False),
) -> Response:
    """Delete Model"""
    logger = ApiDependencies.invoker.services.logger

    try:
        ApiDependencies.invoker.services.model_manager.del_model(key, delete_files=delete_files)
        logger.info(f"Deleted model: {key}")
        return Response(status_code=204)
    except UnknownModelException as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))


@models_router.put(
    "/convert/{key}",
    operation_id="convert_model",
    responses={
        200: {"description": "Model converted successfully"},
        400: {"description": "Bad request"},
        404: {"description": "Model not found"},
    },
    status_code=200,
    response_model=ConvertModelResponse,
)
async def convert_model(
    key: str = Path(description="Unique key of model to remove from model registry."),
    convert_dest_directory: Optional[str] = Query(
        default=None, description="Save the converted model to the designated directory"
    ),
) -> ConvertModelResponse:
    """Convert a checkpoint model into a diffusers model, optionally saving to the indicated destination directory, or `models` if none."""
    try:
        dest = pathlib.Path(convert_dest_directory) if convert_dest_directory else None
        ApiDependencies.invoker.services.model_manager.convert_model(key, convert_dest_directory=dest)
        model_raw = ApiDependencies.invoker.services.model_manager.model_info(key).dict()
        response = parse_obj_as(ConvertModelResponse, model_raw)
    except UnknownModelException as e:
        raise HTTPException(status_code=404, detail=f"Model '{key}' not found: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return response


@models_router.get(
    "/search",
    operation_id="search_for_models",
    responses={
        200: {"description": "Directory searched successfully"},
        404: {"description": "Invalid directory path"},
    },
    status_code=200,
    response_model=List[pathlib.Path],
)
async def search_for_models(
    search_path: pathlib.Path = Query(description="Directory path to search for models"),
) -> List[pathlib.Path]:
    """Search for all models in a server-local path."""
    if not search_path.is_dir():
        raise HTTPException(
            status_code=404, detail=f"The search path '{search_path}' does not exist or is not directory"
        )
    return ApiDependencies.invoker.services.model_manager.search_for_models(search_path)


@models_router.get(
    "/ckpt_confs",
    operation_id="list_ckpt_configs",
    responses={
        200: {"description": "paths retrieved successfully"},
    },
    status_code=200,
    response_model=List[pathlib.Path],
)
async def list_ckpt_configs() -> List[pathlib.Path]:
    """Return a list of the legacy checkpoint configuration files stored in `ROOT/configs/stable-diffusion`, relative to ROOT."""
    return ApiDependencies.invoker.services.model_manager.list_checkpoint_configs()


@models_router.post(
    "/sync",
    operation_id="sync_to_config",
    responses={
        201: {"description": "synchronization successful"},
    },
    status_code=201,
    response_model=bool,
)
async def sync_to_config() -> bool:
    """
    Synchronize model in-memory data structures with disk.

    Call after making changes to models.yaml, autoimport directories
    or models directory.
    """
    ApiDependencies.invoker.services.model_manager.sync_to_config()
    return True


@models_router.put(
    "/merge",
    operation_id="merge_models",
    responses={
        200: {"description": "Model converted successfully"},
        400: {"description": "Incompatible models"},
        404: {"description": "One or more models not found"},
        409: {"description": "An identical merged model is already installed"},
    },
    status_code=200,
    response_model=MergeModelResponse,
)
async def merge_models(
    keys: List[str] = Body(description="model name", min_items=2, max_items=3),
    merged_model_name: Optional[str] = Body(description="Name of destination model", default=None),
    alpha: Optional[float] = Body(description="Alpha weighting strength to apply to 2d and 3d models", default=0.5),
    interp: Optional[MergeInterpolationMethod] = Body(description="Interpolation method"),
    force: Optional[bool] = Body(
        description="Force merging of models created with different versions of diffusers", default=False
    ),
    merge_dest_directory: Optional[str] = Body(
        description="Save the merged model to the designated directory (with 'merged_model_name' appended)",
        default=None,
    ),
) -> MergeModelResponse:
    """Merge the indicated diffusers model."""
    logger = ApiDependencies.invoker.services.logger
    try:
        logger.info(f"Merging models: {keys} into {merge_dest_directory or '<MODELS>'}/{merged_model_name}")
        dest = pathlib.Path(merge_dest_directory) if merge_dest_directory else None
        result: ModelConfigBase = ApiDependencies.invoker.services.model_manager.merge_models(
            model_keys=keys,
            merged_model_name=merged_model_name,
            alpha=alpha,
            interp=interp,
            force=force,
            merge_dest_directory=dest,
        )
        response = parse_obj_as(ConvertModelResponse, result.dict())
    except DuplicateModelException as e:
        raise HTTPException(status_code=409, detail=str(e))
    except UnknownModelException:
        raise HTTPException(status_code=404, detail=f"One or more of the models '{keys}' not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return response
