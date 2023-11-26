# Copyright (c) 2023 Lincoln D. Stein
"""FastAPI route for model configuration records."""


from hashlib import sha1
from random import randbytes
from typing import List, Optional, Any, Dict

from fastapi import Body, Path, Query, Response
from fastapi.routing import APIRouter
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException
from typing_extensions import Annotated

from invokeai.app.services.model_records import (
    DuplicateModelException,
    InvalidModelException,
    UnknownModelException,
)
from invokeai.backend.model_manager.config import (
    AnyModelConfig,
    BaseModelType,
    ModelType,
)
from invokeai.app.services.model_install import ModelInstallJob, ModelSource

from ..dependencies import ApiDependencies

model_records_router = APIRouter(prefix="/v1/model/record", tags=["models"])


class ModelsList(BaseModel):
    """Return list of configs."""

    models: list[AnyModelConfig]

    model_config = ConfigDict(use_enum_values=True)


@model_records_router.get(
    "/",
    operation_id="list_model_records",
)
async def list_model_records(
    base_models: Optional[List[BaseModelType]] = Query(default=None, description="Base models to include"),
    model_type: Optional[ModelType] = Query(default=None, description="The type of model to get"),
) -> ModelsList:
    """Get a list of models."""
    record_store = ApiDependencies.invoker.services.model_records
    found_models: list[AnyModelConfig] = []
    if base_models:
        for base_model in base_models:
            found_models.extend(record_store.search_by_attr(base_model=base_model, model_type=model_type))
    else:
        found_models.extend(record_store.search_by_attr(model_type=model_type))
    return ModelsList(models=found_models)


@model_records_router.get(
    "/i/{key}",
    operation_id="get_model_record",
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad request"},
        404: {"description": "The model could not be found"},
    },
)
async def get_model_record(
    key: str = Path(description="Key of the model record to fetch."),
) -> AnyModelConfig:
    """Get a model record"""
    record_store = ApiDependencies.invoker.services.model_records
    try:
        return record_store.get_model(key)
    except UnknownModelException as e:
        raise HTTPException(status_code=404, detail=str(e))


@model_records_router.patch(
    "/i/{key}",
    operation_id="update_model_record",
    responses={
        200: {"description": "The model was updated successfully"},
        400: {"description": "Bad request"},
        404: {"description": "The model could not be found"},
        409: {"description": "There is already a model corresponding to the new name"},
    },
    status_code=200,
    response_model=AnyModelConfig,
)
async def update_model_record(
    key: Annotated[str, Path(description="Unique key of model")],
    info: Annotated[AnyModelConfig, Body(description="Model config", discriminator="type")],
) -> AnyModelConfig:
    """Update model contents with a new config. If the model name or base fields are changed, then the model is renamed."""
    logger = ApiDependencies.invoker.services.logger
    record_store = ApiDependencies.invoker.services.model_records
    try:
        model_response = record_store.update_model(key, config=info)
        logger.info(f"Updated model: {key}")
    except UnknownModelException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(str(e))
        raise HTTPException(status_code=409, detail=str(e))
    return model_response


@model_records_router.delete(
    "/i/{key}",
    operation_id="del_model_record",
    responses={
        204: {"description": "Model deleted successfully"},
        404: {"description": "Model not found"},
    },
    status_code=204,
)
async def del_model_record(
    key: str = Path(description="Unique key of model to remove from model registry."),
) -> Response:
    """Delete Model"""
    logger = ApiDependencies.invoker.services.logger

    try:
        record_store = ApiDependencies.invoker.services.model_records
        record_store.del_model(key)
        logger.info(f"Deleted model: {key}")
        return Response(status_code=204)
    except UnknownModelException as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))


@model_records_router.post(
    "/i/",
    operation_id="add_model_record",
    responses={
        201: {"description": "The model added successfully"},
        409: {"description": "There is already a model corresponding to this path or repo_id"},
        415: {"description": "Unrecognized file/folder format"},
    },
    status_code=201,
)
async def add_model_record(
    config: Annotated[AnyModelConfig, Body(description="Model config", discriminator="type")]
) -> AnyModelConfig:
    """
    Add a model using the configuration information appropriate for its type.
    """
    logger = ApiDependencies.invoker.services.logger
    record_store = ApiDependencies.invoker.services.model_records
    if config.key == "<NOKEY>":
        config.key = sha1(randbytes(100)).hexdigest()
        logger.info(f"Created model {config.key} for {config.name}")
    try:
        record_store.add_model(config.key, config)
    except DuplicateModelException as e:
        logger.error(str(e))
        raise HTTPException(status_code=409, detail=str(e))
    except InvalidModelException as e:
        logger.error(str(e))
        raise HTTPException(status_code=415)

    # now fetch it out
    return record_store.get_model(config.key)


@model_records_router.post(
    "/import",
    operation_id="import_model_record",
    responses={
        201: {"description": "The model imported successfully"},
        404: {"description": "The model could not be found"},
        415: {"description": "Unrecognized file/folder format"},
        424: {"description": "The model appeared to import successfully, but could not be found in the model manager"},
        409: {"description": "There is already a model corresponding to this path or repo_id"},
    },
    status_code=201,
)
async def import_model(
        source: ModelSource = Body(
            description="A model path, repo_id or URL to import. NOTE: only model path is implemented currently!"
        ),
        metadata: Optional[Dict[str, Any]] = Body(
            description="Dict of fields that override auto-probed values, such as name, description and prediction_type ",
            default=None,
        ),
        variant: Optional[str] = Body(
            description="When fetching a repo_id, force variant type to fetch such as 'fp16'",
            default=None,
        ),
        subfolder: Optional[str] = Body(
            description="When fetching a repo_id, specify subfolder to fetch model from",
            default=None,
        ),
        access_token: Optional[str] = Body(
            description="When fetching a repo_id or URL, access token for web access",
            default=None,
        ),
) -> ModelInstallJob:
    """Add a model using its local path, repo_id, or remote URL.

    Models will be downloaded, probed, configured and installed in a
    series of background threads. The return object has `status` attribute
    that can be used to monitor progress.

    The model's configuration record will be probed and filled in
    automatically.  To override the default guesses, pass "metadata"
    with a Dict containing the attributes you wish to override.

    Listen on the event bus for the following events: 
    "model_install_started", "model_install_completed", and "model_install_error."
    On successful completion, the event's payload will contain the field "key" 
    containing the installed ID of the model. On an error, the event's payload
    will contain the fields "error_type" and "error" describing the nature of the
    error and its traceback, respectively.

    """
    logger = ApiDependencies.invoker.services.logger

    try:
        installer = ApiDependencies.invoker.services.model_install
        result: ModelInstallJob = installer.import_model(
            source,
            metadata=metadata,
            variant=variant,
            subfolder=subfolder,
            access_token=access_token,
        )
    except UnknownModelException as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidModelException as e:
        logger.error(str(e))
        raise HTTPException(status_code=415)
    except ValueError as e:
        logger.error(str(e))
        raise HTTPException(status_code=409, detail=str(e))
    return result

@model_records_router.get(
    "/import",
    operation_id="list_model_install_jobs",
)
async def list_install_jobs(
        source: Optional[str] = Query(description="Filter list by install source, partial string match.",
                                      default=None,
                                      )
) -> List[ModelInstallJob]:
    """
    Return list of model install jobs.

    If the optional 'source' argument is provided, then the list will be filtered
    for partial string matches against the install source.
    """
    jobs: List[ModelInstallJob] = ApiDependencies.invoker.services.model_install.list_jobs(source)
    return jobs
