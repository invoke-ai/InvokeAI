# Copyright (c) 2023 Lincoln D. Stein
"""FastAPI route for model configuration records."""


from hashlib import sha1
from random import randbytes
from typing import Any, Dict, List, Optional

from fastapi import Body, Path, Query, Response
from fastapi.routing import APIRouter
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException
from typing_extensions import Annotated

from invokeai.app.services.model_install import ModelInstallJob, ModelSource
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

from ..dependencies import ApiDependencies

model_records_router = APIRouter(prefix="/v1/model/record", tags=["model_manager_v2_unstable"])


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
    model_name: Optional[str] = Query(default=None, description="Exact match on the name of the model"),
    model_format: Optional[str] = Query(
        default=None, description="Exact match on the format of the model (e.g. 'diffusers')"
    ),
) -> ModelsList:
    """Get a list of models."""
    record_store = ApiDependencies.invoker.services.model_records
    found_models: list[AnyModelConfig] = []
    if base_models:
        for base_model in base_models:
            found_models.extend(
                record_store.search_by_attr(
                    base_model=base_model, model_type=model_type, model_name=model_name, model_format=model_format
                )
            )
    else:
        found_models.extend(
            record_store.search_by_attr(model_type=model_type, model_name=model_name, model_format=model_format)
        )
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
    """
    Delete model record from database.

    The configuration record will be removed. The corresponding weights files will be
    deleted as well if they reside within the InvokeAI "models" directory.
    """
    logger = ApiDependencies.invoker.services.logger

    try:
        installer = ApiDependencies.invoker.services.model_install
        installer.delete(key)
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
    config: Annotated[AnyModelConfig, Body(description="Model config", discriminator="type")],
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
        415: {"description": "Unrecognized file/folder format"},
        424: {"description": "The model appeared to import successfully, but could not be found in the model manager"},
        409: {"description": "There is already a model corresponding to this path or repo_id"},
    },
    status_code=201,
)
async def import_model(
    source: ModelSource,
    config: Optional[Dict[str, Any]] = Body(
        description="Dict of fields that override auto-probed values in the model config record, such as name, description and prediction_type ",
        default=None,
    ),
) -> ModelInstallJob:
    """Add a model using its local path, repo_id, or remote URL.

    Models will be downloaded, probed, configured and installed in a
    series of background threads. The return object has `status` attribute
    that can be used to monitor progress.

    The source object is a discriminated Union of LocalModelSource,
    HFModelSource and URLModelSource. Set the "type" field to the
    appropriate value:

    * To install a local path using LocalModelSource, pass a source of form:
      `{
        "type": "local",
        "path": "/path/to/model",
        "inplace": false
      }`
       The "inplace" flag, if true, will register the model in place in its
       current filesystem location. Otherwise, the model will be copied
       into the InvokeAI models directory.

    * To install a HuggingFace repo_id using HFModelSource, pass a source of form:
      `{
        "type": "hf",
        "repo_id": "stabilityai/stable-diffusion-2.0",
        "variant": "fp16",
        "subfolder": "vae",
        "access_token": "f5820a918aaf01"
      }`
     The `variant`, `subfolder` and `access_token` fields are optional.

    * To install a remote model using an arbitrary URL, pass:
      `{
        "type": "url",
        "url": "http://www.civitai.com/models/123456",
        "access_token": "f5820a918aaf01"
      }`
    The `access_token` field is optonal

    The model's configuration record will be probed and filled in
    automatically.  To override the default guesses, pass "metadata"
    with a Dict containing the attributes you wish to override.

    Installation occurs in the background. Either use list_model_install_jobs()
    to poll for completion, or listen on the event bus for the following events:

      "model_install_started"
      "model_install_completed"
      "model_install_error"

    On successful completion, the event's payload will contain the field "key"
    containing the installed ID of the model. On an error, the event's payload
    will contain the fields "error_type" and "error" describing the nature of the
    error and its traceback, respectively.

    """
    logger = ApiDependencies.invoker.services.logger

    try:
        installer = ApiDependencies.invoker.services.model_install
        result: ModelInstallJob = installer.import_model(
            source=source,
            config=config,
        )
        logger.info(f"Started installation of {source}")
    except UnknownModelException as e:
        logger.error(str(e))
        raise HTTPException(status_code=424, detail=str(e))
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
async def list_model_install_jobs() -> List[ModelInstallJob]:
    """
    Return list of model install jobs.

    If the optional 'source' argument is provided, then the list will be filtered
    for partial string matches against the install source.
    """
    jobs: List[ModelInstallJob] = ApiDependencies.invoker.services.model_install.list_jobs()
    return jobs


@model_records_router.patch(
    "/import",
    operation_id="prune_model_install_jobs",
    responses={
        204: {"description": "All completed and errored jobs have been pruned"},
        400: {"description": "Bad request"},
    },
)
async def prune_model_install_jobs() -> Response:
    """
    Prune all completed and errored jobs from the install job list.
    """
    ApiDependencies.invoker.services.model_install.prune_jobs()
    return Response(status_code=204)


@model_records_router.patch(
    "/sync",
    operation_id="sync_models_to_config",
    responses={
        204: {"description": "Model config record database resynced with files on disk"},
        400: {"description": "Bad request"},
    },
)
async def sync_models_to_config() -> Response:
    """
    Traverse the models and autoimport directories. Model files without a corresponding
    record in the database are added. Orphan records without a models file are deleted.
    """
    ApiDependencies.invoker.services.model_install.sync_to_config()
    return Response(status_code=204)
