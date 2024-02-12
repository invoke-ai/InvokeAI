# Copyright (c) 2023 Lincoln D. Stein
"""FastAPI route for model configuration records."""

import pathlib
from hashlib import sha1
from random import randbytes
from typing import Any, Dict, List, Optional, Set

from fastapi import Body, Path, Query, Response
from fastapi.routing import APIRouter
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException
from typing_extensions import Annotated

from invokeai.app.services.model_install import ModelInstallJob, ModelSource
from invokeai.app.services.model_records import (
    DuplicateModelException,
    InvalidModelException,
    ModelRecordOrderBy,
    ModelSummary,
    UnknownModelException,
)
from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.backend.model_manager.config import (
    AnyModelConfig,
    BaseModelType,
    ModelFormat,
    ModelType,
)
from invokeai.backend.model_manager.merge import MergeInterpolationMethod, ModelMerger
from invokeai.backend.model_manager.metadata import AnyModelRepoMetadata

from ..dependencies import ApiDependencies

model_manager_v2_router = APIRouter(prefix="/v2/models", tags=["model_manager_v2"])


class ModelsList(BaseModel):
    """Return list of configs."""

    models: List[AnyModelConfig]

    model_config = ConfigDict(use_enum_values=True)


class ModelTagSet(BaseModel):
    """Return tags for a set of models."""

    key: str
    name: str
    author: str
    tags: Set[str]


@model_manager_v2_router.get(
    "/",
    operation_id="list_model_records",
)
async def list_model_records(
    base_models: Optional[List[BaseModelType]] = Query(default=None, description="Base models to include"),
    model_type: Optional[ModelType] = Query(default=None, description="The type of model to get"),
    model_name: Optional[str] = Query(default=None, description="Exact match on the name of the model"),
    model_format: Optional[ModelFormat] = Query(
        default=None, description="Exact match on the format of the model (e.g. 'diffusers')"
    ),
) -> ModelsList:
    """Get a list of models."""
    record_store = ApiDependencies.invoker.services.model_manager.store
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


@model_manager_v2_router.get(
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
    record_store = ApiDependencies.invoker.services.model_manager.store
    try:
        config: AnyModelConfig = record_store.get_model(key)
        return config
    except UnknownModelException as e:
        raise HTTPException(status_code=404, detail=str(e))


@model_manager_v2_router.get("/meta", operation_id="list_model_summary")
async def list_model_summary(
    page: int = Query(default=0, description="The page to get"),
    per_page: int = Query(default=10, description="The number of models per page"),
    order_by: ModelRecordOrderBy = Query(default=ModelRecordOrderBy.Default, description="The attribute to order by"),
) -> PaginatedResults[ModelSummary]:
    """Gets a page of model summary data."""
    record_store = ApiDependencies.invoker.services.model_manager.store
    results: PaginatedResults[ModelSummary] = record_store.list_models(page=page, per_page=per_page, order_by=order_by)
    return results


@model_manager_v2_router.get(
    "/meta/i/{key}",
    operation_id="get_model_metadata",
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad request"},
        404: {"description": "No metadata available"},
    },
)
async def get_model_metadata(
    key: str = Path(description="Key of the model repo metadata to fetch."),
) -> Optional[AnyModelRepoMetadata]:
    """Get a model metadata object."""
    record_store = ApiDependencies.invoker.services.model_manager.store
    result: Optional[AnyModelRepoMetadata] = record_store.get_metadata(key)
    if not result:
        raise HTTPException(status_code=404, detail="No metadata for a model with this key")
    return result


@model_manager_v2_router.get(
    "/tags",
    operation_id="list_tags",
)
async def list_tags() -> Set[str]:
    """Get a unique set of all the model tags."""
    record_store = ApiDependencies.invoker.services.model_manager.store
    result: Set[str] = record_store.list_tags()
    return result


@model_manager_v2_router.get(
    "/tags/search",
    operation_id="search_by_metadata_tags",
)
async def search_by_metadata_tags(
    tags: Set[str] = Query(default=None, description="Tags to search for"),
) -> ModelsList:
    """Get a list of models."""
    record_store = ApiDependencies.invoker.services.model_manager.store
    results = record_store.search_by_metadata_tag(tags)
    return ModelsList(models=results)


@model_manager_v2_router.patch(
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
    record_store = ApiDependencies.invoker.services.model_manager.store
    try:
        model_response: AnyModelConfig = record_store.update_model(key, config=info)
        logger.info(f"Updated model: {key}")
    except UnknownModelException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(str(e))
        raise HTTPException(status_code=409, detail=str(e))
    return model_response


@model_manager_v2_router.delete(
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
        installer = ApiDependencies.invoker.services.model_manager.install
        installer.delete(key)
        logger.info(f"Deleted model: {key}")
        return Response(status_code=204)
    except UnknownModelException as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))


@model_manager_v2_router.post(
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
    """Add a model using the configuration information appropriate for its type."""
    logger = ApiDependencies.invoker.services.logger
    record_store = ApiDependencies.invoker.services.model_manager.store
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
    result: AnyModelConfig = record_store.get_model(config.key)
    return result


@model_manager_v2_router.post(
    "/heuristic_import",
    operation_id="heuristic_import_model",
    responses={
        201: {"description": "The model imported successfully"},
        415: {"description": "Unrecognized file/folder format"},
        424: {"description": "The model appeared to import successfully, but could not be found in the model manager"},
        409: {"description": "There is already a model corresponding to this path or repo_id"},
    },
    status_code=201,
)
async def heuristic_import(
    source: str,
    config: Optional[Dict[str, Any]] = Body(
        description="Dict of fields that override auto-probed values in the model config record, such as name, description and prediction_type ",
        default=None,
    ),
    access_token: Optional[str] = None,
) -> ModelInstallJob:
    """Install a model using a string identifier.

    `source` can be any of the following.

    1. A path on the local filesystem ('C:\\users\\fred\\model.safetensors')
    2. A Url pointing to a single downloadable model file
    3. A HuggingFace repo_id with any of the following formats:
       - model/name
       - model/name:fp16:vae
       - model/name::vae          -- use default precision
       - model/name:fp16:path/to/model.safetensors
       - model/name::path/to/model.safetensors

    `config` is an optional dict containing model configuration values that will override
    the ones that are probed automatically.

    `access_token` is an optional access token for use with Urls that require
    authentication.

    Models will be downloaded, probed, configured and installed in a
    series of background threads. The return object has `status` attribute
    that can be used to monitor progress.

    See the documentation for `import_model_record` for more information on
    interpreting the job information returned by this route.
    """
    logger = ApiDependencies.invoker.services.logger

    try:
        installer = ApiDependencies.invoker.services.model_manager.install
        result: ModelInstallJob = installer.heuristic_import(
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


@model_manager_v2_router.post(
    "/import",
    operation_id="import_model",
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
    """Install a model using its local path, repo_id, or remote URL.

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

      "model_install_running"
      "model_install_completed"
      "model_install_error"

    On successful completion, the event's payload will contain the field "key"
    containing the installed ID of the model. On an error, the event's payload
    will contain the fields "error_type" and "error" describing the nature of the
    error and its traceback, respectively.

    """
    logger = ApiDependencies.invoker.services.logger

    try:
        installer = ApiDependencies.invoker.services.model_manager.install
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


@model_manager_v2_router.get(
    "/import",
    operation_id="list_model_install_jobs",
)
async def list_model_install_jobs() -> List[ModelInstallJob]:
    """Return list of model install jobs."""
    jobs: List[ModelInstallJob] = ApiDependencies.invoker.services.model_manager.install.list_jobs()
    return jobs


@model_manager_v2_router.get(
    "/import/{id}",
    operation_id="get_model_install_job",
    responses={
        200: {"description": "Success"},
        404: {"description": "No such job"},
    },
)
async def get_model_install_job(id: int = Path(description="Model install id")) -> ModelInstallJob:
    """Return model install job corresponding to the given source."""
    try:
        result: ModelInstallJob = ApiDependencies.invoker.services.model_manager.install.get_job_by_id(id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@model_manager_v2_router.delete(
    "/import/{id}",
    operation_id="cancel_model_install_job",
    responses={
        201: {"description": "The job was cancelled successfully"},
        415: {"description": "No such job"},
    },
    status_code=201,
)
async def cancel_model_install_job(id: int = Path(description="Model install job ID")) -> None:
    """Cancel the model install job(s) corresponding to the given job ID."""
    installer = ApiDependencies.invoker.services.model_manager.install
    try:
        job = installer.get_job_by_id(id)
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))
    installer.cancel_job(job)


@model_manager_v2_router.patch(
    "/import",
    operation_id="prune_model_install_jobs",
    responses={
        204: {"description": "All completed and errored jobs have been pruned"},
        400: {"description": "Bad request"},
    },
)
async def prune_model_install_jobs() -> Response:
    """Prune all completed and errored jobs from the install job list."""
    ApiDependencies.invoker.services.model_manager.install.prune_jobs()
    return Response(status_code=204)


@model_manager_v2_router.patch(
    "/sync",
    operation_id="sync_models_to_config",
    responses={
        204: {"description": "Model config record database resynced with files on disk"},
        400: {"description": "Bad request"},
    },
)
async def sync_models_to_config() -> Response:
    """
    Traverse the models and autoimport directories.

    Model files without a corresponding
    record in the database are added. Orphan records without a models file are deleted.
    """
    ApiDependencies.invoker.services.model_manager.install.sync_to_config()
    return Response(status_code=204)


@model_manager_v2_router.put(
    "/merge",
    operation_id="merge",
)
async def merge(
    keys: List[str] = Body(description="Keys for two to three models to merge", min_length=2, max_length=3),
    merged_model_name: Optional[str] = Body(description="Name of destination model", default=None),
    alpha: float = Body(description="Alpha weighting strength to apply to 2d and 3d models", default=0.5),
    force: bool = Body(
        description="Force merging of models created with different versions of diffusers",
        default=False,
    ),
    interp: Optional[MergeInterpolationMethod] = Body(description="Interpolation method", default=None),
    merge_dest_directory: Optional[str] = Body(
        description="Save the merged model to the designated directory (with 'merged_model_name' appended)",
        default=None,
    ),
) -> AnyModelConfig:
    """
    Merge diffusers models.

        keys: List of 2-3 model keys to merge together. All models must use the same base type.
        merged_model_name: Name for the merged model [Concat model names]
        alpha: Alpha value (0.0-1.0). Higher values give more weight to the second model [0.5]
        force: If true, force the merge even if the models were generated by different versions of the diffusers library [False]
        interp: Interpolation method. One of "weighted_sum", "sigmoid", "inv_sigmoid" or "add_difference" [weighted_sum]
        merge_dest_directory: Specify a directory to store the merged model in [models directory]
    """
    print(f"here i am, keys={keys}")
    logger = ApiDependencies.invoker.services.logger
    try:
        logger.info(f"Merging models: {keys} into {merge_dest_directory or '<MODELS>'}/{merged_model_name}")
        dest = pathlib.Path(merge_dest_directory) if merge_dest_directory else None
        installer = ApiDependencies.invoker.services.model_manager.install
        merger = ModelMerger(installer)
        model_names = [installer.record_store.get_model(x).name for x in keys]
        response = merger.merge_diffusion_models_and_save(
            model_keys=keys,
            merged_model_name=merged_model_name or "+".join(model_names),
            alpha=alpha,
            interp=interp,
            force=force,
            merge_dest_directory=dest,
        )
    except UnknownModelException:
        raise HTTPException(
            status_code=404,
            detail=f"One or more of the models '{keys}' not found",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return response
