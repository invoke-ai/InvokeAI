# Copyright (c) 2023 Lincoln D. Stein
"""FastAPI route for model configuration records."""

import pathlib
import shutil
from hashlib import sha1
from random import randbytes
import traceback
from typing import Any, Dict, List, Optional, Set

from fastapi import Body, Path, Query, Response
from fastapi.routing import APIRouter
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException
from typing_extensions import Annotated

from invokeai.app.services.model_install import ModelInstallJob
from invokeai.app.services.model_metadata.metadata_store_base import ModelMetadataChanges
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
    MainCheckpointConfig,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.model_manager.merge import MergeInterpolationMethod, ModelMerger
from invokeai.backend.model_manager.metadata import AnyModelRepoMetadata
from invokeai.backend.model_manager.metadata.metadata_base import BaseMetadata
from invokeai.backend.model_manager.search import ModelSearch

from ..dependencies import ApiDependencies

model_manager_router = APIRouter(prefix="/v2/models", tags=["model_manager"])


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


##############################################################################
# These are example inputs and outputs that are used in places where Swagger
# is unable to generate a correct example.
##############################################################################
example_model_config = {
    "path": "string",
    "name": "string",
    "base": "sd-1",
    "type": "main",
    "format": "checkpoint",
    "config": "string",
    "key": "string",
    "original_hash": "string",
    "current_hash": "string",
    "description": "string",
    "source": "string",
    "last_modified": 0,
    "vae": "string",
    "variant": "normal",
    "prediction_type": "epsilon",
    "repo_variant": "fp16",
    "upcast_attention": False,
    "ztsnr_training": False,
}

example_model_input = {
    "path": "/path/to/model",
    "name": "model_name",
    "base": "sd-1",
    "type": "main",
    "format": "checkpoint",
    "config": "configs/stable-diffusion/v1-inference.yaml",
    "description": "Model description",
    "vae": None,
    "variant": "normal",
}

example_model_metadata = {
    "name": "ip_adapter_sd_image_encoder",
    "author": "InvokeAI",
    "tags": [
        "transformers",
        "safetensors",
        "clip_vision_model",
        "endpoints_compatible",
        "region:us",
        "has_space",
        "license:apache-2.0",
    ],
    "files": [
        {
            "url": "https://huggingface.co/InvokeAI/ip_adapter_sd_image_encoder/resolve/main/README.md",
            "path": "ip_adapter_sd_image_encoder/README.md",
            "size": 628,
            "sha256": None,
        },
        {
            "url": "https://huggingface.co/InvokeAI/ip_adapter_sd_image_encoder/resolve/main/config.json",
            "path": "ip_adapter_sd_image_encoder/config.json",
            "size": 560,
            "sha256": None,
        },
        {
            "url": "https://huggingface.co/InvokeAI/ip_adapter_sd_image_encoder/resolve/main/model.safetensors",
            "path": "ip_adapter_sd_image_encoder/model.safetensors",
            "size": 2528373448,
            "sha256": "6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030",
        },
    ],
    "type": "huggingface",
    "id": "InvokeAI/ip_adapter_sd_image_encoder",
    "tag_dict": {"license": "apache-2.0"},
    "last_modified": "2023-09-23T17:33:25Z",
}

##############################################################################
# ROUTES
##############################################################################


@model_manager_router.get(
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


@model_manager_router.get(
    "/get_by_attrs",
    operation_id="get_model_records_by_attrs",
    response_model=AnyModelConfig,
)
async def get_model_records_by_attrs(
    name: str = Query(description="The name of the model"),
    type: ModelType = Query(description="The type of the model"),
    base: BaseModelType = Query(description="The base model of the model"),
) -> AnyModelConfig:
    """Gets a model by its attributes. The main use of this route is to provide backwards compatibility with the old
    model manager, which identified models by a combination of name, base and type."""
    configs = ApiDependencies.invoker.services.model_manager.store.search_by_attr(
        base_model=base, model_type=type, model_name=name
    )
    if not configs:
        raise HTTPException(status_code=404, detail="No model found with these attributes")

    return configs[0]


@model_manager_router.get(
    "/i/{key}",
    operation_id="get_model_record",
    responses={
        200: {
            "description": "The model configuration was retrieved successfully",
            "content": {"application/json": {"example": example_model_config}},
        },
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


@model_manager_router.get("/summary", operation_id="list_model_summary")
async def list_model_summary(
    page: int = Query(default=0, description="The page to get"),
    per_page: int = Query(default=10, description="The number of models per page"),
    order_by: ModelRecordOrderBy = Query(default=ModelRecordOrderBy.Default, description="The attribute to order by"),
) -> PaginatedResults[ModelSummary]:
    """Gets a page of model summary data."""
    record_store = ApiDependencies.invoker.services.model_manager.store
    results: PaginatedResults[ModelSummary] = record_store.list_models(page=page, per_page=per_page, order_by=order_by)
    return results


@model_manager_router.get(
    "/i/{key}/metadata",
    operation_id="get_model_metadata",
    responses={
        200: {
            "description": "The model metadata was retrieved successfully",
            "content": {"application/json": {"example": example_model_metadata}},
        },
        400: {"description": "Bad request"},
    },
)
async def get_model_metadata(
    key: str = Path(description="Key of the model repo metadata to fetch."),
) -> Optional[AnyModelRepoMetadata]:
    """Get a model metadata object."""
    record_store = ApiDependencies.invoker.services.model_manager.store
    result: Optional[AnyModelRepoMetadata] = record_store.get_metadata(key)

    return result

@model_manager_router.patch(
    "/i/{key}/metadata",
    operation_id="update_model_metadata",
    responses={
        201: {
            "description": "The model metadata was updated successfully",
            "content": {"application/json": {"example": example_model_metadata}},
        },
        400: {"description": "Bad request"},
    },
)
async def update_model_metadata(
    key: str = Path(description="Key of the model repo metadata to fetch."),
    changes: ModelMetadataChanges = Body(description="The changes")
) -> Optional[AnyModelRepoMetadata]:
    """Updates or creates a model metadata object."""
    record_store = ApiDependencies.invoker.services.model_manager.store
    metadata_store = ApiDependencies.invoker.services.model_manager.store.metadata_store
    
    try:
        original_metadata = record_store.get_metadata(key)
        if original_metadata:
            original_metadata.trigger_phrases = changes.trigger_phrases

            metadata_store.update_metadata(key, original_metadata)
        else:
            metadata_store.add_metadata(key, BaseMetadata(name="", author="",trigger_phrases=changes.trigger_phrases))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while updating the model metadata: {e}",
        )

    result: Optional[AnyModelRepoMetadata] = record_store.get_metadata(key)

    return result


@model_manager_router.get(
    "/tags",
    operation_id="list_tags",
)
async def list_tags() -> Set[str]:
    """Get a unique set of all the model tags."""
    record_store = ApiDependencies.invoker.services.model_manager.store
    result: Set[str] = record_store.list_tags()
    return result


class FoundModel(BaseModel):
    path: str = Field(description="Path to the model")
    is_installed: bool = Field(description="Whether or not the model is already installed")


@model_manager_router.get(
    "/scan_folder",
    operation_id="scan_for_models",
    responses={
        200: {"description": "Directory scanned successfully"},
        400: {"description": "Invalid directory path"},
    },
    status_code=200,
    response_model=List[FoundModel],
)
async def scan_for_models(
    scan_path: str = Query(description="Directory path to search for models", default=None),
) -> List[FoundModel]:
    path = pathlib.Path(scan_path)
    if not scan_path or not path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"The search path '{scan_path}' does not exist or is not directory",
        )

    search = ModelSearch()
    try:
        found_model_paths = search.search(path)
        models_path = ApiDependencies.invoker.services.configuration.models_path

        # If the search path includes the main models directory, we need to exclude core models from the list.
        # TODO(MM2): Core models should be handled by the model manager so we can determine if they are installed
        # without needing to crawl the filesystem.
        core_models_path = pathlib.Path(models_path, "core").resolve()
        non_core_model_paths = [p for p in found_model_paths if not p.is_relative_to(core_models_path)]

        installed_models = ApiDependencies.invoker.services.model_manager.store.search_by_attr()
        resolved_installed_model_paths: list[str] = []
        installed_model_sources: list[str] = []

        # This call lists all installed models.
        for model in installed_models:
            path = pathlib.Path(model.path)
            # If the model has a source, we need to add it to the list of installed sources.
            if model.source:
                installed_model_sources.append(model.source)
            # If the path is not absolute, that means it is in the app models directory, and we need to join it with
            # the models path before resolving.
            if not path.is_absolute():
                resolved_installed_model_paths.append(str(pathlib.Path(models_path, path).resolve()))
                continue
            resolved_installed_model_paths.append(str(path.resolve()))

        scan_results: list[FoundModel] = []

        # Check if the model is installed by comparing the resolved paths, appending to the scan result.
        for p in non_core_model_paths:
            path = str(p)
            is_installed = path in resolved_installed_model_paths or path in installed_model_sources
            found_model = FoundModel(path=path, is_installed=is_installed)
            scan_results.append(found_model)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while searching the directory: {e}",
        )
    return scan_results


@model_manager_router.get(
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


@model_manager_router.patch(
    "/i/{key}",
    operation_id="update_model_record",
    responses={
        200: {
            "description": "The model was updated successfully",
            "content": {"application/json": {"example": example_model_config}},
        },
        400: {"description": "Bad request"},
        404: {"description": "The model could not be found"},
        409: {"description": "There is already a model corresponding to the new name"},
    },
    status_code=200,
)
async def update_model_record(
    key: Annotated[str, Path(description="Unique key of model")],
    info: Annotated[
        AnyModelConfig, Body(description="Model config", discriminator="type", example=example_model_input)
    ],
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


@model_manager_router.delete(
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


@model_manager_router.post(
    "/i/",
    operation_id="add_model_record",
    responses={
        201: {
            "description": "The model added successfully",
            "content": {"application/json": {"example": example_model_config}},
        },
        409: {"description": "There is already a model corresponding to this path or repo_id"},
        415: {"description": "Unrecognized file/folder format"},
    },
    status_code=201,
)
async def add_model_record(
    config: Annotated[
        AnyModelConfig, Body(description="Model config", discriminator="type", example=example_model_input)
    ],
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


@model_manager_router.post(
    "/install",
    operation_id="install_model",
    responses={
        201: {"description": "The model imported successfully"},
        415: {"description": "Unrecognized file/folder format"},
        424: {"description": "The model appeared to import successfully, but could not be found in the model manager"},
        409: {"description": "There is already a model corresponding to this path or repo_id"},
    },
    status_code=201,
)
async def install_model(
    source: str = Query(description="Model source to install, can be a local path, repo_id, or remote URL"),
    # TODO(MM2): Can we type this?
    config: Optional[Dict[str, Any]] = Body(
        description="Dict of fields that override auto-probed values in the model config record, such as name, description and prediction_type ",
        default=None,
        example={"name": "string", "description": "string"},
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
            access_token=access_token,
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


@model_manager_router.get(
    "/import",
    operation_id="list_model_install_jobs",
)
async def list_model_install_jobs() -> List[ModelInstallJob]:
    """Return the list of model install jobs.

    Install jobs have a numeric `id`, a `status`, and other fields that provide information on
    the nature of the job and its progress. The `status` is one of:

    * "waiting" -- Job is waiting in the queue to run
    * "downloading" -- Model file(s) are downloading
    * "running" -- Model has downloaded and the model probing and registration process is running
    * "completed" -- Installation completed successfully
    * "error" -- An error occurred. Details will be in the "error_type" and "error" fields.
    * "cancelled" -- Job was cancelled before completion.

    Once completed, information about the model such as its size, base
    model, type, and metadata can be retrieved from the `config_out`
    field. For multi-file models such as diffusers, information on individual files
    can be retrieved from `download_parts`.

    See the example and schema below for more information.
    """
    jobs: List[ModelInstallJob] = ApiDependencies.invoker.services.model_manager.install.list_jobs()
    return jobs


@model_manager_router.get(
    "/import/{id}",
    operation_id="get_model_install_job",
    responses={
        200: {"description": "Success"},
        404: {"description": "No such job"},
    },
)
async def get_model_install_job(id: int = Path(description="Model install id")) -> ModelInstallJob:
    """
    Return model install job corresponding to the given source. See the documentation for 'List Model Install Jobs'
    for information on the format of the return value.
    """
    try:
        result: ModelInstallJob = ApiDependencies.invoker.services.model_manager.install.get_job_by_id(id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@model_manager_router.delete(
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


@model_manager_router.patch(
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


@model_manager_router.patch(
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


@model_manager_router.put(
    "/convert/{key}",
    operation_id="convert_model",
    responses={
        200: {
            "description": "Model converted successfully",
            "content": {"application/json": {"example": example_model_config}},
        },
        400: {"description": "Bad request"},
        404: {"description": "Model not found"},
        409: {"description": "There is already a model registered at this location"},
    },
)
async def convert_model(
    key: str = Path(description="Unique key of the safetensors main model to convert to diffusers format."),
) -> AnyModelConfig:
    """
    Permanently convert a model into diffusers format, replacing the safetensors version.
    Note that during the conversion process the key and model hash will change.
    The return value is the model configuration for the converted model.
    """
    model_manager = ApiDependencies.invoker.services.model_manager
    logger = ApiDependencies.invoker.services.logger
    loader = ApiDependencies.invoker.services.model_manager.load
    store = ApiDependencies.invoker.services.model_manager.store
    installer = ApiDependencies.invoker.services.model_manager.install

    try:
        model_config = store.get_model(key)
    except UnknownModelException as e:
        logger.error(str(e))
        raise HTTPException(status_code=424, detail=str(e))

    if not isinstance(model_config, MainCheckpointConfig):
        logger.error(f"The model with key {key} is not a main checkpoint model.")
        raise HTTPException(400, f"The model with key {key} is not a main checkpoint model.")

    # loading the model will convert it into a cached diffusers file
    model_manager.load_model_by_config(model_config, submodel_type=SubModelType.Scheduler)

    # Get the path of the converted model from the loader
    cache_path = loader.convert_cache.cache_path(key)
    assert cache_path.exists()

    # temporarily rename the original safetensors file so that there is no naming conflict
    original_name = model_config.name
    model_config.name = f"{original_name}.DELETE"
    store.update_model(key, config=model_config)

    # install the diffusers
    try:
        new_key = installer.install_path(
            cache_path,
            config={
                "name": original_name,
                "description": model_config.description,
                "original_hash": model_config.original_hash,
                "source": model_config.source,
            },
        )
    except DuplicateModelException as e:
        logger.error(str(e))
        raise HTTPException(status_code=409, detail=str(e))

    # get the original metadata
    if orig_metadata := store.get_metadata(key):
        store.metadata_store.add_metadata(new_key, orig_metadata)

    # delete the original safetensors file
    installer.delete(key)

    # delete the cached version
    shutil.rmtree(cache_path)

    # return the config record for the new diffusers directory
    new_config: AnyModelConfig = store.get_model(new_key)
    return new_config


@model_manager_router.put(
    "/merge",
    operation_id="merge",
    responses={
        200: {
            "description": "Model converted successfully",
            "content": {"application/json": {"example": example_model_config}},
        },
        400: {"description": "Bad request"},
        404: {"description": "Model not found"},
        409: {"description": "There is already a model registered at this location"},
    },
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
    Merge diffusers models. The process is controlled by a set parameters provided in the body of the request.
    ```
    Argument                Description [default]
    --------               ----------------------
    keys                   List of 2-3 model keys to merge together. All models must use the same base type.
    merged_model_name      Name for the merged model [Concat model names]
    alpha                  Alpha value (0.0-1.0). Higher values give more weight to the second model [0.5]
    force                  If true, force the merge even if the models were generated by different versions of the diffusers library [False]
    interp                 Interpolation method. One of "weighted_sum", "sigmoid", "inv_sigmoid" or "add_difference" [weighted_sum]
    merge_dest_directory   Specify a directory to store the merged model in [models directory]
    ```
    """
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
