# Copyright (c) 2023 Lincoln D. Stein
"""FastAPI route for model configuration records."""

import io
import pathlib
import shutil
import traceback
from copy import deepcopy
from enum import Enum
from tempfile import TemporaryDirectory
from typing import List, Optional, Type

from fastapi import Body, Path, Query, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.routing import APIRouter
from PIL import Image
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException
from typing_extensions import Annotated

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.config import get_config
from invokeai.app.services.model_images.model_images_common import ModelImageFileNotFoundException
from invokeai.app.services.model_install.model_install_common import ModelInstallJob
from invokeai.app.services.model_records import (
    InvalidModelException,
    ModelRecordChanges,
    UnknownModelException,
)
from invokeai.backend.model_manager.config import (
    AnyModelConfig,
    BaseModelType,
    MainCheckpointConfig,
    ModelFormat,
    ModelType,
)
from invokeai.backend.model_manager.load.model_cache.model_cache_base import CacheStats
from invokeai.backend.model_manager.metadata.fetch.huggingface import HuggingFaceMetadataFetch
from invokeai.backend.model_manager.metadata.metadata_base import ModelMetadataWithFiles, UnknownMetadataException
from invokeai.backend.model_manager.search import ModelSearch
from invokeai.backend.model_manager.starter_models import (
    STARTER_BUNDLES,
    STARTER_MODELS,
    StarterModel,
    StarterModelWithoutDependencies,
)

model_manager_router = APIRouter(prefix="/v2/models", tags=["model_manager"])

# images are immutable; set a high max-age
IMAGE_MAX_AGE = 31536000


class ModelsList(BaseModel):
    """Return list of configs."""

    models: List[AnyModelConfig]

    model_config = ConfigDict(use_enum_values=True)


class CacheType(str, Enum):
    """Cache type - one of vram or ram."""

    RAM = "RAM"
    VRAM = "VRAM"


def add_cover_image_to_model_config(config: AnyModelConfig, dependencies: Type[ApiDependencies]) -> AnyModelConfig:
    """Add a cover image URL to a model configuration."""
    cover_image = dependencies.invoker.services.model_images.get_url(config.key)
    config.cover_image = cover_image
    return config


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
    "config_path": "string",
    "key": "string",
    "hash": "string",
    "description": "string",
    "source": "string",
    "converted_at": 0,
    "variant": "normal",
    "prediction_type": "epsilon",
    "repo_variant": "fp16",
    "upcast_attention": False,
}

example_model_input = {
    "path": "/path/to/model",
    "name": "model_name",
    "base": "sd-1",
    "type": "main",
    "format": "checkpoint",
    "config_path": "configs/stable-diffusion/v1-inference.yaml",
    "description": "Model description",
    "vae": None,
    "variant": "normal",
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
    for model in found_models:
        model = add_cover_image_to_model_config(model, ApiDependencies)
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
    try:
        config = ApiDependencies.invoker.services.model_manager.store.get_model(key)
        return add_cover_image_to_model_config(config, ApiDependencies)
    except UnknownModelException as e:
        raise HTTPException(status_code=404, detail=str(e))


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

        scan_results: list[FoundModel] = []

        # Check if the model is installed by comparing paths, appending to the scan result.
        for p in non_core_model_paths:
            path = str(p)
            is_installed = any(str(models_path / m.path) == path for m in installed_models)
            found_model = FoundModel(path=path, is_installed=is_installed)
            scan_results.append(found_model)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while searching the directory: {e}",
        )
    return scan_results


class HuggingFaceModels(BaseModel):
    urls: List[AnyHttpUrl] | None = Field(description="URLs for all checkpoint format models in the metadata")
    is_diffusers: bool = Field(description="Whether the metadata is for a Diffusers format model")


@model_manager_router.get(
    "/hugging_face",
    operation_id="get_hugging_face_models",
    responses={
        200: {"description": "Hugging Face repo scanned successfully"},
        400: {"description": "Invalid hugging face repo"},
    },
    status_code=200,
    response_model=HuggingFaceModels,
)
async def get_hugging_face_models(
    hugging_face_repo: str = Query(description="Hugging face repo to search for models", default=None),
) -> HuggingFaceModels:
    try:
        metadata = HuggingFaceMetadataFetch().from_id(hugging_face_repo)
    except UnknownMetadataException:
        raise HTTPException(
            status_code=400,
            detail="No HuggingFace repository found",
        )

    assert isinstance(metadata, ModelMetadataWithFiles)

    return HuggingFaceModels(
        urls=metadata.ckpt_urls,
        is_diffusers=metadata.is_diffusers,
    )


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
    changes: Annotated[ModelRecordChanges, Body(description="Model config", example=example_model_input)],
) -> AnyModelConfig:
    """Update a model's config."""
    logger = ApiDependencies.invoker.services.logger
    record_store = ApiDependencies.invoker.services.model_manager.store
    installer = ApiDependencies.invoker.services.model_manager.install
    try:
        record_store.update_model(key, changes=changes)
        config = installer.sync_model_path(key)
        config = add_cover_image_to_model_config(config, ApiDependencies)
        logger.info(f"Updated model: {key}")
    except UnknownModelException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(str(e))
        raise HTTPException(status_code=409, detail=str(e))
    return config


@model_manager_router.get(
    "/i/{key}/image",
    operation_id="get_model_image",
    responses={
        200: {
            "description": "The model image was fetched successfully",
        },
        400: {"description": "Bad request"},
        404: {"description": "The model image could not be found"},
    },
    status_code=200,
)
async def get_model_image(
    key: str = Path(description="The name of model image file to get"),
) -> FileResponse:
    """Gets an image file that previews the model"""

    try:
        path = ApiDependencies.invoker.services.model_images.get_path(key)

        response = FileResponse(
            path,
            media_type="image/png",
            filename=key + ".png",
            content_disposition_type="inline",
        )
        response.headers["Cache-Control"] = f"max-age={IMAGE_MAX_AGE}"
        return response
    except Exception:
        raise HTTPException(status_code=404)


@model_manager_router.patch(
    "/i/{key}/image",
    operation_id="update_model_image",
    responses={
        200: {
            "description": "The model image was updated successfully",
        },
        400: {"description": "Bad request"},
    },
    status_code=200,
)
async def update_model_image(
    key: Annotated[str, Path(description="Unique key of model")],
    image: UploadFile,
) -> None:
    if not image.content_type or not image.content_type.startswith("image"):
        raise HTTPException(status_code=415, detail="Not an image")

    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))

    except Exception:
        ApiDependencies.invoker.services.logger.error(traceback.format_exc())
        raise HTTPException(status_code=415, detail="Failed to read image")

    logger = ApiDependencies.invoker.services.logger
    model_images = ApiDependencies.invoker.services.model_images
    try:
        model_images.save(pil_image, key)
        logger.info(f"Updated image for model: {key}")
    except ValueError as e:
        logger.error(str(e))
        raise HTTPException(status_code=409, detail=str(e))
    return


@model_manager_router.delete(
    "/i/{key}",
    operation_id="delete_model",
    responses={
        204: {"description": "Model deleted successfully"},
        404: {"description": "Model not found"},
    },
    status_code=204,
)
async def delete_model(
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


@model_manager_router.delete(
    "/i/{key}/image",
    operation_id="delete_model_image",
    responses={
        204: {"description": "Model image deleted successfully"},
        404: {"description": "Model image not found"},
    },
    status_code=204,
)
async def delete_model_image(
    key: str = Path(description="Unique key of model image to remove from model_images directory."),
) -> None:
    logger = ApiDependencies.invoker.services.logger
    model_images = ApiDependencies.invoker.services.model_images
    try:
        model_images.delete(key)
        logger.info(f"Deleted model image: {key}")
        return
    except UnknownModelException as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))


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
    inplace: Optional[bool] = Query(description="Whether or not to install a local model in place", default=False),
    access_token: Optional[str] = Query(description="access token for the remote resource", default=None),
    config: ModelRecordChanges = Body(
        description="Object containing fields that override auto-probed values in the model config record, such as name, description and prediction_type ",
        example={"name": "string", "description": "string"},
    ),
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

    `config` is a ModelRecordChanges object. Fields in this object will override
    the ones that are probed automatically. Pass an empty object to accept
    all the defaults.

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
            inplace=bool(inplace),
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
    "/install/huggingface",
    operation_id="install_hugging_face_model",
    responses={
        201: {"description": "The model is being installed"},
        400: {"description": "Bad request"},
        409: {"description": "There is already a model corresponding to this path or repo_id"},
    },
    status_code=201,
    response_class=HTMLResponse,
)
async def install_hugging_face_model(
    source: str = Query(description="HuggingFace repo_id to install"),
) -> HTMLResponse:
    """Install a Hugging Face model using a string identifier."""

    def generate_html(title: str, heading: str, repo_id: str, is_error: bool, message: str | None = "") -> str:
        if message:
            message = f"<p>{message}</p>"
        title_class = "error" if is_error else "success"
        return f"""
            <html>

            <head>
                <title>{title}</title>
                <style>
                    body {{
                        text-align: center;
                        background-color: hsl(220 12% 10% / 1);
                        font-family: Helvetica, sans-serif;
                        color: hsl(220 12% 86% / 1);
                    }}

                    .repo-id {{
                        color: hsl(220 12% 68% / 1);
                    }}

                    .error {{
                        color: hsl(0 42% 68% / 1)
                    }}

                    .message-box {{
                        display: inline-block;
                        border-radius: 5px;
                        background-color: hsl(220 12% 20% / 1);
                        padding-inline-end: 30px;
                        padding: 20px;
                        padding-inline-start: 30px;
                        padding-inline-end: 30px;
                    }}

                    .container {{
                        display: flex;
                        width: 100%;
                        height: 100%;
                        align-items: center;
                        justify-content: center;
                    }}

                    a {{
                        color: inherit
                    }}

                    a:visited {{
                        color: inherit
                    }}

                    a:active {{
                        color: inherit
                    }}
                </style>
            </head>

            <body style="background-color: hsl(220 12% 10% / 1);">
                <div class="container">
                    <div class="message-box">
                        <h2 class="{title_class}">{heading}</h2>
                        {message}
                        <p class="repo-id">Repo ID: {repo_id}</p>
                    </div>
                </div>
            </body>

            </html>
        """

    try:
        metadata = HuggingFaceMetadataFetch().from_id(source)
        assert isinstance(metadata, ModelMetadataWithFiles)
    except UnknownMetadataException:
        title = "Unable to Install Model"
        heading = "No HuggingFace repository found with that repo ID."
        message = "Ensure the repo ID is correct and try again."
        return HTMLResponse(content=generate_html(title, heading, source, True, message), status_code=400)

    logger = ApiDependencies.invoker.services.logger

    try:
        installer = ApiDependencies.invoker.services.model_manager.install
        if metadata.is_diffusers:
            installer.heuristic_import(
                source=source,
                inplace=False,
            )
        elif metadata.ckpt_urls is not None and len(metadata.ckpt_urls) == 1:
            installer.heuristic_import(
                source=str(metadata.ckpt_urls[0]),
                inplace=False,
            )
        else:
            title = "Unable to Install Model"
            heading = "This HuggingFace repo has multiple models."
            message = "Please use the Model Manager to install this model."
            return HTMLResponse(content=generate_html(title, heading, source, True, message), status_code=200)

        title = "Model Install Started"
        heading = "Your HuggingFace model is installing now."
        message = "You can close this tab and check the Model Manager for installation progress."
        return HTMLResponse(content=generate_html(title, heading, source, False, message), status_code=201)
    except Exception as e:
        logger.error(str(e))
        title = "Unable to Install Model"
        heading = "There was an problem installing this model."
        message = 'Please use the Model Manager directly to install this model. If the issue persists, ask for help on <a href="https://discord.gg/ZmtBAhwWhy">discord</a>.'
        return HTMLResponse(content=generate_html(title, heading, source, True, message), status_code=500)


@model_manager_router.get(
    "/install",
    operation_id="list_model_installs",
)
async def list_model_installs() -> List[ModelInstallJob]:
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
    model and type can be retrieved from the `config_out` field. For multi-file models such as diffusers,
    information on individual files can be retrieved from `download_parts`.

    See the example and schema below for more information.
    """
    jobs: List[ModelInstallJob] = ApiDependencies.invoker.services.model_manager.install.list_jobs()
    return jobs


@model_manager_router.get(
    "/install/{id}",
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
    "/install/{id}",
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


@model_manager_router.delete(
    "/install",
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
    loader = model_manager.load
    logger = ApiDependencies.invoker.services.logger
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

    with TemporaryDirectory(dir=ApiDependencies.invoker.services.configuration.models_path) as tmpdir:
        convert_path = pathlib.Path(tmpdir) / pathlib.Path(model_config.path).stem
        converted_model = loader.load_model(model_config)
        # write the converted file to the convert path
        raw_model = converted_model.model
        assert hasattr(raw_model, "save_pretrained")
        raw_model.save_pretrained(convert_path)  # type: ignore
        assert convert_path.exists()

        # temporarily rename the original safetensors file so that there is no naming conflict
        original_name = model_config.name
        model_config.name = f"{original_name}.DELETE"
        changes = ModelRecordChanges(name=model_config.name)
        store.update_model(key, changes=changes)

        # install the diffusers
        try:
            new_key = installer.install_path(
                convert_path,
                config=ModelRecordChanges(
                    name=original_name,
                    description=model_config.description,
                    hash=model_config.hash,
                    source=model_config.source,
                ),
            )
        except Exception as e:
            logger.error(str(e))
            store.update_model(key, changes=ModelRecordChanges(name=original_name))
            raise HTTPException(status_code=409, detail=str(e))

    # Update the model image if the model had one
    try:
        model_image = ApiDependencies.invoker.services.model_images.get(key)
        ApiDependencies.invoker.services.model_images.save(model_image, new_key)
        ApiDependencies.invoker.services.model_images.delete(key)
    except ModelImageFileNotFoundException:
        pass

    # delete the original safetensors file
    installer.delete(key)

    # delete the temporary directory
    # shutil.rmtree(cache_path)

    # return the config record for the new diffusers directory
    new_config = store.get_model(new_key)
    new_config = add_cover_image_to_model_config(new_config, ApiDependencies)
    return new_config


class StarterModelResponse(BaseModel):
    starter_models: list[StarterModel]
    starter_bundles: dict[str, list[StarterModel]]


def get_is_installed(
    starter_model: StarterModel | StarterModelWithoutDependencies, installed_models: list[AnyModelConfig]
) -> bool:
    for model in installed_models:
        if model.source == starter_model.source:
            return True
        if (
            (model.name == starter_model.name or model.name in starter_model.previous_names)
            and model.base == starter_model.base
            and model.type == starter_model.type
        ):
            return True
    return False


@model_manager_router.get("/starter_models", operation_id="get_starter_models", response_model=StarterModelResponse)
async def get_starter_models() -> StarterModelResponse:
    installed_models = ApiDependencies.invoker.services.model_manager.store.search_by_attr()
    starter_models = deepcopy(STARTER_MODELS)
    starter_bundles = deepcopy(STARTER_BUNDLES)
    for model in starter_models:
        model.is_installed = get_is_installed(model, installed_models)
        # Remove already-installed dependencies
        missing_deps: list[StarterModelWithoutDependencies] = []

        for dep in model.dependencies or []:
            if not get_is_installed(dep, installed_models):
                missing_deps.append(dep)
        model.dependencies = missing_deps

    for bundle in starter_bundles.values():
        for model in bundle:
            model.is_installed = get_is_installed(model, installed_models)
            # Remove already-installed dependencies
            missing_deps: list[StarterModelWithoutDependencies] = []
            for dep in model.dependencies or []:
                if not get_is_installed(dep, installed_models):
                    missing_deps.append(dep)
            model.dependencies = missing_deps

    return StarterModelResponse(starter_models=starter_models, starter_bundles=starter_bundles)


@model_manager_router.get(
    "/model_cache",
    operation_id="get_cache_size",
    response_model=float,
    summary="Get maximum size of model manager RAM or VRAM cache.",
)
async def get_cache_size(cache_type: CacheType = Query(description="The cache type", default=CacheType.RAM)) -> float:
    """Return the current RAM or VRAM cache size setting (in GB)."""
    cache = ApiDependencies.invoker.services.model_manager.load.ram_cache
    value = 0.0
    if cache_type == CacheType.RAM:
        value = cache.max_cache_size
    elif cache_type == CacheType.VRAM:
        value = cache.max_vram_cache_size
    return value


@model_manager_router.put(
    "/model_cache",
    operation_id="set_cache_size",
    response_model=float,
    summary="Set maximum size of model manager RAM or VRAM cache, optionally writing new value out to invokeai.yaml config file.",
)
async def set_cache_size(
    value: float = Query(description="The new value for the maximum cache size"),
    cache_type: CacheType = Query(description="The cache type", default=CacheType.RAM),
    persist: bool = Query(description="Write new value out to invokeai.yaml", default=False),
) -> float:
    """Set the current RAM or VRAM cache size setting (in GB). ."""
    cache = ApiDependencies.invoker.services.model_manager.load.ram_cache
    app_config = get_config()
    # Record initial state.
    vram_old = app_config.vram
    ram_old = app_config.ram

    # Prepare target state.
    vram_new = vram_old
    ram_new = ram_old
    if cache_type == CacheType.RAM:
        ram_new = value
    elif cache_type == CacheType.VRAM:
        vram_new = value
    else:
        raise ValueError(f"Unexpected {cache_type=}.")

    config_path = app_config.config_file_path
    new_config_path = config_path.with_suffix(".yaml.new")

    try:
        # Try to apply the target state.
        cache.max_vram_cache_size = vram_new
        cache.max_cache_size = ram_new
        app_config.ram = ram_new
        app_config.vram = vram_new
        if persist:
            app_config.write_file(new_config_path)
            shutil.move(new_config_path, config_path)
    except Exception as e:
        # If there was a failure, restore the initial state.
        cache.max_cache_size = ram_old
        cache.max_vram_cache_size = vram_old
        app_config.ram = ram_old
        app_config.vram = vram_old

        raise RuntimeError("Failed to update cache size") from e
    return value


@model_manager_router.get(
    "/stats",
    operation_id="get_stats",
    response_model=Optional[CacheStats],
    summary="Get model manager RAM cache performance statistics.",
)
async def get_stats() -> Optional[CacheStats]:
    """Return performance statistics on the model manager's RAM cache. Will return null if no models have been loaded."""

    return ApiDependencies.invoker.services.model_manager.load.ram_cache.stats
