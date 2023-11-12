# Copyright (c) 2023 Lincoln D. Stein
"""FastAPI route for model configuration records."""


from hashlib import sha1
from random import randbytes
from typing import List, Optional

from fastapi import Body, Path, Query, Response
from fastapi.routing import APIRouter
from pydantic import BaseModel, ConfigDict, TypeAdapter
from starlette.exceptions import HTTPException
from typing_extensions import Annotated

from invokeai.app.services.model_records import DuplicateModelException, InvalidModelException, UnknownModelException
from invokeai.backend.model_manager.config import AnyModelConfig, BaseModelType, ModelType

from ..dependencies import ApiDependencies

model_records_router = APIRouter(prefix="/v1/model/record", tags=["models"])


class ModelsList(BaseModel):
    """Return list of configs."""

    models: list[AnyModelConfig]

    model_config = ConfigDict(use_enum_values=True)


ModelsListValidator = TypeAdapter(ModelsList)


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
    if base_models and len(base_models) > 0:
        models_raw = list()
        for base_model in base_models:
            models_raw.extend(
                [x.model_dump() for x in record_store.search_by_attr(base_model=base_model, model_type=model_type)]
            )
    else:
        models_raw = [x.model_dump() for x in record_store.search_by_attr(model_type=model_type)]
    models = ModelsListValidator.validate_python({"models": models_raw})
    return models


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
    except UnknownModelException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(str(e))
        raise HTTPException(status_code=409, detail=str(e))
    return model_response


@model_records_router.delete(
    "/i/{key}",
    operation_id="del_model_record",
    responses={204: {"description": "Model deleted successfully"}, 404: {"description": "Model not found"}},
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
