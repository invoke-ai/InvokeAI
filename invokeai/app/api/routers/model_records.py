# Copyright (c) 2023 Lincoln D. Stein
"""FastAPI route for model configuration records."""


from typing import List, Optional

from fastapi import Body, Path, Query
from fastapi.routing import APIRouter
from pydantic import BaseModel, ConfigDict, TypeAdapter
from starlette.exceptions import HTTPException

from invokeai.app.services.model_records import UnknownModelException
from invokeai.backend.model_manager.config import AnyModelConfig, BaseModelType, ModelType

from ..dependencies import ApiDependencies

model_records_router = APIRouter(prefix="/v1/model_records", tags=["model_records"])

ModelConfigValidator = TypeAdapter(AnyModelConfig)


class ModelsList(BaseModel):
    """Return list of configs."""

    models: list[AnyModelConfig]

    model_config = ConfigDict(use_enum_values=True)


ModelsListValidator = TypeAdapter(ModelsList)


@model_records_router.get(
    "/",
    operation_id="list_model_configs",
    responses={200: {"model": ModelsList}},
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
                [x.model_dump() for x in record_store.search_by_name(base_model=base_model, model_type=model_type)]
            )
    else:
        models_raw = [x.model_dump() for x in record_store.search_by_name(model_type=model_type)]
    models = ModelsListValidator.validate_python({"models": models_raw})
    return models


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
    key: str = Path(description="Unique key of model"),
    info: AnyModelConfig = Body(description="Model configuration"),
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
