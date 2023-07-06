# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654), 2023 Kent Keirsey (https://github.com/hipsterusername), 2024 Lincoln Stein


from typing import Literal, List, Optional, Union

from fastapi import Body, Path, Query, Response
from fastapi.routing import APIRouter
from pydantic import BaseModel, parse_obj_as
from starlette.exceptions import HTTPException

from invokeai.backend import BaseModelType, ModelType
from invokeai.backend.model_management.models import (
    OPENAPI_MODEL_CONFIGS,
    SchedulerPredictionType,
)
from invokeai.backend.model_management import MergeInterpolationMethod
from ..dependencies import ApiDependencies

models_router = APIRouter(prefix="/v1/models", tags=["models"])

UpdateModelResponse = Union[tuple(OPENAPI_MODEL_CONFIGS)]
ImportModelResponse = Union[tuple(OPENAPI_MODEL_CONFIGS)]
ConvertModelResponse = Union[tuple(OPENAPI_MODEL_CONFIGS)]
MergeModelResponse = Union[tuple(OPENAPI_MODEL_CONFIGS)]

class ModelsList(BaseModel):
    models: list[Union[tuple(OPENAPI_MODEL_CONFIGS)]]

@models_router.get(
    "/",
    operation_id="list_models",
    responses={200: {"model": ModelsList }},
)
async def list_models(
    base_model: Optional[BaseModelType] = Query(default=None, description="Base model"),
    model_type: Optional[ModelType] = Query(default=None, description="The type of model to get"),
) -> ModelsList:
    """Gets a list of models"""
    models_raw = ApiDependencies.invoker.services.model_manager.list_models(base_model, model_type)
    models = parse_obj_as(ModelsList, { "models": models_raw })
    return models

@models_router.patch(
    "/{base_model}/{model_type}/{model_name}",
    operation_id="update_model",
    responses={200: {"description" : "The model was updated successfully"},
               404: {"description" : "The model could not be found"},
               400: {"description" : "Bad request"}
               },
    status_code = 200,
    response_model = UpdateModelResponse,
)
async def update_model(
        base_model: BaseModelType = Path(description="Base model"),
        model_type: ModelType = Path(description="The type of model"),
        model_name: str = Path(description="model name"),
        info: Union[tuple(OPENAPI_MODEL_CONFIGS)] = Body(description="Model configuration"),
) -> UpdateModelResponse:
    """ Add Model """
    try:
        ApiDependencies.invoker.services.model_manager.update_model(
            model_name=model_name,
            base_model=base_model,
            model_type=model_type,
            model_attributes=info.dict()
        )
        model_raw = ApiDependencies.invoker.services.model_manager.list_model(
            model_name=model_name,
            base_model=base_model,
            model_type=model_type,
        )
        model_response = parse_obj_as(UpdateModelResponse, model_raw)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return model_response

@models_router.post(
    "/",
    operation_id="import_model",
    responses= {
        201: {"description" : "The model imported successfully"},
        404: {"description" : "The model could not be found"},
        424: {"description" : "The model appeared to import successfully, but could not be found in the model manager"},
        409: {"description" : "There is already a model corresponding to this path or repo_id"},
    },
    status_code=201,
    response_model=ImportModelResponse
)
async def import_model(
        location: str = Body(description="A model path, repo_id or URL to import"),
        prediction_type: Optional[Literal['v_prediction','epsilon','sample']] = \
                Body(description='Prediction type for SDv2 checkpoint files', default="v_prediction"),
) -> ImportModelResponse:
    """ Add a model using its local path, repo_id, or remote URL """
    
    items_to_import = {location}
    prediction_types = { x.value: x for x in SchedulerPredictionType }
    logger = ApiDependencies.invoker.services.logger

    try:
        installed_models = ApiDependencies.invoker.services.model_manager.heuristic_import(
            items_to_import = items_to_import,
            prediction_type_helper = lambda x: prediction_types.get(prediction_type)
        )
        info = installed_models.get(location)

        if not info:
            logger.error("Import failed")
            raise HTTPException(status_code=424)
        
        logger.info(f'Successfully imported {location}, got {info}')
        model_raw = ApiDependencies.invoker.services.model_manager.list_model(
            model_name=info.name,
            base_model=info.base_model,
            model_type=info.model_type
        )
        return parse_obj_as(ImportModelResponse, model_raw)
    
    except KeyError as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(str(e))
        raise HTTPException(status_code=409, detail=str(e))
        

@models_router.delete(
    "/{base_model}/{model_type}/{model_name}",
    operation_id="del_model",
    responses={
        204: {
        "description": "Model deleted successfully"
        }, 
        404: {
        "description": "Model not found"
        }
    },
)
async def delete_model(
        base_model: BaseModelType = Path(description="Base model"),
        model_type: ModelType = Path(description="The type of model"),
        model_name: str = Path(description="model name"),
) -> Response:
    """Delete Model"""
    logger = ApiDependencies.invoker.services.logger
    
    try:
        ApiDependencies.invoker.services.model_manager.del_model(model_name,
                                                                 base_model = base_model,
                                                                 model_type = model_type
                                                                 )
        logger.info(f"Deleted model: {model_name}")
        return Response(status_code=204)
    except KeyError:
        logger.error(f"Model not found: {model_name}")
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

@models_router.put(
    "/convert/{base_model}/{model_type}/{model_name}",
    operation_id="convert_model",
    responses={
        200: { "description": "Model converted successfully" },
        400: {"description" : "Bad request"  },
        404: { "description": "Model not found"  },
    },
    status_code = 200,
    response_model = ConvertModelResponse,
)
async def convert_model(
        base_model: BaseModelType = Path(description="Base model"),
        model_type: ModelType = Path(description="The type of model"),
        model_name: str = Path(description="model name"),
) -> ConvertModelResponse:
    """Convert a checkpoint model into a diffusers model"""
    logger = ApiDependencies.invoker.services.logger
    try:
        logger.info(f"Converting model: {model_name}")
        ApiDependencies.invoker.services.model_manager.convert_model(model_name,
                                                                     base_model = base_model,
                                                                     model_type = model_type
                                                                     )
        model_raw = ApiDependencies.invoker.services.model_manager.list_model(model_name,
                                                                              base_model = base_model,
                                                                              model_type = model_type)
        response = parse_obj_as(ConvertModelResponse, model_raw)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return response
        
@models_router.put(
    "/merge/{base_model}",
    operation_id="merge_models",
    responses={
        200: { "description": "Model converted successfully" },
        400: { "description": "Incompatible models"  },
        404: { "description": "One or more models not found"  },
    },
    status_code = 200,
    response_model = MergeModelResponse,
)
async def merge_models(
        base_model: BaseModelType                  = Path(description="Base model"),
        model_names: List[str]                     = Body(description="model name", min_items=2, max_items=3),
        merged_model_name: Optional[str]           = Body(description="Name of destination model"),
        alpha: Optional[float]                     = Body(description="Alpha weighting strength to apply to 2d and 3d models", default=0.5),
        interp: Optional[MergeInterpolationMethod] = Body(description="Interpolation method"),
        force: Optional[bool]                      = Body(description="Force merging of models created with different versions of diffusers", default=False),
) -> MergeModelResponse:
    """Convert a checkpoint model into a diffusers model"""
    logger = ApiDependencies.invoker.services.logger
    try:
        logger.info(f"Merging models: {model_names}")
        result = ApiDependencies.invoker.services.model_manager.merge_models(model_names,
                                                                             base_model,
                                                                             merged_model_name or "+".join(model_names),
                                                                             alpha,
                                                                             interp,
                                                                             force)
        model_raw = ApiDependencies.invoker.services.model_manager.list_model(result.name,
                                                                              base_model = base_model,
                                                                              model_type = ModelType.Main,
                                                                              )
        response = parse_obj_as(ConvertModelResponse, model_raw)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"One or more of the models '{model_names}' not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return response
