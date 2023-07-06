# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654), 2023 Kent Keirsey (https://github.com/hipsterusername), 2024 Lincoln Stein


from typing import Literal, Optional, Union

from fastapi import Body, Path, Query, Response
from fastapi.routing import APIRouter
from pydantic import BaseModel, parse_obj_as
from starlette.exceptions import HTTPException

from invokeai.backend import BaseModelType, ModelType
from invokeai.backend.model_management.models import (
    OPENAPI_MODEL_CONFIGS,
    SchedulerPredictionType
)

from ..dependencies import ApiDependencies

models_router = APIRouter(prefix="/v1/models", tags=["models"])

UpdateModelResponse = Union[tuple(OPENAPI_MODEL_CONFIGS)]
ImportModelResponse = Union[tuple(OPENAPI_MODEL_CONFIGS)]
ConvertModelResponse = Union[tuple(OPENAPI_MODEL_CONFIGS)]

class ModelsList(BaseModel):
    models: list[Union[tuple(OPENAPI_MODEL_CONFIGS)]]

@models_router.get(
    "/{base_model}/{model_type}",
    operation_id="list_models",
    responses={200: {"model": ModelsList }},
)
async def list_models(
    base_model: Optional[BaseModelType] = Path(
        default=None, description="Base model"
    ),
    model_type: Optional[ModelType] = Path(
        default=None, description="The type of model to get"
    ),
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
        base_model: BaseModelType = Path(default='sd-1', description="Base model"),
        model_type: ModelType = Path(default='main', description="The type of model"),
        model_name: str = Path(default=None, description="model name"),
        info: Union[tuple(OPENAPI_MODEL_CONFIGS)]  = Body(description="Model configuration"),
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
    response_model = Union[tuple(OPENAPI_MODEL_CONFIGS)],
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
        
        # @socketio.on("mergeDiffusersModels")
        # def merge_diffusers_models(model_merge_info: dict):
        #     try:
        #         models_to_merge = model_merge_info["models_to_merge"]
        #         model_ids_or_paths = [
        #             self.generate.model_manager.model_name_or_path(x)
        #             for x in models_to_merge
        #         ]
        #         merged_pipe = merge_diffusion_models(
        #             model_ids_or_paths,
        #             model_merge_info["alpha"],
        #             model_merge_info["interp"],
        #             model_merge_info["force"],
        #         )

        #         dump_path = global_models_dir() / "merged_models"
        #         if model_merge_info["model_merge_save_path"] is not None:
        #             dump_path = Path(model_merge_info["model_merge_save_path"])

        #         os.makedirs(dump_path, exist_ok=True)
        #         dump_path = dump_path / model_merge_info["merged_model_name"]
        #         merged_pipe.save_pretrained(dump_path, safe_serialization=1)

        #         merged_model_config = dict(
        #             model_name=model_merge_info["merged_model_name"],
        #             description=f'Merge of models {", ".join(models_to_merge)}',
        #             commit_to_conf=opt.conf,
        #         )

        #         if vae := self.generate.model_manager.config[models_to_merge[0]].get(
        #             "vae", None
        #         ):
        #             print(f">> Using configured VAE assigned to {models_to_merge[0]}")
        #             merged_model_config.update(vae=vae)

        #         self.generate.model_manager.import_diffuser_model(
        #             dump_path, **merged_model_config
        #         )
        #         new_model_list = self.generate.model_manager.list_models()

        #         socketio.emit(
        #             "modelsMerged",
        #             {
        #                 "merged_models": models_to_merge,
        #                 "merged_model_name": model_merge_info["merged_model_name"],
        #                 "model_list": new_model_list,
        #                 "update": True,
        #             },
        #         )
        #         print(f">> Models Merged: {models_to_merge}")
        #         print(f">> New Model Added: {model_merge_info['merged_model_name']}")
        #     except Exception as e:
