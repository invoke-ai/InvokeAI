# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from typing import Annotated, Any, List, Literal, Optional, Union

from fastapi.routing import APIRouter, HTTPException
from pydantic import BaseModel, Field, parse_obj_as
from pathlib import Path
from ..dependencies import ApiDependencies

models_router = APIRouter(prefix="/v1/models", tags=["models"])


class VaeRepo(BaseModel):
    repo_id: str = Field(description="The repo ID to use for this VAE")
    path: Optional[str] = Field(description="The path to the VAE")
    subfolder: Optional[str] = Field(description="The subfolder to use for this VAE")

class ModelInfo(BaseModel):
    description: Optional[str] = Field(description="A description of the model")
    
class CkptModelInfo(ModelInfo):
    format: Literal['ckpt'] = 'ckpt'

    config: str = Field(description="The path to the model config")
    weights: str = Field(description="The path to the model weights")
    vae: str = Field(description="The path to the model VAE")
    width: Optional[int] = Field(description="The width of the model")
    height: Optional[int] = Field(description="The height of the model")

class DiffusersModelInfo(ModelInfo):
    format: Literal['diffusers'] = 'diffusers'

    vae: Optional[VaeRepo] = Field(description="The VAE repo to use for this model")
    repo_id: Optional[str] = Field(description="The repo ID to use for this model")
    path: Optional[str] = Field(description="The path to the model")

class CreateModelRequest (BaseModel):
    name: str = Field(description="The name of the model")
    info: Union[CkptModelInfo, DiffusersModelInfo] = Field(..., discriminator="format", description="The model info")

class CreateModelResponse (BaseModel):
    name: str = Field(description="The name of the new model")
    info: Union[CkptModelInfo, DiffusersModelInfo] = Field(..., discriminator="format", description="The model info")
    status: str = Field(description="The status of the API response")

class ConvertedModelRequest (BaseModel):
    name: str = Field(description="The name of the new model")
    info: CkptModelInfo = Field(description="The converted model info")

class ConvertedModelResponse (BaseModel):
    name: str = Field(description="The name of the new model")
    info: DiffusersModelInfo = Field(description="The converted model info")

class ModelsList(BaseModel):
    models: dict[str, Annotated[Union[(CkptModelInfo,DiffusersModelInfo)], Field(discriminator="format")]]


@models_router.get(
    "/",
    operation_id="list_models",
    responses={200: {"model": ModelsList }},
)
async def list_models() -> ModelsList:
    """Gets a list of models"""
    models_raw = ApiDependencies.invoker.services.model_manager.list_models()
    models = parse_obj_as(ModelsList, { "models": models_raw })
    return models


@models_router.post(
    "/",
    operation_id="update_model",
    responses={
        201: {
        "model_response": "Model added", 
        },
        202: {
        "description": "Model submission is processing. Check back later."
        }, 
    },
)
async def update_model(
    model_request: CreateModelRequest
) -> CreateModelResponse:
    """ Add Model """
    try:
        model_request_info = model_request.info
        print(f">> Checking for {model_request_info}...")
        info_dict = model_request_info.dict()

        ApiDependencies.invoker.services.model_manager.add_model(
            model_name=model_request.name,
            model_attributes=info_dict,
            clobber=True,
        )
        model_response = CreateModelResponse(name=model_request.name, info=model_request.info, status="success")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return model_response


@models_router.delete(
    "/{model_name}",
    operation_id="del_model",
    responses={
        204: {
        "description": "Model deleted"
        }, 
        404: {
        "description": "Model not found"
        }
    },
)
async def delete_model(model_name: str) -> None:
    """Delete Model"""
    model_names = ApiDependencies.invoker.services.model_manager.model_names()
    model_exists = model_name in model_names
    
    try:
        # check if model exists
        print(f">> Checking for model {model_name}...")

        if not model_exists:
            print(f">> Model not found")
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")                
        
        # delete model
        print(f">> Deleting Model: {model_name}")
        ApiDependencies.invoker.services.model_manager.del_model(model_name, delete_files=True)
        print(f">> Model Deleted: {model_name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
