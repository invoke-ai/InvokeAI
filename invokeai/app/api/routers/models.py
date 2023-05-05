# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654) and 2023 Kent Keirsey (https://github.com/hipsterusername)

import shutil
import asyncio
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

class CreateModelRequest(BaseModel):
    name: str = Field(description="The name of the model")
    info: Union[CkptModelInfo, DiffusersModelInfo] = Field(discriminator="format", description="The model info")

class CreateModelResponse(BaseModel):
    name: str = Field(description="The name of the new model")
    info: Union[CkptModelInfo, DiffusersModelInfo] = Field(discriminator="format", description="The model info")
    status: str = Field(description="The status of the API response")

class ConversionRequest(BaseModel):
    name: str = Field(description="The name of the new model")
    info: CkptModelInfo = Field(description="The converted model info")
    save_location: str = Field(description="The path to save the converted model weights")
    

class ConvertedModelResponse(BaseModel):
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
    responses={200: {"status": "success"}},
)
async def update_model(
    model_request: CreateModelRequest
) -> CreateModelResponse:
    """ Add Model """
    model_request_info = model_request.info
    info_dict = model_request_info.dict()
    model_response = CreateModelResponse(name=model_request.name, info=model_request.info, status="success")

    ApiDependencies.invoker.services.model_manager.add_model(
        model_name=model_request.name,
        model_attributes=info_dict,
        clobber=True,
    )

    return model_response


@models_router.delete(
    "/{model_name}",
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
async def delete_model(model_name: str) -> None:
    """Delete Model"""
    model_names = ApiDependencies.invoker.services.model_manager.model_names()
    logger = ApiDependencies.invoker.services.logger
    model_exists = model_name in model_names

    # check if model exists
    logger.info(f"Checking for model {model_name}...")
           
    if model_exists:
        logger.info(f"Deleting Model: {model_name}")
        ApiDependencies.invoker.services.model_manager.del_model(model_name, delete_files=True)
        logger.info(f"Model Deleted: {model_name}")
        raise HTTPException(status_code=204, detail=f"Model '{model_name}' deleted successfully")
    
    else:
        logger.error(f"Model not found")
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    

            # @socketio.on("convertToDiffusers")
        # def convert_to_diffusers(model_to_convert: dict):
        #     try:
        #         if model_info := self.generate.model_manager.model_info(
        #             model_name=model_to_convert["model_name"]
        #         ):
        #             if "weights" in model_info:
        #                 ckpt_path = Path(model_info["weights"])
        #                 original_config_file = Path(model_info["config"])
        #                 model_name = model_to_convert["model_name"]
        #                 model_description = model_info["description"]
        #             else:
        #                 self.socketio.emit(
        #                     "error", {"message": "Model is not a valid checkpoint file"}
        #                 )
        #         else:
        #             self.socketio.emit(
        #                 "error", {"message": "Could not retrieve model info."}
        #             )

        #         if not ckpt_path.is_absolute():
        #             ckpt_path = Path(Globals.root, ckpt_path)

        #         if original_config_file and not original_config_file.is_absolute():
        #             original_config_file = Path(Globals.root, original_config_file)

        #         diffusers_path = Path(
        #             ckpt_path.parent.absolute(), f"{model_name}_diffusers"
        #         )

        #         if model_to_convert["save_location"] == "root":
        #             diffusers_path = Path(
        #                 global_converted_ckpts_dir(), f"{model_name}_diffusers"
        #             )

        #         if (
        #             model_to_convert["save_location"] == "custom"
        #             and model_to_convert["custom_location"] is not None
        #         ):
        #             diffusers_path = Path(
        #                 model_to_convert["custom_location"], f"{model_name}_diffusers"
        #             )

        #         if diffusers_path.exists():
        #             shutil.rmtree(diffusers_path)

        #         self.generate.model_manager.convert_and_import(
        #             ckpt_path,
        #             diffusers_path,
        #             model_name=model_name,
        #             model_description=model_description,
        #             vae=None,
        #             original_config_file=original_config_file,
        #             commit_to_conf=opt.conf,
        #         )

        #         new_model_list = self.generate.model_manager.list_models()
        #         socketio.emit(
        #             "modelConverted",
        #             {
        #                 "new_model_name": model_name,
        #                 "model_list": new_model_list,
        #                 "update": True,
        #             },
        #         )
        #         print(f">> Model Converted: {model_name}")
        #     except Exception as e:
        #         self.handle_exceptions(e)

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
