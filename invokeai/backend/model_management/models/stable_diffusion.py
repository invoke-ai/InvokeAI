import os
import torch
from pydantic import Field
from typing import Literal, Optional
from .base import (
    ModelBase,
    ModelConfigBase,
    BaseModelType,
    ModelType,
    SubModelType,
    DiffusersModel,
)
from invokeai.app.services.config import InvokeAIAppConfig


# TODO: how to name properly
class StableDiffusion15Model(DiffusersModel):

    # TODO: str -> Path?
    class DiffusersConfig(ModelConfigBase):
        format: Literal["diffusers"]
        vae: Optional[str] = Field(None)

    class CheckpointConfig(ModelConfigBase):
        format: Literal["checkpoint"]
        vae: Optional[str] = Field(None)
        config: Optional[str] = Field(None)


    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert base_model == BaseModelType.StableDiffusion1_5
        assert model_type == ModelType.Pipeline
        super().__init__(
            model_path=model_path,
            base_model=BaseModelType.StableDiffusion1_5,
            model_type=ModelType.Pipeline,
        )

    @classmethod
    def save_to_config(cls) -> bool:
        return True

    @classmethod
    def detect_format(cls, model_path: str):
        if os.path.isdir(model_path):
            return "diffusers"
        else:
            return "checkpoint"

    @classmethod
    def convert_if_required(cls, model_path: str, dst_cache_path: str, config: Optional[dict]) -> str:
        cfg = cls.build_config(**config)
        if isinstance(cfg, cls.CheckpointConfig):
            return _convert_ckpt_and_cache(cfg) # TODO: args
        else:
            return model_path

# all same
class StableDiffusion2BaseModel(StableDiffusion15Model):
    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        # skip StableDiffusion15Model __init__
        assert base_model == BaseModelType.StableDiffusion2Base
        assert model_type == ModelType.Pipeline
        super(StableDiffusion15Model, self).__init__(
            model_path=model_path,
            base_model=BaseModelType.StableDiffusion2Base,
            model_type=ModelType.Pipeline,
        )

class StableDiffusion2Model(DiffusersModel):

    # TODO: str -> Path?
    # overwrite configs
    class DiffusersConfig(ModelConfigBase):
        format: Literal["diffusers"]
        vae: Optional[str] = Field(None)
        attention_upscale: bool = Field(True)

    class CheckpointConfig(ModelConfigBase):
        format: Literal["checkpoint"]
        vae: Optional[str] = Field(None)
        config: Optional[str] = Field(None)
        attention_upscale: bool = Field(True)


    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        # skip StableDiffusion15Model __init__
        assert base_model == BaseModelType.StableDiffusion2
        assert model_type == ModelType.Pipeline
        super().__init__(
            model_path=model_path,
            base_model=BaseModelType.StableDiffusion2,
            model_type=ModelType.Pipeline,
        )


# TODO: rework
DictConfig = dict
def _convert_ckpt_and_cache(self, mconfig: DictConfig) -> str:
    """
    Convert the checkpoint model indicated in mconfig into a
    diffusers, cache it to disk, and return Path to converted
    file. If already on disk then just returns Path.
    """
    app_config = InvokeAIAppConfig.get_config()
    weights = app_config.root_dir / mconfig.path
    config_file = app_config.root_dir / mconfig.config
    diffusers_path = app_config.converted_ckpts_dir / weights.stem

    # return cached version if it exists
    if diffusers_path.exists():
        return diffusers_path

    # TODO: I think that it more correctly to convert with embedded vae
    #       as if user will delete custom vae he will got not embedded but also custom vae
    #vae_ckpt_path, vae_model = self._get_vae_for_conversion(weights, mconfig)
    vae_ckpt_path, vae_model = None, None

    # to avoid circular import errors
    from ..convert_ckpt_to_diffusers import convert_ckpt_to_diffusers
    with SilenceWarnings():        
        convert_ckpt_to_diffusers(
            weights,
            diffusers_path,
            extract_ema=True,
            original_config_file=config_file,
            vae=vae_model,
            vae_path=str(app_config.root_dir / vae_ckpt_path) if vae_ckpt_path else None,
            scan_needed=True,
        )
    return diffusers_path
