import os
import json
from enum import Enum
from pydantic import Field
from pathlib import Path
from typing import Literal, Optional, Union
from .base import (
    ModelBase,
    ModelConfigBase,
    BaseModelType,
    ModelType,
    SubModelType,
    ModelVariantType,
    DiffusersModel,
    SchedulerPredictionType,
    SilenceWarnings,
    read_checkpoint_meta,
    classproperty,
)
from invokeai.app.services.config import InvokeAIAppConfig
from omegaconf import OmegaConf

class StableDiffusion1ModelFormat(str, Enum):
    Checkpoint = "checkpoint"
    Diffusers = "diffusers"

class StableDiffusion1Model(DiffusersModel):

    class DiffusersConfig(ModelConfigBase):
        model_format: Literal[StableDiffusion1ModelFormat.Diffusers]
        vae: Optional[str] = Field(None)
        variant: ModelVariantType

    class CheckpointConfig(ModelConfigBase):
        model_format: Literal[StableDiffusion1ModelFormat.Checkpoint]
        vae: Optional[str] = Field(None)
        config: Optional[str] = Field(None)
        variant: ModelVariantType


    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert base_model == BaseModelType.StableDiffusion1
        assert model_type == ModelType.Pipeline
        super().__init__(
            model_path=model_path,
            base_model=BaseModelType.StableDiffusion1,
            model_type=ModelType.Pipeline,
        )

    @classmethod
    def probe_config(cls, path: str, **kwargs):
        model_format = cls.detect_format(path)
        ckpt_config_path = kwargs.get("config", None)
        if model_format == StableDiffusion1ModelFormat.Checkpoint:
            if ckpt_config_path:
                ckpt_config = OmegaConf.load(ckpt_config_path)
                ckpt_config["model"]["params"]["unet_config"]["params"]["in_channels"]

            else:
                checkpoint = read_checkpoint_meta(path)
                checkpoint = checkpoint.get('state_dict', checkpoint)
                in_channels = checkpoint["model.diffusion_model.input_blocks.0.0.weight"].shape[1]

        elif model_format == StableDiffusion1ModelFormat.Diffusers:
            unet_config_path = os.path.join(path, "unet", "config.json")
            if os.path.exists(unet_config_path):
                with open(unet_config_path, "r") as f:
                    unet_config = json.loads(f.read())
                in_channels = unet_config['in_channels']

            else:
                raise Exception("Not supported stable diffusion diffusers format(possibly onnx?)")

        else:
            raise NotImplementedError(f"Unknown stable diffusion 1.* format: {model_format}")

        if in_channels == 9:
            variant = ModelVariantType.Inpaint
        elif in_channels == 4:
            variant = ModelVariantType.Normal
        else:
            raise Exception("Unkown stable diffusion 1.* model format")


        return cls.create_config(
            path=path,
            model_format=model_format,

            config=ckpt_config_path,
            variant=variant,
        )

    @classproperty
    def save_to_config(cls) -> bool:
        return True

    @classmethod
    def detect_format(cls, model_path: str):
        if os.path.isdir(model_path):
            return StableDiffusion1ModelFormat.Diffusers
        else:
            return StableDiffusion1ModelFormat.Checkpoint

    @classmethod
    def convert_if_required(
        cls,
        model_path: str,
        output_path: str,
        config: ModelConfigBase,
        base_model: BaseModelType,
    ) -> str:
        assert model_path == config.path

        if isinstance(config, cls.CheckpointConfig):
            return _convert_ckpt_and_cache(
                version=BaseModelType.StableDiffusion1,
                model_config=config,
                output_path=output_path,
            ) # TODO: args
        else:
            return model_path

class StableDiffusion2ModelFormat(str, Enum):
    Checkpoint = "checkpoint"
    Diffusers = "diffusers"

class StableDiffusion2Model(DiffusersModel):

    # TODO: check that configs overwriten properly
    class DiffusersConfig(ModelConfigBase):
        model_format: Literal[StableDiffusion2ModelFormat.Diffusers]
        vae: Optional[str] = Field(None)
        variant: ModelVariantType
        prediction_type: SchedulerPredictionType
        upcast_attention: bool

    class CheckpointConfig(ModelConfigBase):
        model_format: Literal[StableDiffusion2ModelFormat.Checkpoint]
        vae: Optional[str] = Field(None)
        config: Optional[str] = Field(None)
        variant: ModelVariantType
        prediction_type: SchedulerPredictionType
        upcast_attention: bool


    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert base_model == BaseModelType.StableDiffusion2
        assert model_type == ModelType.Pipeline
        super().__init__(
            model_path=model_path,
            base_model=BaseModelType.StableDiffusion2,
            model_type=ModelType.Pipeline,
        )

    @classmethod
    def probe_config(cls, path: str, **kwargs):
        model_format = cls.detect_format(path)
        ckpt_config_path = kwargs.get("config", None)
        if model_format == StableDiffusion2ModelFormat.Checkpoint:
            if ckpt_config_path:
                ckpt_config = OmegaConf.load(ckpt_config_path)
                ckpt_config["model"]["params"]["unet_config"]["params"]["in_channels"]

            else:
                checkpoint = read_checkpoint_meta(path)
                checkpoint = checkpoint.get('state_dict', checkpoint)
                in_channels = checkpoint["model.diffusion_model.input_blocks.0.0.weight"].shape[1]

        elif model_format == StableDiffusion2ModelFormat.Diffusers:
            unet_config_path = os.path.join(path, "unet", "config.json")
            if os.path.exists(unet_config_path):
                with open(unet_config_path, "r") as f:
                    unet_config = json.loads(f.read())
                in_channels = unet_config['in_channels']

            else:
                raise Exception("Not supported stable diffusion diffusers format(possibly onnx?)")

        else:
            raise NotImplementedError(f"Unknown stable diffusion 2.* format: {model_format}")

        if in_channels == 9:
            variant = ModelVariantType.Inpaint
        elif in_channels == 5:
            variant = ModelVariantType.Depth
        elif in_channels == 4:
            variant = ModelVariantType.Normal
        else:
            raise Exception("Unkown stable diffusion 2.* model format")

        if variant == ModelVariantType.Normal:
            prediction_type = SchedulerPredictionType.VPrediction
            upcast_attention = True

        else:
            prediction_type = SchedulerPredictionType.Epsilon
            upcast_attention = False

        return cls.create_config(
            path=path,
            model_format=model_format,

            config=ckpt_config_path,
            variant=variant,
            prediction_type=prediction_type,
            upcast_attention=upcast_attention,
        )

    @classproperty
    def save_to_config(cls) -> bool:
        return True

    @classmethod
    def detect_format(cls, model_path: str):
        if os.path.isdir(model_path):
            return StableDiffusion2ModelFormat.Diffusers
        else:
            return StableDiffusion2ModelFormat.Checkpoint

    @classmethod
    def convert_if_required(
        cls,
        model_path: str,
        output_path: str,
        config: ModelConfigBase,
        base_model: BaseModelType,
    ) -> str:
        assert model_path == config.path

        if isinstance(config, cls.CheckpointConfig):
            return _convert_ckpt_and_cache(
                version=BaseModelType.StableDiffusion2,
                model_config=config,
                output_path=output_path,
            ) # TODO: args
        else:
            return model_path

def _select_ckpt_config(version: BaseModelType, variant: ModelVariantType):
    ckpt_configs = {
        BaseModelType.StableDiffusion1: {
            ModelVariantType.Normal: "v1-inference.yaml",
            ModelVariantType.Inpaint: "v1-inpainting-inference.yaml",
        },
        BaseModelType.StableDiffusion2: {
            # code further will manually set upcast_attention and v_prediction
            ModelVariantType.Normal: "v2-inference.yaml",
            ModelVariantType.Inpaint: "v2-inpainting-inference.yaml",
            ModelVariantType.Depth: "v2-midas-inference.yaml",
        }
    }

    try:
        # TODO: path
        #model_config.config = app_config.config_dir / "stable-diffusion" / ckpt_configs[version][model_config.variant]
        #return InvokeAIAppConfig.get_config().legacy_conf_dir / ckpt_configs[version][variant]
        return InvokeAIAppConfig.get_config().root_dir / "configs" / "stable-diffusion" / ckpt_configs[version][variant]
            
    except:
        return None


# TODO: rework
def _convert_ckpt_and_cache(
    version: BaseModelType,
    model_config: Union[StableDiffusion1Model.CheckpointConfig, StableDiffusion2Model.CheckpointConfig],
    output_path: str,
) -> str:
    """
    Convert the checkpoint model indicated in mconfig into a
    diffusers, cache it to disk, and return Path to converted
    file. If already on disk then just returns Path.
    """
    app_config = InvokeAIAppConfig.get_config()

    if model_config.config is None:
        model_config.config = _select_ckpt_config(version, model_config.variant)
        if model_config.config is None:
            raise Exception(f"Model variant {model_config.variant} not supported for {version}")


    weights = app_config.root_dir / model_config.path
    config_file = app_config.root_dir / model_config.config
    output_path = Path(output_path)

    if version == BaseModelType.StableDiffusion1:
        upcast_attention = False
        prediction_type = SchedulerPredictionType.Epsilon

    elif version == BaseModelType.StableDiffusion2:
        upcast_attention = model_config.upcast_attention
        prediction_type = model_config.prediction_type

    else:
        raise Exception(f"Unknown model provided: {version}")


    # return cached version if it exists
    if output_path.exists():
        return output_path

    # TODO: I think that it more correctly to convert with embedded vae
    #       as if user will delete custom vae he will got not embedded but also custom vae
    #vae_ckpt_path, vae_model = self._get_vae_for_conversion(weights, mconfig)

    # to avoid circular import errors
    from ..convert_ckpt_to_diffusers import convert_ckpt_to_diffusers
    with SilenceWarnings():        
        convert_ckpt_to_diffusers(
            weights,
            output_path,
            model_version=version,
            model_variant=model_config.variant,
            original_config_file=config_file,
            extract_ema=True,
            upcast_attention=upcast_attention,
            prediction_type=prediction_type,
            scan_needed=True,
            model_root=app_config.models_path,
        )
    return output_path
