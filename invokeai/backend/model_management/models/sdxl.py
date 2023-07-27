import os
import json
import invokeai.backend.util.logging as logger
from enum import Enum
from pydantic import Field
from typing import Literal, Optional
from .base import (
    ModelConfigBase,
    BaseModelType,
    ModelType,
    ModelVariantType,
    DiffusersModel,
    read_checkpoint_meta,
    classproperty,
)
from omegaconf import OmegaConf


class StableDiffusionXLModelFormat(str, Enum):
    Checkpoint = "checkpoint"
    Diffusers = "diffusers"


class StableDiffusionXLModel(DiffusersModel):
    # TODO: check that configs overwriten properly
    class DiffusersConfig(ModelConfigBase):
        model_format: Literal[StableDiffusionXLModelFormat.Diffusers]
        vae: Optional[str] = Field(None)
        variant: ModelVariantType

    class CheckpointConfig(ModelConfigBase):
        model_format: Literal[StableDiffusionXLModelFormat.Checkpoint]
        vae: Optional[str] = Field(None)
        config: str
        variant: ModelVariantType

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert base_model in {BaseModelType.StableDiffusionXL, BaseModelType.StableDiffusionXLRefiner}
        assert model_type == ModelType.Main
        super().__init__(
            model_path=model_path,
            base_model=BaseModelType.StableDiffusionXL,
            model_type=ModelType.Main,
        )

    @classmethod
    def probe_config(cls, path: str, **kwargs):
        model_format = cls.detect_format(path)
        ckpt_config_path = kwargs.get("config", None)
        if model_format == StableDiffusionXLModelFormat.Checkpoint:
            if ckpt_config_path:
                ckpt_config = OmegaConf.load(ckpt_config_path)
                in_channels = ckpt_config["model"]["params"]["unet_config"]["params"]["in_channels"]

            else:
                checkpoint = read_checkpoint_meta(path)
                checkpoint = checkpoint.get("state_dict", checkpoint)
                in_channels = checkpoint["model.diffusion_model.input_blocks.0.0.weight"].shape[1]

        elif model_format == StableDiffusionXLModelFormat.Diffusers:
            unet_config_path = os.path.join(path, "unet", "config.json")
            if os.path.exists(unet_config_path):
                with open(unet_config_path, "r") as f:
                    unet_config = json.loads(f.read())
                in_channels = unet_config["in_channels"]

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

        if ckpt_config_path is None:
            # TO DO: implement picking
            pass

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
            return StableDiffusionXLModelFormat.Diffusers
        else:
            return StableDiffusionXLModelFormat.Checkpoint

    @classmethod
    def convert_if_required(
        cls,
        model_path: str,
        output_path: str,
        config: ModelConfigBase,
        base_model: BaseModelType,
    ) -> str:
        # The convert script adapted from the diffusers package uses
        # strings for the base model type. To avoid making too many
        # source code changes, we simply translate here
        if isinstance(config, cls.CheckpointConfig):
            from invokeai.backend.model_management.models.stable_diffusion import _convert_ckpt_and_cache

            return _convert_ckpt_and_cache(
                version=base_model,
                model_config=config,
                output_path=output_path,
                use_safetensors=False,  # corrupts sdxl models for some reason
            )
        else:
            return model_path
