import os
import json
from enum import Enum
from pydantic import Field
from pathlib import Path
from typing import Literal, Optional, Union
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline
from .base import (
    ModelConfigBase,
    BaseModelType,
    ModelType,
    ModelVariantType,
    DiffusersModel,
    SilenceWarnings,
    read_checkpoint_meta,
    classproperty,
    InvalidModelException,
    ModelNotFoundException,
)
from .sdxl import StableDiffusionXLModel
import invokeai.backend.util.logging as logger
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
        config: str
        variant: ModelVariantType

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert base_model == BaseModelType.StableDiffusion1
        assert model_type == ModelType.Main
        super().__init__(
            model_path=model_path,
            base_model=BaseModelType.StableDiffusion1,
            model_type=ModelType.Main,
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
                checkpoint = checkpoint.get("state_dict", checkpoint)
                in_channels = checkpoint["model.diffusion_model.input_blocks.0.0.weight"].shape[1]

        elif model_format == StableDiffusion1ModelFormat.Diffusers:
            unet_config_path = os.path.join(path, "unet", "config.json")
            if os.path.exists(unet_config_path):
                with open(unet_config_path, "r") as f:
                    unet_config = json.loads(f.read())
                in_channels = unet_config["in_channels"]

            else:
                raise NotImplementedError(f"{path} is not a supported stable diffusion diffusers format")

        else:
            raise NotImplementedError(f"Unknown stable diffusion 1.* format: {model_format}")

        if in_channels == 9:
            variant = ModelVariantType.Inpaint
        elif in_channels == 4:
            variant = ModelVariantType.Normal
        else:
            raise Exception("Unkown stable diffusion 1.* model format")

        if ckpt_config_path is None:
            ckpt_config_path = _select_ckpt_config(BaseModelType.StableDiffusion1, variant)

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
        if not os.path.exists(model_path):
            raise ModelNotFoundException()

        if os.path.isdir(model_path):
            if os.path.exists(os.path.join(model_path, "model_index.json")):
                return StableDiffusion1ModelFormat.Diffusers

        if os.path.isfile(model_path):
            if any([model_path.endswith(f".{ext}") for ext in ["safetensors", "ckpt", "pt"]]):
                return StableDiffusion1ModelFormat.Checkpoint

        raise InvalidModelException(f"Not a valid model: {model_path}")

    @classmethod
    def convert_if_required(
        cls,
        model_path: str,
        output_path: str,
        config: ModelConfigBase,
        base_model: BaseModelType,
    ) -> str:
        if isinstance(config, cls.CheckpointConfig):
            return _convert_ckpt_and_cache(
                version=BaseModelType.StableDiffusion1,
                model_config=config,
                load_safety_checker=False,
                output_path=output_path,
            )
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

    class CheckpointConfig(ModelConfigBase):
        model_format: Literal[StableDiffusion2ModelFormat.Checkpoint]
        vae: Optional[str] = Field(None)
        config: str
        variant: ModelVariantType

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert base_model == BaseModelType.StableDiffusion2
        assert model_type == ModelType.Main
        super().__init__(
            model_path=model_path,
            base_model=BaseModelType.StableDiffusion2,
            model_type=ModelType.Main,
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
                checkpoint = checkpoint.get("state_dict", checkpoint)
                in_channels = checkpoint["model.diffusion_model.input_blocks.0.0.weight"].shape[1]

        elif model_format == StableDiffusion2ModelFormat.Diffusers:
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
            ckpt_config_path = _select_ckpt_config(BaseModelType.StableDiffusion2, variant)

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
        if not os.path.exists(model_path):
            raise ModelNotFoundException()

        if os.path.isdir(model_path):
            if os.path.exists(os.path.join(model_path, "model_index.json")):
                return StableDiffusion2ModelFormat.Diffusers

        if os.path.isfile(model_path):
            if any([model_path.endswith(f".{ext}") for ext in ["safetensors", "ckpt", "pt"]]):
                return StableDiffusion2ModelFormat.Checkpoint

        raise InvalidModelException(f"Not a valid model: {model_path}")

    @classmethod
    def convert_if_required(
        cls,
        model_path: str,
        output_path: str,
        config: ModelConfigBase,
        base_model: BaseModelType,
    ) -> str:
        if isinstance(config, cls.CheckpointConfig):
            return _convert_ckpt_and_cache(
                version=BaseModelType.StableDiffusion2,
                model_config=config,
                output_path=output_path,
            )
        else:
            return model_path


# TODO: rework
# pass precision - currently defaulting to fp16
def _convert_ckpt_and_cache(
    version: BaseModelType,
    model_config: Union[
        StableDiffusion1Model.CheckpointConfig,
        StableDiffusion2Model.CheckpointConfig,
        StableDiffusionXLModel.CheckpointConfig,
    ],
    output_path: str,
    use_save_model: bool = False,
    **kwargs,
) -> str:
    """
    Convert the checkpoint model indicated in mconfig into a
    diffusers, cache it to disk, and return Path to converted
    file. If already on disk then just returns Path.
    """
    app_config = InvokeAIAppConfig.get_config()

    weights = app_config.models_path / model_config.path
    config_file = app_config.root_path / model_config.config
    output_path = Path(output_path)
    variant = model_config.variant
    pipeline_class = StableDiffusionInpaintPipeline if variant == "inpaint" else StableDiffusionPipeline

    # return cached version if it exists
    if output_path.exists():
        return output_path

    # to avoid circular import errors
    from ..convert_ckpt_to_diffusers import convert_ckpt_to_diffusers
    from ...util.devices import choose_torch_device, torch_dtype

    model_base_to_model_type = {
        BaseModelType.StableDiffusion1: "FrozenCLIPEmbedder",
        BaseModelType.StableDiffusion2: "FrozenOpenCLIPEmbedder",
        BaseModelType.StableDiffusionXL: "SDXL",
        BaseModelType.StableDiffusionXLRefiner: "SDXL-Refiner",
    }
    logger.info(f"Converting {weights} to diffusers format")
    with SilenceWarnings():
        convert_ckpt_to_diffusers(
            weights,
            output_path,
            model_type=model_base_to_model_type[version],
            model_version=version,
            model_variant=model_config.variant,
            original_config_file=config_file,
            extract_ema=True,
            scan_needed=True,
            pipeline_class=pipeline_class,
            from_safetensors=weights.suffix == ".safetensors",
            precision=torch_dtype(choose_torch_device()),
            **kwargs,
        )
    return output_path


def _select_ckpt_config(version: BaseModelType, variant: ModelVariantType):
    ckpt_configs = {
        BaseModelType.StableDiffusion1: {
            ModelVariantType.Normal: "v1-inference.yaml",
            ModelVariantType.Inpaint: "v1-inpainting-inference.yaml",
        },
        BaseModelType.StableDiffusion2: {
            ModelVariantType.Normal: "v2-inference-v.yaml",  # best guess, as we can't differentiate with base(512)
            ModelVariantType.Inpaint: "v2-inpainting-inference.yaml",
            ModelVariantType.Depth: "v2-midas-inference.yaml",
        },
        BaseModelType.StableDiffusionXL: {
            ModelVariantType.Normal: "sd_xl_base.yaml",
            ModelVariantType.Inpaint: None,
            ModelVariantType.Depth: None,
        },
        BaseModelType.StableDiffusionXLRefiner: {
            ModelVariantType.Normal: "sd_xl_refiner.yaml",
            ModelVariantType.Inpaint: None,
            ModelVariantType.Depth: None,
        },
    }

    app_config = InvokeAIAppConfig.get_config()
    try:
        config_path = app_config.legacy_conf_path / ckpt_configs[version][variant]
        if config_path.is_relative_to(app_config.root_path):
            config_path = config_path.relative_to(app_config.root_path)
        return str(config_path)

    except Exception:
        return None
