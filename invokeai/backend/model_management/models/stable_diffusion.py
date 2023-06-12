import os
import json
import torch
import safetensors.torch
from pydantic import Field
from typing import Literal, Optional
from .base import (
    ModelBase,
    ModelConfigBase,
    BaseModelType,
    ModelType,
    SubModelType,
    VariantType,
    DiffusersModel,
)
from invokeai.app.services.config import InvokeAIAppConfig

ModelVariantType = VariantType # TODO:


# TODO: how to name properly
class StableDiffusion15Model(DiffusersModel):

    # TODO: str -> Path?
    class DiffusersConfig(ModelConfigBase):
        format: Literal["diffusers"]
        vae: Optional[str] = Field(None)
        variant: ModelVariantType

    class CheckpointConfig(ModelConfigBase):
        format: Literal["checkpoint"]
        vae: Optional[str] = Field(None)
        config: Optional[str] = Field(None)
        variant: ModelVariantType


    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert base_model == BaseModelType.StableDiffusion1_5
        assert model_type == ModelType.Pipeline
        super().__init__(
            model_path=model_path,
            base_model=BaseModelType.StableDiffusion1_5,
            model_type=ModelType.Pipeline,
        )

    @staticmethod
    def _fast_safetensors_reader(path: str):
        checkpoint = dict()
        device = torch.device("meta")
        with open(path, "rb") as f:
            definition_len = int.from_bytes(f.read(8), 'little')
            definition_json = f.read(definition_len)
            definition = json.loads(definition_json)

            if "__metadata__" in definition and definition["__metadata__"].get("format", "pt") not in {"pt", "torch", "pytorch"}:
                raise Exception("Supported only pytorch safetensors files")
            definition.pop("__metadata__", None)

            for key, info in definition.items():
                dtype = {
                    "I8": torch.int8,
                    "I16": torch.int16,
                    "I32": torch.int32,
                    "I64": torch.int64,
                    "F16": torch.float16,
                    "F32": torch.float32,
                    "F64": torch.float64,
                }[info["dtype"]]

                checkpoint[key] = torch.empty(info["shape"], dtype=dtype, device=device)

        return checkpoint


    @classmethod
    def read_checkpoint_meta(cls, path: str):
        if path.endswith(".safetensors"):
            try:
                checkpoint = cls._fast_safetensors_reader(path)
            except:
                checkpoint = safetensors.torch.load_file(path, device="cpu") # TODO: create issue for support "meta"?
        else:
            checkpoint = torch.load(path, map_location=torch.device("meta"))
        return checkpoint

    @classmethod
    def build_config(cls, **kwargs):
        if "format" not in kwargs:
            kwargs["format"] = cls.detect_format(kwargs["path"])

        if "variant" not in kwargs:
            if kwargs["format"] == "checkpoint":
                if "config" in kwargs:
                    ckpt_config = OmegaConf.load(kwargs["config"])
                    in_channels = ckpt_config["model"]["params"]["unet_config"]["params"]["in_channels"]

                else:
                    checkpoint = cls.read_checkpoint_meta(kwargs["path"])
                    checkpoint = checkpoint.get('state_dict', checkpoint)
                    in_channels = checkpoint["model.diffusion_model.input_blocks.0.0.weight"].shape[1]

            elif kwargs["format"] == "diffusers":
                unet_config_path = os.path.join(kwargs["path"], "unet", "config.json")
                if os.path.exists(unet_config_path):
                    unet_config = json.loads(unet_config_path)
                    in_channels = unet_config['in_channels']

                else:
                    raise Exception("Not supported stable diffusion diffusers format(possibly onnx?)")

            else:
                raise NotImplementedError(f"Unknown stable diffusion format: {kwargs['format']}")

            if in_channels == 9:
                kwargs["variant"] = ModelVariantType.Inpaint
            elif in_channels == 5:
                kwargs["variant"] = ModelVariantType.Depth
            elif in_channels == 4:
                kwargs["variant"] = ModelVariantType.Normal
            else:
                raise Exception("Unkown stable diffusion model format")


        return super().build_config(**kwargs)

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
