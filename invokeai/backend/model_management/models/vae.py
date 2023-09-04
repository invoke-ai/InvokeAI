import os
from enum import Enum
from pathlib import Path
from typing import Optional

import safetensors
import torch
from omegaconf import OmegaConf

from invokeai.app.services.config import InvokeAIAppConfig
from .base import (
    ModelBase,
    ModelConfigBase,
    BaseModelType,
    ModelType,
    SubModelType,
    ModelVariantType,
    EmptyConfigLoader,
    calc_model_size_by_fs,
    calc_model_size_by_data,
    classproperty,
    InvalidModelException,
    ModelNotFoundException,
)


class VaeModelFormat(str, Enum):
    Checkpoint = "checkpoint"
    Diffusers = "diffusers"


class VaeModel(ModelBase):
    # vae_class: Type
    # model_size: int

    class Config(ModelConfigBase):
        model_format: VaeModelFormat

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert model_type == ModelType.Vae
        super().__init__(model_path, base_model, model_type)

        try:
            config = EmptyConfigLoader.load_config(self.model_path, config_name="config.json")
            # config = json.loads(os.path.join(self.model_path, "config.json"))
        except Exception:
            raise Exception("Invalid vae model! (config.json not found or invalid)")

        try:
            vae_class_name = config.get("_class_name", "AutoencoderKL")
            self.vae_class = self._hf_definition_to_type(["diffusers", vae_class_name])
            self.model_size = calc_model_size_by_fs(self.model_path)
        except Exception:
            raise Exception("Invalid vae model! (Unkown vae type)")

    def get_size(self, child_type: Optional[SubModelType] = None):
        if child_type is not None:
            raise Exception("There is no child models in vae model")
        return self.model_size

    def get_model(
        self,
        torch_dtype: Optional[torch.dtype],
        child_type: Optional[SubModelType] = None,
    ):
        if child_type is not None:
            raise Exception("There is no child models in vae model")

        model = self.vae_class.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
        )
        # calc more accurate size
        self.model_size = calc_model_size_by_data(model)
        return model

    @classproperty
    def save_to_config(cls) -> bool:
        return False

    @classmethod
    def detect_format(cls, path: str):
        if not os.path.exists(path):
            raise ModelNotFoundException(f"Does not exist as local file: {path}")

        if os.path.isdir(path):
            if os.path.exists(os.path.join(path, "config.json")):
                return VaeModelFormat.Diffusers

        if os.path.isfile(path):
            if any([path.endswith(f".{ext}") for ext in ["safetensors", "ckpt", "pt"]]):
                return VaeModelFormat.Checkpoint

        raise InvalidModelException(f"Not a valid model: {path}")

    @classmethod
    def convert_if_required(
        cls,
        model_path: str,
        output_path: str,
        config: ModelConfigBase,  # empty config or config of parent model
        base_model: BaseModelType,
    ) -> str:
        if cls.detect_format(model_path) == VaeModelFormat.Checkpoint:
            return _convert_vae_ckpt_and_cache(
                weights_path=model_path,
                output_path=output_path,
                base_model=base_model,
                model_config=config,
            )
        else:
            return model_path


# TODO: rework
def _convert_vae_ckpt_and_cache(
    weights_path: str,
    output_path: str,
    base_model: BaseModelType,
    model_config: ModelConfigBase,
) -> str:
    """
    Convert the VAE indicated in mconfig into a diffusers AutoencoderKL
    object, cache it to disk, and return Path to converted
    file. If already on disk then just returns Path.
    """
    app_config = InvokeAIAppConfig.get_config()
    weights_path = app_config.root_dir / weights_path
    output_path = Path(output_path)

    """
    this size used only in when tiling enabled to separate input in tiles
    sizes in configs from stable diffusion githubs(1 and 2) set to 256
    on huggingface it:
    1.5 - 512
    1.5-inpainting - 256
    2-inpainting - 512
    2-depth - 256
    2-base - 512
    2 - 768
    2.1-base - 768
    2.1 - 768
    """
    image_size = 512

    # return cached version if it exists
    if output_path.exists():
        return output_path

    if base_model in {BaseModelType.StableDiffusion1, BaseModelType.StableDiffusion2}:
        from .stable_diffusion import _select_ckpt_config

        # all sd models use same vae settings
        config_file = _select_ckpt_config(base_model, ModelVariantType.Normal)
    else:
        raise Exception(f"Vae conversion not supported for model type: {base_model}")

    # this avoids circular import error
    from ..convert_ckpt_to_diffusers import convert_ldm_vae_to_diffusers

    if weights_path.suffix == ".safetensors":
        checkpoint = safetensors.torch.load_file(weights_path, device="cpu")
    else:
        checkpoint = torch.load(weights_path, map_location="cpu")

    # sometimes weights are hidden under "state_dict", and sometimes not
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    config = OmegaConf.load(app_config.root_path / config_file)

    vae_model = convert_ldm_vae_to_diffusers(
        checkpoint=checkpoint,
        vae_config=config,
        image_size=image_size,
    )
    vae_model.save_pretrained(output_path, safe_serialization=True)
    return output_path
