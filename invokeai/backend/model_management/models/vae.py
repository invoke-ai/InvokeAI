import os
import torch
from typing import Optional
from .base import (
    ModelBase,
    ModelConfigBase,
    BaseModelType,
    ModelType,
    SubModelType,
    EmptyConfigLoader,
    calc_model_size_by_fs,
    calc_model_size_by_data,
)
from invokeai.app.services.config import InvokeAIAppConfig

class VaeModel(ModelBase):
    #vae_class: Type
    #model_size: int

    class Config(ModelConfigBase):
        format: None

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert model_type == ModelType.Vae
        super().__init__(model_path, base_model, model_type)

        try:
            config = EmptyConfigLoader.load_config(self.model_path, config_name="config.json")
            #config = json.loads(os.path.join(self.model_path, "config.json"))
        except:
            raise Exception("Invalid vae model! (config.json not found or invalid)")

        try:
            vae_class_name = config.get("_class_name", "AutoencoderKL")
            self.vae_class = self._hf_definition_to_type(["diffusers", vae_class_name])
            self.model_size = calc_model_size_by_fs(self.model_path)
        except:
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

    @classmethod
    def save_to_config(cls) -> bool:
        return False

    @classmethod
    def detect_format(cls, path: str):
        if os.path.isdir(path):
            return "diffusers"
        else:
            return "checkpoint"

    @classmethod
    def convert_if_required(cls, model_path: str, dst_cache_path: str, config: Optional[dict]) -> str:
        if cls.detect_format(model_path) != "diffusers":
            # TODO:
            #_convert_vae_ckpt_and_cache
            raise NotImplementedError("TODO: vae convert")
        else:
            return model_path

# TODO: rework
DictConfig = dict
def _convert_vae_ckpt_and_cache(self, mconfig: DictConfig) -> str:
    """
    Convert the VAE indicated in mconfig into a diffusers AutoencoderKL
    object, cache it to disk, and return Path to converted
    file. If already on disk then just returns Path.
    """
    app_config = InvokeAIAppConfig.get_config()
    root = app_config.root_dir
    weights_file = root / mconfig.path
    config_file = root / mconfig.config
    diffusers_path = app_config.converted_ckpts_dir / weights_file.stem
    image_size = mconfig.get('width') or mconfig.get('height') or 512
        
    # return cached version if it exists
    if diffusers_path.exists():
        return diffusers_path

    # this avoids circular import error
    from .convert_ckpt_to_diffusers import convert_ldm_vae_to_diffusers
    if weights_file.suffix == '.safetensors':
        checkpoint = safetensors.torch.load_file(weights_file)
    else:
        checkpoint = torch.load(weights_file, map_location="cpu")

    # sometimes weights are hidden under "state_dict", and sometimes not
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    config = OmegaConf.load(config_file)

    vae_model = convert_ldm_vae_to_diffusers(
        checkpoint = checkpoint,
        vae_config = config,
        image_size = image_size
    )
    vae_model.save_pretrained(
        diffusers_path,
        safe_serialization=is_safetensors_available()
    )
    return diffusers_path
