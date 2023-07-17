import os
import torch
from enum import Enum
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
    classproperty,
    InvalidModelException,
    ModelNotFoundException,
)

class ControlNetModelFormat(str, Enum):
    Checkpoint = "checkpoint"
    Diffusers = "diffusers"

class ControlNetModel(ModelBase):
    #model_class: Type
    #model_size: int

    class Config(ModelConfigBase):
        model_format: ControlNetModelFormat

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert model_type == ModelType.ControlNet
        super().__init__(model_path, base_model, model_type)

        try:
            config = EmptyConfigLoader.load_config(self.model_path, config_name="config.json")
            #config = json.loads(os.path.join(self.model_path, "config.json"))
        except:
            raise Exception("Invalid controlnet model! (config.json not found or invalid)")

        model_class_name = config.get("_class_name", None)
        if model_class_name not in {"ControlNetModel"}:
            raise Exception(f"Invalid ControlNet model! Unknown _class_name: {model_class_name}")

        try:
            self.model_class = self._hf_definition_to_type(["diffusers", model_class_name])
            self.model_size = calc_model_size_by_fs(self.model_path)
        except:
            raise Exception("Invalid ControlNet model!")

    def get_size(self, child_type: Optional[SubModelType] = None):
        if child_type is not None:
            raise Exception("There is no child models in controlnet model")
        return self.model_size

    def get_model(
        self,
        torch_dtype: Optional[torch.dtype],
        child_type: Optional[SubModelType] = None,
    ):
        if child_type is not None:
            raise Exception("There is no child models in controlnet model")

        model = None
        for variant in ['fp16',None]:
            try:
                model = self.model_class.from_pretrained(
                    self.model_path,
                    torch_dtype=torch_dtype,
                    variant=variant,
                )
                break
            except:
                pass
        if not model:
            raise ModelNotFoundException()
        
        # calc more accurate size
        self.model_size = calc_model_size_by_data(model)
        return model

    @classproperty
    def save_to_config(cls) -> bool:
        return False

    @classmethod
    def detect_format(cls, path: str):
        if not os.path.exists(path):
            raise ModelNotFoundException()

        if os.path.isdir(path):
            if os.path.exists(os.path.join(path, "config.json")):
                return ControlNetModelFormat.Diffusers

        if os.path.isfile(path):
            if any([path.endswith(f".{ext}") for ext in ["safetensors", "ckpt", "pt", "pth"]]):
                return ControlNetModelFormat.Checkpoint

        raise InvalidModelException(f"Not a valid model: {path}")

    @classmethod
    def convert_if_required(
        cls,
        model_path: str,
        output_path: str,
        config: ModelConfigBase, # empty config or config of parent model
        base_model: BaseModelType,
    ) -> str:
        if cls.detect_format(model_path) != ControlNetModelFormat.Diffusers:
            raise NotImplementedError("Checkpoint controlnet models currently unsupported")
        else:
            return model_path
