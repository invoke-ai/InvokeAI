import os
import torch
from enum import Enum
from typing import Optional, Union, Literal
from .base import (
    ModelBase,
    ModelConfigBase,
    BaseModelType,
    ModelType,
    SubModelType,
    classproperty,
    InvalidModelException,
    ModelNotFoundException,
)

# TODO: naming
from ..lora import LoRAModel as LoRAModelRaw


class LoRAModelFormat(str, Enum):
    LyCORIS = "lycoris"
    Diffusers = "diffusers"


class LoRAModel(ModelBase):
    # model_size: int

    class Config(ModelConfigBase):
        model_format: LoRAModelFormat  # TODO:

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert model_type == ModelType.Lora
        super().__init__(model_path, base_model, model_type)

        self.model_size = os.path.getsize(self.model_path)

    def get_size(self, child_type: Optional[SubModelType] = None):
        if child_type is not None:
            raise Exception("There is no child models in lora")
        return self.model_size

    def get_model(
        self,
        torch_dtype: Optional[torch.dtype],
        child_type: Optional[SubModelType] = None,
    ):
        if child_type is not None:
            raise Exception("There is no child models in lora")

        model = LoRAModelRaw.from_checkpoint(
            file_path=self.model_path,
            dtype=torch_dtype,
        )

        self.model_size = model.calc_size()
        return model

    @classproperty
    def save_to_config(cls) -> bool:
        return False

    @classmethod
    def detect_format(cls, path: str):
        if not os.path.exists(path):
            raise ModelNotFoundException()

        if os.path.isdir(path):
            if os.path.exists(os.path.join(path, "pytorch_lora_weights.bin")):
                return LoRAModelFormat.Diffusers

        if os.path.isfile(path):
            if any([path.endswith(f".{ext}") for ext in ["safetensors", "ckpt", "pt"]]):
                return LoRAModelFormat.LyCORIS

        raise InvalidModelException(f"Not a valid model: {path}")

    @classmethod
    def convert_if_required(
        cls,
        model_path: str,
        output_path: str,
        config: ModelConfigBase,
        base_model: BaseModelType,
    ) -> str:
        if cls.detect_format(model_path) == LoRAModelFormat.Diffusers:
            # TODO: add diffusers lora when it stabilizes a bit
            raise NotImplementedError("Diffusers lora not supported")
        else:
            return model_path
