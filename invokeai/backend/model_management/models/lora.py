import torch
from typing import Optional
from .base import (
    ModelBase,
    ModelConfigBase,
    BaseModelType,
    ModelType,
    SubModelType,
)
# TODO: naming
from ..lora import LoRAModel as LoRAModelRaw

class LoRAModel(ModelBase):
    #model_size: int

    class Config(ModelConfigBase):
        format: None

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

    @classmethod
    def save_to_config(cls) -> bool:
        return False

    @classmethod
    def detect_format(cls, path: str):
        if os.path.isdir(path):
            return "diffusers"
        else:
            return "lycoris"

    @staticmethod
    def convert_if_required(cls, model_path: str, dst_cache_path: str, config: Optional[dict]) -> str:
        if cls.detect_format(model_path) == "diffusers":
            # TODO: add diffusers lora when it stabilizes a bit
            raise NotImplementedError("Diffusers lora not supported")
        else:
            return model_path
