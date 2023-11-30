import os
from enum import Enum
from typing import Literal, Optional

import torch
from diffusers import T2IAdapter

from invokeai.backend.model_management.models.base import (
    BaseModelType,
    EmptyConfigLoader,
    InvalidModelException,
    ModelBase,
    ModelConfigBase,
    ModelNotFoundException,
    ModelType,
    SubModelType,
    calc_model_size_by_data,
    calc_model_size_by_fs,
    classproperty,
)


class T2IAdapterModelFormat(str, Enum):
    Diffusers = "diffusers"


class T2IAdapterModel(ModelBase):
    class DiffusersConfig(ModelConfigBase):
        model_format: Literal[T2IAdapterModelFormat.Diffusers]

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert model_type == ModelType.T2IAdapter
        super().__init__(model_path, base_model, model_type)

        config = EmptyConfigLoader.load_config(self.model_path, config_name="config.json")

        model_class_name = config.get("_class_name", None)
        if model_class_name not in {"T2IAdapter"}:
            raise InvalidModelException(f"Invalid T2I-Adapter model. Unknown _class_name: '{model_class_name}'.")

        self.model_class = self._hf_definition_to_type(["diffusers", model_class_name])
        self.model_size = calc_model_size_by_fs(self.model_path)

    def get_size(self, child_type: Optional[SubModelType] = None):
        if child_type is not None:
            raise ValueError(f"T2I-Adapters do not have child models. Invalid child type: '{child_type}'.")
        return self.model_size

    def get_model(
        self,
        torch_dtype: Optional[torch.dtype],
        child_type: Optional[SubModelType] = None,
    ) -> T2IAdapter:
        if child_type is not None:
            raise ValueError(f"T2I-Adapters do not have child models. Invalid child type: '{child_type}'.")

        model = None
        for variant in ["fp16", None]:
            try:
                model = self.model_class.from_pretrained(
                    self.model_path,
                    torch_dtype=torch_dtype,
                    variant=variant,
                )
                break
            except Exception:
                pass
        if not model:
            raise ModelNotFoundException()

        # Calculate a more accurate size after loading the model into memory.
        self.model_size = calc_model_size_by_data(model)
        return model

    @classproperty
    def save_to_config(cls) -> bool:
        return False

    @classmethod
    def detect_format(cls, path: str):
        if not os.path.exists(path):
            raise ModelNotFoundException(f"Model not found at '{path}'.")

        if os.path.isdir(path):
            if os.path.exists(os.path.join(path, "config.json")):
                return T2IAdapterModelFormat.Diffusers

        raise InvalidModelException(f"Unsupported T2I-Adapter format: '{path}'.")

    @classmethod
    def convert_if_required(
        cls,
        model_path: str,
        output_path: str,
        config: ModelConfigBase,
        base_model: BaseModelType,
    ) -> str:
        format = cls.detect_format(model_path)
        if format == T2IAdapterModelFormat.Diffusers:
            return model_path
        else:
            raise ValueError(f"Unsupported format: '{format}'.")
