import os
from enum import Enum
from typing import Literal, Optional

import torch
from transformers import CLIPVisionModelWithProjection

from invokeai.backend.model_management.models.base import (
    BaseModelType,
    InvalidModelException,
    ModelBase,
    ModelConfigBase,
    ModelType,
    SubModelType,
    calc_model_size_by_data,
    calc_model_size_by_fs,
    classproperty,
)


class CLIPVisionModelFormat(str, Enum):
    Diffusers = "diffusers"


class CLIPVisionModel(ModelBase):
    class DiffusersConfig(ModelConfigBase):
        model_format: Literal[CLIPVisionModelFormat.Diffusers]

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert model_type == ModelType.CLIPVision
        super().__init__(model_path, base_model, model_type)

        self.model_size = calc_model_size_by_fs(self.model_path)

    @classmethod
    def detect_format(cls, path: str) -> str:
        if not os.path.exists(path):
            raise ModuleNotFoundError(f"No CLIP Vision model at path '{path}'.")

        if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
            return CLIPVisionModelFormat.Diffusers

        raise InvalidModelException(f"Unexpected CLIP Vision model format: {path}")

    @classproperty
    def save_to_config(cls) -> bool:
        return True

    def get_size(self, child_type: Optional[SubModelType] = None) -> int:
        if child_type is not None:
            raise ValueError("There are no child models in a CLIP Vision model.")

        return self.model_size

    def get_model(
        self,
        torch_dtype: Optional[torch.dtype],
        child_type: Optional[SubModelType] = None,
    ) -> CLIPVisionModelWithProjection:
        if child_type is not None:
            raise ValueError("There are no child models in a CLIP Vision model.")

        model = CLIPVisionModelWithProjection.from_pretrained(self.model_path, torch_dtype=torch_dtype)

        # Calculate a more accurate model size.
        self.model_size = calc_model_size_by_data(model)

        return model

    @classmethod
    def convert_if_required(
        cls,
        model_path: str,
        output_path: str,
        config: ModelConfigBase,
        base_model: BaseModelType,
    ) -> str:
        format = cls.detect_format(model_path)
        if format == CLIPVisionModelFormat.Diffusers:
            return model_path
        else:
            raise ValueError(f"Unsupported format: '{format}'.")
