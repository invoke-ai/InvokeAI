import os
from enum import Enum
from typing import Any, Optional

import torch

from invokeai.backend.model_management.models.base import (
    BaseModelType,
    ModelBase,
    ModelType,
    SubModelType,
    classproperty,
)


class IPAdapterModelFormat(Enum):
    # The 'official' IP-Adapter model format from Tencent (i.e. https://huggingface.co/h94/IP-Adapter)
    Tencent = "tencent"


class IPAdapterModel(ModelBase):
    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert model_type == ModelType.IPAdapter
        super().__init__(model_path, base_model, model_type)

        # TODO(ryand): Check correct files for model size calculation.
        self.model_size = os.path.getsize(self.model_path)

    @classmethod
    def detect_format(cls, path: str) -> str:
        if not os.path.exists(path):
            raise ModuleNotFoundError(f"No IP-Adapter model at path '{path}'.")

        raise NotImplementedError()

    @classproperty
    def save_to_config(cls) -> bool:
        raise NotImplementedError()

    def get_size(self, child_type: Optional[SubModelType] = None) -> int:
        if child_type is not None:
            raise ValueError("There are no child models in an IP-Adapter model.")

        raise NotImplementedError()

    def get_model(
        self,
        torch_dtype: Optional[torch.dtype],
        child_type: Optional[SubModelType] = None,
    ) -> Any:
        if child_type is not None:
            raise ValueError("There are no child models in an IP-Adapter model.")
        raise NotImplementedError()
