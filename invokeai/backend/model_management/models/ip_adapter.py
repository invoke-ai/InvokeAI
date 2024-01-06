import os
import typing
from enum import Enum
from typing import Literal, Optional

import torch

from invokeai.backend.ip_adapter.ip_adapter import IPAdapter, IPAdapterPlus, build_ip_adapter
from invokeai.backend.model_management.models.base import (
    BaseModelType,
    InvalidModelException,
    ModelBase,
    ModelConfigBase,
    ModelType,
    SubModelType,
    calc_model_size_by_fs,
    classproperty,
)


class IPAdapterModelFormat(str, Enum):
    # The custom IP-Adapter model format defined by InvokeAI.
    InvokeAI = "invokeai"


class IPAdapterModel(ModelBase):
    class InvokeAIConfig(ModelConfigBase):
        model_format: Literal[IPAdapterModelFormat.InvokeAI]

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert model_type == ModelType.IPAdapter
        super().__init__(model_path, base_model, model_type)

        self.model_size = calc_model_size_by_fs(self.model_path)

    @classmethod
    def detect_format(cls, path: str) -> str:
        if not os.path.exists(path):
            raise ModuleNotFoundError(f"No IP-Adapter model at path '{path}'.")

        if os.path.isdir(path):
            model_file = os.path.join(path, "ip_adapter.bin")
            image_encoder_config_file = os.path.join(path, "image_encoder.txt")
            if os.path.exists(model_file) and os.path.exists(image_encoder_config_file):
                return IPAdapterModelFormat.InvokeAI

        raise InvalidModelException(f"Unexpected IP-Adapter model format: {path}")

    @classproperty
    def save_to_config(cls) -> bool:
        return True

    def get_size(self, child_type: Optional[SubModelType] = None) -> int:
        if child_type is not None:
            raise ValueError("There are no child models in an IP-Adapter model.")

        return self.model_size

    def get_model(
        self,
        torch_dtype: torch.dtype,
        child_type: Optional[SubModelType] = None,
    ) -> typing.Union[IPAdapter, IPAdapterPlus]:
        if child_type is not None:
            raise ValueError("There are no child models in an IP-Adapter model.")

        model = build_ip_adapter(
            ip_adapter_ckpt_path=os.path.join(self.model_path, "ip_adapter.bin"),
            device=torch.device("cpu"),
            dtype=torch_dtype,
        )

        self.model_size = model.calc_size()
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
        if format == IPAdapterModelFormat.InvokeAI:
            return model_path
        else:
            raise ValueError(f"Unsupported format: '{format}'.")


def get_ip_adapter_image_encoder_model_id(model_path: str):
    """Read the ID of the image encoder associated with the IP-Adapter at `model_path`."""
    image_encoder_config_file = os.path.join(model_path, "image_encoder.txt")

    with open(image_encoder_config_file, "r") as f:
        image_encoder_model = f.readline().strip()

    return image_encoder_model
