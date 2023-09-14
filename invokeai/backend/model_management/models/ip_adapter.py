import os
import typing
from enum import Enum
from typing import Any, Literal, Optional

import torch

from invokeai.backend.ip_adapter.ip_adapter import IPAdapter, IPAdapterPlus
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
        image_encoder_model: str

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert model_type == ModelType.IPAdapter
        super().__init__(model_path, base_model, model_type)

        self.model_size = os.path.getsize(self.model_path)

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

    @classmethod
    def probe_config(cls, path: str, **kwargs) -> ModelConfigBase:
        image_encoder_config_file = os.path.join(path, "image_encoder.txt")

        with open(image_encoder_config_file, "r") as f:
            image_encoder_model = f.readline().strip()

        return cls.create_config(
            path=path,
            model_format=cls.detect_format(path),
            image_encoder_model=image_encoder_model,
        )

    @classproperty
    def save_to_config(cls) -> bool:
        return True

    def get_size(self, child_type: Optional[SubModelType] = None) -> int:
        if child_type is not None:
            raise ValueError("There are no child models in an IP-Adapter model.")

        # TODO(ryand): Update self.model_size when the model is loaded from disk.
        return self.model_size

    def get_model(
        self,
        torch_dtype: Optional[torch.dtype],
        child_type: Optional[SubModelType] = None,
    ) -> typing.Union[IPAdapter, IPAdapterPlus]:
        if child_type is not None:
            raise ValueError("There are no child models in an IP-Adapter model.")

        # TODO(ryand): Checking for "plus" in the file path is fragile. It should be possible to infer whether this is a
        # "plus" variant by loading the state_dict.
        if "plus" in str(self.model_path):
            return IPAdapterPlus(
                ip_adapter_ckpt_path=os.path.join(self.model_path, "ip_adapter.bin"), device="cpu", dtype=torch_dtype
            )
        else:
            return IPAdapter(
                ip_adapter_ckpt_path=os.path.join(self.model_path, "ip_adapter.bin"), device="cpu", dtype=torch_dtype
            )

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
