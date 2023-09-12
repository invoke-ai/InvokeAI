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
    classproperty,
)


class IPAdapterModelFormat(str, Enum):
    # Checkpoint is the 'official' IP-Adapter model format from Tencent (i.e. https://huggingface.co/h94/IP-Adapter)
    Checkpoint = "checkpoint"


class IPAdapterModel(ModelBase):
    class CheckpointConfig(ModelConfigBase):
        model_format: Literal[IPAdapterModelFormat.Checkpoint]

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert model_type == ModelType.IPAdapter
        super().__init__(model_path, base_model, model_type)

        # TODO(ryand): Check correct files for model size calculation.
        self.model_size = os.path.getsize(self.model_path)

    @classmethod
    def detect_format(cls, path: str) -> str:
        if not os.path.exists(path):
            raise ModuleNotFoundError(f"No IP-Adapter model at path '{path}'.")

        if os.path.isfile(path):
            if path.endswith((".safetensors", ".ckpt", ".pt", ".pth", ".bin")):
                return IPAdapterModelFormat.Checkpoint

        raise InvalidModelException(f"Unexpected IP-Adapter model format: {path}")

    @classproperty
    def save_to_config(cls) -> bool:
        return True

    def get_size(self, child_type: Optional[SubModelType] = None) -> int:
        if child_type is not None:
            raise ValueError("There are no child models in an IP-Adapter model.")

        # TODO(ryand): Update self.model_size when the model is loaded from disk.
        return self.model_size

    def _get_text_encoder_path(self) -> str:
        # TODO(ryand): Move the CLIP image encoder to its own model directory.
        return os.path.join(os.path.dirname(self.model_path), "image_encoder")

    def get_model(
        self,
        torch_dtype: Optional[torch.dtype],
        child_type: Optional[SubModelType] = None,
    ) -> typing.Union[IPAdapter, IPAdapterPlus]:
        if child_type is not None:
            raise ValueError("There are no child models in an IP-Adapter model.")

        # TODO(ryand): Update IPAdapter to accept a torch_dtype param.

        # TODO(ryand): Checking for "plus" in the file name is fragile. It should be possible to infer whether this is a
        # "plus" variant by loading the state_dict.
        if "plus" in str(self.model_path):
            return IPAdapterPlus(
                image_encoder_path=self._get_text_encoder_path(), ip_adapter_ckpt_path=self.model_path, device="cpu"
            )
        else:
            return IPAdapter(
                image_encoder_path=self._get_text_encoder_path(), ip_adapter_ckpt_path=self.model_path, device="cpu"
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
        if format == IPAdapterModelFormat.Checkpoint:
            return model_path
        else:
            raise ValueError(f"Unsupported format: '{format}'.")
