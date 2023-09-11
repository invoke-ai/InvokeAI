import os
import torch
from typing import Optional
from .base import (
    ModelBase,
    ModelConfigBase,
    BaseModelType,
    ModelType,
    SubModelType,
    classproperty,
    ModelNotFoundException,
    InvalidModelException,
)

# TODO: naming
from ..lora import TextualInversionModel as TextualInversionModelRaw


class TextualInversionModel(ModelBase):
    # model_size: int

    class Config(ModelConfigBase):
        model_format: None

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert model_type == ModelType.TextualInversion
        super().__init__(model_path, base_model, model_type)

        self.model_size = os.path.getsize(self.model_path)

    def get_size(self, child_type: Optional[SubModelType] = None):
        if child_type is not None:
            raise Exception("There is no child models in textual inversion")
        return self.model_size

    def get_model(
        self,
        torch_dtype: Optional[torch.dtype],
        child_type: Optional[SubModelType] = None,
    ):
        if child_type is not None:
            raise Exception("There is no child models in textual inversion")

        checkpoint_path = self.model_path
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, "learned_embeds.bin")

        if not os.path.exists(checkpoint_path):
            raise ModelNotFoundException()

        model = TextualInversionModelRaw.from_checkpoint(
            file_path=checkpoint_path,
            dtype=torch_dtype,
        )

        self.model_size = model.embedding.nelement() * model.embedding.element_size()
        return model

    @classproperty
    def save_to_config(cls) -> bool:
        return False

    @classmethod
    def detect_format(cls, path: str):
        if not os.path.exists(path):
            raise ModelNotFoundException()

        if os.path.isdir(path):
            if os.path.exists(os.path.join(path, "learned_embeds.bin")):
                return None  # diffusers-ti

        if os.path.isfile(path):
            if any([path.endswith(f".{ext}") for ext in ["safetensors", "ckpt", "pt", "bin"]]):
                return None

        raise InvalidModelException(f"Not a valid model: {path}")

    @classmethod
    def convert_if_required(
        cls,
        model_path: str,
        output_path: str,
        config: ModelConfigBase,
        base_model: BaseModelType,
    ) -> str:
        return model_path
