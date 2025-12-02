from typing import Any, Literal, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType


class Qwen3Encoder_Qwen3Encoder_Config(Config_Base):
    """Configuration for Qwen3 Encoder models in a diffusers-like format.

    The model weights are expected to be in a folder called text_encoder inside the model directory,
    compatible with Qwen2VLForConditionalGeneration or similar architectures used by Z-Image.
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.Qwen3Encoder] = Field(default=ModelType.Qwen3Encoder)
    format: Literal[ModelFormat.Qwen3Encoder] = Field(default=ModelFormat.Qwen3Encoder)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # Check for text_encoder config
        expected_config_path = mod.path / "text_encoder" / "config.json"
        # Qwen3 uses Qwen2VLForConditionalGeneration or similar
        raise_for_class_name(
            expected_config_path,
            {
                "Qwen2VLForConditionalGeneration",
                "Qwen2ForCausalLM",
                "Qwen3ForCausalLM",
            },
        )

        return cls(**override_fields)
