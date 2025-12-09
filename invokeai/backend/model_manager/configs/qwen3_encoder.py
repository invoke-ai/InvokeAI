from typing import Any, Literal, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType


def _has_qwen3_keys(state_dict: dict[str | int, Any]) -> bool:
    """Check if state dict contains Qwen3 model keys."""
    # Qwen3 models have keys starting with "model.layers." and "model.embed_tokens"
    qwen3_indicators = ["model.layers.0.", "model.embed_tokens.weight"]
    for key in state_dict.keys():
        if isinstance(key, str):
            for indicator in qwen3_indicators:
                if key.startswith(indicator) or key == indicator:
                    return True
    return False


class Qwen3Encoder_Checkpoint_Config(Checkpoint_Config_Base, Config_Base):
    """Configuration for single-file Qwen3 Encoder models (safetensors)."""

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.Qwen3Encoder] = Field(default=ModelType.Qwen3Encoder)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_qwen3_model(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_looks_like_qwen3_model(cls, mod: ModelOnDisk) -> None:
        has_qwen3_keys = _has_qwen3_keys(mod.load_state_dict())
        if not has_qwen3_keys:
            raise NotAMatchError("state dict does not look like a Qwen3 model")


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
