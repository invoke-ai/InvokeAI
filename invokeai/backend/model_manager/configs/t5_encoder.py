from typing import Any, Literal, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
    state_dict_has_any_keys_ending_with,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType


class T5Encoder_T5Encoder_Config(Config_Base):
    """Configuration for T5 Encoder models in a bespoke, diffusers-like format. The model weights are expected to be in
    a folder called text_encoder_2 inside the model directory, with a config file named model.safetensors.index.json."""

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.T5Encoder] = Field(default=ModelType.T5Encoder)
    format: Literal[ModelFormat.T5Encoder] = Field(default=ModelFormat.T5Encoder)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        expected_config_path = mod.path / "text_encoder_2" / "config.json"
        expected_class_name = "T5EncoderModel"
        raise_for_class_name(expected_config_path, expected_class_name)

        cls.raise_if_doesnt_have_unquantized_config_file(mod)

        return cls(**override_fields)

    @classmethod
    def raise_if_doesnt_have_unquantized_config_file(cls, mod: ModelOnDisk) -> None:
        has_unquantized_config = (mod.path / "text_encoder_2" / "model.safetensors.index.json").exists()

        if not has_unquantized_config:
            raise NotAMatchError("missing text_encoder_2/model.safetensors.index.json")


class T5Encoder_BnBLLMint8_Config(Config_Base):
    """Configuration for T5 Encoder models quantized by bitsandbytes' LLM.int8."""

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.T5Encoder] = Field(default=ModelType.T5Encoder)
    format: Literal[ModelFormat.BnbQuantizedLlmInt8b] = Field(default=ModelFormat.BnbQuantizedLlmInt8b)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        expected_config_path = mod.path / "text_encoder_2" / "config.json"
        expected_class_name = "T5EncoderModel"
        raise_for_class_name(expected_config_path, expected_class_name)

        cls.raise_if_filename_doesnt_look_like_bnb_quantized(mod)

        cls.raise_if_state_dict_doesnt_look_like_bnb_quantized(mod)

        return cls(**override_fields)

    @classmethod
    def raise_if_filename_doesnt_look_like_bnb_quantized(cls, mod: ModelOnDisk) -> None:
        filename_looks_like_bnb = any(x for x in mod.weight_files() if "llm_int8" in x.as_posix())
        if not filename_looks_like_bnb:
            raise NotAMatchError("filename does not look like bnb quantized llm_int8")

    @classmethod
    def raise_if_state_dict_doesnt_look_like_bnb_quantized(cls, mod: ModelOnDisk) -> None:
        has_scb_key_suffix = state_dict_has_any_keys_ending_with(mod.load_state_dict(), "SCB")
        if not has_scb_key_suffix:
            raise NotAMatchError("state dict does not look like bnb quantized llm_int8")
