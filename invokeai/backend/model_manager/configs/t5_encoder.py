from typing import Any, Literal, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
    state_dict_has_any_keys_ending_with,
    state_dict_has_any_keys_starting_with,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor


class T5Encoder_T5Encoder_Config(Config_Base):
    """Configuration for T5 Encoder models in a bespoke, diffusers-like format. The model weights are expected to be in
    a folder called text_encoder_2 inside the model directory, with a config file named model.safetensors.index.json."""

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.T5Encoder] = Field(default=ModelType.T5Encoder)
    format: Literal[ModelFormat.T5Encoder] = Field(default=ModelFormat.T5Encoder)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

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
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

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


class T5Encoder_GGUF_Config(Checkpoint_Config_Base, Config_Base):
    """Configuration for GGUF-quantized T5 text encoder models in a single .gguf file.

    These are conversions like city96/t5-v1_1-xxl-encoder-gguf, which use llama.cpp's T5 encoder
    tensor naming (``enc.blk.N.*``, ``token_embd.weight``, ``enc.output_norm.weight``)."""

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.T5Encoder] = Field(default=ModelType.T5Encoder)
    format: Literal[ModelFormat.GGUFQuantized] = Field(default=ModelFormat.GGUFQuantized)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls.raise_if_doesnt_look_like_t5_encoder(mod)

        cls.raise_if_doesnt_look_like_gguf_quantized(mod)

        return cls(**override_fields)

    @classmethod
    def raise_if_doesnt_look_like_t5_encoder(cls, mod: ModelOnDisk) -> None:
        # llama.cpp T5 encoders use the ``enc.`` prefix on their transformer blocks and final norm. This
        # distinguishes them from decoder-only GGUF models (e.g. Qwen3, which uses bare ``blk.*``).
        state_dict = mod.load_state_dict()
        if not state_dict_has_any_keys_starting_with(state_dict, "enc.blk.") and not state_dict_has_any_keys_ending_with(
            state_dict, "enc.output_norm.weight"
        ):
            raise NotAMatchError("state dict does not look like a T5 encoder (no 'enc.blk.*' keys)")

    @classmethod
    def raise_if_doesnt_look_like_gguf_quantized(cls, mod: ModelOnDisk) -> None:
        has_ggml = any(isinstance(v, GGMLTensor) for v in mod.load_state_dict().values())
        if not has_ggml:
            raise NotAMatchError("state dict does not look like GGUF quantized")
