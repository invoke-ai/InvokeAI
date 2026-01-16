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
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor


def _has_qwen3_keys(state_dict: dict[str | int, Any]) -> bool:
    """Check if state dict contains Qwen3 model keys.

    Supports both:
    - PyTorch/diffusers format: model.layers.0., model.embed_tokens.weight
    - GGUF/llama.cpp format: blk.0., token_embd.weight
    """
    # PyTorch/diffusers format indicators
    pytorch_indicators = ["model.layers.0.", "model.embed_tokens.weight"]
    # GGUF/llama.cpp format indicators
    gguf_indicators = ["blk.0.", "token_embd.weight"]

    for key in state_dict.keys():
        if isinstance(key, str):
            # Check PyTorch format
            for indicator in pytorch_indicators:
                if key.startswith(indicator) or key == indicator:
                    return True
            # Check GGUF format
            for indicator in gguf_indicators:
                if key.startswith(indicator) or key == indicator:
                    return True
    return False


def _has_ggml_tensors(state_dict: dict[str | int, Any]) -> bool:
    """Check if state dict contains GGML tensors (GGUF quantized)."""
    return any(isinstance(v, GGMLTensor) for v in state_dict.values())


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

        cls._validate_does_not_look_like_gguf_quantized(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_looks_like_qwen3_model(cls, mod: ModelOnDisk) -> None:
        has_qwen3_keys = _has_qwen3_keys(mod.load_state_dict())
        if not has_qwen3_keys:
            raise NotAMatchError("state dict does not look like a Qwen3 model")

    @classmethod
    def _validate_does_not_look_like_gguf_quantized(cls, mod: ModelOnDisk) -> None:
        has_ggml = _has_ggml_tensors(mod.load_state_dict())
        if has_ggml:
            raise NotAMatchError("state dict looks like GGUF quantized")


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

        # Exclude full pipeline models - these should be matched as main models, not just Qwen3 encoders.
        # Full pipelines have model_index.json at root (diffusers format) or a transformer subfolder.
        model_index_path = mod.path / "model_index.json"
        transformer_path = mod.path / "transformer"
        if model_index_path.exists() or transformer_path.exists():
            raise NotAMatchError(
                "directory looks like a full diffusers pipeline (has model_index.json or transformer folder), "
                "not a standalone Qwen3 encoder"
            )

        # Check for text_encoder config - support both:
        # 1. Full model structure: model_root/text_encoder/config.json
        # 2. Standalone text_encoder download: model_root/config.json (when text_encoder subfolder is downloaded separately)
        config_path_nested = mod.path / "text_encoder" / "config.json"
        config_path_direct = mod.path / "config.json"

        if config_path_nested.exists():
            expected_config_path = config_path_nested
        elif config_path_direct.exists():
            expected_config_path = config_path_direct
        else:
            raise NotAMatchError(
                f"unable to load config file(s): {{PosixPath('{config_path_nested}'): 'file does not exist'}}"
            )

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


class Qwen3Encoder_GGUF_Config(Checkpoint_Config_Base, Config_Base):
    """Configuration for GGUF-quantized Qwen3 Encoder models."""

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.Qwen3Encoder] = Field(default=ModelType.Qwen3Encoder)
    format: Literal[ModelFormat.GGUFQuantized] = Field(default=ModelFormat.GGUFQuantized)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_qwen3_model(mod)

        cls._validate_looks_like_gguf_quantized(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_looks_like_qwen3_model(cls, mod: ModelOnDisk) -> None:
        has_qwen3_keys = _has_qwen3_keys(mod.load_state_dict())
        if not has_qwen3_keys:
            raise NotAMatchError("state dict does not look like a Qwen3 model")

    @classmethod
    def _validate_looks_like_gguf_quantized(cls, mod: ModelOnDisk) -> None:
        has_ggml = _has_ggml_tensors(mod.load_state_dict())
        if not has_ggml:
            raise NotAMatchError("state dict does not look like GGUF quantized")
