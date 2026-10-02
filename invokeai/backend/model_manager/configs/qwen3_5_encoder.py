"""Identification of Qwen3.5 text encoders (single-file).

Qwen3.5 is a family of its own -- Gated DeltaNet linear attention interleaved with gated full
attention -- so it cannot share `ModelType.Qwen3Encoder` even where widths coincide: the Qwen3.5 4B
and the Qwen3 4B are both 2560 wide. `linear_attn` keys are what tell them apart, and the Qwen3
configs reject them for the same reason this one requires them.
"""

from typing import Any, Literal, Optional, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    raise_for_override_fields,
    raise_if_not_file,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, Qwen35VariantType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor

#: Where a Qwen3.5 language model's `layers.` and `embed_tokens.` sit: bare (ComfyUI text-encoder
#: exports, Anima-3.8B's `qwen35_4b.safetensors`), under a causal LM's `model.`, or under the
#: multimodal checkpoint's `model.language_model.`.
QWEN3_5_KEY_PREFIXES = ("", "model.", "model.language_model.")

_QWEN3_5_HIDDEN_SIZES = {2560: Qwen35VariantType.Qwen35_4B}


def has_qwen3_5_linear_attention(state_dict: dict[str | int, Any]) -> bool:
    """True if the state dict holds Gated DeltaNet (`linear_attn`) layers -- Qwen3.5, never Qwen3."""
    return any(isinstance(key, str) and ".linear_attn." in key for key in state_dict)


def qwen3_5_key_prefix(state_dict: dict[str | int, Any]) -> Optional[str]:
    """The prefix the language model's keys live under, or None if this is not a Qwen3.5 model.

    Layer 0 is a linear-attention layer in every Qwen3.5 size (full attention is every fourth).
    """
    for prefix in QWEN3_5_KEY_PREFIXES:
        if f"{prefix}layers.0.linear_attn.in_proj_qkv.weight" in state_dict and f"{prefix}embed_tokens.weight" in (
            state_dict
        ):
            return prefix
    return None


def get_qwen3_5_variant(state_dict: dict[str | int, Any], prefix: str) -> Optional[Qwen35VariantType]:
    embed = state_dict.get(f"{prefix}embed_tokens.weight")
    shape = getattr(embed, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    return _QWEN3_5_HIDDEN_SIZES.get(int(shape[1]))


class Qwen35Encoder_Checkpoint_Config(Checkpoint_Config_Base, Config_Base):
    """Configuration for single-file Qwen3.5 text encoders (safetensors)."""

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.Qwen35Encoder] = Field(default=ModelType.Qwen35Encoder)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")
    variant: Qwen35VariantType = Field(description="Qwen3.5 model size variant")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        state_dict = mod.load_state_dict()
        prefix = qwen3_5_key_prefix(state_dict)
        if prefix is None:
            raise NotAMatchError("state dict does not look like a Qwen3.5 language model (no linear_attn layer 0)")
        if any(isinstance(v, GGMLTensor) for v in state_dict.values()):
            raise NotAMatchError("state dict looks like GGUF quantized")

        variant = override_fields.pop("variant", None) or get_qwen3_5_variant(state_dict, prefix)
        if variant is None:
            raise NotAMatchError("hidden size does not match a known Qwen3.5 variant")
        return cls(**override_fields, variant=variant)
