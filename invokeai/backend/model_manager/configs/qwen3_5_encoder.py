"""Identification of Qwen3.5 text encoders (single-file).

Qwen3.5 is a family of its own -- Gated DeltaNet linear attention interleaved with gated full
attention -- so it cannot share `ModelType.Qwen3Encoder` even where widths coincide: the Qwen3.5 4B
and the Qwen3 4B are both 2560 wide, the Qwen3.5 9B and the Qwen3 8B both 4096. `linear_attn` keys
are what tell them apart, and the Qwen3 and Qwen3-VL configs reject them for the same reason this
one requires them.
"""

from collections.abc import Container
from typing import Any, Literal, Optional, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    InvalidMatchError,
    NotAMatchError,
    raise_for_override_fields,
    raise_if_not_file,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, Qwen35VariantType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor

#: Where a Qwen3.5 language model's `layers.` and `embed_tokens.` sit: bare (ComfyUI text-encoder
#: exports, Anima-3.8B's `qwen35_4b.safetensors`), under a causal LM's `model.`, under the
#: multimodal checkpoint's `model.language_model.`, or under the `language_model.` of a transformers
#: `Qwen3_5Model` save (MOSS-Image's `text_encoder/` and its ComfyUI export, beside a `visual.*` tower).
QWEN3_5_KEY_PREFIXES = ("", "model.", "model.language_model.", "language_model.")

_QWEN3_5_HIDDEN_SIZES = {2560: Qwen35VariantType.Qwen35_4B, 4096: Qwen35VariantType.Qwen35_9B}

#: The variants read after the final norm (MOSS-Image's 9B), so a checkpoint of one must be whole:
#: every decoder layer and the norm. Not the 4B: Anima-3.8B's file ships its last layer without the
#: MLP and a projection head where the final norm would be.
QWEN3_5_FINAL_NORM_VARIANTS = frozenset({Qwen35VariantType.Qwen35_9B})


def has_qwen3_5_linear_attention(state_dict: dict[str | int, Any]) -> bool:
    """True if the state dict holds Gated DeltaNet (`linear_attn`) layers -- Qwen3.5, never Qwen3."""
    return any(isinstance(key, str) and ".linear_attn." in key for key in state_dict)


def qwen3_5_key_prefix(keys: Container[str | int]) -> Optional[str]:
    """The prefix the language model's keys live under, or None if this is not a Qwen3.5 model.

    Takes a state dict or just its keys. Layer 0 is a linear-attention layer in every Qwen3.5 size
    (full attention is every fourth).
    """
    for prefix in QWEN3_5_KEY_PREFIXES:
        if f"{prefix}layers.0.linear_attn.in_proj_qkv.weight" in keys and f"{prefix}embed_tokens.weight" in keys:
            return prefix
    return None


def get_qwen3_5_variant(state_dict: dict[str | int, Any], prefix: str) -> Optional[Qwen35VariantType]:
    embed = state_dict.get(f"{prefix}embed_tokens.weight")
    shape = getattr(embed, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    return _QWEN3_5_HIDDEN_SIZES.get(int(shape[1]))


def _raise_if_not_whole(state_dict: dict[str | int, Any], prefix: str, variant: Qwen35VariantType) -> None:
    """Refuse a checkpoint that lacks decoder layers or the final norm of a variant read after that norm.

    The first shard of a sharded export is the one that looks like an encoder on its own: it holds the
    embedding and layer 0, and nothing after the layers it ends at. Installed as a single file, it would
    fail only at its first load. `InvalidMatchError` rather than `NotAMatchError`, so it is refused with
    this reason instead of registering as an unknown model.
    """
    # Imported here: transformers' qwen3_5 modeling module costs ~0.25 s, which only a file of such a variant pays.
    from invokeai.backend.qwen3_5.qwen3_5_encoder import qwen3_5_text_config

    num_layers = qwen3_5_text_config(variant).num_hidden_layers
    layers = f"{prefix}layers."
    present = {
        key[len(layers) :].split(".", 1)[0] for key in state_dict if isinstance(key, str) and key.startswith(layers)
    }
    missing = [index for index in range(num_layers) if str(index) not in present]
    norm = f"{prefix}norm.weight"
    if not missing and norm in state_dict:
        return
    lacks = [f"{len(missing)} of its {num_layers} decoder layers (first missing: {missing[0]})"] if missing else []
    if norm not in state_dict:
        lacks.append(f"the final norm '{norm}'")
    raise InvalidMatchError(
        f"this looks like one shard of a sharded Qwen3.5 encoder export: it lacks {' and '.join(lacks)}. "
        "Install the whole model folder, or a merged single-file export."
    )


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
        if variant in QWEN3_5_FINAL_NORM_VARIANTS:
            _raise_if_not_whole(state_dict, prefix, Qwen35VariantType(variant))
        return cls(**override_fields, variant=variant)
