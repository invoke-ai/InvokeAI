"""Identification of Qwen3.5 text encoders (single-file).

Qwen3.5 is a family of its own -- Gated DeltaNet linear attention interleaved with gated full
attention -- so it cannot share `ModelType.Qwen3Encoder` even where widths coincide: the Qwen3.5 4B
and the Qwen3 4B are both 2560 wide, the Qwen3.5 9B and the Qwen3 8B both 4096. `linear_attn` keys
are what tell them apart, and the Qwen3 and Qwen3-VL configs reject them for the same reason this
one requires them.
"""

import re
from collections.abc import Container
from typing import Any, Literal, Optional, Self

import torch
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
from invokeai.backend.quantization.fp8_scaled import FP8_WEIGHT_DTYPES, WEIGHT_SCALE_SUFFIXES, is_scale_metadata_key
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.int8_convrot import INT8_TENSORWISE_FORMAT, read_comfy_quant_markers
from invokeai.backend.qwen3_5.qwen3_5_vl import QWEN3_5_VL_LANGUAGE_MODEL_PREFIX, QWEN3_5_VL_VARIANTS

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

#: The first block of a `Qwen3_5ForConditionalGeneration` vision tower, under transformers' names.
_QWEN3_5_VISION_TOWER_KEY = "model.visual.blocks.0.attn.qkv.weight"

# ComfyUI's w4a8 build of the FLUX.3 prompt upsampler (`asym_w4a8_int8`): packed 4-bit codes beside a codebook and
# per-channel and per-group scales, which no loader here decodes.
_W4A8_SUFFIXES = (".weight_codebook", ".weight_s_channel", ".weight_s_rel")

# mxfp8's per-block scale: one E8M0 exponent per 32 weights, stored as bytes or as the float8 exponent type.
_MXFP8_SCALE_DTYPES = tuple(
    dtype for dtype in (torch.uint8, getattr(torch, "float8_e8m0fnu", None)) if dtype is not None
)


#: Where Comfy-Org publishes MOSS-Image's Qwen3.5 encoders: a `resolve`/`blob` URL of its `text_encoders/` folder at any
#: revision, or the same file as a Hugging Face `repo_id[:variant]::subfolder` source. These files carry the vision
#: tower, `lm_head.weight` and `mtp.*` -- the key set of the FLUX.3 prompt upsampler -- so only where they came from
#: tells the two apart.
_MOSS_IMAGE_ENCODER_SOURCE = re.compile(
    r"^(?:https?://[^/]+/Comfy-Org/MOSS-Image/(?:resolve|blob)/[^/?#]+/|Comfy-Org/MOSS-Image(?::[^:/]*)?::)"
    r"text_encoders/[^/?#]+\.safetensors(?:[?#].*)?$"
)


def is_moss_image_encoder_source(source: Any) -> bool:
    """Whether an install source is one of Comfy-Org's MOSS-Image text encoder files (a Qwen3.5 encoder, never a Text LLM)."""
    return isinstance(source, str) and _MOSS_IMAGE_ENCODER_SOURCE.match(source) is not None


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


def is_qwen3_5_vision_language_lm(state_dict: dict[str | int, Any]) -> bool:
    """Whether this is a whole `Qwen3_5ForConditionalGeneration` that loads as a Text LLM.

    Language model, vision tower and LM head under transformers' names, at a size with a vendored config: the
    FLUX.3 prompt upsampler's single files, which are `TextLLM_Checkpoint_Config`'s. The weights alone cannot tell
    them from Comfy-Org's MOSS-Image encoders, which carry the same tensors (tower, `lm_head`, `mtp.*`);
    `is_moss_image_encoder_source` does that, by where the file came from. Anything else stays the encoder config's,
    LM head or not: a 4B with its tower, which Anima-3.8B can read, or a text-only export.
    """
    return (
        qwen3_5_key_prefix(state_dict) == QWEN3_5_VL_LANGUAGE_MODEL_PREFIX
        and "lm_head.weight" in state_dict
        and _QWEN3_5_VISION_TOWER_KEY in state_dict
        and get_qwen3_5_variant(state_dict, QWEN3_5_VL_LANGUAGE_MODEL_PREFIX) in QWEN3_5_VL_VARIANTS
    )


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


def raise_for_unsupported_qwen3_5_quantization(mod: ModelOnDisk, state_dict: dict[str | int, Any]) -> None:
    """Refuse every quantized Qwen3.5 build but ComfyUI's int8_tensorwise: with bf16, all the loaders can build.

    `InvalidMatchError`: the file is this model, so the installer shows the reason instead of registering a model
    nothing loads. Anima-3.8B's raw fp8 projections carry no scale and pass on purpose: the encoder loader casts them
    exactly. An int8 layer counts when its weight is int8 and its marker -- read from the file, since
    identification sees dtypes and shapes but no bytes -- says int8_tensorwise; any other scale or marker is a scheme
    the loader refuses.
    """
    keys = [key for key in state_dict if isinstance(key, str)]
    int8_layers = {
        key.removesuffix(".weight")
        for key in keys
        if key.endswith(".weight") and getattr(state_dict[key], "dtype", None) is torch.int8
    }
    scheme = _foreign_quantization(state_dict, keys, int8_layers)
    if scheme is None and int8_layers:
        markers = read_comfy_quant_markers(mod.path)
        unmarked = sorted(
            layer for layer in int8_layers if markers.get(layer, {}).get("format") != INT8_TENSORWISE_FORMAT
        )
        if unmarked:
            scheme = f"int8 without an {INT8_TENSORWISE_FORMAT} marker (e.g. '{unmarked[0]}')"
    if scheme is not None:
        raise InvalidMatchError(
            f"{mod.path.name} is quantized as {scheme}, which InvokeAI cannot load for this model. "
            "Install the bf16 or int8_convrot build instead."
        )


def _foreign_quantization(state_dict: dict[str | int, Any], keys: list[str], int8_layers: set[str]) -> str | None:
    """Name a quantization side channel that belongs to no int8 layer, or None if there is none."""
    if any(key.endswith(_W4A8_SUFFIXES) for key in keys):
        return "ComfyUI's w4a8 (asym_w4a8_int8)"
    if any(key.endswith(".weight_scale_2") for key in keys):
        return "nvfp4"
    for key in keys:
        layer = key.rsplit(".", 1)[0]
        if not is_scale_metadata_key(key) or layer in int8_layers:
            continue
        if key.endswith(WEIGHT_SCALE_SUFFIXES) and getattr(state_dict.get(f"{layer}.weight"), "dtype", None) in (
            FP8_WEIGHT_DTYPES
        ):
            return "mxfp8" if getattr(state_dict[key], "dtype", None) in _MXFP8_SCALE_DTYPES else "scaled fp8"
        return "scaled fp8" if key == "scaled_fp8" else f"an unsupported scheme (it carries '{key}')"
    return None


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
        # The FLUX.3 prompt upsampler is a TextLLM. An explicit type, or Comfy-Org's MOSS-Image encoder source, still
        # installs a file of its layout here.
        if (
            override_fields.get("type") != ModelType.Qwen35Encoder
            and not is_moss_image_encoder_source(override_fields.get("source"))
            and is_qwen3_5_vision_language_lm(state_dict)
        ):
            raise NotAMatchError("state dict is a whole Qwen3.5 vision-language model with an LM head: a TextLLM")
        if any(isinstance(v, GGMLTensor) for v in state_dict.values()):
            raise NotAMatchError("state dict looks like GGUF quantized")
        raise_for_unsupported_qwen3_5_quantization(mod, state_dict)

        variant = override_fields.pop("variant", None) or get_qwen3_5_variant(state_dict, prefix)
        if variant is None:
            raise NotAMatchError("hidden size does not match a known Qwen3.5 variant")
        if variant in QWEN3_5_FINAL_NORM_VARIANTS:
            _raise_if_not_whole(state_dict, prefix, Qwen35VariantType(variant))
        return cls(**override_fields, variant=variant)
