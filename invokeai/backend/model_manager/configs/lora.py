import re
from abc import ABC
from pathlib import Path
from typing import (
    Any,
    Literal,
    Self,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator

from invokeai.backend.model_manager.configs.base import (
    Config_Base,
)
from invokeai.backend.model_manager.configs.controlnet import ControlAdapterDefaultSettings
from invokeai.backend.model_manager.configs.flux2_variant import (
    FLUX2_CONTEXT_IN_DIMS,
    flux2_variant_from_context_dim,
    flux2_variant_from_hidden_size,
    flux2_variant_from_vec_dim,
)
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
    state_dict_has_any_keys_ending_with,
    state_dict_has_any_keys_starting_with,
)
from invokeai.backend.model_manager.configs.main import _detect_wan_expert
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.omi import flux_dev_1_lora, stable_diffusion_xl_1_lora
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    Flux2VariantType,
    FluxLoRAFormat,
    ModelFormat,
    ModelType,
    WanLoRAVariantType,
    ZImageVariantType,
)
from invokeai.backend.model_manager.util.model_util import lora_token_vector_length
from invokeai.backend.patches.lora_conversions.anima_lora_constants import (
    has_cosmos_dit_kohya_keys,
    has_cosmos_dit_kohya_keys_strict,
    has_cosmos_dit_peft_keys,
    has_cosmos_dit_peft_keys_strict,
)
from invokeai.backend.patches.lora_conversions.flux_control_lora_utils import is_state_dict_likely_flux_control
from invokeai.backend.patches.lora_conversions.wan_lora_constants import (
    detect_wan_lora_variant,
    has_non_wan_architecture_keys,
    has_wan_kohya_keys,
    has_wan_peft_keys,
)

# Defaults used to compute the effective slider range when one or both bounds
# are unset. These intentionally mirror the frontend's DEFAULT_LORA_WEIGHT_CONFIG
# in invokeai/frontend/web/src/features/controlLayers/store/lorasSlice.ts so that
# bound/weight validation produces the same result whether it runs in the form
# or in this pydantic model.
_DEFAULT_LORA_WEIGHT_SLIDER_MIN = -1.0
_DEFAULT_LORA_WEIGHT_SLIDER_MAX = 2.0


class LoraModelDefaultSettings(BaseModel):
    weight: float | None = Field(default=None, description="Default weight for this model")
    weight_min: float | None = Field(default=None, description="Minimum weight slider value for this model")
    weight_max: float | None = Field(default=None, description="Maximum weight slider value for this model")
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_weight_bounds(self) -> "LoraModelDefaultSettings":
        effective_min = self.weight_min if self.weight_min is not None else _DEFAULT_LORA_WEIGHT_SLIDER_MIN
        effective_max = self.weight_max if self.weight_max is not None else _DEFAULT_LORA_WEIGHT_SLIDER_MAX
        if effective_min >= effective_max:
            raise ValueError(
                f"effective weight range is invalid: min ({effective_min}) must be less than max ({effective_max})"
            )
        if self.weight is not None and not (effective_min <= self.weight <= effective_max):
            raise ValueError(
                f"weight ({self.weight}) must be within the effective range [{effective_min}, {effective_max}]"
            )
        return self


class LoRA_Config_Base(ABC, BaseModel):
    """Base class for LoRA models."""

    type: Literal[ModelType.LoRA] = Field(default=ModelType.LoRA)
    trigger_phrases: set[str] | None = Field(
        default=None,
        description="Set of trigger phrases for this model",
    )
    default_settings: LoraModelDefaultSettings | None = Field(
        default=None,
        description="Default settings for this model",
    )


def _get_flux_lora_format(mod: ModelOnDisk) -> FluxLoRAFormat | None:
    # TODO(psyche): Moving this import to the function to avoid circular imports. Refactor later.
    from invokeai.backend.patches.lora_conversions.formats import flux_format_from_state_dict

    state_dict = mod.load_state_dict()
    value = flux_format_from_state_dict(state_dict, mod.metadata())
    return value


# Matches an SDXL UNet attention key, capturing the transformer_blocks index.
#
# Anchored on the `lora_unet_` prefix because that is exactly the key set
# `convert_sdxl_keys_to_diffusers_format()` can convert at load time — keys outside it
# (e.g. diffusers/PEFT `unet.….lora_A.weight`) would be identified as SDXL here only to
# raise `ValueError: Unrecognized SDXL LoRA key prefix` mid-generation.
#
# Both kohya `sd-scripts` naming conventions are covered: Stability-AI block names
# (`input_blocks_8_1`, `middle_block_1`, `output_blocks_0_1`) and diffusers block names
# (`down_blocks_2_attentions_1`, `mid_block_attentions_0`). Requiring a UNet block name
# in addition to the prefix keeps DiT-style transformers (FLUX/Qwen/Z-Image), which also
# use `transformer_blocks`, from matching.
_SDXL_UNET_ATTENTION_RE = re.compile(
    r"^lora_unet_"
    r"(?:(?:down_blocks|up_blocks)_\d+_attentions_\d+"
    r"|mid_block_attentions_\d+"
    r"|(?:input_blocks|output_blocks)_\d+_\d+"
    r"|middle_block_\d+)"
    r"_transformer_blocks_(\d+)_"
)


def _state_dict_looks_like_sdxl_unet_lora(state_dict: dict[str | int, Any]) -> bool:
    """Detect an SDXL UNet LoRA from its block structure alone.

    SDXL's UNet uses a deep transformer stack (up to 10 transformer blocks) in its
    lower-resolution attention blocks, so `transformer_blocks` indices reach >= 2.
    SD1.x/SD2.x UNets only ever have a single transformer block (index 0) per
    attention. This lets us identify SDXL from UNet-only LoRAs that lack the
    cross-attention / text-encoder keys `lora_token_vector_length()` relies on
    (e.g. self-attention-only "slider" LoRAs).

    Known limitation: the `>= 2` threshold is what separates SDXL from SD1.x/SD2.x, so an
    SDXL LoRA confined to the 2-transformer-block attention blocks (`down_blocks_1` /
    `up_blocks_1`, indices 0-1) is not detected. Such a LoRA is indistinguishable from
    SD1/SD2 by block structure alone.
    """
    for key in state_dict:
        if not isinstance(key, str):
            continue
        match = _SDXL_UNET_ATTENTION_RE.match(key)
        if match is not None and int(match.group(1)) >= 2:
            return True
    return False


# FLUX.2 context_in_dim values (Klein 4B 7680 / Klein 9B 12288 / Dev 15360) come from the
# shared dimension table so this "is it FLUX.2?" check can't drift from variant detection.
_FLUX2_CONTEXT_IN_DIMS = FLUX2_CONTEXT_IN_DIMS

# FLUX.2 vec_in_dim values: text encoder hidden_size
# Klein 4B: 2560 (Qwen3-4B), Klein 9B: 4096 (Qwen3-8B), Dev: 5120 (Mistral Small 3.1)
_FLUX2_VEC_IN_DIMS = {2560, 4096, 5120}

# FLUX.1 hidden_size is 3072. Klein 9B uses 4096, FLUX.2 [dev] uses 6144 (48 heads × 128 head_dim).
# Klein 4B also uses 3072, so hidden_size alone can't distinguish Klein 4B from FLUX.1.
_FLUX1_HIDDEN_SIZE = 3072

# FLUX.1 uses mlp_ratio=4 (ffn_dim=12288 for hidden_size=3072).
# Klein 4B uses mlp_ratio=6 (ffn_dim=18432 for hidden_size=3072).
_FLUX1_MLP_RATIO = 4


def _lokr_in_dim(state_dict: dict[str | int, Any], key_prefix: str) -> int | None:
    """Compute the input dimension of a LOKR layer: w1.shape[1] * w2.shape[1].

    Supports both full LOKR (lokr_w1/lokr_w2) and factorized LOKR (lokr_w1_b/lokr_w2_b).
    Returns None if the required keys are not present.
    """
    if f"{key_prefix}.lokr_w1" in state_dict and f"{key_prefix}.lokr_w2" in state_dict:
        return state_dict[f"{key_prefix}.lokr_w1"].shape[1] * state_dict[f"{key_prefix}.lokr_w2"].shape[1]
    elif f"{key_prefix}.lokr_w1_b" in state_dict and f"{key_prefix}.lokr_w2_b" in state_dict:
        return state_dict[f"{key_prefix}.lokr_w1_b"].shape[1] * state_dict[f"{key_prefix}.lokr_w2_b"].shape[1]
    return None


def _lokr_out_dim(state_dict: dict[str | int, Any], key_prefix: str) -> int | None:
    """Compute the output dimension of a LOKR layer: w1.shape[0] * w2.shape[0].

    Supports both full LOKR (lokr_w1/lokr_w2) and factorized LOKR (lokr_w1_a/lokr_w2_a).
    Returns None if the required keys are not present.
    """
    if f"{key_prefix}.lokr_w1" in state_dict and f"{key_prefix}.lokr_w2" in state_dict:
        return state_dict[f"{key_prefix}.lokr_w1"].shape[0] * state_dict[f"{key_prefix}.lokr_w2"].shape[0]
    elif f"{key_prefix}.lokr_w1_a" in state_dict and f"{key_prefix}.lokr_w2_a" in state_dict:
        return state_dict[f"{key_prefix}.lokr_w1_a"].shape[0] * state_dict[f"{key_prefix}.lokr_w2_a"].shape[0]
    return None


def _is_flux2_lora(mod: ModelOnDisk) -> bool:
    """Check if a FLUX-format LoRA is specifically for FLUX.2 (Klein) rather than FLUX.1.

    Detection is based on:
    1. Tensor shapes of embedding layers (context_embedder, vector_in) that differ between FLUX.1 and FLUX.2
    2. Hidden size of attention layers (3072 for FLUX.1/Klein 4B, 4096 for Klein 9B)

    Returns False for ambiguous LoRAs (e.g. Klein 4B transformer-only LoRAs with no distinguishing layers).
    """
    state_dict = mod.load_state_dict()
    return _is_flux2_lora_state_dict(state_dict)


def _is_flux2_lora_state_dict(state_dict: dict[str | int, Any]) -> bool:
    """Check state dict tensor shapes for FLUX.2 Klein-specific dimensions."""
    # Check diffusers/PEFT format keys (with various prefixes).
    # This covers both Flux.1 diffusers naming AND Flux2 Klein diffusers naming.
    for prefix in ["transformer.", "base_model.model.", ""]:
        # Check context_embedder (txt_in) dimensions
        # FLUX.1: context_in_dim=4096, FLUX.2 Klein 4B: 7680, Klein 9B: 12288
        ctx_key_a = f"{prefix}context_embedder.lora_A.weight"
        if ctx_key_a in state_dict:
            return state_dict[ctx_key_a].shape[1] in _FLUX2_CONTEXT_IN_DIMS

        # Check vector_in (time_text_embed.text_embedder) dimensions
        # FLUX.1: vec_in_dim=768, FLUX.2 Klein 4B: 2560, Klein 9B: 4096
        vec_key_a = f"{prefix}time_text_embed.text_embedder.linear_1.lora_A.weight"
        if vec_key_a in state_dict:
            return state_dict[vec_key_a].shape[1] in _FLUX2_VEC_IN_DIMS

        # Check Flux2 Klein diffusers naming: fused QKV+MLP in single blocks.
        # This key only exists in Flux2 models (Flux.1 uses separate to_q/to_k/to_v + proj_mlp).
        fused_key_a = f"{prefix}single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight"
        if fused_key_a in state_dict:
            return True

        # Check Flux2 Klein diffusers naming: ff.linear_in (Flux.1 uses ff.net.0.proj).
        ff_key_a = f"{prefix}transformer_blocks.0.ff.linear_in.lora_A.weight"
        if ff_key_a in state_dict:
            return True

    # Check BFL PEFT format (diffusion_model.* or base_model.model.* prefix with BFL layer names).
    # Klein 9B has hidden_size=4096 (vs 3072 for FLUX.1 and Klein 4B).
    # Klein 4B has same hidden_size as FLUX.1 (3072) but different mlp_ratio (6 vs 4),
    # and different txt_in/vector_in dimensions.
    _bfl_prefixes = ("diffusion_model.", "base_model.model.")
    bfl_hidden_size: int | None = None
    for key in state_dict:
        if not isinstance(key, str):
            continue
        if not key.startswith(_bfl_prefixes):
            continue

        # BFL PEFT: attention projection → check hidden_size
        if key.endswith(".img_attn.proj.lora_A.weight"):
            bfl_hidden_size = state_dict[key].shape[1]
            if bfl_hidden_size != _FLUX1_HIDDEN_SIZE:
                return True
            # hidden_size=3072 is ambiguous (could be Klein 4B or FLUX.1), keep checking

        # BFL PEFT: context_embedder/txt_in
        elif "txt_in" in key and key.endswith("lora_A.weight"):
            return state_dict[key].shape[1] in _FLUX2_CONTEXT_IN_DIMS

        # BFL PEFT: vector_in
        elif "vector_in" in key and key.endswith("lora_A.weight"):
            return state_dict[key].shape[1] in _FLUX2_VEC_IN_DIMS

        # BFL LyCORIS (LoKR/LoHA): attention projection → check hidden_size via product of dims
        elif key.endswith((".img_attn.proj.lokr_w1", ".img_attn.proj.lokr_w1_b")):
            layer_prefix = key.rsplit(".", 1)[0]
            in_dim = _lokr_in_dim(state_dict, layer_prefix)
            if in_dim is not None:
                if in_dim != _FLUX1_HIDDEN_SIZE:
                    return True
                bfl_hidden_size = in_dim  # ambiguous, keep checking

        # BFL LyCORIS: context_embedder/txt_in
        elif "txt_in" in key and key.endswith((".lokr_w1", ".lokr_w1_b")):
            layer_prefix = key.rsplit(".", 1)[0]
            in_dim = _lokr_in_dim(state_dict, layer_prefix)
            if in_dim is not None:
                return in_dim in _FLUX2_CONTEXT_IN_DIMS

        # BFL LyCORIS: vector_in
        elif "vector_in" in key and key.endswith((".lokr_w1", ".lokr_w1_b")):
            layer_prefix = key.rsplit(".", 1)[0]
            in_dim = _lokr_in_dim(state_dict, layer_prefix)
            if in_dim is not None:
                return in_dim in _FLUX2_VEC_IN_DIMS

    # BFL PEFT/LyCORIS: hidden_size matches FLUX.1. Check MLP ratio to distinguish Klein 4B.
    # Klein 4B uses mlp_ratio=6 (ffn_dim=18432), FLUX.1 uses mlp_ratio=4 (ffn_dim=12288).
    if bfl_hidden_size == _FLUX1_HIDDEN_SIZE:
        for key in state_dict:
            if not isinstance(key, str):
                continue
            if key.startswith(_bfl_prefixes) and key.endswith(".img_mlp.0.lora_B.weight"):
                ffn_dim = state_dict[key].shape[0]
                if ffn_dim != bfl_hidden_size * _FLUX1_MLP_RATIO:
                    return True
                break
            # BFL LyCORIS: check output dim of img_mlp.0 via product of dims
            if key.startswith(_bfl_prefixes) and key.endswith((".img_mlp.0.lokr_w1", ".img_mlp.0.lokr_w1_a")):
                layer_prefix = key.rsplit(".", 1)[0]
                out_dim = _lokr_out_dim(state_dict, layer_prefix)
                if out_dim is not None and out_dim != bfl_hidden_size * _FLUX1_MLP_RATIO:
                    return True
                break

    # Check kohya format: look for context_embedder or vector_in keys
    # Kohya format uses lora_unet_ prefix with underscores instead of dots
    for key in state_dict:
        if not isinstance(key, str):
            continue
        if key.startswith("lora_unet_txt_in.") or key.startswith("lora_unet_context_embedder."):
            if key.endswith("lora_down.weight"):
                return state_dict[key].shape[1] in _FLUX2_CONTEXT_IN_DIMS
            # Kohya LyCORIS (LoKR)
            elif key.endswith((".lokr_w1", ".lokr_w1_b")):
                layer_prefix = key.rsplit(".", 1)[0]
                in_dim = _lokr_in_dim(state_dict, layer_prefix)
                if in_dim is not None:
                    return in_dim in _FLUX2_CONTEXT_IN_DIMS
        if key.startswith("lora_unet_vector_in.") or key.startswith("lora_unet_time_text_embed_text_embedder_"):
            if key.endswith("lora_down.weight"):
                return state_dict[key].shape[1] in _FLUX2_VEC_IN_DIMS
            # Kohya LyCORIS (LoKR)
            elif key.endswith((".lokr_w1", ".lokr_w1_b")):
                layer_prefix = key.rsplit(".", 1)[0]
                in_dim = _lokr_in_dim(state_dict, layer_prefix)
                if in_dim is not None:
                    return in_dim in _FLUX2_VEC_IN_DIMS

    # Kohya format: check transformer block dimensions (hidden_size and MLP ratio).
    # This handles LoRAs that only target transformer blocks (no txt_in/vector_in/context_embedder).
    # Klein 9B has hidden_size=4096 (vs 3072 for FLUX.1 and Klein 4B).
    # Klein 4B has same hidden_size as FLUX.1 (3072) but different mlp_ratio (6 vs 4).
    kohya_hidden_size: int | None = None
    for key in state_dict:
        if not isinstance(key, str):
            continue
        if not key.startswith("lora_unet_"):
            continue

        # Check img_attn_proj hidden_size
        if "_img_attn_proj." in key and key.endswith("lora_down.weight"):
            kohya_hidden_size = state_dict[key].shape[1]
            if kohya_hidden_size != _FLUX1_HIDDEN_SIZE:
                return True
            break
        # LoKR variant
        elif "_img_attn_proj." in key and key.endswith((".lokr_w1", ".lokr_w1_b")):
            layer_prefix = key.rsplit(".", 1)[0]
            in_dim = _lokr_in_dim(state_dict, layer_prefix)
            if in_dim is not None:
                if in_dim != _FLUX1_HIDDEN_SIZE:
                    return True
                kohya_hidden_size = in_dim
            break

    # Kohya format: hidden_size matches FLUX.1. Check MLP ratio to distinguish Klein 4B.
    # Klein 4B uses mlp_ratio=6 (ffn_dim=18432), FLUX.1 uses mlp_ratio=4 (ffn_dim=12288).
    if kohya_hidden_size == _FLUX1_HIDDEN_SIZE:
        for key in state_dict:
            if not isinstance(key, str):
                continue
            if key.startswith("lora_unet_") and "_img_mlp_0." in key and key.endswith("lora_up.weight"):
                ffn_dim = state_dict[key].shape[0]
                if ffn_dim != kohya_hidden_size * _FLUX1_MLP_RATIO:
                    return True
                break
            # LoKR variant
            if key.startswith("lora_unet_") and "_img_mlp_0." in key and key.endswith((".lokr_w1", ".lokr_w1_a")):
                layer_prefix = key.rsplit(".", 1)[0]
                out_dim = _lokr_out_dim(state_dict, layer_prefix)
                if out_dim is not None and out_dim != kohya_hidden_size * _FLUX1_MLP_RATIO:
                    return True
                break

    return False


def _get_flux2_lora_variant(state_dict: dict[str | int, Any]) -> Flux2VariantType | None:
    """Determine FLUX.2 variant (Klein 4B/9B or Dev) from a LoRA state dict.

    Detection is based on tensor dimensions that differ between variants:
    - hidden_size from attention projection: 3072 = Klein 4B, 4096 = Klein 9B, 6144 = Dev
    - context_in_dim from context embedder: 7680 = Klein 4B, 12288 = Klein 9B, 15360 = Dev
    - vec_in_dim from vector embedder: 2560 = Klein 4B, 4096 = Klein 9B, 5120 = Dev

    Returns None if the variant cannot be determined (e.g. LoRA only targets layers
    with identical dimensions across variants).
    """
    # Reverse-lookup helpers come from the shared FLUX.2 dimension table (single source of
    # truth shared with main.py's identification code). Aliased to the original local names
    # to keep the detection code below unchanged.
    _variant_from_context_dim = flux2_variant_from_context_dim
    _variant_from_vec_dim = flux2_variant_from_vec_dim
    _variant_from_hidden_size = flux2_variant_from_hidden_size

    # Check diffusers/PEFT format keys
    for prefix in ["transformer.", "base_model.model.", ""]:
        # Context embedder (txt_in) dimensions
        ctx_key_a = f"{prefix}context_embedder.lora_A.weight"
        if ctx_key_a in state_dict:
            return _variant_from_context_dim(state_dict[ctx_key_a].shape[1])

        # Vector embedder dimensions
        vec_key_a = f"{prefix}time_text_embed.text_embedder.linear_1.lora_A.weight"
        if vec_key_a in state_dict:
            return _variant_from_vec_dim(state_dict[vec_key_a].shape[1])

        # Attention projection hidden_size (Flux.1 diffusers naming)
        attn_key_a = f"{prefix}transformer_blocks.0.attn.to_out.0.lora_A.weight"
        if attn_key_a in state_dict:
            return _variant_from_hidden_size(state_dict[attn_key_a].shape[1])

        # Attention projection hidden_size (Flux2 diffusers naming)
        attn_key_a2 = f"{prefix}transformer_blocks.0.attn.to_add_out.lora_A.weight"
        if attn_key_a2 in state_dict:
            return _variant_from_hidden_size(state_dict[attn_key_a2].shape[1])

        # Fused QKV+MLP hidden_size (Flux2 diffusers naming)
        fused_key_a = f"{prefix}single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight"
        if fused_key_a in state_dict:
            return _variant_from_hidden_size(state_dict[fused_key_a].shape[1])

    # Check BFL PEFT/LyCORIS format (diffusion_model.* or base_model.model.* prefix with BFL names)
    _bfl_prefixes = ("diffusion_model.", "base_model.model.")
    for key in state_dict:
        if not isinstance(key, str):
            continue
        if not key.startswith(_bfl_prefixes):
            continue

        # BFL PEFT: context embedder (txt_in)
        if "txt_in" in key and key.endswith("lora_A.weight"):
            return _variant_from_context_dim(state_dict[key].shape[1])

        # BFL PEFT: vector embedder (vector_in)
        if "vector_in" in key and key.endswith("lora_A.weight"):
            return _variant_from_vec_dim(state_dict[key].shape[1])

        # BFL PEFT: attention projection
        if key.endswith(".img_attn.proj.lora_A.weight"):
            return _variant_from_hidden_size(state_dict[key].shape[1])

        # BFL LyCORIS (LoKR): context embedder (txt_in)
        if "txt_in" in key and key.endswith((".lokr_w1", ".lokr_w1_b")):
            in_dim = _lokr_in_dim(state_dict, key.rsplit(".", 1)[0])
            if in_dim is not None:
                return _variant_from_context_dim(in_dim)

        # BFL LyCORIS (LoKR): vector embedder (vector_in)
        if "vector_in" in key and key.endswith((".lokr_w1", ".lokr_w1_b")):
            in_dim = _lokr_in_dim(state_dict, key.rsplit(".", 1)[0])
            if in_dim is not None:
                return _variant_from_vec_dim(in_dim)

        # BFL LyCORIS (LoKR): attention projection
        if key.endswith((".img_attn.proj.lokr_w1", ".img_attn.proj.lokr_w1_b")):
            in_dim = _lokr_in_dim(state_dict, key.rsplit(".", 1)[0])
            if in_dim is not None:
                return _variant_from_hidden_size(in_dim)

    # Check kohya format
    for key in state_dict:
        if not isinstance(key, str):
            continue
        if key.startswith("lora_unet_txt_in.") or key.startswith("lora_unet_context_embedder."):
            if key.endswith("lora_down.weight"):
                return _variant_from_context_dim(state_dict[key].shape[1])
            # Kohya LyCORIS (LoKR)
            elif key.endswith((".lokr_w1", ".lokr_w1_b")):
                in_dim = _lokr_in_dim(state_dict, key.rsplit(".", 1)[0])
                if in_dim is not None:
                    return _variant_from_context_dim(in_dim)
        if key.startswith("lora_unet_vector_in.") or key.startswith("lora_unet_time_text_embed_text_embedder_"):
            if key.endswith("lora_down.weight"):
                return _variant_from_vec_dim(state_dict[key].shape[1])
            # Kohya LyCORIS (LoKR)
            elif key.endswith((".lokr_w1", ".lokr_w1_b")):
                in_dim = _lokr_in_dim(state_dict, key.rsplit(".", 1)[0])
                if in_dim is not None:
                    return _variant_from_vec_dim(in_dim)

    # Kohya format: check transformer block dimensions (hidden_size from img_attn_proj).
    # This handles LoRAs that only target transformer blocks (no txt_in/vector_in/context_embedder).
    for key in state_dict:
        if not isinstance(key, str):
            continue
        if not key.startswith("lora_unet_"):
            continue

        # Check img_attn_proj hidden_size
        if "_img_attn_proj." in key and key.endswith("lora_down.weight"):
            return _variant_from_hidden_size(state_dict[key].shape[1])
        # LoKR variant
        elif "_img_attn_proj." in key and key.endswith((".lokr_w1", ".lokr_w1_b")):
            in_dim = _lokr_in_dim(state_dict, key.rsplit(".", 1)[0])
            if in_dim is not None:
                return _variant_from_hidden_size(in_dim)

    return None


class LoRA_OMI_Config_Base(LoRA_Config_Base):
    format: Literal[ModelFormat.OMI] = Field(default=ModelFormat.OMI)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_omi_lora(mod)

        cls._validate_base(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatchError(f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _validate_looks_like_omi_lora(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model metadata does not look like an OMI LoRA."""
        flux_format = _get_flux_lora_format(mod)
        if flux_format in [FluxLoRAFormat.Control, FluxLoRAFormat.Diffusers]:
            raise NotAMatchError("model looks like ControlLoRA or Diffusers LoRA")

        metadata = mod.metadata()

        metadata_looks_like_omi_lora = (
            bool(metadata.get("modelspec.sai_model_spec"))
            and metadata.get("ot_branch") == "omi_format"
            and metadata.get("modelspec.architecture", "").split("/")[1].lower() == "lora"
        )

        if not metadata_looks_like_omi_lora:
            raise NotAMatchError("metadata does not look like OMI LoRA")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> Literal[BaseModelType.Flux, BaseModelType.StableDiffusionXL]:
        metadata = mod.metadata()
        architecture = metadata["modelspec.architecture"]

        if architecture == stable_diffusion_xl_1_lora:
            return BaseModelType.StableDiffusionXL
        elif architecture == flux_dev_1_lora:
            return BaseModelType.Flux
        else:
            raise NotAMatchError(f"unrecognised/unsupported architecture for OMI LoRA: {architecture}")


class LoRA_OMI_SDXL_Config(LoRA_OMI_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class LoRA_OMI_FLUX_Config(LoRA_OMI_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)


class LoRA_LyCORIS_Config_Base(LoRA_Config_Base):
    """Model config for LoRA/Lycoris models."""

    type: Literal[ModelType.LoRA] = Field(default=ModelType.LoRA)
    format: Literal[ModelFormat.LyCORIS] = Field(default=ModelFormat.LyCORIS)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_lora(mod)

        cls._validate_base(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatchError(f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _validate_looks_like_lora(cls, mod: ModelOnDisk) -> None:
        # First rule out ControlLoRA
        flux_format = _get_flux_lora_format(mod)
        if flux_format in [FluxLoRAFormat.Control]:
            raise NotAMatchError("model looks like Control LoRA")

        # If it's a recognized Flux LoRA format (Kohya, Diffusers, OneTrainer, AIToolkit, XLabs, etc.),
        # it's valid and we skip the heuristic check
        if flux_format is not None:
            return

        # Note: Existence of these key prefixes/suffixes does not guarantee that this is a LoRA.
        # Some main models have these keys, likely due to the creator merging in a LoRA.
        has_key_with_lora_prefix = state_dict_has_any_keys_starting_with(
            mod.load_state_dict(),
            {
                "lora_te_",
                "lora_unet_",
                "lora_te1_",
                "lora_te2_",
                "lora_transformer_",
            },
        )

        has_key_with_lora_suffix = state_dict_has_any_keys_ending_with(
            mod.load_state_dict(),
            {
                "to_k_lora.up.weight",
                "to_q_lora.down.weight",
                "lora_A.weight",
                "lora_B.weight",
                # LyCORIS LoKR suffixes
                "lokr_w1",
                "lokr_w2",
                # LyCORIS LoHA suffixes
                "hada_w1_a",
                "hada_w2_a",
            },
        )

        if not has_key_with_lora_prefix and not has_key_with_lora_suffix:
            raise NotAMatchError("model does not match LyCORIS LoRA heuristics")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        if _get_flux_lora_format(mod):
            if _is_flux2_lora(mod):
                return BaseModelType.Flux2
            return BaseModelType.Flux

        state_dict = mod.load_state_dict()
        str_keys = [k for k in state_dict.keys() if isinstance(k, str)]

        # Rule out Anima LoRAs — their lora_te_ keys have shapes that
        # lora_token_vector_length() misidentifies as SD2/SDXL.
        if has_cosmos_dit_kohya_keys(str_keys) or has_cosmos_dit_peft_keys(str_keys):
            raise NotAMatchError("model looks like an Anima LoRA, not a Stable Diffusion LoRA")

        # If we've gotten here, we assume that the model is a Stable Diffusion model
        token_vector_length = lora_token_vector_length(state_dict)
        if token_vector_length == 768:
            return BaseModelType.StableDiffusion1
        elif token_vector_length == 1024:
            return BaseModelType.StableDiffusion2
        elif token_vector_length == 1280:
            return BaseModelType.StableDiffusionXL  # recognizes format at https://civitai.com/models/224641
        elif token_vector_length == 2048:
            return BaseModelType.StableDiffusionXL
        # Some SDXL LoRAs (e.g. self-attention-only "slider" LoRAs) target only the UNet
        # and lack the cross-attention / text-encoder keys that lora_token_vector_length()
        # needs. Fall back to detecting SDXL from the UNet's deep transformer-block structure.
        elif _state_dict_looks_like_sdxl_unet_lora(state_dict):
            return BaseModelType.StableDiffusionXL
        else:
            raise NotAMatchError(f"unrecognized token vector length {token_vector_length}")


class LoRA_LyCORIS_SD1_Config(LoRA_LyCORIS_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class LoRA_LyCORIS_SD2_Config(LoRA_LyCORIS_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class LoRA_LyCORIS_SDXL_Config(LoRA_LyCORIS_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class LoRA_LyCORIS_FLUX_Config(LoRA_LyCORIS_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)


class LoRA_LyCORIS_Flux2_Config(LoRA_LyCORIS_Config_Base, Config_Base):
    """Model config for FLUX.2 (Klein) LoRA models in LyCORIS format."""

    base: Literal[BaseModelType.Flux2] = Field(default=BaseModelType.Flux2)
    variant: Flux2VariantType | None = Field(default=None)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)
        raise_for_override_fields(cls, override_fields)
        cls._validate_looks_like_lora(mod)
        cls._validate_base(mod)
        override_fields.setdefault("variant", _get_flux2_lora_variant(mod.load_state_dict()))
        return cls(**override_fields)

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        if _get_flux_lora_format(mod) and _is_flux2_lora(mod):
            return BaseModelType.Flux2
        raise NotAMatchError("model is not a FLUX.2 LoRA")


class LoRA_LyCORIS_ZImage_Config(LoRA_LyCORIS_Config_Base, Config_Base):
    """Model config for Z-Image LoRA models in LyCORIS format."""

    base: Literal[BaseModelType.ZImage] = Field(default=BaseModelType.ZImage)
    variant: ZImageVariantType | None = Field(default=None)

    @classmethod
    def _validate_looks_like_lora(cls, mod: ModelOnDisk) -> None:
        """Z-Image LoRAs have different key patterns than SD/SDXL LoRAs.

        Z-Image LoRAs use keys like:
        - diffusion_model.layers.X.attention.to_k.lora_down.weight (DoRA format)
        - diffusion_model.layers.X.attention.to_k.lora_A.weight (PEFT format)
        - diffusion_model.layers.X.attention.to_k.dora_scale (DoRA scale)
        - lora_unet__layers_X_attention_to_k.lora_down.weight (Kohya format)
        """
        from invokeai.backend.patches.lora_conversions.z_image_lora_conversion_utils import (
            is_state_dict_likely_z_image_kohya_lora,
        )

        state_dict = mod.load_state_dict()

        # Check for Kohya format first
        if is_state_dict_likely_z_image_kohya_lora(state_dict):
            return

        # Check for Z-Image specific LoRA patterns (dot-notation formats)
        has_z_image_lora_keys = state_dict_has_any_keys_starting_with(
            state_dict,
            {
                "diffusion_model.layers.",  # Z-Image S3-DiT layer pattern
                "diffusion_model.context_refiner.",
                "diffusion_model.noise_refiner.",
                "transformer.layers.",  # OneTrainer/diffusers prefix variant
                "base_model.model.transformer.layers.",  # PEFT-wrapped variant
            },
        )

        # Also check for LoRA weight suffixes (various formats)
        has_lora_suffix = state_dict_has_any_keys_ending_with(
            state_dict,
            {
                "lora_A.weight",
                "lora_B.weight",
                "lora_down.weight",
                "lora_up.weight",
                "dora_scale",
            },
        )

        if has_z_image_lora_keys and has_lora_suffix:
            return

        raise NotAMatchError("model does not match Z-Image LoRA heuristics")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        """Z-Image LoRAs are identified by their diffusion_model.layers structure.

        Z-Image uses S3-DiT architecture with layer names like:
        - diffusion_model.layers.0.attention.to_k.lora_A.weight
        - diffusion_model.layers.0.feed_forward.w1.lora_A.weight
        - lora_unet__layers_0_attention_to_k.lora_down.weight (Kohya format)
        """
        from invokeai.backend.patches.lora_conversions.z_image_lora_conversion_utils import (
            is_state_dict_likely_z_image_kohya_lora,
        )

        state_dict = mod.load_state_dict()

        # Check for Kohya format
        if is_state_dict_likely_z_image_kohya_lora(state_dict):
            return BaseModelType.ZImage

        # Check for Z-Image transformer layer patterns (dot-notation formats)
        # Z-Image uses diffusion_model.layers.X structure (unlike Flux which uses double_blocks/single_blocks)
        has_z_image_keys = state_dict_has_any_keys_starting_with(
            state_dict,
            {
                "diffusion_model.layers.",  # Z-Image S3-DiT layer pattern
                "diffusion_model.context_refiner.",
                "diffusion_model.noise_refiner.",
                "transformer.layers.",  # OneTrainer/diffusers prefix variant
                "base_model.model.transformer.layers.",  # PEFT-wrapped variant
            },
        )

        # If it looks like a Z-Image LoRA, return ZImage base
        if has_z_image_keys:
            return BaseModelType.ZImage

        raise NotAMatchError("model does not look like a Z-Image LoRA")


class LoRA_LyCORIS_QwenImage_Config(LoRA_LyCORIS_Config_Base, Config_Base):
    """Model config for Qwen Image Edit LoRA models in LyCORIS format."""

    base: Literal[BaseModelType.QwenImage] = Field(default=BaseModelType.QwenImage)

    @classmethod
    def _validate_looks_like_lora(cls, mod: ModelOnDisk) -> None:
        """Qwen Image Edit LoRAs have keys like transformer_blocks.X.attn.to_k.lora_down.weight."""
        state_dict = mod.load_state_dict()

        has_qwen_ie_keys = state_dict_has_any_keys_starting_with(
            state_dict,
            {
                "transformer_blocks.",
                "transformer.transformer_blocks.",
                "lora_unet_transformer_blocks_",  # Kohya format
            },
        )
        has_lora_suffix = state_dict_has_any_keys_ending_with(
            state_dict,
            {
                "lora_A.weight",
                "lora_B.weight",
                "lora_down.weight",
                "lora_up.weight",
                "dora_scale",
                "lokr_w1",
                "lokr_w2",  # LoKR format
            },
        )
        # Must NOT have diffusion_model.layers (Z-Image) or Flux-style keys.
        # Flux LoRAs can have transformer.single_transformer_blocks or transformer.transformer_blocks
        # (with the "transformer." prefix and "single_" variant) which would falsely match our check.
        # Flux Kohya LoRAs use lora_unet_double_blocks or lora_unet_single_blocks.
        has_z_image_keys = state_dict_has_any_keys_starting_with(state_dict, {"diffusion_model.layers."})
        # Krea-2 LoRAs also carry transformer.transformer_blocks. keys, but uniquely include the
        # text-fusion stage. Exclude them here so they route to LoRA_LyCORIS_Krea2_Config.
        has_krea2_keys = _has_krea2_lora_keys(state_dict)
        has_flux_keys = state_dict_has_any_keys_starting_with(
            state_dict,
            {
                "double_blocks.",
                "single_blocks.",
                "single_transformer_blocks.",
                "transformer.single_transformer_blocks.",
                "lora_unet_double_blocks_",
                "lora_unet_single_blocks_",
                "lora_unet_single_transformer_blocks_",
            },
        )

        if has_qwen_ie_keys and has_lora_suffix and not has_z_image_keys and not has_krea2_keys and not has_flux_keys:
            return

        raise NotAMatchError("model does not match Qwen Image LoRA heuristics")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        state_dict = mod.load_state_dict()
        has_qwen_ie_keys = state_dict_has_any_keys_starting_with(
            state_dict,
            {"transformer_blocks.", "transformer.transformer_blocks.", "lora_unet_transformer_blocks_"},
        )
        has_z_image_keys = state_dict_has_any_keys_starting_with(state_dict, {"diffusion_model.layers."})
        has_krea2_keys = _has_krea2_lora_keys(state_dict)
        has_flux_keys = state_dict_has_any_keys_starting_with(
            state_dict,
            {
                "double_blocks.",
                "single_blocks.",
                "single_transformer_blocks.",
                "transformer.single_transformer_blocks.",
                "lora_unet_double_blocks_",
                "lora_unet_single_blocks_",
                "lora_unet_single_transformer_blocks_",
            },
        )

        if has_qwen_ie_keys and not has_z_image_keys and not has_krea2_keys and not has_flux_keys:
            return BaseModelType.QwenImage
        raise NotAMatchError("model does not look like a Qwen Image Edit LoRA")


def _has_krea2_lora_keys(state_dict: dict[str | int, Any]) -> bool:
    """True if the state dict targets Krea-2's distinctive modules.

    Covers both the diffusers naming (``text_fusion`` / ``time_mod_proj``) and the native/ComfyUI naming
    (``txtfusion``, or the gated attention ``attn.wq`` + ``attn.gate`` unique to Krea-2's single-stream
    blocks) so native-format LoRAs are recognized as Krea-2 rather than falling through to another base.
    """
    str_keys = [k for k in state_dict.keys() if isinstance(k, str)]
    if any(("text_fusion" in k or "txtfusion" in k or "time_mod_proj" in k) for k in str_keys):
        return True
    # Native gated attention identifies a transformer-only Krea-2 LoRA that lacks the text-fusion stage.
    return any(".attn.wq." in k for k in str_keys) and any(".attn.gate." in k for k in str_keys)


# Each LoRA weight half must be accompanied by its partner half. An orphaned half installs successfully
# but crashes later during LoRA conversion, so we reject it at identification time.
_LORA_PAIR_PARTNERS = {
    "lora_A.weight": "lora_B.weight",
    "lora_B.weight": "lora_A.weight",
    "lora_down.weight": "lora_up.weight",
    "lora_up.weight": "lora_down.weight",
}


def _lora_weight_keys_are_all_paired(state_dict: dict[str | int, Any], prefixes: tuple[str, ...] | None = None) -> bool:
    """True if *every* lora_A/lora_B/lora_down/lora_up weight (optionally restricted to `prefixes`) has its
    partner half present. Returns True when there are no such weights at all (nothing to invalidate)."""
    string_keys = {key for key in state_dict if isinstance(key, str)}
    for key in string_keys:
        if prefixes is not None and not key.startswith(prefixes):
            continue
        for suffix, partner_suffix in _LORA_PAIR_PARTNERS.items():
            if key.endswith(suffix):
                if f"{key[: -len(suffix)]}{partner_suffix}" not in string_keys:
                    return False
                break
    return True


def _has_complete_lora_pair(state_dict: dict[str | int, Any], prefixes: tuple[str, ...] | None = None) -> bool:
    """True if at least one complete lora_A/B (or lora_down/up) pair exists, optionally under `prefixes`.

    Note this only requires a *complete* pair to exist; it does not by itself reject dangling halves
    elsewhere — callers pair it with :func:`_lora_weight_keys_are_all_paired` (over the whole state dict)
    to also reject orphaned halves anywhere.
    """
    string_keys = {key for key in state_dict if isinstance(key, str)}
    for key in string_keys:
        if prefixes is not None and not key.startswith(prefixes):
            continue
        for suffix, partner_suffix in _LORA_PAIR_PARTNERS.items():
            if key.endswith(suffix) and f"{key[: -len(suffix)]}{partner_suffix}" in string_keys:
                return True
    return False


# LyCORIS LoKr layers carry their Kronecker factors instead of a lora_A/B (or lora_down/up) pair: either the
# full `lokr_w1`/`lokr_w2`, or the further-factored `lokr_w1_a`/`lokr_w1_b` / `lokr_w2_a`/`lokr_w2_b` (+ the
# optional `lokr_t2` tucker core). Each such layer is self-contained, so there is no "orphaned half" notion to
# check — presence of any factor is enough to call the layer complete.
_LOKR_WEIGHT_SUFFIXES = (
    ".lokr_w1",
    ".lokr_w2",
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_t2",
)


def _has_lokr_layer(state_dict: dict[str | int, Any], prefixes: tuple[str, ...] | None = None) -> bool:
    """True if the state dict contains at least one LoKr layer, optionally restricted to `prefixes`."""
    for key in state_dict:
        if not isinstance(key, str):
            continue
        if prefixes is not None and not key.startswith(prefixes):
            continue
        if key.endswith(_LOKR_WEIGHT_SUFFIXES):
            return True
    return False


# Layouts the converter understands for an explicit Krea-2 override (a transformer-only or text-encoder-only
# LoRA that lacks the auto-detection text_fusion/time_mod_proj keys still installs under an explicit base).
_KREA2_SUPPORTED_LORA_PREFIXES = (
    "transformer.transformer_blocks.",
    "transformer_blocks.",
    # The converter also supports the `diffusion_model.` transformer layout
    # (see krea2_lora_conversion_utils.lora_model_from_krea2_state_dict).
    "diffusion_model.transformer_blocks.",
    "diffusion_model.blocks.",
    "diffusion_model.txtfusion.",
    "diffusion_model.first.",
    "diffusion_model.tmlp.",
    "diffusion_model.tproj.",
    "diffusion_model.txtmlp.",
    "diffusion_model.last.linear.",
    "base_model.model.transformer.transformer_blocks.",
    "text_encoder.",
    "base_model.model.text_encoder.",
)


class LoRA_LyCORIS_Krea2_Config(LoRA_LyCORIS_Config_Base, Config_Base):
    """Model config for Krea-2 LoRA models in LyCORIS (single-file diffusers PEFT) format."""

    base: Literal[BaseModelType.Krea2] = Field(default=BaseModelType.Krea2)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)
        raise_for_override_fields(cls, override_fields)

        state_dict = mod.load_state_dict()
        explicit_krea2_override = override_fields.get("base") is BaseModelType.Krea2
        has_supported_explicit_pair = _has_complete_lora_pair(
            state_dict, _KREA2_SUPPORTED_LORA_PREFIXES
        ) or _has_lokr_layer(state_dict, _KREA2_SUPPORTED_LORA_PREFIXES)
        # Reject an orphaned half *anywhere* in the state dict (e.g. a dangling text_fusion half not under
        # the approved prefixes) — it would install here but fail during LoRA conversion at generation time.
        if explicit_krea2_override and has_supported_explicit_pair and _lora_weight_keys_are_all_paired(state_dict):
            return cls(**override_fields)

        cls._validate_looks_like_lora(mod)
        cls._validate_base(mod)
        return cls(**override_fields)

    @classmethod
    def _validate_looks_like_lora(cls, mod: ModelOnDisk) -> None:
        """Krea-2 LoRAs have keys like transformer.text_fusion.* / transformer.transformer_blocks.* with
        a lora_A/lora_B (or lora_down/lora_up) suffix. The text-fusion stage is unique to Krea-2."""
        state_dict = mod.load_state_dict()
        # Require a *complete* lora_A/B (or lora_down/up) pair, not merely any lora/dora suffix: a file with
        # only ``dora_scale`` and no A/B weights would pass a suffix check but fail later on missing weights.
        if not (
            _has_krea2_lora_keys(state_dict) and (_has_complete_lora_pair(state_dict) or _has_lokr_layer(state_dict))
        ):
            raise NotAMatchError(
                "model does not match Krea-2 LoRA heuristics (no complete lora_A/B, lora_down/up or LoKr layer)"
            )
        # Reject a file with an orphaned LoRA half (a valid layer plus a dangling lora_A/B/down/up); it
        # would install here but fail later during LoRA conversion.
        if not _lora_weight_keys_are_all_paired(state_dict):
            raise NotAMatchError("Krea-2 LoRA has an incomplete lora_A/B (or lora_down/up) weight pair")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        if _has_krea2_lora_keys(mod.load_state_dict()):
            return BaseModelType.Krea2
        raise NotAMatchError("model does not look like a Krea-2 LoRA")


class LoRA_LyCORIS_Anima_Config(LoRA_LyCORIS_Config_Base, Config_Base):
    """Model config for Anima LoRA models in LyCORIS format."""

    base: Literal[BaseModelType.Anima] = Field(default=BaseModelType.Anima)

    @classmethod
    def _validate_looks_like_lora(cls, mod: ModelOnDisk) -> None:
        """Anima LoRAs use Kohya-style keys targeting Cosmos DiT blocks.

        Anima LoRAs have keys like:
        - lora_unet_blocks_0_cross_attn_k_proj.lora_down.weight (Kohya format)
        - diffusion_model.blocks.0.cross_attn.k_proj.lora_A.weight (diffusers PEFT format)
        - transformer.blocks.0.mlp.layer_0.lora_A.weight (Anima-only MLP layer)

        Uses the **strict** Cosmos-DiT detectors, which require an
        Anima-exclusive subcomponent name (``mlp``, ``adaln_modulation``, or
        ``_proj``-suffixed attention). The loose detectors would also accept
        Wan-native LoRAs (which use ``cross_attn``/``self_attn`` too but with
        bare ``.q``/``.k``/``.v``/``.o`` rather than ``_proj``), so they're not
        safe for first-match-wins probing — see the regression tests in
        ``test_wan_lora_probe_independence.py``.
        """
        state_dict = mod.load_state_dict()
        str_keys = [k for k in state_dict.keys() if isinstance(k, str)]

        has_cosmos_keys = has_cosmos_dit_kohya_keys_strict(str_keys) or has_cosmos_dit_peft_keys_strict(str_keys)

        # Also check for LoRA/LoKR weight suffixes
        has_lora_suffix = state_dict_has_any_keys_ending_with(
            state_dict,
            {
                "lora_A.weight",
                "lora_B.weight",
                "lora_down.weight",
                "lora_up.weight",
                "dora_scale",
                ".lokr_w1",
                ".lokr_w2",
            },
        )

        if has_cosmos_keys and has_lora_suffix:
            return

        raise NotAMatchError("model does not match Anima LoRA heuristics")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        """Anima LoRAs target Cosmos DiT blocks (blocks.X.mlp, blocks.X.adaln_modulation,
        blocks.X.cross_attn.q_proj, etc.).

        Uses the strict Cosmos-DiT detectors to be mutually exclusive with
        Wan-LoRA detection — see ``_validate_looks_like_lora`` for rationale.
        """
        state_dict = mod.load_state_dict()
        str_keys = [k for k in state_dict.keys() if isinstance(k, str)]

        if has_cosmos_dit_kohya_keys_strict(str_keys) or has_cosmos_dit_peft_keys_strict(str_keys):
            return BaseModelType.Anima

        raise NotAMatchError("model does not look like an Anima LoRA")


class LoRA_LyCORIS_Wan_Config(LoRA_LyCORIS_Config_Base, Config_Base):
    """Model config for Wan 2.2 LoRA models in LyCORIS format.

    Wan LoRAs target ``WanTransformer3DModel`` blocks. The Wan 2.2 A14B family
    is dual-expert (high-noise + low-noise) — LoRAs are typically trained
    against one expert. ``expert`` records which one so the model loader
    invocation can wire it to the correct ``loras`` / ``loras_low_noise`` list.
    Many LoRAs are expert-agnostic (TI2V-5B family, or community LoRAs that
    just don't tag the expert) — these get ``expert=None`` and are applied to
    both experts by default.
    """

    base: Literal[BaseModelType.Wan] = Field(default=BaseModelType.Wan)
    expert: Literal["high", "low"] | None = Field(
        default=None,
        description="For Wan 2.2 A14B dual-expert LoRAs: 'high' targets the high-noise expert, "
        "'low' targets the low-noise expert. None means the LoRA is expert-agnostic "
        "(TI2V-5B, or community LoRAs without explicit tagging) and is applied to both.",
    )
    variant: WanLoRAVariantType | None = Field(
        default=None,
        description="The Wan model family this LoRA targets, detected from its inner-dim "
        "(5120 -> A14B, 3072 -> TI2V-5B). A14B LoRAs are incompatible with TI2V-5B mains "
        "(and vice versa) — they crash with a shape mismatch in the layer patcher. The "
        "linear-view graph builder filters LoRAs on variant when building the LoRA "
        "collection. None means the LoRA's inner-dim couldn't be identified.",
    )

    @classmethod
    def _validate_looks_like_lora(cls, mod: ModelOnDisk) -> None:
        """Wan LoRAs target attn1/attn2/ffn.net (diffusers form) or self_attn/cross_attn/ffn.N (native form)."""
        state_dict = mod.load_state_dict()
        str_keys = [k for k in state_dict.keys() if isinstance(k, str)]

        has_wan_keys = has_wan_kohya_keys(str_keys) or has_wan_peft_keys(str_keys)
        has_lora_suffix = state_dict_has_any_keys_ending_with(
            state_dict,
            {
                "lora_A.weight",
                "lora_B.weight",
                "lora_down.weight",
                "lora_up.weight",
                "dora_scale",
                ".lokr_w1",
                ".lokr_w2",
            },
        )

        # Reject if any non-Wan architecture signature is present. Without this
        # guard a Wan LoRA could be falsely identified by Anima (cross_attn /
        # self_attn name collision) or vice versa.
        if has_wan_keys and has_lora_suffix and not has_non_wan_architecture_keys(str_keys):
            return

        raise NotAMatchError("model does not match Wan LoRA heuristics")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        state_dict = mod.load_state_dict()
        str_keys = [k for k in state_dict.keys() if isinstance(k, str)]

        if (has_wan_kohya_keys(str_keys) or has_wan_peft_keys(str_keys)) and not has_non_wan_architecture_keys(
            str_keys
        ):
            return BaseModelType.Wan

        raise NotAMatchError("model does not look like a Wan LoRA")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        # Run the base-class probe (file-check, lora-suffix, base detection).
        instance = super().from_model_on_disk(mod, override_fields)

        # Auto-detect the model-family variant from inner_dim in the state
        # dict. The override field skips this if the user has set it.
        #
        # Resolved *before* the expert tag because the expert is only meaningful for
        # A14B — see below.
        if instance.variant is None:
            instance.variant = detect_wan_lora_variant(mod.load_state_dict())

        # Auto-detect the expert tag from the filename if the user didn't override
        # it, using the same helper as the transformer probes so the two can't drift
        # apart. That also picks up the bare ``HIGH``/``LOW`` convention, which
        # matters here: an expert-specific LoRA left untagged is applied to *both*
        # experts by the Wan LoRA loader, which is wrong for the high/low pairs the
        # Lightning-style distills ship in.
        #
        # TI2V-5B is single-transformer, so it has no experts and the denoise path
        # reads only the primary LoRA list. Tagging a 5B LoRA would route it through
        # ``_resolve_target("auto", ...)`` into ``loras_low_noise`` alone, where it is
        # silently inert. The bare-token convention makes that reachable on ordinary
        # names — ``Wan2.2_TI2V_5B_low_light_v2`` has ``low`` as a standalone token —
        # so pin the field the same way ``_resolve_wan_expert`` pins the main-model
        # probe. Only A14B (or an inconclusive variant) gets a tag.
        #
        # Note 'none' vs None: this config uses None for "untagged, apply to both",
        # so a 'none' result must leave the field alone.
        if instance.expert is None and instance.variant != WanLoRAVariantType.Wan5B:
            detected = _detect_wan_expert(mod.path.stem)
            if detected != "none":
                instance.expert = detected

        return instance


class ControlAdapter_Config_Base(ABC, BaseModel):
    default_settings: ControlAdapterDefaultSettings | None = Field(None)


class ControlLoRA_LyCORIS_FLUX_Config(ControlAdapter_Config_Base, Config_Base):
    """Model config for Control LoRA models."""

    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)
    type: Literal[ModelType.ControlLoRa] = Field(default=ModelType.ControlLoRa)
    format: Literal[ModelFormat.LyCORIS] = Field(default=ModelFormat.LyCORIS)

    trigger_phrases: set[str] | None = Field(None)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_control_lora(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_looks_like_control_lora(cls, mod: ModelOnDisk) -> None:
        state_dict = mod.load_state_dict()

        if not is_state_dict_likely_flux_control(state_dict):
            raise NotAMatchError("model state dict does not look like a Flux Control LoRA")


class LoRA_Diffusers_Config_Base(LoRA_Config_Base):
    """Model config for LoRA/Diffusers models."""

    # TODO(psyche): Needs base handling. For FLUX, the Diffusers format does not indicate a folder model; it indicates
    # the weights format. FLUX Diffusers LoRAs are single files.

    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_base(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatchError(f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        if _get_flux_lora_format(mod):
            if _is_flux2_lora(mod):
                return BaseModelType.Flux2
            return BaseModelType.Flux

        # If we've gotten here, we assume that the LoRA is a Stable Diffusion LoRA
        path_to_weight_file = cls._get_weight_file_or_raise(mod)
        state_dict = mod.load_state_dict(path_to_weight_file)
        token_vector_length = lora_token_vector_length(state_dict)

        match token_vector_length:
            case 768:
                return BaseModelType.StableDiffusion1
            case 1024:
                return BaseModelType.StableDiffusion2
            case 1280:
                return BaseModelType.StableDiffusionXL  # recognizes format at https://civitai.com/models/224641
            case 2048:
                return BaseModelType.StableDiffusionXL
            case _:
                # Some SDXL LoRAs (e.g. self-attention-only "slider" LoRAs) target only the
                # UNet and lack the cross-attention / text-encoder keys that
                # lora_token_vector_length() needs. Fall back to detecting SDXL from the
                # UNet's deep transformer-block structure.
                if _state_dict_looks_like_sdxl_unet_lora(state_dict):
                    return BaseModelType.StableDiffusionXL
                raise NotAMatchError(f"unrecognized token vector length {token_vector_length}")

    @classmethod
    def _get_weight_file_or_raise(cls, mod: ModelOnDisk) -> Path:
        suffixes = ["bin", "safetensors"]
        weight_files = [mod.path / f"pytorch_lora_weights.{sfx}" for sfx in suffixes]
        for wf in weight_files:
            if wf.exists():
                return wf
        raise NotAMatchError("missing pytorch_lora_weights.bin or pytorch_lora_weights.safetensors")


class LoRA_Diffusers_SD1_Config(LoRA_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class LoRA_Diffusers_SD2_Config(LoRA_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class LoRA_Diffusers_SDXL_Config(LoRA_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class LoRA_Diffusers_FLUX_Config(LoRA_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)


class LoRA_Diffusers_Flux2_Config(LoRA_Diffusers_Config_Base, Config_Base):
    """Model config for FLUX.2 (Klein) LoRA models in Diffusers format."""

    base: Literal[BaseModelType.Flux2] = Field(default=BaseModelType.Flux2)
    variant: Flux2VariantType | None = Field(default=None)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)
        raise_for_override_fields(cls, override_fields)
        cls._validate_base(mod)
        path_to_weight_file = cls._get_weight_file_or_raise(mod)
        state_dict = mod.load_state_dict(path_to_weight_file)
        override_fields.setdefault("variant", _get_flux2_lora_variant(state_dict))
        return cls(**override_fields)

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        path_to_weight_file = cls._get_weight_file_or_raise(mod)
        state_dict = mod.load_state_dict(path_to_weight_file)
        if _is_flux2_lora_state_dict(state_dict):
            return BaseModelType.Flux2
        raise NotAMatchError("model is not a FLUX.2 Diffusers LoRA")


class LoRA_Diffusers_ZImage_Config(LoRA_Diffusers_Config_Base, Config_Base):
    """Model config for Z-Image LoRA models in Diffusers format."""

    base: Literal[BaseModelType.ZImage] = Field(default=BaseModelType.ZImage)
    variant: ZImageVariantType | None = Field(default=None)
