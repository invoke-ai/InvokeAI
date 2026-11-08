"""Utilities for detecting and converting FLUX LoRAs in BFL PEFT format.

This format uses BFL internal key names (double_blocks, single_blocks, etc.) with a
'diffusion_model.' prefix and PEFT-style LoRA suffixes (lora_A.weight, lora_B.weight).
LyCORIS variants (LoKR, LoHA, etc.) are also supported, using their respective weight key
suffixes (lokr_w1, lokr_w2, hada_w1_a, etc.) in place of the PEFT suffixes.

Example keys (LoRA PEFT):
    diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight
    diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight
    diffusion_model.single_blocks.0.linear1.lora_A.weight

Example keys (LoKR):
    diffusion_model.double_blocks.0.img_attn.proj.lokr_w1
    diffusion_model.double_blocks.0.img_attn.proj.lokr_w2
    diffusion_model.single_blocks.0.linear1.lokr_w1

This format is used by some training tools (e.g. SimpleTuner, ComfyUI-based trainers)
and is common for FLUX.2 Klein LoRAs.
"""

import re
from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.loha_layer import LoHALayer
from invokeai.backend.patches.layers.lokr_layer import LoKRLayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.layers.merged_layer_patch import MergedLayerPatch, Range
from invokeai.backend.patches.layers.swapped_halves_layer_patch import SwappedHalvesLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)

# The prefixes used in BFL PEFT format LoRAs.
# Most commonly "diffusion_model.", but some PEFT-wrapped variants use "base_model.model.".
_BFL_PEFT_PREFIX = "diffusion_model."
_PEFT_BASE_MODEL_PREFIX = "base_model.model."
_BFL_PEFT_PREFIXES = (_BFL_PEFT_PREFIX, _PEFT_BASE_MODEL_PREFIX)

# Key patterns that identify FLUX architecture in BFL format
_BFL_FLUX_BLOCK_PREFIXES = (
    f"{_BFL_PEFT_PREFIX}double_blocks.",
    f"{_BFL_PEFT_PREFIX}single_blocks.",
    f"{_PEFT_BASE_MODEL_PREFIX}double_blocks.",
    f"{_PEFT_BASE_MODEL_PREFIX}single_blocks.",
)

# Regex patterns for converting BFL layer names to diffusers naming (for FLUX.2 Klein).
# BFL uses fused QKV, diffusers uses separate Q/K/V for double blocks.
_DOUBLE_BLOCK_RE = re.compile(r"^double_blocks\.(\d+)\.(.+)$")
_SINGLE_BLOCK_RE = re.compile(r"^single_blocks\.(\d+)\.(.+)$")

# Weight key suffixes used by PEFT LoRA in BFL format.
_BFL_PEFT_LORA_SUFFIXES = ("lora_A.weight", "lora_B.weight")

# Weight key suffixes used by LyCORIS algorithms (LoKR, LoHA, etc.) in BFL format.
# These are single-component suffixes (no dot), unlike the two-component PEFT suffixes.
_BFL_LYCORIS_WEIGHT_SUFFIXES = (
    # LoKR
    "lokr_w1",
    "lokr_w2",
    "lokr_w1_a",
    "lokr_w1_b",
    "lokr_w2_a",
    "lokr_w2_b",
    "lokr_t2",
    # LoHA
    "hada_w1_a",
    "hada_w1_b",
    "hada_w2_a",
    "hada_w2_b",
    "hada_t1",
    "hada_t2",
    # Common to all LyCORIS algorithms
    "alpha",
    "dora_scale",
    "bias_indices",
    "bias_values",
    "bias_size",
    # Full/Diff
    "diff",
)

# All recognized BFL weight key suffixes (both PEFT and LyCORIS).
_BFL_ALL_WEIGHT_SUFFIXES = _BFL_PEFT_LORA_SUFFIXES + _BFL_LYCORIS_WEIGHT_SUFFIXES

# Mapping of BFL double block layer suffixes to diffusers equivalents (simple renames).
_DOUBLE_BLOCK_RENAMES: dict[str, str] = {
    "img_attn.proj": "attn.to_out.0",
    "txt_attn.proj": "attn.to_add_out",
    "img_mlp.0": "ff.linear_in",
    "img_mlp.2": "ff.linear_out",
    "txt_mlp.0": "ff_context.linear_in",
    "txt_mlp.2": "ff_context.linear_out",
}

# Mapping of BFL single block layer suffixes to diffusers equivalents.
_SINGLE_BLOCK_RENAMES: dict[str, str] = {
    "linear1": "attn.to_qkv_mlp_proj",
    "linear2": "attn.to_out",
}

# Mapping of BFL non-block layer names to diffusers equivalents.
# These are top-level modules (embedders, modulations, output layers) that use different
# names in BFL's FLUX.2 model vs the diffusers Flux2Transformer2DModel.
_NON_BLOCK_RENAMES: dict[str, str] = {
    "img_in": "x_embedder",
    "txt_in": "context_embedder",
    "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
    "single_stream_modulation.lin": "single_stream_modulation.linear",
    "final_layer.linear": "proj_out",
    "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
}

# The final-layer modulation also orders its output halves differently: (shift, scale) in BFL's LastLayer,
# (scale, shift) in diffusers' AdaLayerNormContinuous. Its patches are wrapped in a SwappedHalvesLayerPatch.
_BFL_FINAL_MODULATION_KEY = "final_layer.adaLN_modulation.1"
_DIFFUSERS_FINAL_MODULATION_KEY = "norm_out.linear"


def is_state_dict_likely_in_flux_bfl_peft_format(state_dict: dict[str | int, torch.Tensor]) -> bool:
    """Checks if the provided state dict is likely in the BFL PEFT FLUX LoRA/LyCORIS format.

    This format uses BFL key names (double_blocks, single_blocks, img_attn, etc.) with either
    PEFT LoRA suffixes (lora_A.weight, lora_B.weight) or LyCORIS algorithm suffixes (lokr_w1,
    lokr_w2, hada_w1_a, etc.). The keys may be prefixed with either 'diffusion_model.'
    (common for ComfyUI/SimpleTuner) or 'base_model.model.' (PEFT-wrapped variant).
    """
    str_keys = [k for k in state_dict.keys() if isinstance(k, str)]
    if not str_keys:
        return False

    # All keys must use recognized weight suffixes (PEFT LoRA or LyCORIS).
    all_valid_suffixes = all(k.endswith(_BFL_ALL_WEIGHT_SUFFIXES) for k in str_keys)
    if not all_valid_suffixes:
        return False

    # Must have at least some keys with FLUX block structure (double_blocks/single_blocks)
    has_flux_blocks = any(k.startswith(_BFL_FLUX_BLOCK_PREFIXES) for k in str_keys)
    if not has_flux_blocks:
        return False

    # All keys should share the same recognized prefix
    all_have_prefix = all(k.startswith(_BFL_PEFT_PREFIXES) for k in str_keys)

    return all_have_prefix


def _strip_bfl_peft_prefix(key: str) -> str:
    """Strip the BFL PEFT prefix ('diffusion_model.' or 'base_model.model.') from a key."""
    for prefix in _BFL_PEFT_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _split_bfl_key(key: str) -> tuple[str, str]:
    """Split a BFL key (after prefix stripping) into (layer_name, weight_suffix).

    Handles:
    - 2-component suffixes ending with '.weight': e.g., 'lora_A.weight', 'lora_B.weight'
    - 1-component suffixes: e.g., 'lokr_w1', 'lokr_w2', 'alpha', 'dora_scale'
    """
    if key.endswith(".weight"):
        # 2-component suffix: e.g., 'lora_A.weight' → split at last 2 dots
        parts = key.rsplit(".", maxsplit=2)
        return parts[0], f"{parts[1]}.{parts[2]}"
    else:
        # 1-component suffix: e.g., 'lokr_w1', 'alpha' → split at last dot
        parts = key.rsplit(".", maxsplit=1)
        return parts[0], parts[1]


def lora_model_from_flux_bfl_peft_state_dict(
    state_dict: Dict[str, torch.Tensor], alpha: float | None = None
) -> ModelPatchRaw:
    """Convert a BFL PEFT/LyCORIS format FLUX LoRA state dict to a ModelPatchRaw.

    The conversion is straightforward: strip the prefix ('diffusion_model.' or 'base_model.model.')
    to get the BFL internal key names, which are already the format used by InvokeAI internally.
    Supports both PEFT LoRA (lora_A.weight / lora_B.weight) and LyCORIS algorithms (LoKR, LoHA, etc.).
    """
    # Group keys by layer
    grouped_state_dict: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        # Strip the prefix
        if isinstance(key, str):
            key = _strip_bfl_peft_prefix(key)

        layer_name, suffix = _split_bfl_key(key)

        if layer_name not in grouped_state_dict:
            grouped_state_dict[layer_name] = {}

        # Convert PEFT naming to InvokeAI naming; LyCORIS keys pass through unchanged.
        if suffix == "lora_A.weight":
            grouped_state_dict[layer_name]["lora_down.weight"] = value
        elif suffix == "lora_B.weight":
            grouped_state_dict[layer_name]["lora_up.weight"] = value
        else:
            grouped_state_dict[layer_name][suffix] = value

    # Add alpha if provided
    if alpha is not None:
        for layer_state_dict in grouped_state_dict.values():
            layer_state_dict["alpha"] = torch.tensor(alpha)

    # Build LoRA layers with the transformer prefix
    layers = {}
    for layer_key, layer_state_dict in grouped_state_dict.items():
        layers[f"{FLUX_LORA_TRANSFORMER_PREFIX}{layer_key}"] = any_lora_layer_from_state_dict(layer_state_dict)

    return ModelPatchRaw(layers=layers)


def lora_model_from_flux2_bfl_peft_state_dict(
    state_dict: Dict[str, torch.Tensor], alpha: float | None = None
) -> ModelPatchRaw:
    """Convert a BFL PEFT/LyCORIS format FLUX LoRA state dict for use with FLUX.2 Klein (diffusers model).

    FLUX.2 Klein models are loaded as Flux2Transformer2DModel (diffusers), which uses different
    layer naming than BFL's internal format:
      - double_blocks.{i} → transformer_blocks.{i}
      - single_blocks.{i} → single_transformer_blocks.{i}
      - Fused QKV (img_attn.qkv) → separate Q/K/V (attn.to_q, attn.to_k, attn.to_v)

    This function converts BFL PEFT/LyCORIS keys to diffusers naming and splits fused QKV updates
    into separate Q/K/V layers.
    """
    # First, strip the prefix and group by BFL layer name with PEFT→InvokeAI naming.
    grouped_state_dict: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        if isinstance(key, str):
            key = _strip_bfl_peft_prefix(key)

        layer_name, suffix = _split_bfl_key(key)

        if layer_name not in grouped_state_dict:
            grouped_state_dict[layer_name] = {}

        if suffix == "lora_A.weight":
            grouped_state_dict[layer_name]["lora_down.weight"] = value
        elif suffix == "lora_B.weight":
            grouped_state_dict[layer_name]["lora_up.weight"] = value
        else:
            grouped_state_dict[layer_name][suffix] = value

    if alpha is not None:
        for layer_state_dict in grouped_state_dict.values():
            layer_state_dict["alpha"] = torch.tensor(alpha)

    # Now convert BFL layer names to diffusers naming, splitting fused QKV as needed.
    layers: dict[str, BaseLayerPatch] = {}
    for bfl_key, layer_sd in grouped_state_dict.items():
        layer = any_lora_layer_from_state_dict(layer_sd)
        diffusers_layers = _convert_bfl_layer_patch_to_diffusers(bfl_key, layer)
        for diff_key, diff_layer in diffusers_layers:
            layers[f"{FLUX_LORA_TRANSFORMER_PREFIX}{diff_key}"] = diff_layer

    return ModelPatchRaw(layers=layers)


def convert_bfl_lora_patch_to_diffusers(patch: ModelPatchRaw) -> ModelPatchRaw:
    """Convert a ModelPatchRaw with BFL-format layer keys to diffusers-format keys.

    This handles LoRAs that were loaded with the FLUX.1 BFL PEFT converter (which keeps BFL keys)
    but need to be applied to a FLUX.2 Klein model (which uses diffusers module names).

    If the patch doesn't contain BFL-format keys, it is returned unchanged.
    """
    prefix = FLUX_LORA_TRANSFORMER_PREFIX
    prefix_len = len(prefix)

    # Check if any layer keys are in BFL format (contain double_blocks or single_blocks)
    has_bfl_keys = any(
        k.startswith(prefix)
        and (k[prefix_len:].startswith("double_blocks.") or k[prefix_len:].startswith("single_blocks."))
        for k in patch.layers
    )
    if not has_bfl_keys:
        return patch

    new_layers: dict[str, BaseLayerPatch] = {}
    for layer_key, layer in patch.layers.items():
        if not layer_key.startswith(prefix):
            new_layers[layer_key] = layer
            continue

        bfl_key = layer_key[prefix_len:]
        converted = _convert_bfl_layer_patch_to_diffusers(bfl_key, layer)
        for diff_key, diff_layer in converted:
            new_layers[f"{prefix}{diff_key}"] = diff_layer

    return ModelPatchRaw(layers=new_layers)


def _convert_bfl_layer_patch_to_diffusers(bfl_key: str, layer: BaseLayerPatch) -> list[tuple[str, BaseLayerPatch]]:
    """Convert a single BFL-named LoRA layer patch to one or more diffusers-named patches.

    For simple renames, the layer object is reused. Fused QKV LoRA, LoHa, and LoKr patches keep
    their factorized representation after splitting.
    """
    # Double blocks
    m = _DOUBLE_BLOCK_RE.match(bfl_key)
    if m:
        idx, rest = m.group(1), m.group(2)
        diff_prefix = f"transformer_blocks.{idx}"

        # Fused QKV → split into separate Q, K, V
        if rest == "img_attn.qkv":
            return _split_qkv_layer_patch(
                layer,
                q_key=f"{diff_prefix}.attn.to_q",
                k_key=f"{diff_prefix}.attn.to_k",
                v_key=f"{diff_prefix}.attn.to_v",
            )
        if rest == "txt_attn.qkv":
            return _split_qkv_layer_patch(
                layer,
                q_key=f"{diff_prefix}.attn.add_q_proj",
                k_key=f"{diff_prefix}.attn.add_k_proj",
                v_key=f"{diff_prefix}.attn.add_v_proj",
            )
        # Simple renames
        if rest in _DOUBLE_BLOCK_RENAMES:
            return [(f"{diff_prefix}.{_DOUBLE_BLOCK_RENAMES[rest]}", layer)]
        return [(f"{diff_prefix}.{rest}", layer)]

    # Single blocks
    m = _SINGLE_BLOCK_RE.match(bfl_key)
    if m:
        idx, rest = m.group(1), m.group(2)
        diff_prefix = f"single_transformer_blocks.{idx}"

        if rest in _SINGLE_BLOCK_RENAMES:
            return [(f"{diff_prefix}.{_SINGLE_BLOCK_RENAMES[rest]}", layer)]
        return [(f"{diff_prefix}.{rest}", layer)]

    # Non-block keys (embedders, modulations, output layers)
    if bfl_key == _BFL_FINAL_MODULATION_KEY:
        return [(_DIFFUSERS_FINAL_MODULATION_KEY, SwappedHalvesLayerPatch(layer))]
    if bfl_key in _NON_BLOCK_RENAMES:
        return [(_NON_BLOCK_RENAMES[bfl_key], layer)]

    # Fallback: pass through unchanged
    return [(bfl_key, layer)]


def _split_qkv_lora_layer(
    layer: LoRALayer,
    q_key: str,
    k_key: str,
    v_key: str,
) -> list[tuple[str, LoRALayer]]:
    """Split a fused QKV LoRALayer into separate Q, K, V LoRALayers.

    The up weight [3*hidden, rank] is split into 3 parts.
    The down weight [rank, hidden] is cloned for each.
    """
    up_q, up_k, up_v = layer.up.chunk(3, dim=0)

    result = []
    for key, up_part in [(q_key, up_q), (k_key, up_k), (v_key, up_v)]:
        split_layer = LoRALayer(
            up=up_part,
            mid=None,
            down=layer.down.clone(),
            alpha=layer._alpha,
            bias=None,
        )
        result.append((key, split_layer))

    return result


def _split_qkv_bias(
    bias: torch.Tensor | None, out_features: int
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if bias is None:
        return (None, None, None)

    dense_bias = bias.to_dense() if bias.is_sparse else bias
    if dense_bias.shape[0] != out_features or out_features % 3 != 0:
        raise ValueError(
            f"Expected a fused QKV bias with {out_features} output rows, got shape {tuple(dense_bias.shape)}."
        )
    return dense_bias.chunk(3, dim=0)


def _split_qkv_loha_layer(
    layer: LoHALayer,
    q_key: str,
    k_key: str,
    v_key: str,
) -> list[tuple[str, LoHALayer]]:
    """Split a fused QKV LoHa by slicing its output factors without materializing its update."""
    # Standard linear LoHa stores [out, rank]. CP-decomposed convolutional LoHa stores [rank, out].
    out_axis = 0 if layer.t1 is None else 1
    out_features = layer.w1_a.shape[out_axis]
    if layer.w2_a.shape[out_axis] != out_features or out_features % 3 != 0:
        raise ValueError(
            "Expected matching fused QKV LoHa output factors with a dimension divisible by 3, "
            f"got {tuple(layer.w1_a.shape)} and {tuple(layer.w2_a.shape)}."
        )

    w1_a_parts = layer.w1_a.chunk(3, dim=out_axis)
    w2_a_parts = layer.w2_a.chunk(3, dim=out_axis)
    bias_parts = _split_qkv_bias(layer.bias, out_features)
    result: list[tuple[str, LoHALayer]] = []
    for key, w1_a, w2_a, bias in zip((q_key, k_key, v_key), w1_a_parts, w2_a_parts, bias_parts, strict=True):
        result.append(
            (
                key,
                LoHALayer(
                    w1_a=w1_a,
                    w1_b=layer.w1_b,
                    w2_a=w2_a,
                    w2_b=layer.w2_b,
                    t1=layer.t1,
                    t2=layer.t2,
                    alpha=layer._alpha,
                    bias=bias,
                ),
            )
        )
    return result


def _lokr_operand_out_features(layer: LoKRLayer) -> tuple[int, int]:
    if layer.w1 is not None:
        w1_out_features = layer.w1.shape[0]
    else:
        assert layer.w1_a is not None
        w1_out_features = layer.w1_a.shape[0]

    if layer.w2 is not None:
        w2_out_features = layer.w2.shape[0]
    else:
        assert layer.w2_a is not None
        if layer.t2 is None:
            w2_out_features = layer.w2_a.shape[0]
        else:
            # With CP decomposition, einsum reconstructs w2 as [p, r, k, l] from w2_a [i, p].
            w2_out_features = layer.w2_a.shape[1]
    return w1_out_features, w2_out_features


def _slice_lokr_layer(
    layer: LoKRLayer,
    w1_start: int,
    w1_end: int,
    w2_start: int,
    w2_end: int,
    bias: torch.Tensor | None,
) -> LoKRLayer:
    if layer.w1 is not None:
        w1 = layer.w1[w1_start:w1_end]
        w1_a = None
    else:
        w1 = None
        assert layer.w1_a is not None
        w1_a = layer.w1_a[w1_start:w1_end]

    if layer.w2 is not None:
        w2 = layer.w2[w2_start:w2_end]
        w2_a = None
    else:
        w2 = None
        assert layer.w2_a is not None
        w2_a = layer.w2_a[w2_start:w2_end] if layer.t2 is None else layer.w2_a[:, w2_start:w2_end]

    return LoKRLayer(
        w1=w1,
        w1_a=w1_a,
        w1_b=layer.w1_b,
        w2=w2,
        w2_a=w2_a,
        w2_b=layer.w2_b,
        t2=layer.t2,
        alpha=layer._alpha,
        bias=bias,
    )


def _split_qkv_lokr_layer(
    layer: LoKRLayer,
    q_key: str,
    k_key: str,
    v_key: str,
) -> list[tuple[str, BaseLayerPatch]]:
    """Split a fused QKV LoKr while retaining its Kronecker factors.

    Rows of ``kron(w1, w2)`` are grouped into ``w2``-sized blocks, one for each output row of
    ``w1``. The common case has a w1 output dimension divisible by three, so each projection is
    another LoKr obtained by slicing w1. For other layouts, each target row range is represented
    by the exact intersecting LoKr segments in a MergedLayerPatch.
    """
    w1_out_features, w2_out_features = _lokr_operand_out_features(layer)
    fused_out_features = w1_out_features * w2_out_features
    if fused_out_features % 3 != 0:
        raise ValueError(f"Expected fused QKV LoKr output rows divisible by 3, got {fused_out_features}.")

    keys = (q_key, k_key, v_key)
    target_out_features = fused_out_features // 3
    dense_bias = layer.bias.to_dense() if layer.bias is not None and layer.bias.is_sparse else layer.bias
    if dense_bias is not None and dense_bias.shape[0] != fused_out_features:
        raise ValueError(
            f"Expected a fused QKV LoKr bias with {fused_out_features} output rows, "
            f"got shape {tuple(dense_bias.shape)}."
        )

    # Fast path: each Q/K/V range consists of whole w2 row blocks.
    if w1_out_features % 3 == 0:
        w1_rows_per_target = w1_out_features // 3
        result: list[tuple[str, BaseLayerPatch]] = []
        for target_idx, key in enumerate(keys):
            w1_start = target_idx * w1_rows_per_target
            w1_end = w1_start + w1_rows_per_target
            bias = (
                None
                if dense_bias is None
                else dense_bias[target_idx * target_out_features : (target_idx + 1) * target_out_features]
            )
            result.append(
                (
                    key,
                    _slice_lokr_layer(
                        layer,
                        w1_start=w1_start,
                        w1_end=w1_end,
                        w2_start=0,
                        w2_end=w2_out_features,
                        bias=bias,
                    ),
                )
            )
        return result

    result = []
    for target_idx, key in enumerate(keys):
        target_start = target_idx * target_out_features
        target_end = target_start + target_out_features
        sub_layers: list[BaseLayerPatch] = []
        sub_ranges: list[Range] = []

        for w1_row in range(w1_out_features):
            block_start = w1_row * w2_out_features
            block_end = block_start + w2_out_features
            overlap_start = max(target_start, block_start)
            overlap_end = min(target_end, block_end)
            if overlap_start >= overlap_end:
                continue

            w2_start = overlap_start - block_start
            w2_end = overlap_end - block_start
            bias = None if dense_bias is None else dense_bias[overlap_start:overlap_end]
            sub_layers.append(
                _slice_lokr_layer(
                    layer,
                    w1_start=w1_row,
                    w1_end=w1_row + 1,
                    w2_start=w2_start,
                    w2_end=w2_end,
                    bias=bias,
                )
            )
            sub_ranges.append(Range(overlap_start - target_start, overlap_end - target_start))

        if len(sub_layers) == 1 and sub_ranges[0] == Range(0, target_out_features):
            result.append((key, sub_layers[0]))
        else:
            result.append((key, MergedLayerPatch(sub_layers, sub_ranges)))

    return result


def _split_qkv_layer_patch(
    layer: BaseLayerPatch,
    q_key: str,
    k_key: str,
    v_key: str,
) -> list[tuple[str, BaseLayerPatch]]:
    if isinstance(layer, LoRALayer):
        return _split_qkv_lora_layer(layer, q_key=q_key, k_key=k_key, v_key=v_key)
    if isinstance(layer, LoHALayer):
        return _split_qkv_loha_layer(layer, q_key=q_key, k_key=k_key, v_key=v_key)
    if isinstance(layer, LoKRLayer):
        return _split_qkv_lokr_layer(layer, q_key=q_key, k_key=k_key, v_key=v_key)

    logger.warning(f"Unsupported {type(layer).__name__} layer type for a fused QKV split; layer will be skipped.")
    return []
