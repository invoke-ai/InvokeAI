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
from invokeai.backend.patches.layers.lora_layer import LoRALayer
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

    This function converts BFL PEFT/LyCORIS keys to diffusers naming and splits fused QKV LoRAs
    into separate Q/K/V LoRA layers.
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
    layers: dict[str, any] = {}
    for bfl_key, layer_sd in grouped_state_dict.items():
        diffusers_layers = _convert_bfl_layer_to_diffusers(bfl_key, layer_sd)
        for diff_key, diff_sd in diffusers_layers:
            layers[f"{FLUX_LORA_TRANSFORMER_PREFIX}{diff_key}"] = any_lora_layer_from_state_dict(diff_sd)

    return ModelPatchRaw(layers=layers)


def _convert_bfl_layer_to_diffusers(
    bfl_key: str, layer_sd: dict[str, torch.Tensor]
) -> list[tuple[str, dict[str, torch.Tensor]]]:
    """Convert a single BFL-named LoRA/LyCORIS layer to one or more diffusers-named layers.

    Returns a list of (diffusers_key, layer_state_dict) tuples. Most layers produce one entry,
    but fused QKV layers are split into three separate Q/K/V entries.
    """
    # Double blocks
    m = _DOUBLE_BLOCK_RE.match(bfl_key)
    if m:
        idx, rest = m.group(1), m.group(2)
        prefix = f"transformer_blocks.{idx}"

        # Fused image QKV → split into separate Q, K, V
        if rest == "img_attn.qkv":
            if "lora_down.weight" in layer_sd:
                return _split_qkv_lora(
                    layer_sd,
                    q_key=f"{prefix}.attn.to_q",
                    k_key=f"{prefix}.attn.to_k",
                    v_key=f"{prefix}.attn.to_v",
                )
            elif "lokr_w1" in layer_sd or "lokr_w1_a" in layer_sd:
                return _split_qkv_lokr(
                    layer_sd,
                    q_key=f"{prefix}.attn.to_q",
                    k_key=f"{prefix}.attn.to_k",
                    v_key=f"{prefix}.attn.to_v",
                )
            else:
                logger.warning(f"Unsupported layer type for QKV split in {bfl_key}; layer will be skipped.")
                return []
        # Fused text QKV → split into separate Q, K, V
        if rest == "txt_attn.qkv":
            if "lora_down.weight" in layer_sd:
                return _split_qkv_lora(
                    layer_sd,
                    q_key=f"{prefix}.attn.add_q_proj",
                    k_key=f"{prefix}.attn.add_k_proj",
                    v_key=f"{prefix}.attn.add_v_proj",
                )
            elif "lokr_w1" in layer_sd or "lokr_w1_a" in layer_sd:
                return _split_qkv_lokr(
                    layer_sd,
                    q_key=f"{prefix}.attn.add_q_proj",
                    k_key=f"{prefix}.attn.add_k_proj",
                    v_key=f"{prefix}.attn.add_v_proj",
                )
            else:
                logger.warning(f"Unsupported layer type for QKV split in {bfl_key}; layer will be skipped.")
                return []
        # Simple renames
        if rest in _DOUBLE_BLOCK_RENAMES:
            return [(f"{prefix}.{_DOUBLE_BLOCK_RENAMES[rest]}", layer_sd)]

        # Fallback: keep as-is under the new prefix
        return [(f"{prefix}.{rest}", layer_sd)]

    # Single blocks
    m = _SINGLE_BLOCK_RE.match(bfl_key)
    if m:
        idx, rest = m.group(1), m.group(2)
        prefix = f"single_transformer_blocks.{idx}"

        if rest in _SINGLE_BLOCK_RENAMES:
            return [(f"{prefix}.{_SINGLE_BLOCK_RENAMES[rest]}", layer_sd)]

        return [(f"{prefix}.{rest}", layer_sd)]

    # Non-block keys (embedders, modulations, output layers)
    if bfl_key in _NON_BLOCK_RENAMES:
        return [(_NON_BLOCK_RENAMES[bfl_key], layer_sd)]

    # Fallback: pass through unchanged
    return [(bfl_key, layer_sd)]


def _split_qkv_lora(
    layer_sd: dict[str, torch.Tensor],
    q_key: str,
    k_key: str,
    v_key: str,
) -> list[tuple[str, dict[str, torch.Tensor]]]:
    """Split a fused QKV LoRA layer into separate Q, K, V LoRA layers.

    BFL uses fused QKV: lora_down [rank, hidden], lora_up [3*hidden, rank].
    Diffusers uses separate layers: each gets lora_down (shared/cloned) and a third of lora_up.
    """
    lora_down = layer_sd["lora_down.weight"]  # [rank, hidden]
    lora_up = layer_sd["lora_up.weight"]  # [3*hidden, rank]
    alpha = layer_sd.get("alpha")

    # Split lora_up into 3 equal parts along dim 0
    up_q, up_k, up_v = lora_up.chunk(3, dim=0)

    result = []
    for key, up_part in [(q_key, up_q), (k_key, up_k), (v_key, up_v)]:
        sd: dict[str, torch.Tensor] = {
            "lora_down.weight": lora_down.clone(),
            "lora_up.weight": up_part,
        }
        if alpha is not None:
            sd["alpha"] = alpha
        result.append((key, sd))

    return result


def _split_qkv_lokr(
    layer_sd: dict[str, torch.Tensor],
    q_key: str,
    k_key: str,
    v_key: str,
) -> list[tuple[str, dict[str, torch.Tensor]]]:
    """Split a fused QKV LoKR layer into separate Q, K, V full (diff) layers.

    LoKR uses a Kronecker product which cannot be split cleanly, so we compute the full weight
    matrix and store each third as a full weight update (diff).

    BFL uses fused QKV: full weight [3*hidden, hidden].
    Diffusers uses separate layers: each gets a [hidden, hidden] weight slice.

    For factorized LOKR (w1_a/w1_b), the alpha/rank scale is baked into the diff weights because
    FullLayer always uses scale=1.0.
    """
    w1 = layer_sd.get("lokr_w1")
    w1_a = layer_sd.get("lokr_w1_a")
    w1_b = layer_sd.get("lokr_w1_b")
    w2 = layer_sd.get("lokr_w2")
    w2_a = layer_sd.get("lokr_w2_a")
    w2_b = layer_sd.get("lokr_w2_b")
    t2 = layer_sd.get("lokr_t2")
    alpha = layer_sd.get("alpha")

    # Compute rank for scaling (only valid for factorized LOKR).
    if w1_b is not None:
        rank: int | None = w1_b.shape[0]
    elif w2_b is not None:
        rank = w2_b.shape[0]
    else:
        rank = None

    if w1 is None:
        assert w1_a is not None and w1_b is not None
        w1 = w1_a @ w1_b
    if w2 is None:
        assert w2_a is not None and w2_b is not None
        if t2 is not None:
            w2 = torch.einsum("i j k l, i p, j r -> p r k l", t2, w2_a, w2_b)
        else:
            w2 = w2_a @ w2_b

    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)

    full_weight = torch.kron(w1, w2)  # [3*hidden, hidden]

    # For factorized LOKR, bake the alpha/rank scale into the weight because FullLayer.scale()
    # always returns 1.0 (it has no alpha). For non-factorized LOKR, rank is None and scale is 1.0.
    if rank is not None and alpha is not None:
        scale = alpha.item() / rank
        full_weight = full_weight * scale

    weight_q, weight_k, weight_v = full_weight.chunk(3, dim=0)

    result = []
    for key, weight_part in [(q_key, weight_q), (k_key, weight_k), (v_key, weight_v)]:
        result.append((key, {"diff": weight_part}))

    return result


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

    For simple renames, the layer object is reused. For QKV splits, new LoRALayer objects
    are created with split up-weights and cloned down-weights.
    """
    # Double blocks
    m = _DOUBLE_BLOCK_RE.match(bfl_key)
    if m:
        idx, rest = m.group(1), m.group(2)
        diff_prefix = f"transformer_blocks.{idx}"

        # Fused QKV → split into separate Q, K, V
        if rest == "img_attn.qkv" and isinstance(layer, LoRALayer):
            return _split_qkv_lora_layer(
                layer,
                q_key=f"{diff_prefix}.attn.to_q",
                k_key=f"{diff_prefix}.attn.to_k",
                v_key=f"{diff_prefix}.attn.to_v",
            )
        if rest == "txt_attn.qkv" and isinstance(layer, LoRALayer):
            return _split_qkv_lora_layer(
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
