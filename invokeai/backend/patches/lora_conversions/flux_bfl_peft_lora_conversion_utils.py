"""Utilities for detecting and converting FLUX LoRAs in BFL PEFT format.

This format uses BFL internal key names (double_blocks, single_blocks, etc.) with a
'diffusion_model.' prefix and PEFT-style LoRA suffixes (lora_A.weight, lora_B.weight).

Example keys:
    diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight
    diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight
    diffusion_model.single_blocks.0.linear1.lora_A.weight

This format is used by some training tools (e.g. SimpleTuner, ComfyUI-based trainers)
and is common for FLUX.2 Klein LoRAs.
"""

import re
from typing import Dict

import torch

from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

# The prefix used in BFL PEFT format LoRAs
_BFL_PEFT_PREFIX = "diffusion_model."

# Key patterns that identify FLUX architecture in BFL format
_BFL_FLUX_BLOCK_PREFIXES = (
    f"{_BFL_PEFT_PREFIX}double_blocks.",
    f"{_BFL_PEFT_PREFIX}single_blocks.",
)

# Regex patterns for converting BFL layer names to diffusers naming (for FLUX.2 Klein).
# BFL uses fused QKV, diffusers uses separate Q/K/V for double blocks.
_DOUBLE_BLOCK_RE = re.compile(r"^double_blocks\.(\d+)\.(.+)$")
_SINGLE_BLOCK_RE = re.compile(r"^single_blocks\.(\d+)\.(.+)$")

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


def is_state_dict_likely_in_flux_bfl_peft_format(state_dict: dict[str | int, torch.Tensor]) -> bool:
    """Checks if the provided state dict is likely in the BFL PEFT FLUX LoRA format.

    This format uses 'diffusion_model.' prefix with BFL key names and PEFT LoRA suffixes.
    """
    str_keys = [k for k in state_dict.keys() if isinstance(k, str)]
    if not str_keys:
        return False

    # All keys must be in PEFT format (lora_A.weight / lora_B.weight)
    all_peft = all(k.endswith(("lora_A.weight", "lora_B.weight")) for k in str_keys)
    if not all_peft:
        return False

    # Must have at least some keys with the diffusion_model. prefix and FLUX block structure
    has_flux_blocks = any(k.startswith(_BFL_FLUX_BLOCK_PREFIXES) for k in str_keys)
    if not has_flux_blocks:
        return False

    # All keys should start with the diffusion_model. prefix
    all_have_prefix = all(k.startswith(_BFL_PEFT_PREFIX) for k in str_keys)

    return all_have_prefix


def lora_model_from_flux_bfl_peft_state_dict(
    state_dict: Dict[str, torch.Tensor], alpha: float | None = None
) -> ModelPatchRaw:
    """Convert a BFL PEFT format FLUX LoRA state dict to a ModelPatchRaw.

    The conversion is straightforward: strip the 'diffusion_model.' prefix to get
    the BFL internal key names, which are already the format used by InvokeAI internally.
    """
    # Group keys by layer
    grouped_state_dict: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        # Strip the diffusion_model. prefix
        if isinstance(key, str) and key.startswith(_BFL_PEFT_PREFIX):
            key = key[len(_BFL_PEFT_PREFIX) :]

        # Split off the lora_A.weight / lora_B.weight suffix
        parts = key.rsplit(".", maxsplit=2)
        layer_name = parts[0]
        suffix = ".".join(parts[1:])

        if layer_name not in grouped_state_dict:
            grouped_state_dict[layer_name] = {}

        # Convert PEFT naming to InvokeAI naming
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
    """Convert a BFL PEFT format FLUX LoRA state dict for use with FLUX.2 Klein (diffusers model).

    FLUX.2 Klein models are loaded as Flux2Transformer2DModel (diffusers), which uses different
    layer naming than BFL's internal format:
      - double_blocks.{i} → transformer_blocks.{i}
      - single_blocks.{i} → single_transformer_blocks.{i}
      - Fused QKV (img_attn.qkv) → separate Q/K/V (attn.to_q, attn.to_k, attn.to_v)

    This function converts BFL PEFT keys to diffusers naming and splits fused QKV LoRAs
    into separate Q/K/V LoRA layers.
    """
    # First, strip the diffusion_model. prefix and group by BFL layer name with PEFT→InvokeAI naming.
    grouped_state_dict: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        if isinstance(key, str) and key.startswith(_BFL_PEFT_PREFIX):
            key = key[len(_BFL_PEFT_PREFIX) :]

        parts = key.rsplit(".", maxsplit=2)
        layer_name = parts[0]
        suffix = ".".join(parts[1:])

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
    """Convert a single BFL-named LoRA layer to one or more diffusers-named layers.

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
            return _split_qkv_lora(
                layer_sd,
                q_key=f"{prefix}.attn.to_q",
                k_key=f"{prefix}.attn.to_k",
                v_key=f"{prefix}.attn.to_v",
            )
        # Fused text QKV → split into separate Q, K, V
        if rest == "txt_attn.qkv":
            return _split_qkv_lora(
                layer_sd,
                q_key=f"{prefix}.attn.add_q_proj",
                k_key=f"{prefix}.attn.add_k_proj",
                v_key=f"{prefix}.attn.add_v_proj",
            )
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

    # Non-block keys (embedders, etc.) - pass through unchanged
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
