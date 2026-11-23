"""Convert an SD 3 / 3.5 LoRA into a ``ModelPatchRaw`` for diffusers' ``SD3Transformer2DModel`` and the CLIP encoders.

InvokeAI loads SD 3 only from Diffusers folders, so diffusers' module names are the target for every layout:

- diffusers / ai-toolkit / PEFT files already use them behind a container prefix;
- SimpleTuner's LyCORIS files use them flattened (`lycoris_transformer_blocks_0_attn_to_q`), resolved against the
  module templates in ``sd3_lora_constants``;
- kohya sd-scripts files use the original MMDiT names. Their fused attention projections are split the way diffusers'
  single-file conversion splits the weights (rows in thirds, the shared ``down`` reused), and the adaLN projections
  whose two output halves diffusers stores swapped -- the last context block's and the final layer's, recognized by an
  output twice the input -- have their ``up`` rows swapped to match.

CLIP-L and CLIP-G layers are kept under their own prefixes; T5 layers are dropped with a warning, because the T5
encoder takes no LoRA here.
"""

from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.sd3_lora_constants import (
    SD3_LORA_CLIP_G_PREFIX,
    SD3_LORA_CLIP_L_PREFIX,
    SD3_LORA_TRANSFORMER_PREFIX,
    group_sd3_lora_layers,
    kohya_mmdit_swaps_halves,
    kohya_mmdit_targets,
    resolve_sd3_transformer_module,
    sd3_lora_refusal,
    sd3_text_encoder_target,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)

_CLIP_PREFIXES = {"clip_l": SD3_LORA_CLIP_L_PREFIX, "clip_g": SD3_LORA_CLIP_G_PREFIX}


def lora_model_from_sd3_state_dict(state_dict: Dict[str, torch.Tensor]) -> ModelPatchRaw:
    """Convert an SD 3 LoRA state dict into a ``ModelPatchRaw`` over the transformer and the two CLIP encoders."""
    # The probe refuses these at install; this covers a state dict that reaches the conversion another way.
    refusal = sd3_lora_refusal(state_dict)
    if refusal is not None:
        raise ValueError(f"SD 3 LoRA cannot be applied: {refusal}")

    layers: dict[str, BaseLayerPatch] = {}
    dropped_t5_layers = 0
    for layer_path, values in group_sd3_lora_layers(state_dict).items():
        values = _normalize_lora_param_names(values)
        if (targets := kohya_mmdit_targets(layer_path)) is not None:
            for target, target_values in zip(
                targets, _split_kohya_layer(layer_path, values, len(targets)), strict=True
            ):
                layers[f"{SD3_LORA_TRANSFORMER_PREFIX}{target}"] = any_lora_layer_from_state_dict(target_values)
        elif (module := resolve_sd3_transformer_module(layer_path)) is not None:
            layers[f"{SD3_LORA_TRANSFORMER_PREFIX}{module}"] = any_lora_layer_from_state_dict(values)
        else:
            target = sd3_text_encoder_target(layer_path)
            assert target is not None  # anything else was refused above
            encoder, module = target
            if module is None:
                dropped_t5_layers += 1
                continue
            layers[f"{_CLIP_PREFIXES[encoder]}{module}"] = any_lora_layer_from_state_dict(values)

    if dropped_t5_layers:
        logger.warning(
            f"SD 3 LoRA: dropped {dropped_t5_layers} T5 text-encoder layer(s); the T5 encoder does not take LoRAs."
        )
    return ModelPatchRaw(layers=layers)


def _split_kohya_layer(layer_path: str, values: dict[str, torch.Tensor], parts: int) -> list[dict[str, torch.Tensor]]:
    """A kohya layer's tensors per diffusers target: fused rows split in ``parts``, swapped halves put back.

    Only a low-rank layer gets here with ``parts > 1`` or swapped halves; the refusal turns anything else away.
    """
    swaps = kohya_mmdit_swaps_halves(layer_path, values)
    if parts == 1 and not swaps:
        return [values]
    up = values["lora_up.weight"]
    if swaps:
        shift, scale = up.chunk(2, dim=0)
        up = torch.cat([scale, shift], dim=0)
    return [{**values, "lora_up.weight": chunk} for chunk in up.chunk(parts, dim=0)]


def _normalize_lora_param_names(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map PEFT ``lora_A``/``lora_B`` onto the ``lora_down``/``lora_up`` names the layer factory reads."""
    if "lora_A.weight" not in values:
        return values
    normalized = {k: v for k, v in values.items() if k not in ("lora_A.weight", "lora_B.weight")}
    normalized["lora_down.weight"] = values["lora_A.weight"]
    normalized["lora_up.weight"] = values["lora_B.weight"]
    return normalized
