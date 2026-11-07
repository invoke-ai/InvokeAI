"""FLUX.3 Image LoRAs onto diffusers' `Flux3Transformer2DModel`; which keys a file may hold is `flux3_lora_constants`."""

from collections.abc import Mapping

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.swapped_halves_layer_patch import SwappedHalvesLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.flux3_lora_constants import (
    FLUX3_LORA_TRANSFORMER_PREFIX,
    describe_refusal,
    peft_scale,
    read_flux3_lora_keys,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)


def lora_model_from_flux3_state_dict(
    state_dict: dict[str, torch.Tensor], *, name: str, metadata: Mapping[str, str] | None = None
) -> ModelPatchRaw:
    """Convert the FLUX.3 LoRA `name`, in any of the layouts `flux3_lora_constants` reads, to a patch for the diffusers
    transformer, keyed `FLUX3_LORA_TRANSFORMER_PREFIX` + module path. `metadata` is the file's safetensors header.

    A layer's `alpha` in the file sets its scale. A low-rank layer without one takes the scale PEFT gives it where the
    header holds diffusers' PEFT config (`peft_scale`), and 1 otherwise (its alpha is its rank), as ComfyUI applies it.
    A layer on BFL's output modulation is applied with its row halves swapped. Shapes are not checked here (see
    `flux3_lora_constants`).
    """
    keys = read_flux3_lora_keys(state_dict)
    if keys.refused:
        # Identification refuses these files, so one reaches here only by another path; applying the rest would make a
        # partial LoRA that passes for a whole one.
        raise ValueError(describe_refusal(keys.refused))
    scale = peft_scale(metadata)
    if keys.pdd_heads:
        logger.warning(
            f"LoRA '{name}' carries PDD's per-step output heads ({', '.join(keys.pdd_heads)}), which InvokeAI does "
            "not use: it runs as a plain few-step LoRA, not with PDD's own sampler."
        )

    layers: dict[str, BaseLayerPatch] = {}
    for module, sources in keys.layers.items():
        values = {value_key: state_dict[key] for value_key, key in sources.items()}
        if scale is not None and "alpha" not in values and "lora_down.weight" in values:
            # The layer classes scale by alpha / rank.
            values["alpha"] = torch.tensor(scale * values["lora_down.weight"].shape[0])
        layer = any_lora_layer_from_state_dict(values)
        if module in keys.swapped:
            layer = SwappedHalvesLayerPatch(layer)
        layers[f"{FLUX3_LORA_TRANSFORMER_PREFIX}{module}"] = layer
    return ModelPatchRaw(layers=layers)
