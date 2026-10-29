"""Convert an ERNIE-Image LoRA into a ``ModelPatchRaw`` for the ERNIE-Image transformer.

Published ERNIE-Image LoRAs already use the module names of diffusers' ``ErnieImageTransformer2DModel``, which
is also the layout every ERNIE-Image transformer build loads into: the Diffusers folder, the single-file
checkpoint and the community GGUFs all keep diffusers' key names (see ``load/model_loaders/ernie_image.py``).
So there is no name map to keep in step with a loader -- the conversion strips the one container prefix the
trainer wrote (``diffusion_model.`` for ai-toolkit and Comfy, ``transformer.`` for diffusers/PEFT exports,
``base_model.model.`` for PEFT adapters) and builds one layer per module.

ai-toolkit writes PEFT ``lora_A``/``lora_B`` pairs with no ``.alpha`` tensors (alpha == rank, so the patch
applies unscaled); a kohya-style ``alpha`` tensor in the file wins over that default.
"""

from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.ernie_image_lora_constants import (
    ERNIE_IMAGE_LORA_KEY_PREFIXES,
    ERNIE_IMAGE_LORA_TRANSFORMER_PREFIX,
    ernie_image_lora_refusal,
    split_ernie_image_lora_key,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw


def lora_model_from_ernie_image_state_dict(state_dict: Dict[str, torch.Tensor]) -> ModelPatchRaw:
    """Convert an ERNIE-Image LoRA state dict into a ``ModelPatchRaw``.

    Layer keys in the returned patch are the transformer's module paths, prefixed with
    ``ERNIE_IMAGE_LORA_TRANSFORMER_PREFIX``.
    """
    # The probe refuses these at install; this covers a state dict that reaches the conversion another way.
    refusal = ernie_image_lora_refusal([k for k in state_dict.keys() if isinstance(k, str)])
    if refusal is not None:
        raise ValueError(f"ERNIE-Image LoRA cannot be applied: {refusal}")

    layers: dict[str, BaseLayerPatch] = {}
    for layer_path, values in _group_by_layer(state_dict).items():
        layers[f"{ERNIE_IMAGE_LORA_TRANSFORMER_PREFIX}{layer_path}"] = any_lora_layer_from_state_dict(
            _normalize_lora_param_names(values)
        )
    return ModelPatchRaw(layers=layers)


def _strip_prefix(layer_path: str) -> str:
    for prefix in ERNIE_IMAGE_LORA_KEY_PREFIXES:
        if layer_path.startswith(prefix):
            return layer_path[len(prefix) :]
    return layer_path


def _group_by_layer(state_dict: Dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Group the tensors by module path, after stripping the container prefix."""
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        split = split_ernie_image_lora_key(key)
        assert split is not None  # unsupported tails were refused by the caller
        layer_path, param_name = split
        grouped.setdefault(_strip_prefix(layer_path), {})[param_name] = value

    # A low-rank pair split across the file would otherwise surface as a bare KeyError inside the layer
    # factory, after the transformer has been loaded for the generation.
    for layer_path, values in grouped.items():
        has_down = bool({"lora_A.weight", "lora_down.weight"} & values.keys())
        has_up = bool({"lora_B.weight", "lora_up.weight"} & values.keys())
        if has_down != has_up:
            raise ValueError(
                f"ERNIE-Image LoRA layer {layer_path!r} is missing one half of its low-rank pair "
                f"(has {sorted(values)}). The file is truncated or was edited."
            )
    return grouped


def _normalize_lora_param_names(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map PEFT ``lora_A``/``lora_B`` onto the ``lora_down``/``lora_up`` names the layer factory reads."""
    if "lora_A.weight" not in values:
        return values
    normalized = {k: v for k, v in values.items() if k not in ("lora_A.weight", "lora_B.weight")}
    normalized["lora_down.weight"] = values["lora_A.weight"]
    normalized["lora_up.weight"] = values["lora_B.weight"]
    return normalized
