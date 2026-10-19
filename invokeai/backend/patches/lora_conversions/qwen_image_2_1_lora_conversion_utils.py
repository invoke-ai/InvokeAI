"""Qwen-Image-2.1 LoRAs onto the diffusers `QwenImage21Transformer2DModel`; see `qwen_image_2_1_lora_constants`."""

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.qwen_image_2_1_lora_constants import (
    GATE_UP,
    QWEN_IMAGE_21_LORA_NORM_PREFIX,
    QWEN_IMAGE_21_LORA_TRANSFORMER_PREFIX,
    is_norm,
    split_key,
    unsupported_keys,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

# What splits along with the output rows of a fused gate_up layer.
_ROW_KEYS = ("lora_up.weight", "dora_magnitude", "diff")


def _split_gate_up(module: str, values: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """The fused gated-MLP input as diffusers' two Linears: `gate_layer` takes the first half of the output rows.

    Every row-indexed tensor splits in two; the `down` matrix reads the shared input and goes to both halves.
    """
    stem = module[: -len("gate_up")]
    gate, proj = dict(values), dict(values)
    for key in _ROW_KEYS:
        if key in values:
            rows = values[key]
            if rows.shape[0] % 2:
                raise ValueError(f"LoRA layer {module!r} has {rows.shape[0]} output rows; gate_up has two halves.")
            gate[key], proj[key] = rows[: rows.shape[0] // 2], rows[rows.shape[0] // 2 :]
    if "lora_down.weight" in values:
        # A copy of its own, so the model cache, which sizes a patch by its tensors' elements, counts what it holds.
        proj["lora_down.weight"] = values["lora_down.weight"].clone()
    return {f"{stem}gate_layer": gate, f"{stem}proj": proj}


def lora_model_from_qwen_image_21_state_dict(state_dict: dict[str, torch.Tensor]) -> ModelPatchRaw:
    """Convert a Qwen-Image-2.1 LoRA to a patch for the diffusers transformer.

    A layer without an `alpha` scales by 1 (its alpha is its rank), as ComfyUI applies it. Norm diffs go under
    `QWEN_IMAGE_21_LORA_NORM_PREFIX`, the Linears under `QWEN_IMAGE_21_LORA_TRANSFORMER_PREFIX`.
    """
    unsupported = unsupported_keys(state_dict)
    if unsupported:
        # Identification refuses these, so a file reaches here only by another path; applying the rest would make a
        # partial LoRA that passes for a whole one.
        raise ValueError(
            f"{len(unsupported)} part(s) of this LoRA do not apply to Qwen-Image-2.1, e.g. {unsupported[0]!r}: "
            "it targets layers this model does not have, a norm with anything but a full diff, or a LoKR, LoHA "
            "or LyCORIS DoRA layer on the fused gate_up projection."
        )

    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        split = split_key(key)
        assert split is not None  # unsupported_keys checked every key
        module, value_key = split
        grouped.setdefault(module, {})[value_key] = value

    layers: dict[str, BaseLayerPatch] = {}
    for module, values in grouped.items():
        targets = _split_gate_up(module, values) if module.endswith(GATE_UP) else {module: values}
        prefix = QWEN_IMAGE_21_LORA_NORM_PREFIX if is_norm(module) else QWEN_IMAGE_21_LORA_TRANSFORMER_PREFIX
        for target, target_values in targets.items():
            layers[f"{prefix}{target}"] = any_lora_layer_from_state_dict(target_values)
    return ModelPatchRaw(layers=layers)
