"""Convert an Ideogram 4 LoRA into a ``ModelPatchRaw`` for the vendored Ideogram 4 transformer.

Every Ideogram 4 build -- the Diffusers pair, the single-file checkpoints and the GGUFs -- loads into
``backend/ideogram4/modeling_ideogram4.py``, whose attention keeps one fused ``attention.qkv`` and an ``attention.o``.
ai-toolkit LoRAs (the bulk of what is published) already use those names behind ComfyUI's ``diffusion_model.``, so
they convert one-to-one, LoKR included.

diffusers' training script splits the projections (``attention.to_q`` / ``to_k`` / ``to_v`` / ``to_out.0``). Those
are fused into a single low-rank layer on ``attention.qkv``: the three ``down`` matrices stacked, the three ``up``
matrices on a block diagonal, each block carrying its own ``alpha / rank``. That is exact -- the fused weight is the
row-wise concatenation of the three, and so is the product -- and it keeps the patch an ordinary ``LoRALayer``, which
the sidecar of every quantized build applies in activation space instead of materializing a 13824 x 4608 delta.
"""

from typing import Dict

import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict
from invokeai.backend.patches.lora_conversions.ideogram4_lora_constants import (
    IDEOGRAM4_LORA_TRANSFORMER_PREFIX,
    group_ideogram4_lora_layers,
    ideogram4_lora_refusal,
    strip_ideogram4_prefix,
)
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

_SPLIT_PROJECTIONS = ("to_q", "to_k", "to_v")


def lora_model_from_ideogram4_state_dict(state_dict: Dict[str, torch.Tensor]) -> ModelPatchRaw:
    """Convert an Ideogram 4 LoRA state dict into a ``ModelPatchRaw`` over the vendored transformer's modules."""
    # The probe refuses these at install; this covers a state dict that reaches the conversion another way.
    refusal = ideogram4_lora_refusal(state_dict)
    if refusal is not None:
        raise ValueError(f"Ideogram 4 LoRA cannot be applied: {refusal}")

    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for layer_path, values in group_ideogram4_lora_layers(state_dict).items():
        grouped[strip_ideogram4_prefix(layer_path)] = _normalize_lora_param_names(values)

    layers: dict[str, BaseLayerPatch] = {}
    split_attention: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for path, values in grouped.items():
        head, _, leaf = path.rpartition(".")
        if head.endswith(".attention") and leaf in _SPLIT_PROJECTIONS:
            split_attention.setdefault(head, {})[leaf] = values
            continue
        if path.endswith(".attention.to_out.0"):
            path = path.removesuffix(".attention.to_out.0") + ".attention.o"
        layers[f"{IDEOGRAM4_LORA_TRANSFORMER_PREFIX}{path}"] = any_lora_layer_from_state_dict(values)

    for attention, projections in split_attention.items():
        layers[f"{IDEOGRAM4_LORA_TRANSFORMER_PREFIX}{attention}.qkv"] = _fused_qkv_layer(projections)

    return ModelPatchRaw(layers=layers)


def _fused_qkv_layer(projections: dict[str, dict[str, torch.Tensor]]) -> LoRALayer:
    """One low-rank layer on the fused `qkv` that adds exactly what the split q/k/v layers would.

    A projection the file does not train keeps a zero block, so its rows of the fused weight stay untouched.
    """
    present = [name for name in _SPLIT_PROJECTIONS if name in projections]
    out_features = projections[present[0]]["lora_up.weight"].shape[0]
    downs = [projections[name]["lora_down.weight"] for name in present]
    up = torch.zeros(
        3 * out_features, sum(d.shape[0] for d in downs), dtype=projections[present[0]]["lora_up.weight"].dtype
    )
    col = 0
    for name, down in zip(present, downs, strict=True):
        values = projections[name]
        rank = down.shape[0]
        scale = float(values["alpha"]) / rank if "alpha" in values else 1.0
        row = _SPLIT_PROJECTIONS.index(name) * out_features
        up[row : row + out_features, col : col + rank] = values["lora_up.weight"] * scale
        col += rank
    return LoRALayer(up=up, mid=None, down=torch.cat(downs, dim=0), alpha=None, bias=None)


def _normalize_lora_param_names(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map PEFT ``lora_A``/``lora_B`` onto the ``lora_down``/``lora_up`` names the layer factory reads."""
    if "lora_A.weight" not in values:
        return values
    normalized = {k: v for k, v in values.items() if k not in ("lora_A.weight", "lora_B.weight")}
    normalized["lora_down.weight"] = values["lora_A.weight"]
    normalized["lora_up.weight"] = values["lora_B.weight"]
    return normalized
