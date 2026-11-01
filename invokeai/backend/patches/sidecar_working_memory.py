"""The room a transformer's LoRAs need while it samples, beside its activations."""

from collections.abc import Iterable, Mapping

import torch

from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.dora_layer import DoRALayer
from invokeai.backend.patches.layers.full_layer import FullLayer
from invokeai.backend.patches.layers.loha_layer import LoHALayer
from invokeai.backend.patches.layers.lokr_layer import LoKRLayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.layers.swapped_halves_layer_patch import SwappedHalvesLayerPatch
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.dequantizing_linear import requires_sidecar_patching
from invokeai.backend.quantization.fp8_scaled import count_fp8_weights

# `out_features x in_features` tensors a sidecar holds at once while it turns a layer of each kind into a weight
# update (`autocast_linear_forward_sidecar_patches`): the weight cast or dequantized to the compute dtype, plus the
# layer's own. A full layer scales its diff into a new tensor; LoKR builds kron(w1, w2), then scales it; LoHA holds
# its two products and theirs. DoRA holds its scaled delta, the updated weight and their difference, 4 in all; the
# input-dim (LyCORIS) magnitude's column norm, over a transposed view, measured another third of a copy on CUDA
# (gate_layer, bf16), so DoRA is priced at 5. A low-rank layer forms none: it adds its output only.
_SIDECAR_WEIGHT_COPIES: dict[type[BaseLayerPatch], int] = {
    FullLayer: 2,
    LoKRLayer: 3,
    LoHALayer: 4,
    DoRALayer: 5,
}
# The same for a layer in a `SwappedHalvesLayerPatch`, which swaps the weight its layer reads and the update it
# returns, one copy each, and takes no low-rank shortcut: the update of every kind is formed. Measured on the CPU with
# an fp8 weight: 4 copies for a low-rank, full or LoKR layer and 5 for LoHA and DoRA; DoRA is priced one higher for
# the column norm CUDA adds, as above.
_SWAPPED_SIDECAR_WEIGHT_COPIES: dict[type[BaseLayerPatch], int] = {
    LoRALayer: 4,
    FullLayer: 4,
    LoKRLayer: 4,
    LoHALayer: 5,
    DoRALayer: 6,
}


def _elements(layer: BaseLayerPatch) -> int:
    """Elements of a patch layer's tensors, which a sidecar holds in the compute dtype whatever it was loaded in."""
    if isinstance(layer, SwappedHalvesLayerPatch):
        layer = layer.layer
    return sum(value.numel() for value in vars(layer).values() if isinstance(value, torch.Tensor))


def _weight_copies(layer: BaseLayerPatch) -> int:
    if isinstance(layer, SwappedHalvesLayerPatch):
        return _SWAPPED_SIDECAR_WEIGHT_COPIES[type(layer.layer)]
    return _SIDECAR_WEIGHT_COPIES[type(layer)]


def lora_sidecar_bytes(
    patches: Iterable[ModelPatchRaw],
    transformer: torch.nn.Module,
    model_format: ModelFormat,
    *,
    prefix: str,
    rows: int,
    per_sample_rows: Mapping[str, int],
    dtype: torch.dtype,
) -> int:
    """Room the LoRA `patches` need on `transformer` while it samples in `dtype`, beside the activations.

    A layer merged into the weights computes its update once, before the first step, in room the activations take
    only afterwards. A layer run as a sidecar holds its tensors in the compute dtype, and each forward of its module
    adds:

    - an update of the module's output, `rows` rows (those `per_sample_rows` names: its own count, for a Linear that
      reads the timestep embedding), with a low-rank layer's `rank` columns beside it;
    - for any layer but a low-rank one, the `out_features x in_features` tensors `_SIDECAR_WEIGHT_COPIES` counts for
      its kind, one more from the second such layer on the module (the running sum) and two from the third (the
      previous layer's result as well). An int8 Linear hands over only its weight's shape, so there one copy is
      counted that is not made.

    Modules run one after another, so only the largest such update counts. On GGUF, int8 and fp8 weights every layer
    is a sidecar, kept on the device: all the sidecars' tensors are counted, plus that update. Other formats merge, but
    the patcher wraps any layer partial loading leaves on the CPU, and that sidecar copies its tensors to the device
    for each forward; which layers stay on the CPU is unknown before the transformer is locked, so one module's worth
    is reserved: the largest of its tensors plus its update. Only layers keyed `prefix` + module path are counted, and
    only on modules `transformer` has: the patcher skips the others.
    """
    patches = list(patches)
    if not patches:
        return 0
    modules = dict(transformer.named_modules())
    # Per patched module: its layers' elements, the largest rank among its low-rank layers, and the weight copies of
    # each of its other layers.
    elements: dict[str, int] = {}
    ranks: dict[str, int] = {}
    copies: dict[str, list[int]] = {}
    for patch in patches:
        for key, layer in patch.layers.items():
            if not key.startswith(prefix):
                continue
            path = key[len(prefix) :]
            if path not in modules:
                continue
            elements[path] = elements.get(path, 0) + _elements(layer)
            if isinstance(layer, LoRALayer):
                ranks[path] = max(ranks.get(path, 0), layer.down.shape[0])
            else:
                copies.setdefault(path, []).append(_weight_copies(layer))
    updates: dict[str, int] = {}
    for path in elements:
        module = modules[path]
        summed = copies.get(path, [])
        weight_copies = max(summed) + min(len(summed) - 1, 2) if summed else 0
        updates[path] = (
            per_sample_rows.get(path, rows) * (module.out_features + ranks.get(path, 0))
            + weight_copies * module.out_features * module.in_features
        )
    if requires_sidecar_patching(transformer, model_format) or count_fp8_weights(transformer):
        return (sum(elements.values()) + max(updates.values(), default=0)) * dtype.itemsize
    return max((elements[path] + updates[path] for path in elements), default=0) * dtype.itemsize
