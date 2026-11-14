import copy

import torch

from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.layers.merged_layer_patch import MergedLayerPatch, Range


def _lora(out_features: int, dtype: torch.dtype) -> LoRALayer:
    return LoRALayer(
        up=torch.ones(out_features, 2, dtype=dtype),
        mid=None,
        down=torch.ones(2, 4, dtype=dtype),
        alpha=None,
        bias=None,
    )


def _placement(lora: LoRALayer) -> tuple[torch.device, torch.dtype, torch.device, torch.dtype]:
    return lora.up.device, lora.up.dtype, lora.down.device, lora.down.dtype


def test_moving_a_copy_leaves_the_original_sub_layers_in_place() -> None:
    """The sidecar forwards move `copy.copy(patch)` to the input device; the cached patch must stay put."""
    # Different dtypes only make the sub-layers' placements differ from each other.
    layers = [_lora(3, torch.float16), _lora(5, torch.float32)]
    merged = MergedLayerPatch(layers, [Range(0, 3), Range(3, 8)])
    before = [_placement(layer) for layer in layers]

    moved = copy.copy(merged)
    # The copy is shallow: new sub-layer objects, same tensors.
    assert all(new is not old for new, old in zip(moved.lora_layers, layers, strict=True))
    assert moved.lora_layers[0].up is layers[0].up  # type: ignore[attr-defined]
    moved.to(torch.device("meta"))

    assert [_placement(layer) for layer in merged.lora_layers] == before  # type: ignore[arg-type]
    assert all(layer.up.device.type == "meta" and layer.down.device.type == "meta" for layer in moved.lora_layers)  # type: ignore[attr-defined]
