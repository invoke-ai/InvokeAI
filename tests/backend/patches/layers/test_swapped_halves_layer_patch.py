import torch

from invokeai.backend.patches.layers.dora_layer import DoRALayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.layers.swapped_halves_layer_patch import SwappedHalvesLayerPatch

OUT_FEATURES = 12  # Two halves of 6, like the (shift, scale) / (scale, shift) modulation output.
IN_FEATURES = 5
RANK = 3


def _swap(t: torch.Tensor) -> torch.Tensor:
    half = t.shape[0] // 2
    return torch.cat([t[half:], t[:half]], dim=0)


@torch.no_grad()
def test_lora_weight_and_bias_land_in_the_target_halves() -> None:
    torch.manual_seed(0)
    up = torch.randn(OUT_FEATURES, RANK)
    down = torch.randn(RANK, IN_FEATURES)
    bias = torch.randn(OUT_FEATURES)
    alpha = 6.0
    patch_weight = 0.7
    target_weight = torch.randn(OUT_FEATURES, IN_FEATURES)
    target_bias = torch.randn(OUT_FEATURES)

    patch = SwappedHalvesLayerPatch(LoRALayer(up=up, mid=None, down=down, alpha=alpha, bias=bias))
    params = patch.get_parameters({"weight": target_weight, "bias": target_bias}, weight=patch_weight)

    # The deltas in the LoRA's own (source) layout, then moved to the target layout.
    source_weight_delta = up @ down * (alpha / RANK) * patch_weight
    source_bias_delta = bias * (alpha / RANK) * patch_weight
    assert torch.allclose(params["weight"], _swap(source_weight_delta), atol=1e-6)
    assert torch.allclose(params["bias"], _swap(source_bias_delta), atol=1e-6)


@torch.no_grad()
def test_dora_normalizes_against_the_original_weight_in_its_own_layout() -> None:
    torch.manual_seed(0)
    up = torch.randn(OUT_FEATURES, RANK) * 0.5
    down = torch.randn(RANK, IN_FEATURES) * 0.5
    magnitude = torch.rand(OUT_FEATURES) + 0.5
    source_weight = torch.randn(OUT_FEATURES, IN_FEATURES)

    patch = SwappedHalvesLayerPatch(
        DoRALayer.from_state_dict_values({"lora_up.weight": up, "lora_down.weight": down, "dora_magnitude": magnitude})
    )
    params = patch.get_parameters({"weight": _swap(source_weight)}, weight=1.0)

    # PEFT DoRA in the source layout: each output row of W + up @ down rescaled to its magnitude.
    direction = source_weight + up @ down
    source_patched = direction * (magnitude / direction.norm(dim=1))[:, None]
    assert torch.allclose(params["weight"], _swap(source_patched - source_weight), atol=1e-5)


@torch.no_grad()
def test_sparse_lycoris_bias_is_swapped() -> None:
    torch.manual_seed(0)
    up = torch.randn(OUT_FEATURES, RANK)
    down = torch.randn(RANK, IN_FEATURES)
    patch = SwappedHalvesLayerPatch(
        LoRALayer.from_state_dict_values(
            {
                "lora_up.weight": up,
                "lora_down.weight": down,
                "bias_indices": torch.tensor([[1, 8]]),
                "bias_values": torch.tensor([2.0, 3.0]),
                "bias_size": torch.tensor([OUT_FEATURES]),
            }
        )
    )
    params = patch.get_parameters(
        {"weight": torch.zeros(OUT_FEATURES, IN_FEATURES), "bias": torch.zeros(OUT_FEATURES)}, weight=1.0
    )

    source_bias_delta = torch.zeros(OUT_FEATURES)
    source_bias_delta[1] = 2.0
    source_bias_delta[8] = 3.0
    assert torch.equal(params["bias"].to_dense(), _swap(source_bias_delta))


@torch.no_grad()
def test_meta_original_parameters_give_the_same_lora_delta() -> None:
    """The sidecar hands shape-only meta tensors to patches when the base weights are quantized."""
    torch.manual_seed(0)
    patch = SwappedHalvesLayerPatch(
        LoRALayer(
            up=torch.randn(OUT_FEATURES, RANK), mid=None, down=torch.randn(RANK, IN_FEATURES), alpha=None, bias=None
        )
    )

    real = patch.get_parameters({"weight": torch.randn(OUT_FEATURES, IN_FEATURES)}, weight=1.0)
    meta = patch.get_parameters({"weight": torch.empty(OUT_FEATURES, IN_FEATURES, device="meta")}, weight=1.0)

    assert torch.equal(meta["weight"], real["weight"])
