import torch

from invokeai.backend.patches.layers.dora_layer import DoRALayer


def _reference_aitoolkit_forward(
    x: torch.Tensor, orig_weight: torch.Tensor, down: torch.Tensor, up: torch.Tensor, magnitude: torch.Tensor
) -> torch.Tensor:
    """The output ai-toolkit's DoRAModule produces (multiplier=1, alpha=rank so scale=1).

    See ``ToolkitModuleMixin.forward`` + ``DoRAModule.apply_dora`` in ostris/ai-toolkit: the module output is
    ``org_forward(x) + lora_output + (magnitude / ||W + dV||_row - 1) * F.linear(x, W + dV)``.
    """
    delta_v = up @ down
    weight_norm = torch.linalg.norm(orig_weight + delta_v, dim=1)
    return (
        torch.nn.functional.linear(x, orig_weight)
        + torch.nn.functional.linear(x, delta_v)
        + (magnitude / weight_norm).view(1, -1) * torch.nn.functional.linear(x, orig_weight + delta_v)
        - torch.nn.functional.linear(x, orig_weight + delta_v)
    )


@torch.no_grad()
def test_out_dim_magnitude_matches_peft_aitoolkit_math() -> None:
    """A PEFT/ai-toolkit DoRA magnitude (one entry per output row) must reproduce their forward pass.

    Covers non-square layers in both directions: applying the LyCORIS (input-dim) math to an output-dim
    magnitude raises a broadcast error there, and is silently wrong on square layers.
    """
    torch.manual_seed(0)
    for out_features, in_features, rank in [(12, 20, 4), (16, 16, 4), (20, 12, 4)]:
        orig_weight = torch.randn(out_features, in_features)
        down = torch.randn(rank, in_features) * 0.05
        up = torch.randn(out_features, rank) * 0.05
        magnitude = torch.randn(out_features).abs() + 1.0

        layer = DoRALayer.from_state_dict_values(
            {"lora_down.weight": down, "lora_up.weight": up, "dora_magnitude": magnitude}
        )
        assert layer.magnitude_is_out_dim is True

        patched_weight = orig_weight + layer.get_weight(orig_weight)
        x = torch.randn(7, in_features)
        expected = _reference_aitoolkit_forward(x, orig_weight, down, up, magnitude)
        assert torch.allclose(torch.nn.functional.linear(x, patched_weight), expected, atol=1e-5)


@torch.no_grad()
def test_in_dim_magnitude_keeps_lycoris_math() -> None:
    """The LyCORIS/kohya ``dora_scale`` (one entry per input column) keeps its original normalization."""
    torch.manual_seed(0)
    out_features, in_features, rank = 12, 20, 4
    orig_weight = torch.randn(out_features, in_features)
    down = torch.randn(rank, in_features) * 0.05
    up = torch.randn(out_features, rank) * 0.05
    dora_scale = torch.randn(1, in_features).abs() + 1.0

    layer = DoRALayer.from_state_dict_values({"lora_down.weight": down, "lora_up.weight": up, "dora_scale": dora_scale})
    assert layer.magnitude_is_out_dim is False

    direction = orig_weight + up @ down
    direction_norm = direction.transpose(0, 1).norm(dim=1, keepdim=True).transpose(0, 1)
    expected = direction * (dora_scale / direction_norm)
    assert torch.allclose(orig_weight + layer.get_weight(orig_weight), expected, atol=1e-6)
