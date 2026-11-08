import pytest
import torch

from invokeai.backend.patches.layers.dora_layer import DoRALayer
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict


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


@torch.no_grad()
@pytest.mark.parametrize(
    ["out_features", "in_features"],
    [(12, 20), (16, 16), (1, 8)],
    ids=["non-square", "square", "single-output-row"],
)
def test_out_dim_dora_scale_matches_lycoris_wd_on_out(out_features: int, in_features: int) -> None:
    """A ``.dora_scale`` stored per output row, ``(out_features, 1)``, is LyCORIS' default ``wd_on_out=True`` layout
    (and OneTrainer's "DoRA on output axis"). It must be applied per output row, as LyCORIS' ``weight_decompose``
    does, not per input column (#10016)."""
    torch.manual_seed(0)
    rank, alpha = 2, 1.0
    orig_weight = torch.randn(out_features, in_features)
    up = torch.randn(out_features, rank) * 0.3
    down = torch.randn(rank, in_features) * 0.3
    dora_scale = orig_weight.norm(dim=1, keepdim=True)

    layer = any_lora_layer_from_state_dict(
        {"lora_up.weight": up, "lora_down.weight": down, "dora_scale": dora_scale, "alpha": torch.tensor(alpha)}
    )
    assert isinstance(layer, DoRALayer)
    assert layer.magnitude_is_out_dim is True

    direction = orig_weight + up @ down * (alpha / rank)
    expected = dora_scale * direction / direction.norm(dim=1, keepdim=True)
    torch.testing.assert_close(orig_weight + layer.get_weight(orig_weight.clone()), expected, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_out_dim_dora_scale_matches_lycoris_wd_on_out_for_conv() -> None:
    """For a conv layer, LyCORIS' ``wd_on_out=True`` ``.dora_scale`` is ``(out_channels, 1, 1, 1)``."""
    torch.manual_seed(0)
    out_channels, in_channels, rank = 8, 4, 2
    orig_weight = torch.randn(out_channels, in_channels, 3, 3)
    up = torch.randn(out_channels, rank, 1, 1) * 0.3
    down = torch.randn(rank, in_channels, 3, 3) * 0.3
    dora_scale = orig_weight.flatten(1).norm(dim=1).reshape(out_channels, 1, 1, 1)

    layer = DoRALayer.from_state_dict_values({"lora_up.weight": up, "lora_down.weight": down, "dora_scale": dora_scale})
    assert layer.magnitude_is_out_dim is True

    direction = orig_weight + (up.flatten(1) @ down.flatten(1)).reshape(orig_weight.shape)
    direction_norm = direction.flatten(1).norm(dim=1).reshape(out_channels, 1, 1, 1)
    expected = dora_scale * direction / direction_norm
    torch.testing.assert_close(orig_weight + layer.get_weight(orig_weight.clone()), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ["dora_scale_shape", "expected_out_dim"],
    [
        ((1, 20), False),  # LyCORIS wd_on_out=False, linear
        ((12, 1), True),  # LyCORIS wd_on_out=True, linear
        ((1, 20, 1, 1), False),  # LyCORIS wd_on_out=False, conv
        ((12, 1, 1, 1), True),  # LyCORIS wd_on_out=True, conv
        ((20,), False),  # 1-D: no axis to tell from, keeps the input-axis reading
    ],
)
def test_dora_scale_orientation_follows_its_shape(dora_scale_shape: tuple[int, ...], expected_out_dim: bool) -> None:
    out_features, in_features, rank = 12, 20, 4
    up = torch.ones(out_features, rank, *([1] * (len(dora_scale_shape) - 2)))
    down = torch.ones(rank, in_features, *([1] * (len(dora_scale_shape) - 2)))
    layer = DoRALayer.from_state_dict_values(
        {"lora_up.weight": up, "lora_down.weight": down, "dora_scale": torch.ones(dora_scale_shape)}
    )
    assert layer.magnitude_is_out_dim is expected_out_dim
