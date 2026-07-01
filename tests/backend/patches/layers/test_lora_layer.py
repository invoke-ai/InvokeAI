import logging

import pytest
import torch

from invokeai.backend.patches.layers.lora_layer import LoRALayer


def test_lora_layer_init_from_state_dict():
    """Test initializing a LoRALayer from state dict values."""
    # Create mock state dict values
    in_features = 8
    out_features = 16
    rank = 4
    alpha = 16.0
    values = {
        "lora_up.weight": torch.ones(out_features, rank),
        "lora_down.weight": torch.ones(rank, in_features),
        "alpha": torch.tensor(alpha),
    }
    layer = LoRALayer.from_state_dict_values(values)

    assert layer.up.shape == (out_features, rank)
    assert layer.down.shape == (rank, in_features)
    assert layer._alpha == alpha
    assert layer.bias is None


def test_lora_layer_init_from_state_dict_with_unhandled_keys_logs_warning(caplog: pytest.LogCaptureFixture):
    """Test initializing a LoRALayer from state dict values with an unhandled key."""
    in_features = 8
    out_features = 16
    rank = 4
    alpha = 16.0
    values = {
        "lora_up.weight": torch.ones(out_features, rank),
        "lora_down.weight": torch.ones(rank, in_features),
        "alpha": torch.tensor(alpha),
        "unhandled_key": torch.randn(4, 4),
    }

    with caplog.at_level(logging.WARNING):
        _ = LoRALayer.from_state_dict_values(values)

    assert (
        "Unexpected keys found in LoRA/LyCORIS layer, model might work incorrectly! Unexpected keys: {'unhandled_key'}"
        in caplog.text
    )


@pytest.mark.parametrize(
    ["device"],
    [
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")),
        pytest.param(
            "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS device")
        ),
    ],
)
def test_lora_layer_to(device: str):
    in_features = 8
    out_features = 16
    rank = 4
    alpha = 16.0
    values = {
        "lora_up.weight": torch.ones(out_features, rank),
        "lora_down.weight": torch.ones(rank, in_features),
        "alpha": torch.tensor(alpha),
    }
    layer = LoRALayer.from_state_dict_values(values)

    # Layer is initialized on the CPU.
    assert layer.up.device.type == "cpu"
    assert layer.down.device.type == "cpu"

    # Test moving to device.
    layer.to(device=torch.device(device))
    assert layer.up.device.type == device
    assert layer.down.device.type == device


def test_lora_layer_calc_size():
    """Test calculating memory size of LoRALayer tensors."""
    # Initialize weights with random shapes.
    up = torch.randn(1, 2)
    mid = torch.randn(2, 3)
    down = torch.randn(3, 4)
    bias = torch.randn(5)
    layer = LoRALayer(up=up, mid=mid, down=down, alpha=8.0, bias=bias)

    assert layer.calc_size() == sum(tensor.numel() * tensor.element_size() for tensor in [up, mid, down, bias])


def test_lora_layer_get_parameters():
    """Test getting weight and bias parameters from LoRALayer."""
    in_features = 8
    out_features = 16
    rank = 4
    alpha = 16.0
    values = {
        "lora_up.weight": torch.ones(out_features, rank),
        "lora_down.weight": torch.ones(rank, in_features),
        "alpha": torch.tensor(alpha),
    }
    layer = LoRALayer.from_state_dict_values(values)

    # Create mock original module
    orig_module = torch.nn.Linear(in_features, out_features)

    params = layer.get_parameters(dict(orig_module.named_parameters(recurse=False)), weight=1.0)
    assert "weight" in params
    assert params["weight"].shape == orig_module.weight.shape
    assert params["weight"].allclose(torch.ones(out_features, in_features) * alpha)
    assert "bias" not in params  # No bias in this case
