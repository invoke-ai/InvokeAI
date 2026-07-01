import torch

from invokeai.backend.patches.layers.flux_control_lora_layer import FluxControlLoRALayer


def test_flux_control_lora_layer_get_parameters():
    """Test getting weight and bias parameters from FluxControlLoRALayer."""
    small_in_features = 4
    big_in_features = 8
    out_features = 16
    rank = 4
    alpha = 16.0
    layer = FluxControlLoRALayer(
        up=torch.ones(out_features, rank), mid=None, down=torch.ones(rank, big_in_features), alpha=alpha, bias=None
    )

    # Create mock original module
    orig_module = torch.nn.Linear(small_in_features, out_features)

    # Test that get_parameters() behaves as expected in spite of the difference in in_features shapes.
    params = layer.get_parameters(dict(orig_module.named_parameters(recurse=False)), weight=1.0)
    assert "weight" in params
    assert params["weight"].shape == (out_features, big_in_features)
    assert params["weight"].allclose(torch.ones(out_features, big_in_features) * alpha)
    assert "bias" not in params  # No bias in this case
