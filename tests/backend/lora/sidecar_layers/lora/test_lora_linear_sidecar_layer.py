import copy

import torch

from invokeai.backend.lora.layers.lora_layer import LoRALayer
from invokeai.backend.lora.sidecar_layers.lora.lora_linear_sidecar_layer import LoRALinearSidecarLayer
from invokeai.backend.lora.sidecar_layers.lora_sidecar_module import LoRASidecarModule


@torch.no_grad()
def test_lora_linear_sidecar_layer():
    """Test that a LoRALinearSidecarLayer is equivalent to patching a linear layer with the LoRA layer."""

    # Create a linear layer.
    in_features = 10
    out_features = 20
    linear = torch.nn.Linear(in_features, out_features)

    # Create a LoRA layer.
    rank = 4
    down = torch.randn(rank, in_features)
    up = torch.randn(out_features, rank)
    bias = torch.randn(out_features)
    lora_layer = LoRALayer(up=up, mid=None, down=down, alpha=1.0, bias=bias)

    # Patch the LoRA layer into the linear layer.
    linear_patched = copy.deepcopy(linear)
    linear_patched.weight.data += lora_layer.get_weight(linear_patched.weight) * lora_layer.scale()
    linear_patched.bias.data += lora_layer.get_bias(linear_patched.bias) * lora_layer.scale()
    # Create a LoRALinearSidecarLayer.
    lora_linear_sidecar_layer = LoRALinearSidecarLayer(lora_layer, weight=1.0)
    linear_with_sidecar = LoRASidecarModule(linear, [lora_linear_sidecar_layer])

    # Run the LoRA-patched linear layer and the LoRALinearSidecarLayer and assert they are equal.
    input = torch.randn(1, in_features)
    output_patched = linear_patched(input)
    output_sidecar = linear_with_sidecar(input)
    assert torch.allclose(output_patched, output_sidecar, atol=1e-6)
