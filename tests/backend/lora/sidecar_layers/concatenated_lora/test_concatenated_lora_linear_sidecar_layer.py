import copy

import torch

from invokeai.backend.lora.layers.concatenated_lora_layer import ConcatenatedLoRALayer
from invokeai.backend.lora.layers.lora_layer import LoRALayer
from invokeai.backend.lora.sidecar_layers.concatenated_lora.concatenated_lora_linear_sidecar_layer import (
    ConcatenatedLoRALinearSidecarLayer,
)
from invokeai.backend.lora.sidecar_layers.lora_sidecar_module import LoRASidecarModule


def test_concatenated_lora_linear_sidecar_layer():
    """Test that a ConcatenatedLoRALinearSidecarLayer is equivalent to patching a linear layer with the ConcatenatedLoRA
    layer.
    """

    # Create a linear layer.
    in_features = 5
    sub_layer_out_features = [5, 10, 15]
    linear = torch.nn.Linear(in_features, sum(sub_layer_out_features))

    # Create a ConcatenatedLoRA layer.
    rank = 4
    sub_layers: list[LoRALayer] = []
    for out_features in sub_layer_out_features:
        down = torch.randn(rank, in_features)
        up = torch.randn(out_features, rank)
        bias = torch.randn(out_features)
        sub_layers.append(LoRALayer(up=up, mid=None, down=down, alpha=1.0, bias=bias))
    concatenated_lora_layer = ConcatenatedLoRALayer(sub_layers, concat_axis=0)

    # Patch the ConcatenatedLoRA layer into the linear layer.
    linear_patched = copy.deepcopy(linear)
    linear_patched.weight.data += (
        concatenated_lora_layer.get_weight(linear_patched.weight) * concatenated_lora_layer.scale()
    )
    linear_patched.bias.data += concatenated_lora_layer.get_bias(linear_patched.bias) * concatenated_lora_layer.scale()

    # Create a ConcatenatedLoRALinearSidecarLayer.
    concatenated_lora_linear_sidecar_layer = ConcatenatedLoRALinearSidecarLayer(concatenated_lora_layer, weight=1.0)
    linear_with_sidecar = LoRASidecarModule(linear, [concatenated_lora_linear_sidecar_layer])

    # Run the ConcatenatedLoRA-patched linear layer and the ConcatenatedLoRALinearSidecarLayer and assert they are
    # equal.
    input = torch.randn(1, in_features)
    output_patched = linear_patched(input)
    output_sidecar = linear_with_sidecar(input)
    assert torch.allclose(output_patched, output_sidecar, atol=1e-6)
