import copy

import torch

from invokeai.backend.lora.layers.concatenated_lora_layer import ConcatenatedLoRALayer
from invokeai.backend.lora.layers.lora_layer import LoRALayer
from invokeai.backend.lora.lora_layer_wrappers import LoRALinearWrapper


@torch.no_grad()
def test_lora_linear_wrapper():
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

    # Create a LoRALinearWrapper.
    lora_wrapped = LoRALinearWrapper(linear, [lora_layer], [1.0])

    # Run the LoRA-patched linear layer and the LoRALinearWrapper and assert they are equal.
    input = torch.randn(1, in_features)
    output_patched = linear_patched(input)
    output_wrapped = lora_wrapped(input)
    assert torch.allclose(output_patched, output_wrapped, atol=1e-6)


def test_concatenated_lora_linear_wrapper():
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

    # Create a LoRALinearWrapper.
    lora_wrapped = LoRALinearWrapper(linear, [concatenated_lora_layer], [1.0])

    # Run the ConcatenatedLoRA-patched linear layer and the LoRALinearWrapper and assert they are equal.
    input = torch.randn(1, in_features)
    output_patched = linear_patched(input)
    output_wrapped = lora_wrapped(input)
    assert torch.allclose(output_patched, output_wrapped, atol=1e-6)
