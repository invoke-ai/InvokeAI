import copy

import torch

from invokeai.backend.patches.layers.concatenated_lora_layer import ConcatenatedLoRALayer
from invokeai.backend.patches.layers.flux_control_lora_layer import FluxControlLoRALayer
from invokeai.backend.patches.layers.full_layer import FullLayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.pad_with_zeros import pad_with_zeros
from invokeai.backend.patches.sidecar_wrappers.linear_sidecar_wrapper import LinearSidecarWrapper


@torch.no_grad()
def test_linear_sidecar_wrapper_lora():
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

    # Create a LinearSidecarWrapper.
    lora_wrapped = LinearSidecarWrapper(linear, [(lora_layer, 1.0)])

    # Run the LoRA-patched linear layer and the LinearSidecarWrapper and assert they are equal.
    input = torch.randn(1, in_features)
    output_patched = linear_patched(input)
    output_wrapped = lora_wrapped(input)
    assert torch.allclose(output_patched, output_wrapped, atol=1e-6)


@torch.no_grad()
def test_linear_sidecar_wrapper_multiple_loras():
    # Create a linear layer.
    in_features = 10
    out_features = 20
    linear = torch.nn.Linear(in_features, out_features)

    # Create two LoRA layers.
    rank = 4
    lora_layer = LoRALayer(
        up=torch.randn(out_features, rank),
        mid=None,
        down=torch.randn(rank, in_features),
        alpha=1.0,
        bias=torch.randn(out_features),
    )
    lora_layer_2 = LoRALayer(
        up=torch.randn(out_features, rank),
        mid=None,
        down=torch.randn(rank, in_features),
        alpha=1.0,
        bias=torch.randn(out_features),
    )
    # We use different weights for the two LoRA layers to ensure this is working.
    lora_weight = 1.0
    lora_weight_2 = 0.5

    # Patch the LoRA layers into the linear layer.
    linear_patched = copy.deepcopy(linear)
    linear_patched.weight.data += lora_layer.get_weight(linear_patched.weight) * (lora_layer.scale() * lora_weight)
    linear_patched.bias.data += lora_layer.get_bias(linear_patched.bias) * (lora_layer.scale() * lora_weight)
    linear_patched.weight.data += lora_layer_2.get_weight(linear_patched.weight) * (
        lora_layer_2.scale() * lora_weight_2
    )
    linear_patched.bias.data += lora_layer_2.get_bias(linear_patched.bias) * (lora_layer_2.scale() * lora_weight_2)

    # Create a LinearSidecarWrapper.
    lora_wrapped = LinearSidecarWrapper(linear, [(lora_layer, lora_weight), (lora_layer_2, lora_weight_2)])

    # Run the LoRA-patched linear layer and the LinearSidecarWrapper and assert they are equal.
    input = torch.randn(1, in_features)
    output_patched = linear_patched(input)
    output_wrapped = lora_wrapped(input)
    assert torch.allclose(output_patched, output_wrapped, atol=1e-6)


@torch.no_grad()
def test_linear_sidecar_wrapper_concatenated_lora():
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

    # Create a LinearSidecarWrapper.
    lora_wrapped = LinearSidecarWrapper(linear, [(concatenated_lora_layer, 1.0)])

    # Run the ConcatenatedLoRA-patched linear layer and the LinearSidecarWrapper and assert they are equal.
    input = torch.randn(1, in_features)
    output_patched = linear_patched(input)
    output_wrapped = lora_wrapped(input)
    assert torch.allclose(output_patched, output_wrapped, atol=1e-6)


def test_linear_sidecar_wrapper_full_layer():
    # Create a linear layer.
    in_features = 10
    out_features = 20
    linear = torch.nn.Linear(in_features, out_features)

    # Create a FullLayer.
    full_layer = FullLayer(weight=torch.randn(out_features, in_features), bias=torch.randn(out_features))

    # Patch the FullLayer into the linear layer.
    linear_patched = copy.deepcopy(linear)
    linear_patched.weight.data += full_layer.get_weight(linear_patched.weight)
    linear_patched.bias.data += full_layer.get_bias(linear_patched.bias)

    # Create a LinearSidecarWrapper.
    full_wrapped = LinearSidecarWrapper(linear, [(full_layer, 1.0)])

    # Run the FullLayer-patched linear layer and the LinearSidecarWrapper and assert they are equal.
    input = torch.randn(1, in_features)
    output_patched = linear_patched(input)
    output_wrapped = full_wrapped(input)
    assert torch.allclose(output_patched, output_wrapped, atol=1e-6)


def test_linear_sidecar_wrapper_flux_control_lora_layer():
    # Create a linear layer.
    orig_in_features = 10
    out_features = 40
    linear = torch.nn.Linear(orig_in_features, out_features)

    # Create a FluxControlLoRALayer.
    patched_in_features = 20
    rank = 4
    lora_layer = FluxControlLoRALayer(
        up=torch.randn(out_features, rank),
        mid=None,
        down=torch.randn(rank, patched_in_features),
        alpha=1.0,
        bias=torch.randn(out_features),
    )

    # Patch the FluxControlLoRALayer into the linear layer.
    linear_patched = copy.deepcopy(linear)
    # Expand the existing weight.
    expanded_weight = pad_with_zeros(linear_patched.weight, torch.Size([out_features, patched_in_features]))
    linear_patched.weight = torch.nn.Parameter(expanded_weight, requires_grad=linear_patched.weight.requires_grad)
    # Expand the existing bias.
    expanded_bias = pad_with_zeros(linear_patched.bias, torch.Size([out_features]))
    linear_patched.bias = torch.nn.Parameter(expanded_bias, requires_grad=linear_patched.bias.requires_grad)
    # Add the residuals.
    linear_patched.weight.data += lora_layer.get_weight(linear_patched.weight) * lora_layer.scale()
    linear_patched.bias.data += lora_layer.get_bias(linear_patched.bias) * lora_layer.scale()

    # Create a LinearSidecarWrapper.
    lora_wrapped = LinearSidecarWrapper(linear, [(lora_layer, 1.0)])

    # Run the FluxControlLoRA-patched linear layer and the LinearSidecarWrapper and assert they are equal.
    input = torch.randn(1, patched_in_features)
    output_patched = linear_patched(input)
    output_wrapped = lora_wrapped(input)
    assert torch.allclose(output_patched, output_wrapped, atol=1e-6)
