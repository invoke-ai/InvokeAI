import copy

import gguf
import pytest
import torch

from invokeai.backend.flux.modules.layers import RMSNorm
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    AUTOCAST_MODULE_TYPE_MAPPING,
    AUTOCAST_MODULE_TYPE_MAPPING_INVERSE,
    unwrap_custom_layer,
    wrap_custom_layer,
)
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch
from invokeai.backend.patches.layers.flux_control_lora_layer import FluxControlLoRALayer
from invokeai.backend.patches.layers.lokr_layer import LoKRLayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.layers.merged_layer_patch import MergedLayerPatch, Range
from invokeai.backend.util.original_weights_storage import OriginalWeightsStorage
from tests.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.test_custom_invoke_linear_8_bit_lt import (
    build_linear_8bit_lt_layer,
)
from tests.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.test_custom_invoke_linear_nf4 import (
    build_linear_nf4_layer,
)
from tests.backend.quantization.gguf.test_ggml_tensor import quantize_tensor


def build_linear_layer_with_ggml_quantized_tensor(orig_layer: torch.nn.Linear | None = None):
    if orig_layer is None:
        orig_layer = torch.nn.Linear(32, 64)

    ggml_quantized_weight = quantize_tensor(orig_layer.weight, gguf.GGMLQuantizationType.Q8_0)
    orig_layer.weight = torch.nn.Parameter(ggml_quantized_weight)
    ggml_quantized_bias = quantize_tensor(orig_layer.bias, gguf.GGMLQuantizationType.Q8_0)
    orig_layer.bias = torch.nn.Parameter(ggml_quantized_bias)
    return orig_layer


parameterize_all_devices = pytest.mark.parametrize(
    ("device"),
    [
        pytest.param("cpu"),
        pytest.param(
            "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available.")
        ),
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")),
    ],
)

parameterize_cuda_and_mps = pytest.mark.parametrize(
    ("device"),
    [
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")),
        pytest.param(
            "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available.")
        ),
    ],
)


LayerUnderTest = tuple[torch.nn.Module, torch.Tensor, bool]


@pytest.fixture(
    params=[
        "linear",
        "conv1d",
        "conv2d",
        "group_norm",
        "embedding",
        "flux_rms_norm",
        "linear_with_ggml_quantized_tensor",
        "invoke_linear_8_bit_lt",
        "invoke_linear_nf4",
    ]
)
def layer_under_test(request: pytest.FixtureRequest) -> LayerUnderTest:
    """A fixture that returns a tuple of (layer, input, supports_cpu_inference) for the layer under test."""
    layer_type = request.param
    if layer_type == "linear":
        return (torch.nn.Linear(8, 16), torch.randn(1, 8), True)
    elif layer_type == "conv1d":
        return (torch.nn.Conv1d(8, 16, 3), torch.randn(1, 8, 5), True)
    elif layer_type == "conv2d":
        return (torch.nn.Conv2d(8, 16, 3), torch.randn(1, 8, 5, 5), True)
    elif layer_type == "group_norm":
        return (torch.nn.GroupNorm(2, 8), torch.randn(1, 8, 5), True)
    elif layer_type == "embedding":
        return (torch.nn.Embedding(4, 8), torch.tensor([0, 1], dtype=torch.long), True)
    elif layer_type == "flux_rms_norm":
        return (RMSNorm(8), torch.randn(1, 8), True)
    elif layer_type == "linear_with_ggml_quantized_tensor":
        return (build_linear_layer_with_ggml_quantized_tensor(), torch.randn(1, 32), True)
    elif layer_type == "invoke_linear_8_bit_lt":
        return (build_linear_8bit_lt_layer(), torch.randn(1, 32), False)
    elif layer_type == "invoke_linear_nf4":
        return (build_linear_nf4_layer(), torch.randn(1, 64), False)
    else:
        raise ValueError(f"Unsupported layer_type: {layer_type}")


def layer_to_device_via_state_dict(layer: torch.nn.Module, device: str):
    """A helper function to move a layer to a device by roundtripping through a state dict. This most closely matches
    how models are moved in the app. Some of the quantization types have broken semantics around calling .to() on the
    layer directly, so this is a workaround.

    We should fix this in the future.
    Relevant article: https://pytorch.org/tutorials/recipes/recipes/swap_tensors.html
    """
    state_dict = layer.state_dict()
    state_dict = {k: v.to(device) for k, v in state_dict.items()}
    layer.load_state_dict(state_dict, assign=True)


def wrap_single_custom_layer(layer: torch.nn.Module):
    custom_layer_type = AUTOCAST_MODULE_TYPE_MAPPING[type(layer)]
    return wrap_custom_layer(layer, custom_layer_type)


def unwrap_single_custom_layer(layer: torch.nn.Module):
    orig_layer_type = AUTOCAST_MODULE_TYPE_MAPPING_INVERSE[type(layer)]
    return unwrap_custom_layer(layer, orig_layer_type)


def test_isinstance(layer_under_test: LayerUnderTest):
    """Test that isinstance() and type() behave as expected after wrapping a layer in a custom layer."""
    orig_layer, _, _ = layer_under_test
    orig_type = type(orig_layer)

    custom_layer = wrap_single_custom_layer(orig_layer)

    assert isinstance(custom_layer, orig_type)
    assert type(custom_layer) is not orig_type


def test_wrap_and_unwrap(layer_under_test: LayerUnderTest):
    """Test that wrapping and unwrapping a layer behaves as expected."""
    orig_layer, _, _ = layer_under_test
    orig_type = type(orig_layer)

    # Wrap the original layer and assert that attributes of the custom layer can be accessed.
    custom_layer = wrap_single_custom_layer(orig_layer)
    custom_layer.set_device_autocasting_enabled(True)
    assert custom_layer._device_autocasting_enabled

    # Unwrap the custom layer.
    # Assert that the methods of the wrapped layer are no longer accessible.
    unwrapped_layer = unwrap_single_custom_layer(custom_layer)
    with pytest.raises(AttributeError):
        _ = unwrapped_layer.set_device_autocasting_enabled(True)
    # For now, we have chosen to allow attributes to persist. We may revisit this in the future.
    assert unwrapped_layer._device_autocasting_enabled
    assert type(unwrapped_layer) is orig_type


@parameterize_all_devices
def test_state_dict(device: str, layer_under_test: LayerUnderTest):
    """Test that .state_dict() behaves the same on the original layer and the wrapped layer."""
    orig_layer, _, _ = layer_under_test

    # Get the original layer on the test device.
    orig_layer.to(device)
    orig_state_dict = orig_layer.state_dict()

    # Wrap the original layer.
    custom_layer = copy.deepcopy(orig_layer)
    custom_layer = wrap_single_custom_layer(custom_layer)

    custom_state_dict = custom_layer.state_dict()

    assert set(orig_state_dict.keys()) == set(custom_state_dict.keys())
    for k in orig_state_dict:
        assert orig_state_dict[k].shape == custom_state_dict[k].shape
        assert orig_state_dict[k].dtype == custom_state_dict[k].dtype
        assert orig_state_dict[k].device == custom_state_dict[k].device
        assert torch.allclose(orig_state_dict[k], custom_state_dict[k])


@parameterize_all_devices
def test_load_state_dict(device: str, layer_under_test: LayerUnderTest):
    """Test that .load_state_dict() behaves the same on the original layer and the wrapped layer."""
    orig_layer, _, _ = layer_under_test

    orig_layer.to(device)

    custom_layer = copy.deepcopy(orig_layer)
    custom_layer = wrap_single_custom_layer(custom_layer)

    # Do a state dict roundtrip.
    orig_state_dict = orig_layer.state_dict()
    custom_state_dict = custom_layer.state_dict()

    orig_layer.load_state_dict(custom_state_dict, assign=True)
    custom_layer.load_state_dict(orig_state_dict, assign=True)

    orig_state_dict = orig_layer.state_dict()
    custom_state_dict = custom_layer.state_dict()

    # Assert that the state dicts are the same after the roundtrip.
    assert set(orig_state_dict.keys()) == set(custom_state_dict.keys())
    for k in orig_state_dict:
        assert orig_state_dict[k].shape == custom_state_dict[k].shape
        assert orig_state_dict[k].dtype == custom_state_dict[k].dtype
        assert orig_state_dict[k].device == custom_state_dict[k].device
        assert torch.allclose(orig_state_dict[k], custom_state_dict[k])


@parameterize_all_devices
def test_inference_on_device(device: str, layer_under_test: LayerUnderTest):
    """Test that inference behaves the same on the original layer and the wrapped layer when all weights are on the
    device.
    """
    orig_layer, layer_input, supports_cpu_inference = layer_under_test

    if device == "cpu" and not supports_cpu_inference:
        pytest.skip("Layer does not support CPU inference.")

    layer_to_device_via_state_dict(orig_layer, device)

    custom_layer = copy.deepcopy(orig_layer)
    custom_layer = wrap_single_custom_layer(custom_layer)

    # Run inference with the original layer.
    x = layer_input.to(device)
    orig_output = orig_layer(x)

    # Run inference with the wrapped layer.
    custom_output = custom_layer(x)

    assert torch.allclose(orig_output, custom_output)


@parameterize_cuda_and_mps
def test_inference_autocast_from_cpu_to_device(device: str, layer_under_test: LayerUnderTest):
    """Test that inference behaves the same on the original layer and the wrapped layer when all weights are on the
    device.
    """
    orig_layer, layer_input, supports_cpu_inference = layer_under_test

    if device == "cpu" and not supports_cpu_inference:
        pytest.skip("Layer does not support CPU inference.")

    # Make sure the original layer is on the device.
    layer_to_device_via_state_dict(orig_layer, device)

    x = layer_input.to(device)

    # Run inference with the original layer on the device.
    orig_output = orig_layer(x)

    # Move the original layer to the CPU.
    layer_to_device_via_state_dict(orig_layer, "cpu")

    # Inference should fail with an input on the device.
    with pytest.raises(RuntimeError):
        _ = orig_layer(x)

    # Wrap the original layer.
    custom_layer = copy.deepcopy(orig_layer)
    custom_layer = wrap_single_custom_layer(custom_layer)

    # Inference should still fail with autocasting disabled.
    custom_layer.set_device_autocasting_enabled(False)
    with pytest.raises(RuntimeError):
        _ = custom_layer(x)

    # Run inference with the wrapped layer on the device.
    custom_layer.set_device_autocasting_enabled(True)
    custom_output = custom_layer(x)
    assert custom_output.device.type == device

    assert torch.allclose(orig_output, custom_output)


PatchUnderTest = tuple[list[tuple[BaseLayerPatch, float]], torch.Tensor]


@pytest.fixture(
    params=[
        "single_lora",
        "multiple_loras",
        "concatenated_lora",
        "flux_control_lora",
        "single_lokr",
    ]
)
def patch_under_test(request: pytest.FixtureRequest) -> PatchUnderTest:
    """A fixture that returns a tuple of (patches, input) for the patch under test."""
    layer_type = request.param
    torch.manual_seed(0)

    # The assumed in/out features of the base linear layer.
    in_features = 32
    out_features = 64

    rank = 4

    if layer_type == "single_lora":
        lora_layer = LoRALayer(
            up=torch.randn(out_features, rank),
            mid=None,
            down=torch.randn(rank, in_features),
            alpha=1.0,
            bias=torch.randn(out_features),
        )
        input = torch.randn(1, in_features)
        return ([(lora_layer, 0.7)], input)
    elif layer_type == "multiple_loras":
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

        input = torch.randn(1, in_features)
        return ([(lora_layer, 1.0), (lora_layer_2, 0.5)], input)
    elif layer_type == "concatenated_lora":
        sub_layer_out_features = [16, 16, 32]

        # Create a MergedLayerPatch.
        sub_layers: list[LoRALayer] = []
        sub_layer_ranges: list[Range] = []
        dim_0_offset = 0
        for out_features in sub_layer_out_features:
            down = torch.randn(rank, in_features)
            up = torch.randn(out_features, rank)
            bias = torch.randn(out_features)
            sub_layers.append(LoRALayer(up=up, mid=None, down=down, alpha=1.0, bias=bias))
            sub_layer_ranges.append(Range(dim_0_offset, dim_0_offset + out_features))
            dim_0_offset += out_features
        merged_layer_patch = MergedLayerPatch(sub_layers, sub_layer_ranges)

        input = torch.randn(1, in_features)
        return ([(merged_layer_patch, 0.7)], input)
    elif layer_type == "flux_control_lora":
        # Create a FluxControlLoRALayer.
        patched_in_features = 40
        lora_layer = FluxControlLoRALayer(
            up=torch.randn(out_features, rank),
            mid=None,
            down=torch.randn(rank, patched_in_features),
            alpha=1.0,
            bias=torch.randn(out_features),
        )

        input = torch.randn(1, patched_in_features)
        return ([(lora_layer, 0.7)], input)
    elif layer_type == "single_lokr":
        lokr_layer = LoKRLayer(
            w1=torch.randn(rank, rank),
            w1_a=None,
            w1_b=None,
            w2=torch.randn(out_features // rank, in_features // rank),
            w2_a=None,
            w2_b=None,
            t2=None,
            alpha=1.0,
            bias=torch.randn(out_features),
        )
        input = torch.randn(1, in_features)
        return ([(lokr_layer, 0.7)], input)
    else:
        raise ValueError(f"Unsupported layer_type: {layer_type}")


@parameterize_all_devices
def test_linear_sidecar_patches(device: str, patch_under_test: PatchUnderTest):
    patches, input = patch_under_test

    torch.manual_seed(0)

    # Build the base layer under test.
    layer = torch.nn.Linear(32, 64)

    # Move the layer and input to the device.
    layer_to_device_via_state_dict(layer, device)
    input = input.to(torch.device(device))

    # Patch the LoRA layer into the linear layer.
    layer_patched = copy.deepcopy(layer)
    for patch, weight in patches:
        LayerPatcher._apply_model_layer_patch(
            module_to_patch=layer_patched,
            module_to_patch_key="",
            patch=patch,
            patch_weight=weight,
            original_weights=OriginalWeightsStorage(),
        )

    # Wrap the original layer in a custom layer and add the patch to it as a sidecar.
    custom_layer = wrap_single_custom_layer(layer)
    for patch, weight in patches:
        patch.to(torch.device(device))
        custom_layer.add_patch(patch, weight)

    # Run inference with the original layer and the patched layer and assert they are equal.
    output_patched = layer_patched(input)
    output_custom = custom_layer(input)
    assert torch.allclose(output_patched, output_custom, atol=1e-6)


@parameterize_cuda_and_mps
def test_linear_sidecar_patches_with_autocast_from_cpu_to_device(device: str, patch_under_test: PatchUnderTest):
    """Test that the output of a linear layer with sidecar patches is the same when the layer is on the device and
    when the layer is on the CPU and the patches are autocasted to the device.
    """
    patches, input = patch_under_test

    # Build the base layer under test.
    layer = torch.nn.Linear(32, 64)

    # Move the layer and input to the device.
    layer_to_device_via_state_dict(layer, device)
    input = input.to(torch.device(device))

    # Wrap the original layer in a custom layer and add the patch to it.
    custom_layer = wrap_single_custom_layer(layer)
    for patch, weight in patches:
        patch.to(torch.device(device))
        custom_layer.add_patch(patch, weight)

    # Run inference with the custom layer on the device.
    expected_output = custom_layer(input)

    # Move the custom layer to the CPU.
    layer_to_device_via_state_dict(custom_layer, "cpu")

    # Move the patches to the CPU.
    custom_layer.clear_patches()
    for patch, weight in patches:
        patch.to(torch.device("cpu"))
        custom_layer.add_patch(patch, weight)

    # Run inference with an input on the device, and all layer weights on the CPU. The weights should be autocasted to
    # the device.
    autocast_output = custom_layer(input)
    assert autocast_output.device.type == device

    # Assert that the outputs with and without autocasting are the same.
    assert torch.allclose(expected_output, autocast_output, atol=1e-6)


@pytest.fixture(
    params=[
        "linear_ggml_quantized",
        "invoke_linear_8_bit_lt",
        "invoke_linear_nf4",
    ]
)
def quantized_linear_layer_under_test(request: pytest.FixtureRequest):
    in_features = 32
    out_features = 64
    torch.manual_seed(0)
    layer_type = request.param
    orig_layer = torch.nn.Linear(in_features, out_features)
    if layer_type == "linear_ggml_quantized":
        return orig_layer, build_linear_layer_with_ggml_quantized_tensor(orig_layer)
    elif layer_type == "invoke_linear_8_bit_lt":
        return orig_layer, build_linear_8bit_lt_layer(orig_layer)
    elif layer_type == "invoke_linear_nf4":
        return orig_layer, build_linear_nf4_layer(orig_layer)
    else:
        raise ValueError(f"Unsupported layer_type: {layer_type}")


@parameterize_cuda_and_mps
def test_quantized_linear_sidecar_patches(
    device: str,
    quantized_linear_layer_under_test: tuple[torch.nn.Module, torch.nn.Module],
    patch_under_test: PatchUnderTest,
):
    """Test that patches can be applied to quantized linear layers and that the output is the same as when the patch is
    applied to a non-quantized linear layer.
    """
    patches, input = patch_under_test

    linear_layer, quantized_linear_layer = quantized_linear_layer_under_test

    # Move everything to the device.
    layer_to_device_via_state_dict(linear_layer, device)
    layer_to_device_via_state_dict(quantized_linear_layer, device)
    input = input.to(torch.device(device))

    # Wrap both layers in custom layers.
    linear_layer_custom = wrap_single_custom_layer(linear_layer)
    quantized_linear_layer_custom = wrap_single_custom_layer(quantized_linear_layer)

    # Apply the patches to the custom layers.
    for patch, weight in patches:
        patch.to(torch.device(device))
        linear_layer_custom.add_patch(patch, weight)
        quantized_linear_layer_custom.add_patch(patch, weight)

    # Run inference with the original layer and the patched layer and assert they are equal.
    output_linear_patched = linear_layer_custom(input)
    output_quantized_patched = quantized_linear_layer_custom(input)
    assert torch.allclose(output_linear_patched, output_quantized_patched, rtol=0.2, atol=0.2)


@parameterize_cuda_and_mps
def test_quantized_linear_sidecar_patches_with_autocast_from_cpu_to_device(
    device: str,
    quantized_linear_layer_under_test: tuple[torch.nn.Module, torch.nn.Module],
    patch_under_test: PatchUnderTest,
):
    """Test that the output of a linear layer with sidecar patches is the same when the layer is on the device and
    when the layer is on the CPU and the patches are autocasted to the device.
    """
    patches, input = patch_under_test

    _, quantized_linear_layer = quantized_linear_layer_under_test

    # Move everything to the device.
    layer_to_device_via_state_dict(quantized_linear_layer, device)
    input = input.to(torch.device(device))

    # Wrap the quantized linear layer in a custom layer and add the patch to it.
    quantized_linear_layer_custom = wrap_single_custom_layer(quantized_linear_layer)
    for patch, weight in patches:
        patch.to(torch.device(device))
        quantized_linear_layer_custom.add_patch(patch, weight)

    # Run inference with the custom layer on the device.
    expected_output = quantized_linear_layer_custom(input)

    # Move the custom layer to the CPU.
    layer_to_device_via_state_dict(quantized_linear_layer_custom, "cpu")

    # Move the patches to the CPU.
    quantized_linear_layer_custom.clear_patches()
    for patch, weight in patches:
        patch.to(torch.device("cpu"))
        quantized_linear_layer_custom.add_patch(patch, weight)

    # Run inference with an input on the device, and all layer weights on the CPU. The weights should be autocasted to
    # the device.
    autocast_output = quantized_linear_layer_custom(input)
    assert autocast_output.device.type == device

    # Assert that the outputs with and without autocasting are the same.
    assert torch.allclose(expected_output, autocast_output, atol=1e-6)
