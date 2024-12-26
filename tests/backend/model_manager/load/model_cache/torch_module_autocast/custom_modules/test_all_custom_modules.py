import copy

import gguf
import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from tests.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.test_custom_invoke_linear_8_bit_lt import (
    build_linear_8bit_lt_layer,
)
from tests.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.test_custom_invoke_linear_nf4 import (
    build_linear_nf4_layer,
)
from tests.backend.quantization.gguf.test_ggml_tensor import quantize_tensor


def build_linear_layer_with_ggml_quantized_tensor():
    layer = torch.nn.Linear(32, 64)
    ggml_quantized_weight = quantize_tensor(layer.weight, gguf.GGMLQuantizationType.Q8_0)
    layer.weight = torch.nn.Parameter(ggml_quantized_weight)
    ggml_quantized_bias = quantize_tensor(layer.bias, gguf.GGMLQuantizationType.Q8_0)
    layer.bias = torch.nn.Parameter(ggml_quantized_bias)
    return layer


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


def test_isinstance(layer_under_test: LayerUnderTest):
    """Test that isinstance() and type() behave as expected after wrapping a layer in a custom layer."""
    orig_layer, _, _ = layer_under_test
    orig_type = type(orig_layer)

    apply_custom_layers_to_model(orig_layer)

    assert isinstance(orig_layer, orig_type)
    assert type(orig_layer) is not orig_type


@parameterize_all_devices
def test_state_dict(device: str, layer_under_test: LayerUnderTest):
    """Test that .state_dict() behaves the same on the original layer and the wrapped layer."""
    orig_layer, _, _ = layer_under_test

    # Get the original layer on the test device.
    orig_layer.to(device)
    orig_state_dict = orig_layer.state_dict()

    # Wrap the original layer.
    custom_layer = copy.deepcopy(orig_layer)
    apply_custom_layers_to_model(custom_layer)

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
    apply_custom_layers_to_model(custom_layer)

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
    apply_custom_layers_to_model(custom_layer)

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
    apply_custom_layers_to_model(custom_layer)

    # Inference should still fail with autocasting disabled.
    custom_layer.set_device_autocasting_enabled(False)
    with pytest.raises(RuntimeError):
        _ = custom_layer(x)

    # Run inference with the wrapped layer on the device.
    custom_layer.set_device_autocasting_enabled(True)
    custom_output = custom_layer(x)
    assert custom_output.device.type == device

    assert torch.allclose(orig_output, custom_output)
