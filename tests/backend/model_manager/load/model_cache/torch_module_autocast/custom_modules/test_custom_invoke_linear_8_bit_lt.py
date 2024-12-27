import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    wrap_custom_layer,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)
else:
    from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_invoke_linear_8_bit_lt import (
        CustomInvokeLinear8bitLt,
    )
    from invokeai.backend.quantization.bnb_llm_int8 import InvokeLinear8bitLt


def build_linear_8bit_lt_layer(orig_layer: torch.nn.Linear | None = None):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(1)

    if orig_layer is None:
        orig_layer = torch.nn.Linear(32, 64)
    orig_layer_state_dict = orig_layer.state_dict()

    # Prepare a quantized InvokeLinear8bitLt layer.
    quantized_layer = InvokeLinear8bitLt(
        input_features=orig_layer.in_features, output_features=orig_layer.out_features, has_fp16_weights=False
    )
    quantized_layer.load_state_dict(orig_layer_state_dict)
    quantized_layer.to("cuda")

    # Assert that the InvokeLinear8bitLt layer is quantized.
    assert quantized_layer.weight.CB is not None
    assert quantized_layer.weight.SCB is not None
    assert quantized_layer.weight.CB.dtype == torch.int8

    return quantized_layer


@pytest.fixture
def linear_8bit_lt_layer():
    return build_linear_8bit_lt_layer()


def test_custom_invoke_linear_8bit_lt_all_weights_on_cuda(linear_8bit_lt_layer: InvokeLinear8bitLt):
    """Test CustomInvokeLinear8bitLt inference with all weights on the GPU."""
    # Run inference on the original layer.
    x = torch.randn(1, 32).to("cuda")
    y_quantized = linear_8bit_lt_layer(x)

    # Wrap the InvokeLinear8bitLt layer in a CustomInvokeLinear8bitLt layer, and run inference on it.
    custom_linear_8bit_lt_layer = wrap_custom_layer(linear_8bit_lt_layer, CustomInvokeLinear8bitLt)
    y_custom = custom_linear_8bit_lt_layer(x)

    # Assert that the quantized and custom layers produce the same output.
    assert torch.allclose(y_quantized, y_custom, atol=1e-5)


def test_custom_invoke_linear_8bit_lt_all_weights_on_cpu(linear_8bit_lt_layer: InvokeLinear8bitLt):
    """Test CustomInvokeLinear8bitLt inference with all weights on the CPU (streaming to the GPU)."""
    # Run inference on the original layer.
    x = torch.randn(1, 32).to("cuda")
    y_quantized = linear_8bit_lt_layer(x)

    # Copy the state dict to the CPU and reload it.
    state_dict = linear_8bit_lt_layer.state_dict()
    state_dict = {k: v.to("cpu") for k, v in state_dict.items()}
    linear_8bit_lt_layer.load_state_dict(state_dict)

    # Inference of the original layer should fail.
    with pytest.raises(RuntimeError):
        linear_8bit_lt_layer(x)

    # Wrap the InvokeLinear8bitLt layer in a CustomInvokeLinear8bitLt layer, and run inference on it.
    custom_linear_8bit_lt_layer = wrap_custom_layer(linear_8bit_lt_layer, CustomInvokeLinear8bitLt)
    custom_linear_8bit_lt_layer.set_device_autocasting_enabled(True)
    y_custom = custom_linear_8bit_lt_layer(x)

    # Assert that the quantized and custom layers produce the same output.
    assert torch.allclose(y_quantized, y_custom, atol=1e-5)
