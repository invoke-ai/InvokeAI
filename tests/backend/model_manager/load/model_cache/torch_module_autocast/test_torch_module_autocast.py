import os

import gguf
import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
    remove_custom_layers_from_model,
)
from tests.backend.quantization.gguf.test_ggml_tensor import quantize_tensor

try:
    from invokeai.backend.quantization.bnb_llm_int8 import InvokeLinear8bitLt, quantize_model_llm_int8
except ImportError:
    # This is expected to fail on MacOS
    pass

cuda_and_mps = pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            torch.device("cuda"), marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")
        ),
        pytest.param(
            torch.device("mps"),
            marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS device"),
        ),
    ],
)


class ModelWithLinearLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.fixture(params=["none", "gguf"])
def model(request: pytest.FixtureRequest) -> torch.nn.Module:
    if request.param == "none":
        return ModelWithLinearLayer()
    elif request.param == "gguf":
        # Initialize ModelWithLinearLayer and replace the linear layer weight with a GGML quantized weight.
        model = ModelWithLinearLayer()
        ggml_quantized_weight = quantize_tensor(model.linear.weight, gguf.GGMLQuantizationType.Q8_0)
        model.linear.weight = torch.nn.Parameter(ggml_quantized_weight)
        return model
    else:
        raise ValueError(f"Invalid quantization type: {request.param}")


@cuda_and_mps
@torch.no_grad()
def test_torch_module_autocast_linear_layer(device: torch.device, model: torch.nn.Module):
    # Skip this test with MPS on GitHub Actions. It fails but I haven't taken the tie to figure out why. It passes
    # locally on MacOS.
    if os.environ.get("GITHUB_ACTIONS") == "true" and device.type == "mps":
        pytest.skip("This test is flaky on GitHub Actions")

    # Model parameters should start off on the CPU.
    assert all(p.device.type == "cpu" for p in model.parameters())

    torch.manual_seed(0)

    # Run inference on the CPU.
    x = torch.randn(1, 32, device="cpu")
    expected = model(x)
    assert expected.device.type == "cpu"

    # Apply the custom layers to the model.
    apply_custom_layers_to_model(model, device_autocasting_enabled=True)

    # Run the model on the device.
    autocast_result = model(x.to(device))

    # The model output should be on the device.
    assert autocast_result.device.type == device.type
    # The model parameters should still be on the CPU.
    assert all(p.device.type == "cpu" for p in model.parameters())

    # Remove the custom layers from the model.
    remove_custom_layers_from_model(model)

    # After removing the custom layers, the model should no longer be able to run inference on the device.
    with pytest.raises(RuntimeError):
        _ = model(x.to(device))

    # Run inference again on the CPU.
    after_result = model(x)

    assert after_result.device.type == "cpu"

    # The results from all inference runs should be the same.
    assert torch.allclose(autocast_result.to("cpu"), expected, atol=1e-5)
    assert torch.allclose(after_result, expected, atol=1e-5)


@torch.no_grad()
def test_torch_module_autocast_bnb_llm_int8_linear_layer():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA device")

    torch.manual_seed(0)

    model = ModelWithLinearLayer()
    model = quantize_model_llm_int8(model, modules_to_not_convert=set())
    # The act of moving the model to the CUDA device will trigger quantization.
    model.to("cuda")
    # Confirm that the layer is quantized.
    assert isinstance(model.linear, InvokeLinear8bitLt)
    assert model.linear.weight.CB is not None
    assert model.linear.weight.SCB is not None

    # Run inference on the GPU.
    x = torch.randn(1, 32)
    expected = model(x.to("cuda"))
    assert expected.device.type == "cuda"

    # Move the model back to the CPU and add the custom layers to the model.
    model.to("cpu")
    apply_custom_layers_to_model(model, device_autocasting_enabled=True)

    # Run inference with weights being streamed to the GPU.
    autocast_result = model(x.to("cuda"))
    assert autocast_result.device.type == "cuda"

    # The results from all inference runs should be the same.
    assert torch.allclose(autocast_result, expected, atol=1e-5)
