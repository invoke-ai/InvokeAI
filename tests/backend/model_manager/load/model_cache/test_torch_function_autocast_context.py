import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.torch_function_autocast_context import (
    TorchFunctionAutocastDeviceContext,
    add_autocast_to_module_forward,
)
from tests.backend.model_manager.load.model_cache.dummy_module import DummyModule


def test_torch_function_autocast_device_context():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    model = DummyModule()
    # Model parameters should start off on the CPU.
    assert all(p.device.type == "cpu" for p in model.parameters())

    with TorchFunctionAutocastDeviceContext(to_device=torch.device("cuda")):
        x = torch.randn(10, 10, device="cuda")
        y = model(x)

    # The model output should be on the GPU.
    assert y.device.type == "cuda"

    # The model parameters should still be on the CPU.
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_add_autocast_to_module_forward():
    model = DummyModule()
    assert all(p.device.type == "cpu" for p in model.parameters())

    add_autocast_to_module_forward(model, torch.device("cuda"))
    # After adding autocast, the model parameters should still be on the CPU.
    assert all(p.device.type == "cpu" for p in model.parameters())

    x = torch.randn(10, 10, device="cuda")
    y = model(x)

    # The model output should be on the GPU.
    assert y.device.type == "cuda"

    # The model parameters should still be on the CPU.
    assert all(p.device.type == "cpu" for p in model.parameters())

    # The autocast context should automatically be disabled after the model forward call completes.
    # So, attempting to perform an operation with comflicting devices should raise an error.
    with pytest.raises(RuntimeError):
        _ = torch.randn(10, device="cuda") * torch.randn(10, device="cpu")
