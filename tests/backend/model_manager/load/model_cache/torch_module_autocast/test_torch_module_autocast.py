import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
    remove_custom_layers_from_model,
)
from tests.backend.model_manager.load.model_cache.dummy_module import DummyModule

mps_and_cuda = pytest.mark.parametrize(
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


@mps_and_cuda
def test_torch_module_autocast(device: torch.device):
    model = DummyModule()
    # Model parameters should start off on the CPU.
    assert all(p.device.type == "cpu" for p in model.parameters())

    # Run inference on the CPU.
    x = torch.randn(10, 10, device="cpu")
    expected = model(x)
    assert expected.device.type == "cpu"

    # Apply the custom layers to the model.
    apply_custom_layers_to_model(model)

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
    assert torch.allclose(autocast_result.to("cpu"), expected)
    assert torch.allclose(after_result, expected)
