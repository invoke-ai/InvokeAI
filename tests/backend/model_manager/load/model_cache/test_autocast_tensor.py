import itertools

import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.autocast_tensor import AutocastTensor
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
def test_autocast_tensor_multiply(device: torch.device):
    x = torch.randn(32, 64)
    y = torch.randn(32, 64)
    y_autocast = AutocastTensor(y, device)

    # Multiply on the CPU to get the expected result.
    expected = x * y

    # Now, multiply on the GPU using the AutocastTensor class.
    result = x.to(device) * y_autocast

    assert type(result) is torch.Tensor
    assert expected.device.type == "cpu"
    assert result.device.type == device.type
    assert torch.allclose(result.to("cpu"), expected)


@mps_and_cuda
def test_autocast_tensor_to_device(device: torch.device):
    x = torch.randn(32, 64)
    x_autocast = AutocastTensor(x, device)
    with pytest.raises(RuntimeError):
        x_autocast.to(device=device)


@mps_and_cuda
def test_autocast_tensor_to_dtype(device: torch.device):
    x = torch.randn(32, 64)
    x_autocast = AutocastTensor(x, device)
    assert x_autocast.dtype == torch.float32
    x_autocast_new = x_autocast.to(dtype=torch.float16)
    assert isinstance(x_autocast_new, AutocastTensor)
    assert x_autocast_new.dtype == torch.float16


@mps_and_cuda
def test_autocast_tensor_state_dict_roundtrip(device: torch.device):
    model = DummyModule()
    # Model parameters should start off on the CPU.
    assert all(p.device.type == "cpu" for p in model.parameters())

    # Trying to run inference on an on-device tensor should raise an error.
    x = torch.randn(10, 10, device=device)
    with pytest.raises(RuntimeError):
        model(x)

    # Extract the state dict and wrap the tensors in AutocastTensor.
    state_dict = model.state_dict()
    state_dict = {k: AutocastTensor(v, device) for k, v in state_dict.items()}

    # Load the wrapped state_dict into the model.
    model.load_state_dict(state_dict, assign=True)

    # The model parameters and buffers should all be wrapped in AutocastTensor.
    assert all(isinstance(p, AutocastTensor) for p in itertools.chain(model.parameters(), model.buffers()))
    assert all(
        p.get_original_tensor().device.type == "cpu" for p in itertools.chain(model.parameters(), model.buffers())
    )

    # Run inference on the model.
    x = torch.randn(10, 10, device=device)
    result = model(x)

    # The result should be on the device.
    assert result.device.type == device.type

    # Verify that we can extract the state dict from the model after adding the AutocastTensor wrappers.
    state_dict = model.state_dict()
    assert all(isinstance(v, AutocastTensor) for v in state_dict.values())
