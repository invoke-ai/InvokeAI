"""
Test abstract device class.
"""

import pytest
import torch

from invokeai.app.services.config import get_config
from invokeai.backend.util.devices import TorchDeviceSelect

devices = ["cpu", "cuda:0", "cuda:1", "mps"]
device_types = [
    ("cpu", torch.float32),
    ("cuda:0", torch.float16 if torch.cuda.is_available() else torch.float32),
    ("mps", torch.float16 if torch.backends.mps.is_available() else torch.float32),
]


@pytest.mark.parametrize("device_name", devices)
def test_device_choice(device_name):
    config = get_config()
    config.device = device_name
    torch_device = TorchDeviceSelect.choose_torch_device()
    assert torch_device == torch.device(device_name)


@pytest.mark.parametrize("device_dtype_pair", device_types)
def test_device_dtype(device_dtype_pair):
    device_name, dtype = device_dtype_pair
    config = get_config()
    config.device = device_name
    torch_dtype = TorchDeviceSelect.choose_torch_dtype()
    assert torch_dtype == dtype


@pytest.mark.parametrize("device_dtype_pair", device_types)
def test_device_dtype_override(device_dtype_pair):
    device_name, dtype = device_dtype_pair
    config = get_config()
    config.device = device_name
    config.precision = "float32"
    torch_dtype = TorchDeviceSelect.choose_torch_dtype()
    assert torch_dtype == torch.float32


def test_normalize():
    assert (
        TorchDeviceSelect.normalize("cuda") == torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cuda")
    )
    assert (
        TorchDeviceSelect.normalize("cuda:0") == torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cuda")
    )
    assert (
        TorchDeviceSelect.normalize("cuda:1") == torch.device("cuda:1")
        if torch.cuda.is_available()
        else torch.device("cuda")
    )
    assert TorchDeviceSelect.normalize("mps") == torch.device("mps")
    assert TorchDeviceSelect.normalize("cpu") == torch.device("cpu")
