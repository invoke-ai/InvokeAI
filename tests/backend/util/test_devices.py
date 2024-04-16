"""
Test abstract device class.
"""

from unittest.mock import patch

import pytest
import torch

from invokeai.app.services.config import get_config
from invokeai.backend.model_manager.load import ModelCache
from invokeai.backend.util.devices import TorchDevice, choose_precision, choose_torch_device, torch_dtype

devices = ["cpu", "cuda:0", "cuda:1", "mps"]
device_types_cpu = [("cpu", torch.float32), ("cuda:0", torch.float32), ("mps", torch.float32)]
device_types_cuda = [("cpu", torch.float32), ("cuda:0", torch.float16), ("mps", torch.float32)]
device_types_mps = [("cpu", torch.float32), ("cuda:0", torch.float32), ("mps", torch.float16)]


@pytest.mark.parametrize("device_name", devices)
def test_device_choice(device_name):
    config = get_config()
    config.device = device_name
    torch_device = TorchDevice.choose_torch_device()
    assert torch_device == torch.device(device_name)


@pytest.mark.parametrize("device_dtype_pair", device_types_cpu)
def test_device_dtype_cpu(device_dtype_pair):
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        device_name, dtype = device_dtype_pair
        config = get_config()
        config.device = device_name
        torch_dtype = TorchDevice.choose_torch_dtype()
        assert torch_dtype == dtype


@pytest.mark.parametrize("device_dtype_pair", device_types_cuda)
def test_device_dtype_cuda(device_dtype_pair):
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_name", return_value="RTX4070"),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        device_name, dtype = device_dtype_pair
        config = get_config()
        config.device = device_name
        torch_dtype = TorchDevice.choose_torch_dtype()
        assert torch_dtype == dtype


@pytest.mark.parametrize("device_dtype_pair", device_types_mps)
def test_device_dtype_mps(device_dtype_pair):
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        device_name, dtype = device_dtype_pair
        config = get_config()
        config.device = device_name
        torch_dtype = TorchDevice.choose_torch_dtype()
        assert torch_dtype == dtype


@pytest.mark.parametrize("device_dtype_pair", device_types_cuda)
def test_device_dtype_override(device_dtype_pair):
    with (
        patch("torch.cuda.get_device_name", return_value="RTX4070"),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        device_name, dtype = device_dtype_pair
        config = get_config()
        config.device = device_name
        config.precision = "float32"
        torch_dtype = TorchDevice.choose_torch_dtype()
        assert torch_dtype == torch.float32


def test_normalize():
    assert (
        TorchDevice.normalize("cuda") == torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cuda")
    )
    assert (
        TorchDevice.normalize("cuda:0") == torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cuda")
    )
    assert (
        TorchDevice.normalize("cuda:1") == torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cuda")
    )
    assert TorchDevice.normalize("mps") == torch.device("mps")
    assert TorchDevice.normalize("cpu") == torch.device("cpu")


@pytest.mark.parametrize("device_name", devices)
def test_legacy_device_choice(device_name):
    config = get_config()
    config.device = device_name
    with pytest.deprecated_call():
        torch_device = choose_torch_device()
    assert torch_device == torch.device(device_name)


@pytest.mark.parametrize("device_dtype_pair", device_types_cpu)
def test_legacy_device_dtype_cpu(device_dtype_pair):
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.get_device_name", return_value="RTX9090"),
    ):
        device_name, dtype = device_dtype_pair
        config = get_config()
        config.device = device_name
        with pytest.deprecated_call():
            torch_device = choose_torch_device()
            returned_dtype = torch_dtype(torch_device)
        assert returned_dtype == dtype


def test_legacy_precision_name():
    config = get_config()
    config.precision = "auto"
    with (
        pytest.deprecated_call(),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.cuda.get_device_name", return_value="RTX9090"),
    ):
        assert "float16" == choose_precision(torch.device("cuda"))
        assert "float16" == choose_precision(torch.device("mps"))
        assert "float32" == choose_precision(torch.device("cpu"))


def test_multi_device_support_1():
    config = get_config()
    config.devices = ["cuda:0", "cuda:1"]
    assert TorchDevice.execution_devices() == {torch.device("cuda:0"), torch.device("cuda:1")}


def test_multi_device_support_2():
    config = get_config()
    config.devices = None
    with (
        patch("torch.cuda.device_count", return_value=3),
        patch("torch.cuda.is_available", return_value=True),
    ):
        assert TorchDevice.execution_devices() == {
            torch.device("cuda:0"),
            torch.device("cuda:1"),
            torch.device("cuda:2"),
        }


def test_multi_device_support_3():
    config = get_config()
    config.devices = ["cuda:0", "cuda:1"]
    cache = ModelCache()
    with cache.reserve_execution_device() as gpu:
        assert gpu in [torch.device(x) for x in config.devices]
        assert TorchDevice.choose_torch_device() == gpu
