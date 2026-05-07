"""
Test abstract device class.
"""

from unittest.mock import patch

import pytest
import torch

from invokeai.app.services.config import get_config
from invokeai.backend.util.devices import TorchDevice, choose_precision, choose_torch_device, torch_dtype

devices = ["cpu", "cuda:0", "cuda:1", "cuda:2", "mps"]
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


# --- resolve_model_precision tests ---


def test_resolve_model_precision_auto_delegates_to_bfloat16_safe():
    """When override is 'auto', the helper should defer to choose_bfloat16_safe_dtype."""
    sentinel = torch.bfloat16
    with patch.object(TorchDevice, "choose_bfloat16_safe_dtype", return_value=sentinel) as mock_safe:
        result = TorchDevice.resolve_model_precision("auto", torch.device("cpu"))
    assert result is sentinel
    mock_safe.assert_called_once_with(torch.device("cpu"))


def test_resolve_model_precision_explicit_bfloat16():
    """An explicit 'bfloat16' override should return torch.bfloat16 unchanged."""
    result = TorchDevice.resolve_model_precision("bfloat16", torch.device("cpu"))
    assert result == torch.bfloat16


def test_resolve_model_precision_explicit_float32():
    """An explicit 'float32' override should return torch.float32 unchanged."""
    result = TorchDevice.resolve_model_precision("float32", torch.device("cpu"))
    assert result == torch.float32


def test_resolve_model_precision_does_not_consult_safe_dtype_for_explicit_overrides():
    """Explicit overrides must bypass choose_bfloat16_safe_dtype entirely."""
    with patch.object(TorchDevice, "choose_bfloat16_safe_dtype") as mock_safe:
        TorchDevice.resolve_model_precision("float32", torch.device("cpu"))
        TorchDevice.resolve_model_precision("bfloat16", torch.device("cpu"))
    mock_safe.assert_not_called()
