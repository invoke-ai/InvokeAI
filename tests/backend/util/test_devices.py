"""
Test abstract device class.
"""

import threading
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


# ===== per-thread session device (multi-GPU worker pinning) ================


def test_session_device_overrides_config():
    """A per-thread session device takes precedence over the global config.device."""
    config = get_config()
    config.device = "cpu"
    try:
        TorchDevice.set_session_device("cuda:1")
        assert TorchDevice.choose_torch_device() == torch.device("cuda:1")
    finally:
        TorchDevice.clear_session_device()
    # Once cleared, we fall back to the global config.
    assert TorchDevice.choose_torch_device() == torch.device("cpu")


def test_session_device_is_thread_local():
    """Each thread sees only its own pinned device; the main thread is unaffected."""
    config = get_config()
    config.device = "cpu"
    results: dict[str, torch.device] = {}
    barrier = threading.Barrier(2)

    def worker(name: str, device: str):
        TorchDevice.set_session_device(device)
        # Wait so both threads have set their device before either reads it, proving isolation.
        barrier.wait()
        results[name] = TorchDevice.choose_torch_device()
        TorchDevice.clear_session_device()

    t0 = threading.Thread(target=worker, args=("a", "cuda:0"))
    t1 = threading.Thread(target=worker, args=("b", "cuda:1"))
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    assert results["a"] == torch.device("cuda:0")
    assert results["b"] == torch.device("cuda:1")
    # The main thread never set a session device, so it still uses the global config.
    assert TorchDevice.get_session_device() is None
    assert TorchDevice.choose_torch_device() == torch.device("cpu")


# ===== generation_devices resolution (config -> concrete device list) =======


def test_get_generation_devices_auto_expands_to_all_cuda():
    """`auto` enumerates every visible CUDA device."""
    with (
        patch("invokeai.backend.util.devices.torch.cuda.is_available", return_value=True),
        patch("invokeai.backend.util.devices.torch.cuda.device_count", return_value=3),
    ):
        assert TorchDevice.get_generation_devices("auto") == [
            torch.device("cuda:0"),
            torch.device("cuda:1"),
            torch.device("cuda:2"),
        ]


def test_get_generation_devices_auto_without_cuda():
    """`auto` falls back to the single best device when CUDA is unavailable."""
    config = get_config()
    config.device = "cpu"
    with (
        patch("invokeai.backend.util.devices.torch.cuda.is_available", return_value=False),
        patch("invokeai.backend.util.devices.torch.backends.mps.is_available", return_value=False),
    ):
        assert TorchDevice.get_generation_devices("auto") == [torch.device("cpu")]


def test_get_generation_devices_explicit_list_is_deduplicated():
    """An explicit list is normalized and deduplicated, preserving order."""
    assert TorchDevice.get_generation_devices(["cuda:0", "cuda:0", "cuda:1"]) == [
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    ]


@pytest.mark.parametrize("value", [None, []])
def test_get_generation_devices_empty(value):
    """`None` or an empty list resolves to an empty list (caller handles the single-device fallback)."""
    assert TorchDevice.get_generation_devices(value) == []


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


# ===== choose_anima_inference_dtype (config.precision honoring) ============


def test_choose_anima_inference_dtype_float16():
    """precision='float16' returns torch.float16 without touching hardware."""
    config = get_config()
    config.precision = "float16"
    result = TorchDevice.choose_anima_inference_dtype(torch.device("cpu"))
    assert result is torch.float16


def test_choose_anima_inference_dtype_bfloat16():
    """precision='bfloat16' returns torch.bfloat16 without touching hardware."""
    config = get_config()
    config.precision = "bfloat16"
    result = TorchDevice.choose_anima_inference_dtype(torch.device("cpu"))
    assert result is torch.bfloat16


def test_choose_anima_inference_dtype_float32():
    """precision='float32' returns torch.float32 without touching hardware."""
    config = get_config()
    config.precision = "float32"
    result = TorchDevice.choose_anima_inference_dtype(torch.device("cpu"))
    assert result is torch.float32


def test_choose_anima_inference_dtype_auto_delegates_to_safe_dtype():
    """precision='auto' delegates to choose_bfloat16_safe_dtype (current behavior)."""
    config = get_config()
    config.precision = "auto"
    device = torch.device("cpu")
    sentinel = torch.bfloat16
    with patch.object(TorchDevice, "choose_bfloat16_safe_dtype", return_value=sentinel) as mock_safe:
        result = TorchDevice.choose_anima_inference_dtype(device)
    assert result is sentinel
    mock_safe.assert_called_once_with(device)
