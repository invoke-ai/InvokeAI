import pytest


@pytest.fixture(autouse=True)
def _no_windows_video_memory_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the cache's VRAM arithmetic on whatever each test patches into ``torch.cuda``.

    Many of these tests fake a ``cuda`` execution device and patch ``torch.cuda.mem_get_info``. On a ROCm build under
    Windows, ``TorchDevice.cuda_mem_get_info`` would additionally cap that fake with the real adapter's WDDM budget,
    making the expected figures depend on the machine running the suite. Tests of the cap patch it back in.
    """
    monkeypatch.setattr("invokeai.backend.util.devices.video_memory_budget", lambda device: None)


@pytest.fixture(autouse=True)
def _native_cuda_allocator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Budget these tests as with torch's native caching allocator, whatever the machine configures.

    A cudaMallocAsync machine would otherwise change the credit and the offload policy under the fake devices. Tests
    of the async pool patch the backend back in.
    """
    monkeypatch.setattr("torch.cuda.get_allocator_backend", lambda: "native")
