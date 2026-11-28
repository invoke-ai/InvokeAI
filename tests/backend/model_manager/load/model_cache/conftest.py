from contextlib import nullcontext

import pytest
import torch


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


class FakeRuntime:
    """A page-locking runtime that registers nothing for real. `log` records every call, in order."""

    def __init__(self) -> None:
        self.log: list[tuple[str, int]] = []
        self.registered: set[int] = set()

    def register(self, ptr: int, nbytes: int) -> int:
        self.log.append(("register", ptr))
        self.registered.add(ptr)
        return 0

    def unregister(self, ptr: int) -> int:
        self.log.append(("unregister", ptr))
        self.registered.discard(ptr)
        return 0


@pytest.fixture
def fake_streaming(monkeypatch: pytest.MonkeyPatch):
    """A fresh registry on a `FakeRuntime`, with streaming possible on whatever device: `(registry, runtime)`."""
    from invokeai.backend.model_manager.load.model_cache.torch_module_autocast import weight_streaming

    registry = weight_streaming.PinnedStorageRegistry()
    runtime = FakeRuntime()
    monkeypatch.setattr(weight_streaming, "PINNED_STORAGES", registry)
    monkeypatch.setattr(weight_streaming, "_runtime", runtime)
    monkeypatch.setattr(weight_streaming, "_registration_self_test", lambda device=None: None)
    monkeypatch.setattr(weight_streaming, "can_stream_weights_to", lambda device, has_ram_copy: has_ram_copy)
    monkeypatch.setattr(weight_streaming, "_streamers", {})
    # Registration selects the model's device, which does not exist here.
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    return registry, runtime
