"""Tests that ModelLoadService routes to the per-device cache for the calling thread (multi-GPU)."""

import threading

import torch

from invokeai.app.services.config.config_default import InvokeAIAppConfig, get_config
from invokeai.app.services.model_load.model_load_default import ModelLoadService
from invokeai.backend.util.devices import TorchDevice


class _FakeCache:
    """Stand-in for ModelCache; ModelLoadService only needs `.execution_device` for keying."""

    def __init__(self, device: str):
        self.execution_device = torch.device(device)


def _build_service() -> tuple[ModelLoadService, _FakeCache, _FakeCache]:
    cache0 = _FakeCache("cuda:0")
    cache1 = _FakeCache("cuda:1")
    service = ModelLoadService(
        app_config=InvokeAIAppConfig(),
        ram_cache=cache0,  # type: ignore[arg-type]
        ram_caches={"cuda:0": cache0, "cuda:1": cache1},  # type: ignore[arg-type]
    )
    return service, cache0, cache1


def test_ram_cache_routes_to_pinned_device():
    """A thread pinned to cuda:1 resolves to that device's cache; the default thread to cuda:0."""
    service, cache0, cache1 = _build_service()

    # The default thread has no session device; point config.device at cuda:0 so it resolves there.
    get_config().device = "cuda:0"
    assert service.ram_cache is cache0

    results: dict[str, object] = {}

    def worker():
        TorchDevice.set_session_device("cuda:1")
        try:
            results["cache"] = service.ram_cache
        finally:
            TorchDevice.clear_session_device()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert results["cache"] is cache1
    # Main thread is unaffected by the worker's pinning.
    assert service.ram_cache is cache0


def test_ram_caches_exposes_all_devices():
    service, cache0, cache1 = _build_service()
    caches = service.ram_caches
    assert set(caches.keys()) == {"cuda:0", "cuda:1"}
    assert caches["cuda:0"] is cache0
    assert caches["cuda:1"] is cache1


def test_unknown_device_falls_back_to_default():
    """A thread pinned to a device with no cache falls back to the default cache."""
    service, cache0, _ = _build_service()

    results: dict[str, object] = {}

    def worker():
        TorchDevice.set_session_device("cuda:7")
        try:
            results["cache"] = service.ram_cache
        finally:
            TorchDevice.clear_session_device()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert results["cache"] is cache0
