"""The worker's end-of-session VRAM release: a canceled session asks for an empty_cache (its
denoise working set is orphaned mid-node with nothing downstream to release it); a completed
one only flushes a release a peer deferred onto it; neither may fail the worker."""

import threading
from types import SimpleNamespace

import torch

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.model_load.model_load_default import ModelLoadService
from invokeai.app.services.session_processor.session_processor_default import DefaultSessionProcessor
from invokeai.backend.util.devices import TorchDevice


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str, **kwargs) -> None:
        self.warnings.append(message)


class _Cache:
    def __init__(self) -> None:
        self.offloaded: list[tuple[str, ...]] = []

    def offload_models_from_vram_except(self, keep_model_keys) -> int:
        self.offloaded.append(tuple(keep_model_keys))
        return 0


def _processor(
    clear_vram_after_session: bool = False, cache: "_Cache | None" = None
) -> tuple[DefaultSessionProcessor, _Logger]:
    logger = _Logger()
    processor = DefaultSessionProcessor()
    services = SimpleNamespace(
        logger=logger,
        configuration=SimpleNamespace(clear_vram_after_session=clear_vram_after_session),
        model_manager=SimpleNamespace(load=SimpleNamespace(ram_cache=cache or _Cache())),
    )
    processor._invoker = SimpleNamespace(services=services)  # type: ignore[attr-defined]
    return processor, logger


def _worker(canceled: bool) -> SimpleNamespace:
    event = threading.Event()
    if canceled:
        event.set()
    return SimpleNamespace(cancel_event=event, label="worker (cuda:0)")


def test_canceled_session_requests_an_empty_cache(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(TorchDevice, "empty_cache", classmethod(lambda cls: calls.append("empty")))
    monkeypatch.setattr(TorchDevice, "flush_deferred_empty_cache", classmethod(lambda cls: calls.append("flush")))
    processor, logger = _processor()

    processor._release_vram_after_session(_worker(canceled=True))  # type: ignore[arg-type]

    assert calls == ["empty"]
    assert logger.warnings == []


def test_completed_session_only_flushes_a_deferred_release(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(TorchDevice, "empty_cache", classmethod(lambda cls: calls.append("empty")))
    monkeypatch.setattr(TorchDevice, "flush_deferred_empty_cache", classmethod(lambda cls: calls.append("flush")))
    processor, logger = _processor()

    processor._release_vram_after_session(_worker(canceled=False))  # type: ignore[arg-type]

    assert calls == ["flush"]
    assert logger.warnings == []


def test_release_failure_is_logged_not_raised(monkeypatch) -> None:
    def boom(cls) -> None:
        raise RuntimeError("HIP error: invalid device context")

    monkeypatch.setattr(TorchDevice, "empty_cache", classmethod(boom))
    processor, logger = _processor()

    processor._release_vram_after_session(_worker(canceled=True))  # type: ignore[arg-type]

    assert len(logger.warnings) == 1
    assert "cuda:0" in logger.warnings[0]


def test_clear_vram_after_session_moves_every_model_to_ram_and_releases_the_memory(monkeypatch) -> None:
    """The setting for a session that runs fine once and runs out of memory the second time: every unlocked model
    leaves VRAM (staying cached in RAM) and the allocator hands its blocks back, even after a completed session."""
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        TorchDevice, "empty_cache", classmethod(lambda cls, force=False: calls.append(("empty", force)))
    )
    monkeypatch.setattr(
        TorchDevice, "flush_deferred_empty_cache", classmethod(lambda cls: calls.append(("flush", False)))
    )
    cache = _Cache()
    processor, logger = _processor(clear_vram_after_session=True, cache=cache)

    processor._release_vram_after_session(_worker(canceled=False))  # type: ignore[arg-type]

    assert cache.offloaded == [()]
    # Forced: deferred behind a busy peer, the next session would budget against memory the allocator still holds.
    assert calls == [("empty", True)]
    assert logger.warnings == []


def test_clear_vram_after_session_leaves_other_gpus_caches_alone(monkeypatch) -> None:
    """Multi-GPU: a worker clears its own device's cache only -- a peer may be mid-session on its own."""
    monkeypatch.setattr(TorchDevice, "empty_cache", classmethod(lambda cls, force=False: True))
    caches = {"cuda:0": _Cache(), "cuda:1": _Cache()}
    for device, cache in caches.items():
        cache.execution_device = torch.device(device)  # type: ignore[attr-defined]
    load = ModelLoadService(app_config=InvokeAIAppConfig(), ram_cache=caches["cuda:0"], ram_caches=caches)  # type: ignore[arg-type]
    processor, logger = _processor(clear_vram_after_session=True)
    processor._invoker.services.model_manager.load = load  # type: ignore[attr-defined]

    def worker() -> None:
        TorchDevice.set_session_device("cuda:1")
        try:
            processor._release_vram_after_session(_worker(canceled=False))  # type: ignore[arg-type]
        finally:
            TorchDevice.clear_session_device()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert caches["cuda:1"].offloaded == [()]
    assert caches["cuda:0"].offloaded == []
    assert logger.warnings == []
