"""The worker's end-of-session VRAM release: a canceled session asks for an empty_cache (its
denoise working set is orphaned mid-node with nothing downstream to release it); a completed
one only flushes a release a peer deferred onto it; neither may fail the worker."""

import threading
from types import SimpleNamespace

from invokeai.app.services.session_processor.session_processor_default import DefaultSessionProcessor
from invokeai.backend.util.devices import TorchDevice


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str, **kwargs) -> None:
        self.warnings.append(message)


def _processor() -> tuple[DefaultSessionProcessor, _Logger]:
    logger = _Logger()
    processor = DefaultSessionProcessor()
    processor._invoker = SimpleNamespace(services=SimpleNamespace(logger=logger))  # type: ignore[attr-defined]
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
