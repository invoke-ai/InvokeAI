"""The end-of-session warning when Windows keeps part of a worker's VRAM in shared system memory.

It must count only what stayed paged across two session ends, speak once per episode (again only after that grew by
the threshold, re-armed once a reading fell below it), say nothing where the answer is unknown, and never fail the
worker.
"""

from types import SimpleNamespace

import pytest
import torch

import invokeai.app.services.session_processor.session_processor_default as processor_module
from invokeai.app.services.session_processor.session_processor_default import (
    _VRAM_PAGING_WARNING_BYTES,
    DefaultSessionProcessor,
    _SessionWorker,
)

MIB = 2**20
DEVICE = torch.device("cuda", 0)


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.debugs: list[str] = []

    def warning(self, message: str, **kwargs) -> None:
        self.warnings.append(message)

    def debug(self, message: str, **kwargs) -> None:
        self.debugs.append(message)


@pytest.fixture
def processor() -> tuple[DefaultSessionProcessor, _Logger]:
    logger = _Logger()
    processor = DefaultSessionProcessor()
    processor._invoker = SimpleNamespace(services=SimpleNamespace(logger=logger))  # type: ignore[attr-defined]
    return processor, logger


def _warned_after_each_session(monkeypatch, processor, readings) -> list[bool]:
    """End one session per scripted `paged_bytes` reading on one worker; whether each one warned."""
    processor, logger = processor
    worker = _SessionWorker(DEVICE, runner=SimpleNamespace())  # type: ignore[arg-type]
    values = iter(readings)

    def paged_bytes(device):
        assert device == DEVICE, "the worker's own device is the one asked"
        return next(values)

    monkeypatch.setattr(processor_module, "paged_bytes", paged_bytes)
    warned = []
    for _ in readings:
        before = len(logger.warnings)
        processor._warn_if_vram_paged(worker)
        warned.append(len(logger.warnings) > before)
    return warned


def test_warns_once_per_episode_and_again_only_after_growth(monkeypatch, processor):
    threshold = _VRAM_PAGING_WARNING_BYTES
    readings = [
        6 * MIB,  # an idle process
        threshold + 100 * MIB,  # paged, once so far
        threshold + 300 * MIB,  # still paged: warn
        2 * threshold + 400 * MIB,  # what stayed paged grew by less than the threshold since the warning
        2 * threshold + 500 * MIB,  # grew by the threshold: warn again
        100 * MIB,  # recovered below the threshold: re-armed
        threshold,
        threshold,  # a new episode: warn
    ]

    warned = _warned_after_each_session(monkeypatch, processor, readings)

    assert warned == [False, False, True, False, True, False, False, True]
    assert "cuda:0" in processor[1].warnings[0]
    assert "0.6 GiB" in processor[1].warnings[0]  # what stayed paged (0.6), not the latest reading (0.8)


def test_paging_that_clears_between_sessions_is_not_reported(monkeypatch, processor):
    """A decode can overflow for a few seconds at the end of every session and then return to VRAM."""
    paged, cleared = _VRAM_PAGING_WARNING_BYTES + 400 * MIB, 6 * MIB
    readings = [paged, cleared, paged, cleared, paged]

    assert _warned_after_each_session(monkeypatch, processor, readings) == [False] * 5


def test_silent_where_the_answer_is_unknown(monkeypatch, processor):
    assert _warned_after_each_session(monkeypatch, processor, [None, None]) == [False, False]


def test_a_failing_counter_is_logged_at_debug_and_does_not_fail_the_worker(monkeypatch, processor):
    processor_instance, logger = processor

    def broken(device):
        raise OSError("PDH query failed")

    monkeypatch.setattr(processor_module, "paged_bytes", broken)

    processor_instance._warn_if_vram_paged(_SessionWorker(DEVICE, runner=SimpleNamespace()))  # type: ignore[arg-type]

    assert logger.warnings == []
    assert len(logger.debugs) == 1


def test_the_worker_checks_after_each_session(monkeypatch):
    from tests.app.services.session_processor.test_session_processor_cancel_guard import _run_guard_scenario

    checked = []
    monkeypatch.setattr(DefaultSessionProcessor, "_warn_if_vram_paged", lambda self, worker: checked.append(worker))

    _canceled, run_items, worker = _run_guard_scenario(statuses=["in_progress"], set_cancel_event=False)

    assert run_items == [42]
    assert checked == [worker]
