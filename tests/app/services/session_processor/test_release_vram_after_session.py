"""The worker's end-of-session VRAM release: a session that ended early (canceled, failed in a node, or aborted by an
error out of the runner) asks for an empty_cache, its working set being orphaned mid-node with nothing downstream to
release it; a completed one only flushes a release a peer deferred onto it; neither may fail the worker."""

import asyncio
import threading
import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Callable, Iterator

import pytest
import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.fields import InputField, OutputField
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_load.model_load_default import ModelLoadService
from invokeai.app.services.session_processor.session_processor_default import (
    DefaultSessionProcessor,
    DefaultSessionRunner,
)
from invokeai.app.services.session_queue.session_queue_common import Batch
from invokeai.app.services.session_queue.session_queue_default import SessionQueue
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.util.devices import TorchDevice

# Bounds a hang, not worker latency.
_TIMEOUT = 30


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str, **kwargs) -> None:
        self.warnings.append(message)


class _DeviceRequests:
    """The calls the worker makes to `TorchDevice` for a release, as (name, force), and an event set by each one.

    The worker releases on its own thread after the queue status is final, so a test waits on `wait()`. `sample`, when
    set, is called as each release is requested and its results are collected in `samples`."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[str, bool]] = []
        self.sample: Callable[[], object] | None = None
        self.samples: list[object] = []
        self._released = threading.Event()
        monkeypatch.setattr(TorchDevice, "empty_cache", self._record("empty_cache"))
        monkeypatch.setattr(TorchDevice, "flush_deferred_empty_cache", self._record("flush"))

    def _record(self, name: str):
        def stub(cls, force: bool = False) -> bool:
            self.calls.append((name, force))
            if self.sample is not None:
                self.samples.append(self.sample())
            self._released.set()
            return True

        return classmethod(stub)

    def wait(self) -> list[tuple[str, bool]]:
        assert self._released.wait(_TIMEOUT)
        return self.calls


@pytest.fixture
def device_requests(monkeypatch: pytest.MonkeyPatch) -> _DeviceRequests:
    return _DeviceRequests(monkeypatch)


class _Stats:
    @contextmanager
    def collect_stats(self, invocation, graph_execution_state_id) -> Iterator[None]:
        yield

    def log_stats(self, graph_execution_state_id) -> None:
        pass

    def reset_stats(self, graph_execution_state_id) -> None:
        pass


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


def _worker(outcome: str = "completed") -> SimpleNamespace:
    """A worker whose session ended as `outcome`: completed, canceled, or failed in a node."""
    event = threading.Event()
    if outcome == "canceled":
        event.set()
    session = GraphExecutionState(graph=Graph())
    if outcome == "failed":
        session.errors["node"] = "RuntimeError: out of memory"
    return SimpleNamespace(cancel_event=event, label="worker (cuda:0)", queue_item=SimpleNamespace(session=session))


def test_canceled_session_requests_an_unforced_empty_cache(device_requests) -> None:
    processor, logger = _processor()

    processor._release_vram_after_session(_worker("canceled"), run_returned=True)  # type: ignore[arg-type]

    assert device_requests.calls == [("empty_cache", False)]
    assert logger.warnings == []


@pytest.mark.parametrize("outcome", ["completed", "canceled", "failed"])
def test_clear_vram_after_session_moves_every_model_to_ram_and_forces_the_release(device_requests, outcome) -> None:
    """The setting for a session that runs fine once and runs out of memory the second time: every unlocked model
    leaves VRAM (staying cached in RAM) and the allocator hands its blocks back, whatever the session's outcome."""
    cache = _Cache()
    processor, logger = _processor(clear_vram_after_session=True, cache=cache)

    processor._release_vram_after_session(_worker(outcome), run_returned=True)  # type: ignore[arg-type]

    assert cache.offloaded == [()]
    # Forced: deferred behind a busy peer, the next session would budget against memory the allocator still holds.
    assert device_requests.calls == [("empty_cache", True)]
    assert logger.warnings == []


def test_clear_vram_after_session_logs_a_refused_forced_release_instead_of_failing_the_worker(monkeypatch) -> None:
    def refuse(cls, force: bool = False) -> bool:
        raise RuntimeError("HIP error: invalid device context")

    monkeypatch.setattr(TorchDevice, "empty_cache", classmethod(refuse))
    processor, logger = _processor(clear_vram_after_session=True)

    processor._release_vram_after_session(_worker(), run_returned=True)  # type: ignore[arg-type]

    assert len(logger.warnings) == 1
    assert "cuda:0" in logger.warnings[0]


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
            processor._release_vram_after_session(_worker(), run_returned=True)  # type: ignore[arg-type]
        finally:
            TorchDevice.clear_session_device()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert caches["cuda:1"].offloaded == [()]
    assert caches["cuda:0"].offloaded == []
    assert logger.warnings == []


@invocation_output("test_release_vram_output")
class _ReleaseVramOutput(BaseInvocationOutput):
    value: int = OutputField(description="The node's value")


@invocation("test_release_vram_node", version="1.0.0")
class _ReleaseVramInvocation(BaseInvocation):
    fail: bool = InputField(default=False, description="Raise instead of returning")

    def invoke(self, context: InvocationContext) -> _ReleaseVramOutput:
        if self.fail:
            raise RuntimeError("out of memory")
        return _ReleaseVramOutput(value=1)


class _NodeMemory:
    """Stands in for the tensors a node holds."""


class _RaisingRunner(DefaultSessionRunner):
    def __init__(self) -> None:
        super().__init__()
        self.node_memory: weakref.ref[_NodeMemory] | None = None

    def run(self, queue_item) -> None:
        node_memory = _NodeMemory()
        self.node_memory = weakref.ref(node_memory)
        raise RuntimeError("runner broke")


def _run_one_session(
    invoker: Invoker,
    database: Database,
    device_requests: _DeviceRequests,
    *,
    fail: bool = False,
    runner: DefaultSessionRunner | None = None,
) -> tuple[SessionQueue, list[tuple[str, bool]]]:
    """Run one session on a real processor and return the queue and what the worker then asked of the device.

    The worker releases after the item's status is final, so the item is read back once the release was requested.
    Stopping the processor sets the worker's cancel event, so it waits for the release first or the session would
    look canceled."""
    queue = SessionQueue(database)
    invoker.services.session_queue = queue
    invoker.services.performance_statistics = _Stats()
    queue.start(invoker)
    graph = Graph()
    graph.add_node(_ReleaseVramInvocation(id="node", fail=fail))
    asyncio.run(queue.enqueue_batch("default", Batch(graph=graph, origin="canvas"), prepend=False))
    processor = DefaultSessionProcessor(session_runner=runner or DefaultSessionRunner(), polling_interval=0)
    try:
        processor.start(invoker)
        calls = device_requests.wait()
    finally:
        processor.stop()
        for worker in processor._workers:
            assert worker.thread is not None
            worker.thread.join(timeout=_TIMEOUT)
            assert not worker.thread.is_alive()
    return queue, calls


def test_session_failed_in_a_node_requests_an_unforced_empty_cache(
    mock_invoker: Invoker, mock_sqlite_database: Database, device_requests: _DeviceRequests
) -> None:
    queue, calls = _run_one_session(mock_invoker, mock_sqlite_database, device_requests, fail=True)

    assert calls == [("empty_cache", False)]
    item = queue.get_queue_item(1)
    assert (item.status, item.error_message) == ("failed", "out of memory")


def test_completed_session_only_flushes_a_deferred_release(
    mock_invoker: Invoker, mock_sqlite_database: Database, device_requests: _DeviceRequests
) -> None:
    queue, calls = _run_one_session(mock_invoker, mock_sqlite_database, device_requests)

    assert calls == [("flush", False)]
    assert queue.get_queue_item(1).status == "completed"


def test_error_out_of_the_runner_fails_the_item_and_requests_an_unforced_empty_cache(
    mock_invoker: Invoker, mock_sqlite_database: Database, device_requests: _DeviceRequests
) -> None:
    queue, calls = _run_one_session(mock_invoker, mock_sqlite_database, device_requests, runner=_RaisingRunner())

    assert calls == [("empty_cache", False)]
    item = queue.get_queue_item(1)
    assert (item.status, item.error_message) == ("failed", "runner broke")


def test_release_after_an_error_out_of_the_runner_runs_once_the_failed_frames_are_gone(
    mock_invoker: Invoker, mock_sqlite_database: Database, device_requests: _DeviceRequests
) -> None:
    """While the exception is in flight its traceback keeps what the failed frames hold alive, so a release made then
    would free none of it."""
    runner = _RaisingRunner()
    device_requests.sample = lambda: runner.node_memory is not None and runner.node_memory() is not None

    _run_one_session(mock_invoker, mock_sqlite_database, device_requests, runner=runner)

    assert runner.node_memory is not None
    assert device_requests.samples == [False]
