from contextlib import contextmanager
from threading import Event

import pytest

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.services.session_processor.session_processor_default import DefaultSessionRunner
from invokeai.app.services.shared.graph import WorkflowCallFrame
from tests.dangerously_run_function_in_subprocess import dangerously_run_function_in_subprocess


@invocation_output("test_interrupt_output")
class InterruptTestOutput(BaseInvocationOutput):
    pass


@invocation("test_keyboard_interrupt", version="1.0.0")
class KeyboardInterruptInvocation(BaseInvocation):
    def invoke(self, context) -> InterruptTestOutput:
        raise KeyboardInterrupt


class _DummyStats:
    @contextmanager
    def collect_stats(self, invocation: BaseInvocation, graph_execution_state_id: str):
        yield

    def log_stats(self, graph_execution_state_id: str) -> None:
        pass

    def reset_stats(self, graph_execution_state_id: str) -> None:
        pass


class _DummyEvents:
    def emit_invocation_started(self, queue_item, invocation) -> None:
        pass

    def emit_invocation_complete(self, invocation, queue_item, output) -> None:
        pass

    def emit_invocation_error(self, queue_item, invocation, error_type, error_message, error_traceback) -> None:
        pass


class _DummyLogger:
    def debug(self, msg) -> None:
        pass

    def error(self, msg) -> None:
        pass


class _DummyConfig:
    node_cache_size = 0


def _build_runner(monkeypatch: pytest.MonkeyPatch) -> DefaultSessionRunner:
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.build_invocation_context",
        lambda data, services, is_canceled: None,
    )

    runner = DefaultSessionRunner()
    runner.start(
        services=type(
            "Services",
            (),
            {
                "performance_statistics": _DummyStats(),
                "events": _DummyEvents(),
                "logger": _DummyLogger(),
                "configuration": _DummyConfig(),
            },
        )(),
        cancel_event=Event(),
    )
    return runner


def _build_queue_item(invocation: BaseInvocation):
    return type(
        "QueueItem",
        (),
        {
            "item_id": 1,
            "session_id": "test-session",
            "session": type("Session", (), {"prepared_source_mapping": {invocation.id: invocation.id}})(),
        },
    )()


class _DummySessionQueue:
    def __init__(self) -> None:
        self.completed_item_ids: list[int] = []
        self.session_updates: list[tuple[int, object]] = []

    def set_queue_item_session(self, item_id: int, session):
        self.session_updates.append((item_id, session))
        return type("QueueItem", (), {"item_id": item_id, "status": "in_progress", "session": session})()

    def complete_queue_item(self, item_id: int):
        self.completed_item_ids.append(item_id)
        return type("QueueItem", (), {"item_id": item_id, "status": "completed"})()


class _WaitingSession:
    def __init__(self) -> None:
        self.id = "session-id"
        self.prepared_source_mapping = {}
        self._next_calls = 0
        self.waiting_workflow_call = WorkflowCallFrame(
            prepared_call_node_id="prepared-call",
            source_call_node_id="source-call",
            workflow_id="workflow-a",
            depth=1,
        )

    def next(self):
        self._next_calls += 1
        return None

    def is_complete(self) -> bool:
        return False


def test_run_node_propagates_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _build_runner(monkeypatch)
    invocation = KeyboardInterruptInvocation(id="node")
    queue_item = _build_queue_item(invocation)

    with pytest.raises(KeyboardInterrupt):
        runner.run_node(invocation=invocation, queue_item=queue_item)


def test_run_node_does_not_swallow_sigint_in_subprocess() -> None:
    def test_func():
        import os
        import signal
        import threading
        import time
        from contextlib import contextmanager
        from threading import Event

        import invokeai.app.services.session_processor.session_processor_default as session_processor_default
        from invokeai.app.invocations.baseinvocation import (
            BaseInvocation,
            BaseInvocationOutput,
            invocation,
            invocation_output,
        )
        from invokeai.app.services.session_processor.session_processor_default import DefaultSessionRunner

        @invocation_output("test_interrupt_output_subprocess")
        class InterruptTestOutput(BaseInvocationOutput):
            pass

        @invocation("test_sigint_during_node", version="1.0.0")
        class SigIntDuringNodeInvocation(BaseInvocation):
            def invoke(self, context) -> InterruptTestOutput:
                timer = threading.Thread(target=lambda: (time.sleep(0.1), os.kill(os.getpid(), signal.SIGINT)))
                timer.daemon = True
                timer.start()
                time.sleep(5)
                return InterruptTestOutput()

        class DummyStats:
            @contextmanager
            def collect_stats(self, invocation: BaseInvocation, graph_execution_state_id: str):
                yield

        class DummyEvents:
            def emit_invocation_started(self, queue_item, invocation) -> None:
                pass

            def emit_invocation_complete(self, invocation, queue_item, output) -> None:
                pass

            def emit_invocation_error(self, queue_item, invocation, error_type, error_message, error_traceback) -> None:
                pass

        class DummyLogger:
            def debug(self, msg) -> None:
                pass

            def error(self, msg) -> None:
                pass

        class DummyConfig:
            node_cache_size = 0

        session_processor_default.build_invocation_context = lambda data, services, is_canceled: None

        runner = DefaultSessionRunner()
        runner.start(
            services=type(
                "Services",
                (),
                {
                    "performance_statistics": DummyStats(),
                    "events": DummyEvents(),
                    "logger": DummyLogger(),
                    "configuration": DummyConfig(),
                },
            )(),
            cancel_event=Event(),
        )

        invocation = SigIntDuringNodeInvocation(id="node")
        queue_item = type(
            "QueueItem",
            (),
            {
                "item_id": 1,
                "session_id": "test-session",
                "session": type("Session", (), {"prepared_source_mapping": {invocation.id: invocation.id}})(),
            },
        )()

        runner.run_node(invocation=invocation, queue_item=queue_item)
        print("swallowed")

    stdout, stderr, returncode = dangerously_run_function_in_subprocess(test_func)

    assert stdout.strip() == ""
    assert returncode != 0, stderr


def test_on_after_run_session_does_not_complete_incomplete_session(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()

    runner = DefaultSessionRunner()
    runner.start(
        services=type(
            "Services",
            (),
            {
                "performance_statistics": _DummyStats(),
                "events": _DummyEvents(),
                "logger": _DummyLogger(),
                "configuration": _DummyConfig(),
                "session_queue": session_queue,
            },
        )(),
        cancel_event=Event(),
    )

    session = type("Session", (), {"id": "session-id", "is_complete": lambda self: False})()
    queue_item = type(
        "QueueItem",
        (),
        {
            "item_id": 1,
            "status": "in_progress",
            "session": session,
            "session_id": "session-id",
        },
    )()

    runner._on_after_run_session(queue_item=queue_item)

    assert session_queue.session_updates == [(1, session)]
    assert session_queue.completed_item_ids == []


def test_run_persists_waiting_session_without_completing_queue_item(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner = DefaultSessionRunner()
    runner.start(
        services=type(
            "Services",
            (),
            {
                "performance_statistics": _DummyStats(),
                "events": _DummyEvents(),
                "logger": _DummyLogger(),
                "configuration": _DummyConfig(),
                "session_queue": session_queue,
            },
        )(),
        cancel_event=Event(),
    )

    session = _WaitingSession()
    queue_item = type(
        "QueueItem",
        (),
        {
            "item_id": 1,
            "status": "in_progress",
            "session": session,
            "session_id": "session-id",
        },
    )()

    runner.run(queue_item=queue_item)

    assert session._next_calls == 1
    assert session_queue.session_updates == [(1, session)]
    assert session_queue.completed_item_ids == []
