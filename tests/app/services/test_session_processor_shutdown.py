from contextlib import contextmanager
from threading import Event

import pytest

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.services.session_processor.session_processor_default import DefaultSessionRunner
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
