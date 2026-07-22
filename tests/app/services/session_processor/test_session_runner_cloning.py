"""Tests for per-worker session-runner cloning in multi-GPU mode.

Each worker needs its own runner instance because start() stores the worker's cancel event on the
runner. Cloning must preserve the SessionRunnerBase contract: a DefaultSessionRunner subclass must
not be silently downgraded to a plain DefaultSessionRunner (losing its overrides), and an unrelated
SessionRunnerBase implementation must not be silently shared across workers (only the last worker's
cancel event would ever fire).
"""

import pytest

from invokeai.app.services.session_processor.session_processor_base import SessionRunnerBase
from invokeai.app.services.session_processor.session_processor_default import (
    DefaultSessionProcessor,
    DefaultSessionRunner,
)


def test_clone_default_runner_carries_callbacks():
    def on_before_session(**kwargs):
        pass

    template = DefaultSessionRunner(on_before_run_session_callbacks=[on_before_session])
    processor = DefaultSessionProcessor(session_runner=template)

    clone = processor._clone_session_runner(template)

    assert type(clone) is DefaultSessionRunner
    assert clone is not template
    assert clone._on_before_run_session_callbacks == [on_before_session]


def test_clone_rejects_default_runner_subclass():
    class CustomizedRunner(DefaultSessionRunner):
        def run(self, queue_item):
            raise NotImplementedError

    template = CustomizedRunner()
    processor = DefaultSessionProcessor(session_runner=template)

    with pytest.raises(ValueError, match="CustomizedRunner"):
        processor._clone_session_runner(template)


def test_clone_rejects_unrelated_runner_implementation():
    class IndependentRunner(SessionRunnerBase):
        # The processor reads this attribute off the runner at construction time.
        workflow_call_queue_lifecycle = None

        def start(self, services, cancel_event, profiler=None):
            self.cancel_event = cancel_event

        def run(self, queue_item):
            pass

        def run_node(self, invocation, queue_item):
            pass

    template = IndependentRunner()
    processor = DefaultSessionProcessor(session_runner=template)

    with pytest.raises(ValueError, match="IndependentRunner"):
        processor._clone_session_runner(template)
