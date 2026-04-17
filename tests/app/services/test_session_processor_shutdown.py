from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace

import pytest

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.invocations.math import AddInvocation
from invokeai.app.services.session_processor.session_processor_default import DefaultSessionRunner
from invokeai.app.services.shared.graph import Graph, GraphExecutionState, WorkflowCallFrame
from invokeai.app.services.workflow_records.workflow_records_common import WorkflowCategory
from tests.dangerously_run_function_in_subprocess import dangerously_run_function_in_subprocess
from tests.test_nodes import create_edge


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
    def __init__(self) -> None:
        self.started: list[tuple[object, object]] = []
        self.completed: list[tuple[object, object, object]] = []
        self.errors: list[tuple[object, object, str, str, str]] = []

    def emit_invocation_started(self, queue_item, invocation) -> None:
        self.started.append((queue_item, invocation))

    def emit_invocation_complete(self, invocation, queue_item, output) -> None:
        self.completed.append((invocation, queue_item, output))

    def emit_invocation_error(self, queue_item, invocation, error_type, error_message, error_traceback) -> None:
        self.errors.append((queue_item, invocation, error_type, error_message, error_traceback))


class _DummyLogger:
    def debug(self, msg) -> None:
        pass

    def error(self, msg) -> None:
        pass


class _DummyConfig:
    node_cache_size = 0
    multiuser = False


class _DummyWorkflowRecords:
    def __init__(self) -> None:
        self.return_invalid_workflow = False
        self.return_batch_special_workflow = False
        self.exposed_field_name = "a"

    def get(self, workflow_id: str):
        workflow = SimpleNamespace(
            name="Child Workflow",
            author="Tester",
            description="",
            version="1.0.0",
            contact="",
            tags="",
            notes="",
            exposedFields=[{"nodeId": "child-add", "fieldName": self.exposed_field_name}],
            meta=SimpleNamespace(category=WorkflowCategory.User),
            form=None,
            nodes=[
                {
                    "id": "child-add",
                    "type": "invocation",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "id": "child-add",
                        "type": "add",
                        "version": "1.0.0",
                        "nodePack": "invokeai",
                        "label": "",
                        "notes": "",
                        "isOpen": True,
                        "isIntermediate": False,
                        "useCache": True,
                        "dynamicInputTemplates": {},
                        "inputs": {
                            "a": {"value": 1},
                            "b": {"value": 2},
                        },
                    },
                }
            ],
            edges=[],
        )
        workflow.model_dump = lambda: {
            "name": workflow.name,
            "author": workflow.author,
            "description": workflow.description,
            "version": workflow.version,
            "contact": workflow.contact,
            "tags": workflow.tags,
            "notes": workflow.notes,
            "exposedFields": workflow.exposedFields,
            "meta": {"category": workflow.meta.category, "version": "1.0.0"},
            "nodes": workflow.nodes,
            "edges": workflow.edges,
            "form": workflow.form,
        }
        if self.return_invalid_workflow:
            workflow.model_dump = lambda: {
                "name": workflow.name,
                "author": workflow.author,
                "description": workflow.description,
                "version": workflow.version,
                "contact": workflow.contact,
                "tags": workflow.tags,
                "notes": workflow.notes,
                "exposedFields": workflow.exposedFields,
                "meta": {"category": workflow.meta.category, "version": "1.0.0"},
                "nodes": workflow.nodes,
                "edges": [
                    {
                        "id": "edge-invalid",
                        "type": "default",
                        "source": "child-add",
                        "sourceHandle": "value",
                        "target": "child-add",
                        "targetHandle": "missing_input",
                    }
                ],
                "form": workflow.form,
            }
        if self.return_batch_special_workflow:
            workflow.model_dump = lambda: {
                "name": workflow.name,
                "author": workflow.author,
                "description": workflow.description,
                "version": workflow.version,
                "contact": workflow.contact,
                "tags": workflow.tags,
                "notes": workflow.notes,
                "exposedFields": workflow.exposedFields,
                "meta": {"category": workflow.meta.category, "version": "1.0.0"},
                "nodes": [
                    {
                        "id": "child-image-batch",
                        "type": "invocation",
                        "position": {"x": 0, "y": 0},
                        "data": {
                            "id": "child-image-batch",
                            "type": "image_batch",
                            "version": "1.0.0",
                            "nodePack": "invokeai",
                            "label": "",
                            "notes": "",
                            "isOpen": True,
                            "isIntermediate": False,
                            "useCache": True,
                            "dynamicInputTemplates": {},
                            "inputs": {
                                "images": {"value": []},
                            },
                        },
                    }
                ],
                "edges": [],
                "form": workflow.form,
            }
        return SimpleNamespace(
            user_id="user-1",
            is_public=False,
            workflow=workflow,
        )


class _DummyUsers:
    def get(self, user_id: str):
        return None


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


def _build_workflow_runner(monkeypatch: pytest.MonkeyPatch, session_queue=None):
    monkeypatch.setattr(
        "invokeai.app.services.session_processor.session_processor_default.build_invocation_context",
        lambda data, services, is_canceled: SimpleNamespace(_services=services, _data=data),
    )

    events = _DummyEvents()
    runner = DefaultSessionRunner()
    workflow_records = _DummyWorkflowRecords()
    runner.start(
        services=type(
            "Services",
            (),
            {
                "performance_statistics": _DummyStats(),
                "events": events,
                "logger": _DummyLogger(),
                "configuration": _DummyConfig(),
                "workflow_records": workflow_records,
                "users": _DummyUsers(),
                "session_queue": session_queue or _DummySessionQueue(),
            },
        )(),
        cancel_event=Event(),
    )
    return runner, events, workflow_records


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
        self.failed_item_ids: list[int] = []

    def set_queue_item_session(self, item_id: int, session):
        self.session_updates.append((item_id, session))
        return type("QueueItem", (), {"item_id": item_id, "status": "in_progress", "session": session})()

    def complete_queue_item(self, item_id: int):
        self.completed_item_ids.append(item_id)
        session = self.session_updates[-1][1]
        return type("QueueItem", (), {"item_id": item_id, "status": "completed", "session": session})()

    def fail_queue_item(self, item_id: int, error_type: str, error_message: str, error_traceback: str):
        self.failed_item_ids.append(item_id)
        session = self.session_updates[-1][1]
        return type(
            "QueueItem",
            (),
            {
                "item_id": item_id,
                "status": "failed",
                "session": session,
                "error_type": error_type,
                "error_message": error_message,
                "error_traceback": error_traceback,
            },
        )()


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


class _WorkflowCallBoundarySession:
    def __init__(self, invocation_id: str) -> None:
        self.id = "session-id"
        self.prepared_source_mapping = {invocation_id: "source-call"}
        self.completed: list[tuple[str, object]] = []
        self.frames: list[WorkflowCallFrame] = []
        self.waiting: WorkflowCallFrame | None = None
        self.waiting_workflow_call_child_session = None
        self.errors: dict[str, str] = {}
        self.execution_graph = Graph()
        self.results = {}

    def build_workflow_call_frame(self, exec_node_id: str, workflow_id: str) -> WorkflowCallFrame:
        frame = WorkflowCallFrame(
            prepared_call_node_id=exec_node_id,
            source_call_node_id=self.prepared_source_mapping[exec_node_id],
            workflow_id=workflow_id,
            depth=1,
        )
        self.frames.append(frame)
        return frame

    def begin_waiting_on_workflow_call(self, frame: WorkflowCallFrame) -> None:
        self.waiting = frame

    def create_child_workflow_execution_state(self, graph: Graph, frame: WorkflowCallFrame):
        return GraphExecutionState(graph=graph, workflow_call_stack=[frame])

    def attach_waiting_workflow_call_child_session(self, child_session: GraphExecutionState) -> None:
        self.waiting_workflow_call_child_session = child_session

    def end_waiting_on_workflow_call(self) -> None:
        self.waiting = None
        self.waiting_workflow_call_child_session = None

    def complete(self, node_id: str, output) -> None:
        self.completed.append((node_id, output))

    def is_waiting_on_workflow_call(self) -> bool:
        return self.waiting is not None

    def set_node_error(self, node_id: str, error: str) -> None:
        self.errors[node_id] = error


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


def test_run_node_executes_child_workflow_and_clears_waiting_state(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch)
    invocation = CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-a")
    session = _WorkflowCallBoundarySession(invocation.id)
    queue_item = type(
        "QueueItem",
        (),
        {
            "item_id": 1,
            "session_id": "test-session",
            "user_id": "user-1",
            "session": session,
        },
    )()

    monkeypatch.setattr(
        CallSavedWorkflowInvocation,
        "invoke_internal",
        lambda self, context, services: (_ for _ in ()).throw(AssertionError("invoke_internal should not be called")),
    )

    runner.run_node(invocation=invocation, queue_item=queue_item)

    assert len(session.frames) == 1
    assert session.waiting is None
    assert session.frames[0].prepared_call_node_id == invocation.id
    assert session.frames[0].workflow_id == "workflow-a"
    assert len(session.completed) == 1
    assert len(events.started) == 2
    assert len(events.completed) == 2
    assert events.errors == []


def test_run_node_fails_cleanly_for_unsupported_batch_special_child_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, events, workflow_records = _build_workflow_runner(monkeypatch)
    workflow_records.return_batch_special_workflow = True
    invocation = CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-a")
    session = _WorkflowCallBoundarySession(invocation.id)
    queue_item = type(
        "QueueItem",
        (),
        {
            "item_id": 1,
            "session_id": "test-session",
            "user_id": "user-1",
            "status": "in_progress",
            "session": session,
        },
    )()

    runner.run_node(invocation=invocation, queue_item=queue_item)

    assert session.waiting is None
    assert session.waiting_workflow_call_child_session is None
    assert session.completed == []
    assert len(events.started) == 1
    assert events.completed == []
    assert len(events.errors) == 1
    _queue_item, _invocation, error_type, error_message, _traceback = events.errors[0]
    assert error_type == "UnsupportedWorkflowNodeError"
    assert "call_saved_workflow does not yet support batch-special" in error_message


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


def test_run_completes_call_saved_workflow_and_runs_downstream_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-a"))
    graph.add_node(AddInvocation(id="downstream-add", b=2))
    graph.add_edge(create_edge("call-node", "value", "downstream-add", "a"))

    session = GraphExecutionState(graph=graph)
    queue_item = type(
        "QueueItem",
        (),
        {
            "item_id": 1,
            "status": "in_progress",
            "session": session,
            "session_id": "session-id",
            "user_id": "user-1",
        },
    )()

    runner.run(queue_item=queue_item)

    assert not session.is_waiting_on_workflow_call()
    assert "downstream-add" in session.executed
    assert len(events.started) == 3
    assert [invocation.get_type() for _queue_item, invocation in events.started] == [
        "call_saved_workflow",
        "add",
        "add",
    ]
    assert len(events.completed) == 3
    assert events.errors == []
    assert session_queue.completed_item_ids == [1]
    assert session_queue.session_updates == [(1, session)]


def test_run_node_records_child_execution_state_for_call_saved_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-a"))
    session = GraphExecutionState(graph=graph)
    invocation = session.next()
    assert invocation is not None

    queue_item = type(
        "QueueItem",
        (),
        {
            "item_id": 1,
            "session_id": "session-id",
            "user_id": "user-1",
            "session": session,
        },
    )()

    runner.run_node(invocation=invocation, queue_item=queue_item)

    assert not session.is_waiting_on_workflow_call()
    assert session.waiting_workflow_call_child_session is None
    assert invocation.id in session.executed
    assert len(events.started) == 2
    assert len(events.completed) == 2
    assert events.errors == []


def test_run_executes_child_workflow_and_completes_parent_queue_item(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-a"))

    session = GraphExecutionState(graph=graph)
    queue_item = type(
        "QueueItem",
        (),
        {
            "item_id": 1,
            "status": "in_progress",
            "session": session,
            "session_id": "session-id",
            "user_id": "user-1",
        },
    )()

    runner.run(queue_item=queue_item)

    assert not session.is_waiting_on_workflow_call()
    assert session_queue.completed_item_ids == [1]
    assert "call-node" in session.executed
    child_add_outputs = [
        output
        for invocation, child_queue_item, output in events.completed
        if invocation.get_type() == "add"
        and child_queue_item.session.prepared_source_mapping[invocation.id] == "child-add"
    ]
    assert len(child_add_outputs) == 1
    assert child_add_outputs[0].value == 3
    parent_outputs = [
        output for invocation, _queue_item, output in events.completed if invocation.get_type() == "call_saved_workflow"
    ]
    assert len(parent_outputs) == 1
    assert parent_outputs[0].value == 0
    assert events.errors == []


def test_run_forwards_literal_dynamic_workflow_inputs_to_child_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)

    graph = Graph()
    graph.add_node(
        CallSavedWorkflowInvocation(
            id="call-node",
            workflow_id="workflow-a",
            workflow_inputs={"saved_workflow_input::child-add::a": 7},
        )
    )

    session = GraphExecutionState(graph=graph)
    queue_item = type(
        "QueueItem",
        (),
        {
            "item_id": 1,
            "status": "in_progress",
            "session": session,
            "session_id": "session-id",
            "user_id": "user-1",
        },
    )()

    runner.run(queue_item=queue_item)

    child_add_outputs = [
        output
        for invocation, child_queue_item, output in events.completed
        if invocation.get_type() == "add"
        and child_queue_item.session.prepared_source_mapping[invocation.id] == "child-add"
    ]
    assert len(child_add_outputs) == 1
    assert child_add_outputs[0].value == 9
    assert session_queue.completed_item_ids == [1]
    assert events.errors == []


def test_run_forwards_connected_dynamic_workflow_inputs_to_child_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)

    graph = Graph()
    graph.add_node(AddInvocation(id="source-add", a=2, b=3))
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-a"))
    graph.add_edge(create_edge("source-add", "value", "call-node", "saved_workflow_input::child-add::a"))

    session = GraphExecutionState(graph=graph)
    queue_item = type(
        "QueueItem",
        (),
        {
            "item_id": 1,
            "status": "in_progress",
            "session": session,
            "session_id": "session-id",
            "user_id": "user-1",
        },
    )()

    runner.run(queue_item=queue_item)

    child_add_outputs = [
        output
        for invocation, child_queue_item, output in events.completed
        if invocation.get_type() == "add"
        and child_queue_item.session.prepared_source_mapping[invocation.id] == "child-add"
    ]
    assert len(child_add_outputs) == 1
    assert child_add_outputs[0].value == 7
    assert session_queue.completed_item_ids == [1]
    assert events.errors == []


def test_run_rejects_non_exposed_dynamic_workflow_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    workflow_records.exposed_field_name = "a"

    graph = Graph()
    graph.add_node(
        CallSavedWorkflowInvocation(
            id="call-node",
            workflow_id="workflow-a",
            workflow_inputs={"saved_workflow_input::child-add::b": 11},
        )
    )

    session = GraphExecutionState(graph=graph)
    queue_item = type(
        "QueueItem",
        (),
        {
            "item_id": 1,
            "status": "in_progress",
            "session": session,
            "session_id": "session-id",
            "user_id": "user-1",
        },
    )()

    runner.run(queue_item=queue_item)

    assert session.has_error()
    assert session_queue.failed_item_ids == [1]
    assert events.completed == []
    assert len(events.errors) == 1
    assert "not exposed" in events.errors[0][3]


def test_run_fails_call_saved_workflow_when_child_workflow_graph_cannot_be_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_queue = _DummySessionQueue()
    runner, events, workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    workflow_records.return_invalid_workflow = True

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-a"))

    session = GraphExecutionState(graph=graph)
    queue_item = type(
        "QueueItem",
        (),
        {
            "item_id": 1,
            "status": "in_progress",
            "session": session,
            "session_id": "session-id",
            "user_id": "user-1",
        },
    )()

    runner.run(queue_item=queue_item)

    assert not session.is_waiting_on_workflow_call()
    assert session.waiting_workflow_call_child_session is None
    assert session.has_error()
    assert session_queue.failed_item_ids == [1]
    assert len(events.started) == 1
    assert events.completed == []
    assert len(events.errors) == 1


def test_run_fails_call_saved_workflow_with_invalid_selection_without_entering_waiting_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id=""))

    session = GraphExecutionState(graph=graph)
    queue_item = type(
        "QueueItem",
        (),
        {
            "item_id": 1,
            "status": "in_progress",
            "session": session,
            "session_id": "session-id",
            "user_id": "user-1",
        },
    )()

    runner.run(queue_item=queue_item)

    assert not session.is_waiting_on_workflow_call()
    assert session.has_error()
    assert session_queue.failed_item_ids == [1]
    assert len(events.started) == 1
    assert events.completed == []
    assert len(events.errors) == 1
    assert events.errors[0][2] == "ValueError"


def test_run_fails_call_saved_workflow_when_depth_limit_is_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-a"))

    session = GraphExecutionState(
        graph=graph,
        workflow_call_stack=[
            WorkflowCallFrame(
                prepared_call_node_id=f"prepared-{i}",
                source_call_node_id=f"source-{i}",
                workflow_id=f"workflow-{i}",
                depth=i + 1,
            )
            for i in range(4)
        ],
    )
    queue_item = type(
        "QueueItem",
        (),
        {
            "item_id": 1,
            "status": "in_progress",
            "session": session,
            "session_id": "session-id",
            "user_id": "user-1",
        },
    )()

    runner.run(queue_item=queue_item)

    assert not session.is_waiting_on_workflow_call()
    assert session.has_error()
    assert session_queue.failed_item_ids == [1]
    assert len(events.started) == 1
    assert events.completed == []
    assert len(events.errors) == 1
    assert events.errors[0][2] == "ValueError"
