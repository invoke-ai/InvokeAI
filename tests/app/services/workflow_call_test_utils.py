from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.invocations.fields import InputField, OutputField
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.math import AddInvocation
from invokeai.app.services.session_processor.session_processor_default import (
    DefaultSessionProcessor,
    DefaultSessionRunner,
)
from invokeai.app.services.session_processor.workflow_call_runtime import (
    WorkflowCallCoordinator,
    WorkflowCallQueueLifecycle,
)
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


@invocation_output("test_fail_on_integer_output")
class FailOnIntegerOutput(BaseInvocationOutput):
    value: int = OutputField(description="The validated integer")


@invocation("test_fail_on_integer", version="1.0.0")
class FailOnIntegerInvocation(BaseInvocation):
    value: int = InputField(default=0, description="The integer to validate")

    def invoke(self, context) -> FailOnIntegerOutput:
        if self.value == 2:
            raise ValueError("Refusing integer value 2")
        return FailOnIntegerOutput(value=self.value)


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
    max_queue_size = 1000


class _DummyWorkflowRecords:
    def __init__(self) -> None:
        self.return_invalid_workflow = False
        self.return_batch_special_workflow = False
        self.exposed_field_name = "a"

    @staticmethod
    def _invocation_node(node_id: str, invocation_type: str, inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": node_id,
            "type": "invocation",
            "position": {"x": 0, "y": 0},
            "data": {
                "id": node_id,
                "type": invocation_type,
                "version": "1.0.0",
                "nodePack": "invokeai",
                "label": "",
                "notes": "",
                "isOpen": True,
                "isIntermediate": False,
                "useCache": True,
                "dynamicInputTemplates": {},
                "inputs": inputs,
            },
        }

    @classmethod
    def _workflow_dump(
        cls,
        *,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        exposed_fields: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        return {
            "name": "Child Workflow",
            "author": "Tester",
            "description": "",
            "version": "1.0.0",
            "contact": "",
            "tags": "",
            "notes": "",
            "exposedFields": exposed_fields or [],
            "meta": {"category": WorkflowCategory.User, "version": "1.0.0"},
            "nodes": nodes,
            "edges": edges,
            "form": None,
        }

    def get(self, workflow_id: str):
        workflow_dump = self._workflow_dump(
            nodes=[
                self._invocation_node(
                    "child-add",
                    "add",
                    {
                        "a": {"value": 1},
                        "b": {"value": 2},
                    },
                ),
                self._invocation_node(
                    "child-collection",
                    "integer_collection",
                    {"collection": {"value": [3]}},
                ),
                self._invocation_node(
                    "child-return",
                    "workflow_return",
                    {"collection": {"value": []}},
                ),
            ],
            edges=[
                {
                    "id": "edge-default-return",
                    "type": "default",
                    "source": "child-collection",
                    "sourceHandle": "collection",
                    "target": "child-return",
                    "targetHandle": "collection",
                }
            ],
            exposed_fields=[{"nodeId": "child-add", "fieldName": self.exposed_field_name}],
        )
        if self.return_invalid_workflow:
            workflow_dump = {
                **workflow_dump,
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
            }
        if self.return_batch_special_workflow:
            workflow_dump = {
                **workflow_dump,
                "nodes": [
                    self._invocation_node(
                        "child-image-batch",
                        "image_batch",
                        {
                            "images": {"value": []},
                        },
                    )
                ],
                "edges": [],
            }
        if workflow_id == "workflow-dependent":
            workflow_dump = self._workflow_dump(
                nodes=[
                    self._invocation_node("child-add-1", "add", {"a": {"value": 1}, "b": {"value": 2}}),
                    self._invocation_node("child-add-2", "add", {"a": {"value": 0}, "b": {"value": 4}}),
                    self._invocation_node(
                        "child-collection",
                        "integer_collection",
                        {"collection": {"value": [7]}},
                    ),
                    self._invocation_node(
                        "child-return",
                        "workflow_return",
                        {"collection": {"value": []}},
                    ),
                ],
                edges=[
                    {
                        "id": "edge-dependent",
                        "type": "default",
                        "source": "child-add-1",
                        "sourceHandle": "value",
                        "target": "child-add-2",
                        "targetHandle": "a",
                    },
                    {
                        "id": "edge-dependent-return",
                        "type": "default",
                        "source": "child-collection",
                        "sourceHandle": "collection",
                        "target": "child-return",
                        "targetHandle": "collection",
                    },
                ],
            )
        elif workflow_id == "workflow-if":
            workflow_dump = self._workflow_dump(
                nodes=[
                    self._invocation_node("child-bool", "boolean", {"value": {"value": True}}),
                    self._invocation_node("child-add", "add", {"a": {"value": 2}, "b": {"value": 3}}),
                    self._invocation_node(
                        "child-collection",
                        "integer_collection",
                        {"collection": {"value": [5]}},
                    ),
                    self._invocation_node(
                        "child-if",
                        "if",
                        {
                            "condition": {"value": False},
                            "true_input": {"value": None},
                            "false_input": {"value": 11},
                        },
                    ),
                    self._invocation_node(
                        "child-return",
                        "workflow_return",
                        {"collection": {"value": []}},
                    ),
                ],
                edges=[
                    {
                        "id": "edge-if-condition",
                        "type": "default",
                        "source": "child-bool",
                        "sourceHandle": "value",
                        "target": "child-if",
                        "targetHandle": "condition",
                    },
                    {
                        "id": "edge-if-true",
                        "type": "default",
                        "source": "child-add",
                        "sourceHandle": "value",
                        "target": "child-if",
                        "targetHandle": "true_input",
                    },
                    {
                        "id": "edge-if-return",
                        "type": "default",
                        "source": "child-collection",
                        "sourceHandle": "collection",
                        "target": "child-return",
                        "targetHandle": "collection",
                    },
                ],
            )
        elif workflow_id == "workflow-nested":
            workflow_dump = self._workflow_dump(
                nodes=[
                    self._invocation_node(
                        "nested-call",
                        "call_saved_workflow",
                        {
                            "workflow_id": {"value": "workflow-leaf"},
                            "workflow_inputs": {"value": {}},
                        },
                    ),
                    self._invocation_node("nested-add", "add", {"a": {"value": 0}, "b": {"value": 4}}),
                    self._invocation_node(
                        "nested-collection",
                        "integer_collection",
                        {"collection": {"value": [4]}},
                    ),
                    self._invocation_node(
                        "nested-return",
                        "workflow_return",
                        {"collection": {"value": []}},
                    ),
                ],
                edges=[
                    {
                        "id": "edge-nested-return",
                        "type": "default",
                        "source": "nested-collection",
                        "sourceHandle": "collection",
                        "target": "nested-return",
                        "targetHandle": "collection",
                    }
                ],
            )
        elif workflow_id == "workflow-leaf":
            workflow_dump = self._workflow_dump(
                nodes=[
                    self._invocation_node("leaf-add", "add", {"a": {"value": 5}, "b": {"value": 6}}),
                    self._invocation_node(
                        "leaf-collection",
                        "integer_collection",
                        {"collection": {"value": [11]}},
                    ),
                    self._invocation_node(
                        "leaf-return",
                        "workflow_return",
                        {"collection": {"value": []}},
                    ),
                ],
                edges=[
                    {
                        "id": "edge-leaf-return",
                        "type": "default",
                        "source": "leaf-collection",
                        "sourceHandle": "collection",
                        "target": "leaf-return",
                        "targetHandle": "collection",
                    }
                ],
            )
        elif workflow_id == "workflow-nested-no-return":
            workflow_dump = self._workflow_dump(
                nodes=[
                    self._invocation_node(
                        "nested-call",
                        "call_saved_workflow",
                        {
                            "workflow_id": {"value": "workflow-no-return"},
                            "workflow_inputs": {"value": {}},
                        },
                    ),
                    self._invocation_node(
                        "nested-collection",
                        "integer_collection",
                        {"collection": {"value": [4]}},
                    ),
                    self._invocation_node(
                        "nested-return",
                        "workflow_return",
                        {"collection": {"value": []}},
                    ),
                ],
                edges=[
                    {
                        "id": "edge-nested-return",
                        "type": "default",
                        "source": "nested-collection",
                        "sourceHandle": "collection",
                        "target": "nested-return",
                        "targetHandle": "collection",
                    }
                ],
            )
        elif workflow_id == "workflow-return":
            workflow_dump = self._workflow_dump(
                nodes=[
                    self._invocation_node(
                        "child-collection",
                        "integer_collection",
                        {"collection": {"value": [7, 8]}},
                    ),
                    self._invocation_node(
                        "child-return",
                        "workflow_return",
                        {"collection": {"value": []}},
                    ),
                ],
                edges=[
                    {
                        "id": "edge-return-collection",
                        "type": "default",
                        "source": "child-collection",
                        "sourceHandle": "collection",
                        "target": "child-return",
                        "targetHandle": "collection",
                    }
                ],
            )
        elif workflow_id == "workflow-no-return":
            workflow_dump = self._workflow_dump(
                nodes=[
                    self._invocation_node(
                        "child-add",
                        "add",
                        {
                            "a": {"value": 1},
                            "b": {"value": 2},
                        },
                    )
                ],
                edges=[],
                exposed_fields=[{"nodeId": "child-add", "fieldName": self.exposed_field_name}],
            )
        elif workflow_id == "workflow-batch-direct":
            workflow_dump = self._workflow_dump(
                nodes=[
                    self._invocation_node(
                        "child-batch",
                        "integer_batch",
                        {
                            "integers": {"value": [2, 4, 6]},
                            "batch_group_id": {"value": "None"},
                        },
                    ),
                    self._invocation_node("child-int", "integer", {"value": {"value": 0}}),
                    self._invocation_node("child-collect", "collect", {"collection": {"value": []}}),
                    self._invocation_node("child-return", "workflow_return", {"collection": {"value": []}}),
                ],
                edges=[
                    {
                        "id": "edge-batch-int",
                        "type": "default",
                        "source": "child-batch",
                        "sourceHandle": "integers",
                        "target": "child-int",
                        "targetHandle": "value",
                    },
                    {
                        "id": "edge-int-collect",
                        "type": "default",
                        "source": "child-int",
                        "sourceHandle": "value",
                        "target": "child-collect",
                        "targetHandle": "item",
                    },
                    {
                        "id": "edge-collect-return",
                        "type": "default",
                        "source": "child-collect",
                        "sourceHandle": "collection",
                        "target": "child-return",
                        "targetHandle": "collection",
                    },
                ],
            )
        elif workflow_id == "workflow-batch-grouped":
            workflow_dump = self._workflow_dump(
                nodes=[
                    self._invocation_node(
                        "child-batch-a",
                        "integer_batch",
                        {
                            "integers": {"value": [1, 2, 3]},
                            "batch_group_id": {"value": "Group 1"},
                        },
                    ),
                    self._invocation_node(
                        "child-batch-b",
                        "integer_batch",
                        {
                            "integers": {"value": [10, 20, 30]},
                            "batch_group_id": {"value": "Group 1"},
                        },
                    ),
                    self._invocation_node("child-add", "add", {"a": {"value": 0}, "b": {"value": 0}}),
                    self._invocation_node("child-collect", "collect", {"collection": {"value": []}}),
                    self._invocation_node("child-return", "workflow_return", {"collection": {"value": []}}),
                ],
                edges=[
                    {
                        "id": "edge-group-a",
                        "type": "default",
                        "source": "child-batch-a",
                        "sourceHandle": "integers",
                        "target": "child-add",
                        "targetHandle": "a",
                    },
                    {
                        "id": "edge-group-b",
                        "type": "default",
                        "source": "child-batch-b",
                        "sourceHandle": "integers",
                        "target": "child-add",
                        "targetHandle": "b",
                    },
                    {
                        "id": "edge-group-collect",
                        "type": "default",
                        "source": "child-add",
                        "sourceHandle": "value",
                        "target": "child-collect",
                        "targetHandle": "item",
                    },
                    {
                        "id": "edge-group-return",
                        "type": "default",
                        "source": "child-collect",
                        "sourceHandle": "collection",
                        "target": "child-return",
                        "targetHandle": "collection",
                    },
                ],
            )
        elif workflow_id == "workflow-batch-cartesian":
            workflow_dump = self._workflow_dump(
                nodes=[
                    self._invocation_node(
                        "child-batch-a",
                        "integer_batch",
                        {
                            "integers": {"value": [1, 2]},
                            "batch_group_id": {"value": "None"},
                        },
                    ),
                    self._invocation_node(
                        "child-batch-b",
                        "integer_batch",
                        {
                            "integers": {"value": [10, 20]},
                            "batch_group_id": {"value": "None"},
                        },
                    ),
                    self._invocation_node("child-add", "add", {"a": {"value": 0}, "b": {"value": 0}}),
                    self._invocation_node("child-collect", "collect", {"collection": {"value": []}}),
                    self._invocation_node("child-return", "workflow_return", {"collection": {"value": []}}),
                ],
                edges=[
                    {
                        "id": "edge-cart-a",
                        "type": "default",
                        "source": "child-batch-a",
                        "sourceHandle": "integers",
                        "target": "child-add",
                        "targetHandle": "a",
                    },
                    {
                        "id": "edge-cart-b",
                        "type": "default",
                        "source": "child-batch-b",
                        "sourceHandle": "integers",
                        "target": "child-add",
                        "targetHandle": "b",
                    },
                    {
                        "id": "edge-cart-collect",
                        "type": "default",
                        "source": "child-add",
                        "sourceHandle": "value",
                        "target": "child-collect",
                        "targetHandle": "item",
                    },
                    {
                        "id": "edge-cart-return",
                        "type": "default",
                        "source": "child-collect",
                        "sourceHandle": "collection",
                        "target": "child-return",
                        "targetHandle": "collection",
                    },
                ],
            )
        elif workflow_id == "workflow-batch-failure":
            workflow_dump = self._workflow_dump(
                nodes=[
                    self._invocation_node(
                        "child-batch",
                        "integer_batch",
                        {
                            "integers": {"value": [1, 2, 3]},
                            "batch_group_id": {"value": "None"},
                        },
                    ),
                    self._invocation_node("child-guard", "test_fail_on_integer", {"value": {"value": 0}}),
                    self._invocation_node("child-collect", "collect", {"collection": {"value": []}}),
                    self._invocation_node("child-return", "workflow_return", {"collection": {"value": []}}),
                ],
                edges=[
                    {
                        "id": "edge-failure-value",
                        "type": "default",
                        "source": "child-batch",
                        "sourceHandle": "integers",
                        "target": "child-guard",
                        "targetHandle": "value",
                    },
                    {
                        "id": "edge-failure-collect",
                        "type": "default",
                        "source": "child-guard",
                        "sourceHandle": "value",
                        "target": "child-collect",
                        "targetHandle": "item",
                    },
                    {
                        "id": "edge-failure-return",
                        "type": "default",
                        "source": "child-collect",
                        "sourceHandle": "collection",
                        "target": "child-return",
                        "targetHandle": "collection",
                    },
                ],
            )
        elif workflow_id == "workflow-batch-generator":
            workflow_dump = self._workflow_dump(
                nodes=[
                    self._invocation_node("child-int", "integer", {"value": {"value": 7}}),
                    self._invocation_node(
                        "child-batch",
                        "integer_batch",
                        {
                            "integers": {"value": []},
                            "batch_group_id": {"value": "None"},
                        },
                    ),
                    self._invocation_node("child-int", "integer", {"value": {"value": 0}}),
                    self._invocation_node("child-collect", "collect", {"collection": {"value": []}}),
                    self._invocation_node("child-return", "workflow_return", {"collection": {"value": []}}),
                ],
                edges=[
                    {
                        "id": "edge-int-batch",
                        "type": "default",
                        "source": "child-int",
                        "sourceHandle": "value",
                        "target": "child-batch",
                        "targetHandle": "integers",
                    },
                    {
                        "id": "edge-generator-value",
                        "type": "default",
                        "source": "child-batch",
                        "sourceHandle": "integers",
                        "target": "child-int",
                        "targetHandle": "value",
                    },
                    {
                        "id": "edge-generator-collect",
                        "type": "default",
                        "source": "child-int",
                        "sourceHandle": "value",
                        "target": "child-collect",
                        "targetHandle": "item",
                    },
                    {
                        "id": "edge-generator-return",
                        "type": "default",
                        "source": "child-collect",
                        "sourceHandle": "collection",
                        "target": "child-return",
                        "targetHandle": "collection",
                    },
                ],
            )
        elif workflow_id == "workflow-batch-generator-integer":
            workflow_dump = self._workflow_dump(
                nodes=[
                    self._invocation_node(
                        "child-generator",
                        "integer_generator",
                        {
                            "generator": {
                                "value": {
                                    "type": "integer_generator_arithmetic_sequence",
                                    "start": 2,
                                    "step": 2,
                                    "count": 3,
                                }
                            }
                        },
                    ),
                    self._invocation_node(
                        "child-batch",
                        "integer_batch",
                        {
                            "integers": {"value": []},
                            "batch_group_id": {"value": "None"},
                        },
                    ),
                    self._invocation_node("child-int", "integer", {"value": {"value": 0}}),
                    self._invocation_node("child-collect", "collect", {"collection": {"value": []}}),
                    self._invocation_node("child-return", "workflow_return", {"collection": {"value": []}}),
                ],
                edges=[
                    {
                        "id": "edge-generator-batch",
                        "type": "default",
                        "source": "child-generator",
                        "sourceHandle": "integers",
                        "target": "child-batch",
                        "targetHandle": "integers",
                    },
                    {
                        "id": "edge-generator-value",
                        "type": "default",
                        "source": "child-batch",
                        "sourceHandle": "integers",
                        "target": "child-int",
                        "targetHandle": "value",
                    },
                    {
                        "id": "edge-generator-collect",
                        "type": "default",
                        "source": "child-int",
                        "sourceHandle": "value",
                        "target": "child-collect",
                        "targetHandle": "item",
                    },
                    {
                        "id": "edge-generator-return",
                        "type": "default",
                        "source": "child-collect",
                        "sourceHandle": "collection",
                        "target": "child-return",
                        "targetHandle": "collection",
                    },
                ],
            )
        elif workflow_id == "workflow-batch-generator-image":
            workflow_dump = self._workflow_dump(
                nodes=[
                    self._invocation_node(
                        "child-generator",
                        "image_generator",
                        {
                            "generator": {
                                "value": {
                                    "type": "image_generator_images_from_board",
                                    "board_id": "board-1",
                                    "category": "images",
                                }
                            }
                        },
                    ),
                    self._invocation_node(
                        "child-batch",
                        "image_batch",
                        {
                            "images": {"value": []},
                            "batch_group_id": {"value": "None"},
                        },
                    ),
                    self._invocation_node("child-image", "image", {"image": {"value": None}}),
                    self._invocation_node("child-collect", "collect", {"collection": {"value": []}}),
                    self._invocation_node("child-return", "workflow_return", {"collection": {"value": []}}),
                ],
                edges=[
                    {
                        "id": "edge-generator-batch",
                        "type": "default",
                        "source": "child-generator",
                        "sourceHandle": "images",
                        "target": "child-batch",
                        "targetHandle": "images",
                    },
                    {
                        "id": "edge-generator-value",
                        "type": "default",
                        "source": "child-batch",
                        "sourceHandle": "images",
                        "target": "child-image",
                        "targetHandle": "image",
                    },
                    {
                        "id": "edge-generator-collect",
                        "type": "default",
                        "source": "child-image",
                        "sourceHandle": "image",
                        "target": "child-collect",
                        "targetHandle": "item",
                    },
                    {
                        "id": "edge-generator-return",
                        "type": "default",
                        "source": "child-collect",
                        "sourceHandle": "collection",
                        "target": "child-return",
                        "targetHandle": "collection",
                    },
                ],
            )

        workflow = SimpleNamespace(
            name="Child Workflow",
            author="Tester",
            description="",
            version="1.0.0",
            contact="",
            tags="",
            notes="",
            exposedFields=workflow_dump["exposedFields"],
            meta=SimpleNamespace(category=WorkflowCategory.User),
            form=workflow_dump["form"],
            nodes=workflow_dump["nodes"],
            edges=workflow_dump["edges"],
        )
        workflow.model_dump = lambda: workflow_dump
        return SimpleNamespace(
            user_id="user-1",
            is_public=False,
            workflow=workflow,
        )


class _DummyUsers:
    def get(self, user_id: str):
        return None


class _DummyBoardImages:
    def get_all_board_image_names_for_board(self, board_id: str, categories, is_intermediate: bool | None):
        if board_id == "board-1":
            return ["img-a", "img-b"]
        return []


class _DummyImages:
    def get_dto(self, image_name: str):
        return SimpleNamespace(image_name=image_name, width=64, height=64)


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
        lambda data, services, is_canceled: SimpleNamespace(
            _services=services,
            _data=data,
            images=SimpleNamespace(get_dto=services.images.get_dto),
            boards=SimpleNamespace(
                get_all_image_names_for_board=services.board_images.get_all_board_image_names_for_board
            ),
        ),
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
                "board_images": _DummyBoardImages(),
                "images": _DummyImages(),
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
        self.items: dict[int, object] = {}
        self.next_item_id = 100
        self.completed_item_ids: list[int] = []
        self.session_updates: list[tuple[int, object]] = []
        self.failed_item_ids: list[int] = []
        self.waiting_item_ids: list[int] = []
        self.resumed_item_ids: list[int] = []
        self.enqueued_child_item_ids: list[int] = []
        self.canceled_item_ids: list[int] = []
        self.deleted_item_ids: list[int] = []
        self.pending_count = 0
        self.fail_enqueue_after: int | None = None

    def add_queue_item(self, queue_item):
        self.items[queue_item.item_id] = queue_item
        return queue_item

    def _ensure_queue_item(self, item_id: int, session):
        queue_item = self.items.get(item_id)
        if queue_item is None:
            queue_item = SimpleNamespace(
                item_id=item_id,
                status="in_progress",
                session=session,
                session_id=getattr(session, "id", f"session-{item_id}"),
                user_id="user-1",
                queue_id="default",
                batch_id="batch-1",
                parent_item_id=None,
                parent_session_id=None,
                workflow_call_id=None,
                root_item_id=None,
                workflow_call_depth=None,
            )
            self.items[item_id] = queue_item
        return queue_item

    def get_queue_item(self, item_id: int):
        return self.items[item_id]

    def set_queue_item_session(self, item_id: int, session):
        queue_item = self._ensure_queue_item(item_id, session)
        queue_item.session = session
        self.session_updates.append((item_id, session))
        return queue_item

    def suspend_queue_item(self, item_id: int):
        queue_item = self._ensure_queue_item(item_id, None)
        queue_item.status = "waiting"
        self.waiting_item_ids.append(item_id)
        return queue_item

    def resume_queue_item(self, item_id: int):
        queue_item = self._ensure_queue_item(item_id, None)
        queue_item.status = "pending"
        self.resumed_item_ids.append(item_id)
        return queue_item

    def complete_queue_item(self, item_id: int):
        queue_item = self._ensure_queue_item(item_id, None)
        queue_item.status = "completed"
        self.completed_item_ids.append(item_id)
        return queue_item

    def fail_queue_item(self, item_id: int, error_type: str, error_message: str, error_traceback: str):
        queue_item = self._ensure_queue_item(item_id, None)
        queue_item.status = "failed"
        queue_item.error_type = error_type
        queue_item.error_message = error_message
        queue_item.error_traceback = error_traceback
        self.failed_item_ids.append(item_id)
        return queue_item

    def cancel_workflow_call_children(
        self, workflow_call_id: str, exclude_item_ids: set[int] | None = None
    ) -> list[int]:
        exclude_item_ids = exclude_item_ids or set()
        canceled_item_ids: list[int] = []
        for item_id, queue_item in self.items.items():
            if item_id in exclude_item_ids:
                continue
            if getattr(queue_item, "workflow_call_id", None) != workflow_call_id:
                continue
            if queue_item.status in {"completed", "failed", "canceled"}:
                continue
            queue_item.status = "canceled"
            canceled_item_ids.append(item_id)
            self.canceled_item_ids.append(item_id)
        return canceled_item_ids

    def enqueue_workflow_call_child(self, parent_queue_item, child_session, field_values=None):
        if self.fail_enqueue_after is not None and len(self.enqueued_child_item_ids) >= self.fail_enqueue_after:
            raise RuntimeError("Injected child enqueue failure")
        workflow_call_execution = parent_queue_item.session.waiting_workflow_call_execution
        item_id = self.next_item_id
        self.next_item_id += 1
        child_queue_item = type(
            "QueueItem",
            (),
            {
                "item_id": item_id,
                "status": "pending",
                "session": child_session,
                "session_id": child_session.id,
                "user_id": getattr(parent_queue_item, "user_id", "user-1"),
                "queue_id": getattr(parent_queue_item, "queue_id", "default"),
                "batch_id": getattr(parent_queue_item, "batch_id", "batch-1"),
                "origin": getattr(parent_queue_item, "origin", None),
                "destination": getattr(parent_queue_item, "destination", None),
                "priority": getattr(parent_queue_item, "priority", 0),
                "field_values": field_values,
                "workflow_call_id": workflow_call_execution.id if workflow_call_execution is not None else None,
                "parent_item_id": parent_queue_item.item_id,
                "parent_session_id": parent_queue_item.session_id,
                "root_item_id": getattr(parent_queue_item, "root_item_id", None) or parent_queue_item.item_id,
                "workflow_call_depth": (workflow_call_execution.depth if workflow_call_execution is not None else None),
                "workflow": None,
            },
        )()
        self.add_queue_item(child_queue_item)
        self.enqueued_child_item_ids.append(item_id)
        return child_queue_item

    def delete_queue_items_by_id(self, item_ids: list[int]):
        for item_id in item_ids:
            if item_id in self.items:
                del self.items[item_id]
                self.deleted_item_ids.append(item_id)

    def get_queue_status(self, queue_id: str):
        return SimpleNamespace(pending=self.pending_count)

    def dequeue(self):
        pending_item_ids = sorted(item_id for item_id, item in self.items.items() if item.status == "pending")
        if not pending_item_ids:
            return None
        queue_item = self.items[pending_item_ids[0]]
        queue_item.status = "in_progress"
        return queue_item


def _drain_workflow_call_queue(
    lifecycle: WorkflowCallQueueLifecycle, session_queue: _DummySessionQueue, queue_item
) -> None:
    session_queue.add_queue_item(queue_item)
    lifecycle.run_queue_item(queue_item)
    while True:
        next_queue_item = session_queue.dequeue()
        if next_queue_item is None:
            return
        lifecycle.run_queue_item(next_queue_item)


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
        self.waiting_workflow_call_execution = None
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
        self.waiting_workflow_call_execution = SimpleNamespace(id="workflow-call-1", depth=1)
        self.waiting_workflow_call_child_session = child_session

    def attach_waiting_workflow_call_child_sessions(self, child_sessions: list[GraphExecutionState]) -> None:
        self.waiting_workflow_call_execution = SimpleNamespace(id="workflow-call-1", depth=1)
        self.waiting_workflow_call_child_session = child_sessions[0] if len(child_sessions) == 1 else None

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
    session_queue.add_queue_item(queue_item)

    runner._on_after_run_session(queue_item=queue_item)

    assert session_queue.session_updates == [(1, session)]
    assert session_queue.completed_item_ids == []


def test_run_node_enters_waiting_state_without_executing_child_inline(monkeypatch: pytest.MonkeyPatch) -> None:
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
            "status": "in_progress",
            "session": session,
            "queue_id": "default",
            "batch_id": "batch-1",
            "priority": 0,
            "origin": None,
            "destination": None,
            "root_item_id": None,
        },
    )()

    monkeypatch.setattr(
        CallSavedWorkflowInvocation,
        "invoke_internal",
        lambda self, context, services: (_ for _ in ()).throw(AssertionError("invoke_internal should not be called")),
    )

    runner.run_node(invocation=invocation, queue_item=queue_item)

    assert len(session.frames) == 1
    assert session.waiting == session.frames[0]
    assert session.frames[0].prepared_call_node_id == invocation.id
    assert session.frames[0].workflow_id == "workflow-a"
    assert session.waiting_workflow_call_child_session is not None
    assert session.completed == []
    assert len(events.started) == 1
    assert events.completed == []
    assert events.errors == []


def test_run_node_fails_cleanly_for_invalid_batch_child_workflow(
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
            "queue_id": "default",
            "batch_id": "batch-1",
            "priority": 0,
            "origin": None,
            "destination": None,
            "root_item_id": None,
        },
    )()
    runner._services.session_queue.add_queue_item(queue_item)

    runner.run_node(invocation=invocation, queue_item=queue_item)

    assert session.waiting is None
    assert session.waiting_workflow_call_child_session is None
    assert session.completed == []
    assert len(events.started) == 1
    assert events.completed == []
    assert len(events.errors) == 1
    _queue_item, _invocation, error_type, error_message, _traceback = events.errors[0]
    assert error_type == "UnsupportedWorkflowNodeError"
    assert "must provide at least one batch item" in error_message


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
    session_queue.add_queue_item(queue_item)

    runner.run(queue_item=queue_item)

    assert session._next_calls == 1
    assert session_queue.session_updates == [(1, session)]
    assert session_queue.completed_item_ids == []


def test_workflow_call_coordinator_suspends_parent_and_enqueues_child_queue_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-a"))
    graph.add_node(IfInvocation(id="downstream-if", condition=True, false_input=0))
    graph.add_edge(create_edge("call-node", "collection", "downstream-if", "true_input"))

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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    session_queue.add_queue_item(queue_item)
    runner.run(queue_item)

    assert session.is_waiting_on_workflow_call()
    assert session_queue.waiting_item_ids == [1]
    assert session_queue.enqueued_child_item_ids == [100]
    child_queue_item = session_queue.get_queue_item(100)
    assert child_queue_item.status == "pending"
    assert child_queue_item.parent_item_id == queue_item.item_id
    assert child_queue_item.workflow_call_id == session.waiting_workflow_call_execution.id
    assert "downstream-if" not in session.executed
    assert events.completed == []
    child_started_queue_items = [
        child_queue_item
        for child_queue_item, invocation in events.started
        if invocation.get_type() != "call_saved_workflow" and child_queue_item.session_id != queue_item.session_id
    ]
    assert child_started_queue_items == []
    assert session.workflow_call_history == []
    assert events.errors == []


def test_workflow_call_queue_lifecycle_leaves_non_call_workflows_on_normal_execution_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    lifecycle = WorkflowCallQueueLifecycle(runner)

    graph = Graph()
    graph.add_node(AddInvocation(id="source-add", a=2, b=3))
    graph.add_node(IfInvocation(id="downstream-if", condition=True, false_input=0))
    graph.add_edge(create_edge("source-add", "value", "downstream-if", "true_input"))

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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    session_queue.add_queue_item(queue_item)
    lifecycle.run_queue_item(queue_item)

    assert not session.is_waiting_on_workflow_call()
    assert "source-add" in session.executed
    assert "downstream-if" in session.executed
    assert session_queue.enqueued_child_item_ids == []
    assert session_queue.waiting_item_ids == []
    assert session_queue.resumed_item_ids == []
    assert session_queue.completed_item_ids == [1]
    assert [invocation.get_type() for _queue_item, invocation in events.started] == ["add", "if"]
    assert [invocation.get_type() for invocation, _queue_item, _output in events.completed] == ["add", "if"]
    assert events.errors == []


def test_workflow_call_queue_lifecycle_resumes_parent_from_completed_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    processor = DefaultSessionProcessor(session_runner=runner)

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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    _drain_workflow_call_queue(processor.session_runner.workflow_call_queue_lifecycle, session_queue, queue_item)

    assert session_queue.completed_item_ids == [100, 1]
    assert session_queue.resumed_item_ids == []
    parent_outputs = [
        output for invocation, _queue_item, output in events.completed if invocation.get_type() == "call_saved_workflow"
    ]
    assert len(parent_outputs) == 1
    assert parent_outputs[0].collection == [3]


def test_workflow_call_coordinator_cleans_up_enqueued_children_when_boundary_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_queue = _DummySessionQueue()
    session_queue.fail_enqueue_after = 1
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    coordinator = WorkflowCallCoordinator(runner)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-batch-direct"))
    session = GraphExecutionState(graph=graph)
    invocation = session.next()
    assert isinstance(invocation, CallSavedWorkflowInvocation)
    queue_item = SimpleNamespace(
        item_id=1,
        status="in_progress",
        session=session,
        session_id="session-id",
        user_id="user-1",
        queue_id="default",
        batch_id="batch-1",
        priority=0,
        origin=None,
        destination=None,
        root_item_id=None,
    )
    session_queue.add_queue_item(queue_item)
    workflow_record = runner._services.workflow_records.get(invocation.workflow_id)

    with pytest.raises(RuntimeError, match="Injected child enqueue failure"):
        coordinator.begin_workflow_call_boundary(invocation, queue_item, workflow_record)

    assert session_queue.enqueued_child_item_ids == [100]
    assert session_queue.deleted_item_ids == [100]
    assert session_queue.items[1].status == "in_progress"
    assert session.waiting_workflow_call is None
    assert session.waiting_workflow_call_execution is None
    assert session.waiting_workflow_call_child_session is None
    assert events.completed == []
    assert events.errors == []


def test_workflow_call_coordinator_rejects_child_expansion_that_exceeds_remaining_queue_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_queue = _DummySessionQueue()
    session_queue.pending_count = 999
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    runner._services.configuration.max_queue_size = 1000
    coordinator = WorkflowCallCoordinator(runner)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-batch-direct"))
    session = GraphExecutionState(graph=graph)
    invocation = session.next()
    assert isinstance(invocation, CallSavedWorkflowInvocation)
    queue_item = SimpleNamespace(
        item_id=1,
        status="in_progress",
        session=session,
        session_id="session-id",
        user_id="user-1",
        queue_id="default",
        batch_id="batch-1",
        priority=0,
        origin=None,
        destination=None,
        root_item_id=None,
    )
    session_queue.add_queue_item(queue_item)
    workflow_record = runner._services.workflow_records.get(invocation.workflow_id)

    with pytest.raises(ValueError, match="remaining queue capacity"):
        coordinator.begin_workflow_call_boundary(invocation, queue_item, workflow_record)

    assert session_queue.enqueued_child_item_ids == []
    assert session_queue.waiting_item_ids == []
    assert session.waiting_workflow_call is None
    assert session.waiting_workflow_call_execution is None
    assert events.completed == []
    assert events.errors == []


def test_workflow_call_coordinator_builds_child_queue_item_with_relationship_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_queue = _DummySessionQueue()
    runner, _events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    coordinator = WorkflowCallCoordinator(runner)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-a"))
    parent_session = GraphExecutionState(graph=graph)
    invocation = parent_session.next()
    assert isinstance(invocation, CallSavedWorkflowInvocation)

    queue_item = type(
        "QueueItem",
        (),
        {
            "item_id": 42,
            "status": "in_progress",
            "session": parent_session,
            "session_id": parent_session.id,
            "user_id": "user-1",
            "queue_id": "default",
            "batch_id": "batch-1",
            "priority": 0,
            "origin": None,
            "destination": None,
            "workflow_call_id": None,
            "parent_item_id": None,
            "parent_session_id": None,
            "root_item_id": None,
            "workflow_call_depth": None,
        },
    )()

    workflow_record = runner._services.workflow_records.get(invocation.workflow_id)
    coordinator.begin_workflow_call_boundary(invocation, queue_item, workflow_record)
    child_session = parent_session.waiting_workflow_call_child_session
    assert child_session is not None

    child_queue_item = coordinator.build_child_queue_item(queue_item, child_session)

    assert child_queue_item.workflow_call_id == parent_session.waiting_workflow_call_execution.id
    assert child_queue_item.parent_item_id == queue_item.item_id
    assert child_queue_item.parent_session_id == queue_item.session_id
    assert child_queue_item.root_item_id == queue_item.item_id
    assert child_queue_item.workflow_call_depth == 1
    assert child_queue_item.session_id == child_session.id


def test_run_completes_call_saved_workflow_and_runs_downstream_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    processor = DefaultSessionProcessor(session_runner=runner)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-a"))
    graph.add_node(IfInvocation(id="downstream-if", condition=True, false_input=0))
    graph.add_edge(create_edge("call-node", "collection", "downstream-if", "true_input"))

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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    _drain_workflow_call_queue(processor.session_runner.workflow_call_queue_lifecycle, session_queue, queue_item)

    assert not session.is_waiting_on_workflow_call()
    assert "downstream-if" in session.executed
    assert len(events.started) == 5
    assert [invocation.get_type() for _queue_item, invocation in events.started] == [
        "call_saved_workflow",
        "add",
        "integer_collection",
        "workflow_return",
        "if",
    ]
    assert len(events.completed) == 5
    parent_outputs = [
        output for invocation, _queue_item, output in events.completed if invocation.get_type() == "call_saved_workflow"
    ]
    downstream_outputs = [
        output for invocation, _queue_item, output in events.completed if invocation.get_type() == "if"
    ]
    assert len(parent_outputs) == 1
    assert parent_outputs[0].collection == [3]
    assert len(downstream_outputs) == 1
    assert downstream_outputs[0].value == [3]
    assert events.errors == []
    assert session_queue.completed_item_ids == [100, 1]
    assert session_queue.waiting_item_ids == [1]
    assert session_queue.resumed_item_ids == [1]


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
            "status": "in_progress",
            "session": session,
            "queue_id": "default",
            "batch_id": "batch-1",
            "priority": 0,
            "origin": None,
            "destination": None,
            "root_item_id": None,
        },
    )()

    runner.run_node(invocation=invocation, queue_item=queue_item)

    assert session.is_waiting_on_workflow_call()
    assert session.waiting_workflow_call_child_session is not None
    assert invocation.id not in session.executed
    assert len(events.started) == 1
    assert events.completed == []
    assert events.errors == []


def test_run_executes_child_workflow_and_completes_parent_queue_item(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    processor = DefaultSessionProcessor(session_runner=runner)

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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    _drain_workflow_call_queue(processor.session_runner.workflow_call_queue_lifecycle, session_queue, queue_item)

    assert not session.is_waiting_on_workflow_call()
    assert session_queue.completed_item_ids == [100, 1]
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
    assert parent_outputs[0].collection == [3]
    assert events.errors == []


def test_run_completes_call_saved_workflow_with_child_return_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    processor = DefaultSessionProcessor(session_runner=runner)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-return"))

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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    _drain_workflow_call_queue(processor.session_runner.workflow_call_queue_lifecycle, session_queue, queue_item)

    child_return_outputs = [
        output
        for invocation, child_queue_item, output in events.completed
        if child_queue_item.session is not session
        and child_queue_item.session.prepared_source_mapping[invocation.id] == "child-return"
    ]
    parent_outputs = [
        output for invocation, _queue_item, output in events.completed if invocation.get_type() == "call_saved_workflow"
    ]

    assert len(child_return_outputs) == 1
    assert child_return_outputs[0].collection == [7, 8]
    assert len(parent_outputs) == 1
    assert parent_outputs[0].collection == [7, 8]
    assert session_queue.completed_item_ids == [100, 1]
    assert events.errors == []


def test_run_completes_call_saved_workflow_with_batched_child_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    processor = DefaultSessionProcessor(session_runner=runner)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-batch-direct"))

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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    _drain_workflow_call_queue(processor.session_runner.workflow_call_queue_lifecycle, session_queue, queue_item)

    parent_outputs = [
        output for invocation, _queue_item, output in events.completed if invocation.get_type() == "call_saved_workflow"
    ]

    assert session_queue.enqueued_child_item_ids == [100, 101, 102]
    assert [
        [
            (field_value.node_path, field_value.field_name, field_value.value)
            for field_value in session_queue.items[item_id].field_values
        ]
        for item_id in session_queue.enqueued_child_item_ids
    ] == [[("child-int", "value", 2)], [("child-int", "value", 4)], [("child-int", "value", 6)]]
    assert session_queue.completed_item_ids == [100, 101, 102, 1]
    assert len(parent_outputs) == 1
    assert parent_outputs[0].collection == [2, 4, 6]
    assert events.errors == []


def test_run_zips_grouped_batch_children(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    processor = DefaultSessionProcessor(session_runner=runner)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-batch-grouped"))

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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    _drain_workflow_call_queue(processor.session_runner.workflow_call_queue_lifecycle, session_queue, queue_item)

    parent_outputs = [
        output for invocation, _queue_item, output in events.completed if invocation.get_type() == "call_saved_workflow"
    ]

    assert session_queue.enqueued_child_item_ids == [100, 101, 102]
    assert len(parent_outputs) == 1
    assert parent_outputs[0].collection == [11, 22, 33]


def test_run_expands_ungrouped_batch_children_as_cartesian_product(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    processor = DefaultSessionProcessor(session_runner=runner)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-batch-cartesian"))

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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    _drain_workflow_call_queue(processor.session_runner.workflow_call_queue_lifecycle, session_queue, queue_item)

    parent_outputs = [
        output for invocation, _queue_item, output in events.completed if invocation.get_type() == "call_saved_workflow"
    ]

    assert session_queue.enqueued_child_item_ids == [100, 101, 102, 103]
    assert len(parent_outputs) == 1
    assert parent_outputs[0].collection == [11, 21, 12, 22]


def test_run_fails_batched_child_workflow_and_cancels_remaining_siblings(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    processor = DefaultSessionProcessor(session_runner=runner)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-batch-failure"))

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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    _drain_workflow_call_queue(processor.session_runner.workflow_call_queue_lifecycle, session_queue, queue_item)

    assert session.has_error()
    assert session_queue.failed_item_ids == [101, 1]
    assert session_queue.canceled_item_ids == [102]
    assert len(events.errors) == 2
    assert "Refusing integer value 2" in events.errors[0][3]


def test_run_supports_generator_backed_integer_batched_child_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    processor = DefaultSessionProcessor(session_runner=runner)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-batch-generator-integer"))

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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    _drain_workflow_call_queue(processor.session_runner.workflow_call_queue_lifecycle, session_queue, queue_item)

    parent_outputs = [
        output for invocation, _queue_item, output in events.completed if invocation.get_type() == "call_saved_workflow"
    ]

    assert session_queue.enqueued_child_item_ids == [100, 101, 102]
    assert len(parent_outputs) == 1
    assert parent_outputs[0].collection == [2, 4, 6]
    assert events.errors == []


def test_run_supports_generator_backed_image_batched_child_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    processor = DefaultSessionProcessor(session_runner=runner)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-batch-generator-image"))

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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    _drain_workflow_call_queue(processor.session_runner.workflow_call_queue_lifecycle, session_queue, queue_item)

    parent_outputs = [
        output for invocation, _queue_item, output in events.completed if invocation.get_type() == "call_saved_workflow"
    ]

    assert session_queue.enqueued_child_item_ids == [100, 101]
    assert len(parent_outputs) == 1
    assert [item.image_name for item in parent_outputs[0].collection] == ["img-a", "img-b"]
    assert events.errors == []


def test_run_rejects_non_generator_connected_batched_child_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-batch-generator"))

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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    session_queue.add_queue_item(queue_item)
    runner.run(queue_item=queue_item)

    assert session.has_error()
    assert session_queue.enqueued_child_item_ids == []
    assert session_queue.failed_item_ids == [1]
    assert len(events.errors) == 1
    assert "connected batch child workflow inputs" in events.errors[0][3]


def test_run_fails_call_saved_workflow_when_child_has_no_workflow_return(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-no-return"))

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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    session_queue.add_queue_item(queue_item)
    runner.run(queue_item=queue_item)

    assert not session.is_waiting_on_workflow_call()
    assert session.waiting_workflow_call_child_session is None
    assert session.has_error()
    assert session.workflow_call_history == []
    assert session_queue.enqueued_child_item_ids == []
    assert session_queue.waiting_item_ids == []
    assert session_queue.completed_item_ids == []
    assert session_queue.failed_item_ids == [1]
    assert len(events.errors) == 1
    assert "workflow_return" in events.errors[0][3]


def test_run_respects_child_dependency_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    processor = DefaultSessionProcessor(session_runner=runner)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-dependent"))

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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    _drain_workflow_call_queue(processor.session_runner.workflow_call_queue_lifecycle, session_queue, queue_item)

    child_completions = [
        (child_queue_item.session.prepared_source_mapping[invocation.id], output)
        for invocation, child_queue_item, output in events.completed
        if child_queue_item.session is not session and invocation.get_type() == "add"
    ]
    assert [source_id for source_id, _output in child_completions] == ["child-add-1", "child-add-2"]
    assert child_completions[0][1].value == 3
    assert child_completions[1][1].value == 7
    assert session_queue.completed_item_ids == [100, 1]
    assert events.errors == []


def test_run_respects_child_if_branching(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    processor = DefaultSessionProcessor(session_runner=runner)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-if"))

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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    _drain_workflow_call_queue(processor.session_runner.workflow_call_queue_lifecycle, session_queue, queue_item)

    child_if_outputs = [
        output
        for invocation, child_queue_item, output in events.completed
        if child_queue_item.session is not session and invocation.get_type() == "if"
    ]
    assert len(child_if_outputs) == 1
    assert child_if_outputs[0].value == 5
    assert session_queue.completed_item_ids == [100, 1]
    assert events.errors == []


def test_run_supports_nested_call_saved_workflow_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    processor = DefaultSessionProcessor(session_runner=runner)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-nested"))

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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    _drain_workflow_call_queue(processor.session_runner.workflow_call_queue_lifecycle, session_queue, queue_item)

    call_started = [
        queue_item.session.prepared_source_mapping[invocation.id]
        for queue_item, invocation in events.started
        if invocation.get_type() == "call_saved_workflow"
    ]
    call_completed = [
        queue_item.session.prepared_source_mapping[invocation.id]
        for invocation, queue_item, _output in events.completed
        if invocation.get_type() == "call_saved_workflow"
    ]
    nested_add_outputs = [
        output
        for invocation, child_queue_item, output in events.completed
        if child_queue_item.session is not session
        and child_queue_item.session.prepared_source_mapping[invocation.id] == "nested-add"
    ]
    leaf_add_outputs = [
        output
        for invocation, child_queue_item, output in events.completed
        if child_queue_item.session is not session
        and child_queue_item.session.prepared_source_mapping[invocation.id] == "leaf-add"
    ]

    assert call_started == ["call-node", "nested-call"]
    assert call_completed == ["nested-call", "call-node"]
    assert len(leaf_add_outputs) == 1
    assert leaf_add_outputs[0].value == 11
    assert len(nested_add_outputs) == 1
    assert nested_add_outputs[0].value == 4
    assert session_queue.completed_item_ids == [101, 100, 1]
    assert events.errors == []


def test_run_cascades_nested_child_workflow_failures_to_all_parents(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    processor = DefaultSessionProcessor(session_runner=runner)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-nested-no-return"))

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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    _drain_workflow_call_queue(processor.session_runner.workflow_call_queue_lifecycle, session_queue, queue_item)

    assert session.has_error()
    assert session_queue.completed_item_ids == []
    assert session_queue.failed_item_ids == [100, 1]
    assert len(events.errors) == 2
    assert "workflow_return" in events.errors[0][3]
    assert "workflow_return" in events.errors[1][3]


def test_run_preserves_canceled_child_workflow_chain_without_failing_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    lifecycle = WorkflowCallQueueLifecycle(runner)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-a"))

    session = GraphExecutionState(graph=graph)
    invocation = session.next()
    assert isinstance(invocation, CallSavedWorkflowInvocation)
    queue_item = type(
        "QueueItem",
        (),
        {
            "item_id": 1,
            "status": "in_progress",
            "session": session,
            "session_id": "session-id",
            "user_id": "user-1",
            "queue_id": "default",
            "batch_id": "batch-1",
            "priority": 0,
            "origin": None,
            "destination": None,
            "root_item_id": None,
        },
    )()

    workflow_record = runner._services.workflow_records.get(invocation.workflow_id)
    child_queue_item = runner.workflow_call_coordinator.begin_workflow_call_boundary(
        invocation, queue_item, workflow_record
    )

    session_queue.items[queue_item.item_id].status = "canceled"
    session_queue.items[child_queue_item.item_id].status = "canceled"
    monkeypatch.setattr(runner, "run", lambda queue_item: None)

    lifecycle.run_queue_item(child_queue_item)

    assert not session.has_error()
    assert session_queue.failed_item_ids == []
    assert events.errors == []


def test_run_does_not_resume_canceled_parent_after_completed_child(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    lifecycle = WorkflowCallQueueLifecycle(runner)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-a"))

    session = GraphExecutionState(graph=graph)
    invocation = session.next()
    assert isinstance(invocation, CallSavedWorkflowInvocation)
    queue_item = type(
        "QueueItem",
        (),
        {
            "item_id": 1,
            "status": "in_progress",
            "session": session,
            "session_id": "session-id",
            "user_id": "user-1",
            "queue_id": "default",
            "batch_id": "batch-1",
            "priority": 0,
            "origin": None,
            "destination": None,
            "root_item_id": None,
        },
    )()

    workflow_record = runner._services.workflow_records.get("workflow-a")
    child_queue_item = runner.workflow_call_coordinator.begin_workflow_call_boundary(
        invocation, queue_item, workflow_record
    )
    original_run = runner.run

    def run_child_then_cancel_parent(queue_item):
        original_run(queue_item)
        session_queue.items[1].status = "canceled"

    monkeypatch.setattr(runner, "run", run_child_then_cancel_parent)

    lifecycle.run_queue_item(child_queue_item)

    assert session_queue.items[1].status == "canceled"
    assert session_queue.completed_item_ids == [child_queue_item.item_id]
    assert session_queue.resumed_item_ids == []
    assert [event for event in events.completed if event[0].get_type() == "call_saved_workflow"] == []


def test_run_does_not_fail_canceled_parent_after_child_return_error(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    lifecycle = WorkflowCallQueueLifecycle(runner)

    graph = Graph()
    graph.add_node(CallSavedWorkflowInvocation(id="call-node", workflow_id="workflow-a"))

    session = GraphExecutionState(graph=graph)
    invocation = session.next()
    assert isinstance(invocation, CallSavedWorkflowInvocation)
    queue_item = type(
        "QueueItem",
        (),
        {
            "item_id": 1,
            "status": "in_progress",
            "session": session,
            "session_id": "session-id",
            "user_id": "user-1",
            "queue_id": "default",
            "batch_id": "batch-1",
            "priority": 0,
            "origin": None,
            "destination": None,
            "root_item_id": None,
        },
    )()

    workflow_record = runner._services.workflow_records.get("workflow-a")
    child_queue_item = runner.workflow_call_coordinator.begin_workflow_call_boundary(
        invocation, queue_item, workflow_record
    )

    def fail_child_after_parent_canceled(queue_item):
        queue_item.status = "failed"
        queue_item.error_message = "child failed after parent cancel"
        session_queue.items[queue_item.item_id] = queue_item
        session_queue.items[1].status = "canceled"

    monkeypatch.setattr(runner, "run", fail_child_after_parent_canceled)

    lifecycle.run_queue_item(child_queue_item)

    assert session_queue.items[1].status == "canceled"
    assert session_queue.failed_item_ids == []
    assert session.errors == {}
    assert [event for event in events.errors if event[1].get_type() == "call_saved_workflow"] == []


def test_run_forwards_literal_dynamic_workflow_inputs_to_child_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    processor = DefaultSessionProcessor(session_runner=runner)

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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    _drain_workflow_call_queue(processor.session_runner.workflow_call_queue_lifecycle, session_queue, queue_item)

    child_add_outputs = [
        output
        for invocation, child_queue_item, output in events.completed
        if invocation.get_type() == "add"
        and child_queue_item.session.prepared_source_mapping[invocation.id] == "child-add"
    ]
    assert len(child_add_outputs) == 1
    assert child_add_outputs[0].value == 9
    assert session_queue.completed_item_ids == [100, 1]
    assert events.errors == []


def test_run_forwards_connected_dynamic_workflow_inputs_to_child_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    session_queue = _DummySessionQueue()
    runner, events, _workflow_records = _build_workflow_runner(monkeypatch, session_queue=session_queue)
    processor = DefaultSessionProcessor(session_runner=runner)

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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    _drain_workflow_call_queue(processor.session_runner.workflow_call_queue_lifecycle, session_queue, queue_item)

    child_add_outputs = [
        output
        for invocation, child_queue_item, output in events.completed
        if invocation.get_type() == "add"
        and child_queue_item.session.prepared_source_mapping[invocation.id] == "child-add"
    ]
    assert len(child_add_outputs) == 1
    assert child_add_outputs[0].value == 7
    assert session_queue.completed_item_ids == [100, 1]
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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    session_queue.add_queue_item(queue_item)
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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    session_queue.add_queue_item(queue_item)
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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    session_queue.add_queue_item(queue_item)
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
            "queue_id": "default",
            "batch_id": "batch-1",
        },
    )()

    session_queue.add_queue_item(queue_item)
    runner.run(queue_item=queue_item)

    assert not session.is_waiting_on_workflow_call()
    assert session.has_error()
    assert session_queue.failed_item_ids == [1]
    assert len(events.started) == 1
    assert events.completed == []
    assert len(events.errors) == 1
    assert events.errors[0][2] == "ValueError"
