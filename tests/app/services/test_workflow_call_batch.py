from typing import Any

import pytest

from invokeai.app.services.session_processor.workflow_call_batch import build_child_workflow_sessions
from invokeai.app.services.shared.graph import Graph, GraphExecutionState, WorkflowCallFrame
from invokeai.app.services.shared.workflow_graph_builder import UnsupportedWorkflowNodeError


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


def _workflow_dump(
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
        "meta": {"category": "user", "version": "1.0.0"},
        "nodes": nodes,
        "edges": edges,
        "form": None,
    }


def _call_frame() -> WorkflowCallFrame:
    return WorkflowCallFrame(
        prepared_call_node_id="prepared-call",
        source_call_node_id="source-call",
        workflow_id="workflow-batch",
        depth=1,
    )


def test_build_child_workflow_sessions_expands_direct_integer_batch() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node(
                "batch",
                "integer_batch",
                {"integers": {"value": [2, 4, 6]}, "batch_group_id": {"value": "None"}},
            ),
            _invocation_node("target", "integer", {"value": {"value": 0}}),
            _invocation_node("collect", "collect", {"collection": {"value": []}}),
            _invocation_node("return", "workflow_return", {"collection": {"value": []}}),
        ],
        edges=[
            {
                "id": "edge-batch-target",
                "type": "default",
                "source": "batch",
                "sourceHandle": "integers",
                "target": "target",
                "targetHandle": "value",
            },
            {
                "id": "edge-target-collect",
                "type": "default",
                "source": "target",
                "sourceHandle": "value",
                "target": "collect",
                "targetHandle": "item",
            },
            {
                "id": "edge-collect-return",
                "type": "default",
                "source": "collect",
                "sourceHandle": "collection",
                "target": "return",
                "targetHandle": "collection",
            },
        ],
    )

    parent_session = GraphExecutionState(graph=Graph())
    child_sessions = build_child_workflow_sessions(
        parent_session=parent_session,
        workflow=workflow,
        workflow_inputs={},
        call_frame=_call_frame(),
        maximum_children=10,
    )

    assert len(child_sessions) == 3
    assert [child_session.graph.nodes["target"].value for child_session in child_sessions] == [2, 4, 6]


def test_build_child_workflow_sessions_zips_grouped_batches() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node(
                "batch-a",
                "integer_batch",
                {"integers": {"value": [1, 2, 3]}, "batch_group_id": {"value": "Group 1"}},
            ),
            _invocation_node(
                "batch-b",
                "integer_batch",
                {"integers": {"value": [10, 20, 30]}, "batch_group_id": {"value": "Group 1"}},
            ),
            _invocation_node("target", "add", {"a": {"value": 0}, "b": {"value": 0}}),
            _invocation_node("collect", "collect", {"collection": {"value": []}}),
            _invocation_node("return", "workflow_return", {"collection": {"value": []}}),
        ],
        edges=[
            {
                "id": "edge-a",
                "type": "default",
                "source": "batch-a",
                "sourceHandle": "integers",
                "target": "target",
                "targetHandle": "a",
            },
            {
                "id": "edge-b",
                "type": "default",
                "source": "batch-b",
                "sourceHandle": "integers",
                "target": "target",
                "targetHandle": "b",
            },
            {
                "id": "edge-target-collect",
                "type": "default",
                "source": "target",
                "sourceHandle": "value",
                "target": "collect",
                "targetHandle": "item",
            },
            {
                "id": "edge-collect-return",
                "type": "default",
                "source": "collect",
                "sourceHandle": "collection",
                "target": "return",
                "targetHandle": "collection",
            },
        ],
    )

    child_sessions = build_child_workflow_sessions(
        parent_session=GraphExecutionState(graph=Graph()),
        workflow=workflow,
        workflow_inputs={},
        call_frame=_call_frame(),
        maximum_children=10,
    )

    assert len(child_sessions) == 3
    assert [(child.graph.nodes["target"].a, child.graph.nodes["target"].b) for child in child_sessions] == [
        (1, 10),
        (2, 20),
        (3, 30),
    ]


def test_build_child_workflow_sessions_expands_cartesian_batches() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node(
                "batch-a",
                "integer_batch",
                {"integers": {"value": [1, 2]}, "batch_group_id": {"value": "None"}},
            ),
            _invocation_node(
                "batch-b",
                "integer_batch",
                {"integers": {"value": [10, 20]}, "batch_group_id": {"value": "None"}},
            ),
            _invocation_node("target", "add", {"a": {"value": 0}, "b": {"value": 0}}),
            _invocation_node("collect", "collect", {"collection": {"value": []}}),
            _invocation_node("return", "workflow_return", {"collection": {"value": []}}),
        ],
        edges=[
            {
                "id": "edge-a",
                "type": "default",
                "source": "batch-a",
                "sourceHandle": "integers",
                "target": "target",
                "targetHandle": "a",
            },
            {
                "id": "edge-b",
                "type": "default",
                "source": "batch-b",
                "sourceHandle": "integers",
                "target": "target",
                "targetHandle": "b",
            },
            {
                "id": "edge-target-collect",
                "type": "default",
                "source": "target",
                "sourceHandle": "value",
                "target": "collect",
                "targetHandle": "item",
            },
            {
                "id": "edge-collect-return",
                "type": "default",
                "source": "collect",
                "sourceHandle": "collection",
                "target": "return",
                "targetHandle": "collection",
            },
        ],
    )

    child_sessions = build_child_workflow_sessions(
        parent_session=GraphExecutionState(graph=Graph()),
        workflow=workflow,
        workflow_inputs={},
        call_frame=_call_frame(),
        maximum_children=10,
    )

    assert len(child_sessions) == 4
    assert [(child.graph.nodes["target"].a, child.graph.nodes["target"].b) for child in child_sessions] == [
        (1, 10),
        (1, 20),
        (2, 10),
        (2, 20),
    ]


def test_build_child_workflow_sessions_applies_parent_inputs_before_batch_expansion() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node(
                "batch",
                "integer_batch",
                {"integers": {"value": [1]}, "batch_group_id": {"value": "None"}},
            ),
            _invocation_node("target", "integer", {"value": {"value": 0}}),
            _invocation_node("collect", "collect", {"collection": {"value": []}}),
            _invocation_node("return", "workflow_return", {"collection": {"value": []}}),
        ],
        edges=[
            {
                "id": "edge-batch-target",
                "type": "default",
                "source": "batch",
                "sourceHandle": "integers",
                "target": "target",
                "targetHandle": "value",
            },
            {
                "id": "edge-target-collect",
                "type": "default",
                "source": "target",
                "sourceHandle": "value",
                "target": "collect",
                "targetHandle": "item",
            },
            {
                "id": "edge-collect-return",
                "type": "default",
                "source": "collect",
                "sourceHandle": "collection",
                "target": "return",
                "targetHandle": "collection",
            },
        ],
        exposed_fields=[{"nodeId": "batch", "fieldName": "integers"}],
    )

    child_sessions = build_child_workflow_sessions(
        parent_session=GraphExecutionState(graph=Graph()),
        workflow=workflow,
        workflow_inputs={"saved_workflow_input::batch::integers": [9, 10]},
        call_frame=_call_frame(),
        maximum_children=10,
    )

    assert [child_session.graph.nodes["target"].value for child_session in child_sessions] == [9, 10]


def test_build_child_workflow_sessions_rejects_generator_backed_batch_nodes() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node("generator", "integer_generator", {"generator": {"value": {}}}),
            _invocation_node(
                "batch",
                "integer_batch",
                {"integers": {"value": []}, "batch_group_id": {"value": "None"}},
            ),
            _invocation_node("target", "integer", {"value": {"value": 0}}),
            _invocation_node("collect", "collect", {"collection": {"value": []}}),
            _invocation_node("return", "workflow_return", {"collection": {"value": []}}),
        ],
        edges=[
            {
                "id": "edge-generator-batch",
                "type": "default",
                "source": "generator",
                "sourceHandle": "integers",
                "target": "batch",
                "targetHandle": "integers",
            }
        ],
    )

    with pytest.raises(UnsupportedWorkflowNodeError, match="generator-backed batch child workflow nodes"):
        build_child_workflow_sessions(
            parent_session=GraphExecutionState(graph=Graph()),
            workflow=workflow,
            workflow_inputs={},
            call_frame=_call_frame(),
            maximum_children=10,
        )
