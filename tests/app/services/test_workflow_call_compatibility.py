from typing import Any

from invokeai.app.services.shared.workflow_call_compatibility import (
    WorkflowCallCompatibilityReason,
    get_workflow_call_compatibility,
)


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


def _workflow_dump(*, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": "Child Workflow",
        "author": "Tester",
        "description": "",
        "version": "1.0.0",
        "contact": "",
        "tags": "",
        "notes": "",
        "exposedFields": [],
        "meta": {"category": "user", "version": "1.0.0"},
        "nodes": nodes,
        "edges": edges,
        "form": None,
    }


def _services():
    return type(
        "Services",
        (),
        {
            "board_images": type(
                "BoardImages",
                (),
                {
                    "get_all_board_image_names_for_board": staticmethod(
                        lambda board_id, categories, is_intermediate: ["img-a", "img-b"]
                    )
                },
            )(),
        },
    )()


def test_get_workflow_call_compatibility_returns_ok_for_simple_callable_workflow() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node("collection", "integer_collection", {"collection": {"value": [1]}}),
            _invocation_node("return", "workflow_return", {"collection": {"value": []}}),
        ],
        edges=[
            {
                "id": "edge-return",
                "type": "default",
                "source": "collection",
                "sourceHandle": "collection",
                "target": "return",
                "targetHandle": "collection",
            }
        ],
    )

    compatibility = get_workflow_call_compatibility(
        workflow=workflow,
        workflow_id="workflow-a",
        services=_services(),
        user_id="user-1",
        maximum_children=1000,
    )

    assert compatibility.is_callable is True
    assert compatibility.reason is WorkflowCallCompatibilityReason.Ok
    assert compatibility.message is None


def test_get_workflow_call_compatibility_reports_missing_workflow_return() -> None:
    workflow = _workflow_dump(nodes=[_invocation_node("add", "add", {"a": {"value": 1}, "b": {"value": 2}})], edges=[])

    compatibility = get_workflow_call_compatibility(
        workflow=workflow,
        workflow_id="workflow-a",
        services=_services(),
        user_id="user-1",
        maximum_children=1000,
    )

    assert compatibility.is_callable is False
    assert compatibility.reason is WorkflowCallCompatibilityReason.MissingWorkflowReturn
    assert compatibility.message == "The workflow must contain exactly one workflow_return node."


def test_get_workflow_call_compatibility_reports_multiple_workflow_return_nodes() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node("return-a", "workflow_return", {"collection": {"value": []}}),
            _invocation_node("return-b", "workflow_return", {"collection": {"value": []}}),
        ],
        edges=[],
    )

    compatibility = get_workflow_call_compatibility(
        workflow=workflow,
        workflow_id="workflow-a",
        services=_services(),
        user_id="user-1",
        maximum_children=1000,
    )

    assert compatibility.is_callable is False
    assert compatibility.reason is WorkflowCallCompatibilityReason.MultipleWorkflowReturn
    assert compatibility.message == "The workflow must not contain more than one workflow_return node."


def test_get_workflow_call_compatibility_reports_unsupported_connected_batch_input() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node("source", "integer", {"value": {"value": 7}}),
            _invocation_node(
                "batch", "integer_batch", {"integers": {"value": []}, "batch_group_id": {"value": "None"}}
            ),
            _invocation_node("target", "integer", {"value": {"value": 0}}),
            _invocation_node("collect", "collect", {"collection": {"value": []}}),
            _invocation_node("return", "workflow_return", {"collection": {"value": []}}),
        ],
        edges=[
            {
                "id": "edge-source-batch",
                "type": "default",
                "source": "source",
                "sourceHandle": "value",
                "target": "batch",
                "targetHandle": "integers",
            },
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

    compatibility = get_workflow_call_compatibility(
        workflow=workflow,
        workflow_id="workflow-a",
        services=_services(),
        user_id="user-1",
        maximum_children=1000,
    )

    assert compatibility.is_callable is False
    assert compatibility.reason is WorkflowCallCompatibilityReason.UnsupportedBatchInput
    assert "connected batch child workflow inputs" in (compatibility.message or "")


def test_get_workflow_call_compatibility_allows_workflow_with_required_exposed_input() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node("target", "integer", {"value": {}}),
            _invocation_node("collect", "collect", {"collection": {"value": []}}),
            _invocation_node("return", "workflow_return", {"collection": {"value": []}}),
        ],
        edges=[
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
    workflow["exposedFields"] = [{"nodeId": "target", "fieldName": "value"}]

    compatibility = get_workflow_call_compatibility(
        workflow=workflow,
        workflow_id="workflow-a",
        services=_services(),
        user_id="user-1",
        maximum_children=1000,
    )

    assert compatibility.is_callable is True
    assert compatibility.reason is WorkflowCallCompatibilityReason.Ok
    assert compatibility.message is None
