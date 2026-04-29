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


def _return_nodes() -> list[dict[str, Any]]:
    return [
        _invocation_node(
            "return-value", "workflow_return_value", {"key": {"value": "result"}, "value": {"value": None}}
        ),
        _invocation_node("return-collect", "collect", {"collection": {"value": []}}),
        _invocation_node("return", "workflow_return", {"values": {"value": []}}),
    ]


def _return_edges(source: str, source_handle: str) -> list[dict[str, str]]:
    return [
        {
            "id": "edge-return-value",
            "type": "default",
            "source": source,
            "sourceHandle": source_handle,
            "target": "return-value",
            "targetHandle": "value",
        },
        {
            "id": "edge-return-collect",
            "type": "default",
            "source": "return-value",
            "sourceHandle": "value",
            "target": "return-collect",
            "targetHandle": "item",
        },
        {
            "id": "edge-return-values",
            "type": "default",
            "source": "return-collect",
            "sourceHandle": "collection",
            "target": "return",
            "targetHandle": "values",
        },
    ]


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


def _services_that_fail_on_image_enumeration():
    def fail(*args: Any, **kwargs: Any) -> list[str]:
        raise AssertionError("image names should not be enumerated for structural compatibility")

    return type(
        "Services",
        (),
        {
            "board_images": type(
                "BoardImages",
                (),
                {"get_all_board_image_names_for_board": staticmethod(fail)},
            )(),
        },
    )()


def test_get_workflow_call_compatibility_returns_ok_for_simple_callable_workflow() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node("collection", "integer_collection", {"collection": {"value": [1]}}),
            *_return_nodes(),
        ],
        edges=_return_edges("collection", "collection"),
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
            _invocation_node("return-a", "workflow_return", {"values": {"value": []}}),
            _invocation_node("return-b", "workflow_return", {"values": {"value": []}}),
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
            *_return_nodes(),
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
            *_return_edges("collect", "collection"),
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


def test_get_workflow_call_compatibility_allows_batch_returned_by_name() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node(
                "batch", "integer_batch", {"integers": {"value": [2, 4]}, "batch_group_id": {"value": "None"}}
            ),
            *_return_nodes(),
        ],
        edges=[
            {
                "id": "edge-batch-return",
                "type": "default",
                "source": "batch",
                "sourceHandle": "integers",
                "target": "return-value",
                "targetHandle": "value",
            },
            *_return_edges("return-value", "value")[1:],
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


def test_get_workflow_call_compatibility_can_skip_generator_expansion_for_list_views() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node(
                "generator",
                "image_generator",
                {
                    "generator": {
                        "value": {
                            "type": "image_generator_images_from_board",
                            "board_id": "board-a",
                            "category": "images",
                        }
                    }
                },
            ),
            _invocation_node("batch", "image_batch", {"images": {"value": []}, "batch_group_id": {"value": "None"}}),
            *_return_nodes(),
        ],
        edges=[
            {
                "id": "edge-generator-batch",
                "type": "default",
                "source": "generator",
                "sourceHandle": "collection",
                "target": "batch",
                "targetHandle": "images",
            },
            {
                "id": "edge-batch-return",
                "type": "default",
                "source": "batch",
                "sourceHandle": "images",
                "target": "return-value",
                "targetHandle": "value",
            },
            *_return_edges("return-value", "value")[1:],
        ],
    )

    compatibility = get_workflow_call_compatibility(
        workflow=workflow,
        workflow_id="workflow-a",
        services=_services_that_fail_on_image_enumeration(),
        user_id="user-1",
        maximum_children=1000,
        resolve_generator_items=False,
    )

    assert compatibility.is_callable is True
    assert compatibility.reason is WorkflowCallCompatibilityReason.Ok
    assert compatibility.message is None


def test_get_workflow_call_compatibility_reports_multiple_batch_inputs_as_unsupported_batch_input() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node("source-a", "integer", {"value": {"value": 7}}),
            _invocation_node("source-b", "integer", {"value": {"value": 8}}),
            _invocation_node(
                "batch", "integer_batch", {"integers": {"value": []}, "batch_group_id": {"value": "None"}}
            ),
            _invocation_node("target", "integer", {"value": {"value": 0}}),
            _invocation_node("collect", "collect", {"collection": {"value": []}}),
            *_return_nodes(),
        ],
        edges=[
            {
                "id": "edge-source-a-batch",
                "type": "default",
                "source": "source-a",
                "sourceHandle": "value",
                "target": "batch",
                "targetHandle": "integers",
            },
            {
                "id": "edge-source-b-batch",
                "type": "default",
                "source": "source-b",
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
            *_return_edges("collect", "collection"),
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
    assert "multiple connected batch inputs" in (compatibility.message or "")


def test_get_workflow_call_compatibility_allows_workflow_with_required_exposed_input() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node("target", "integer", {"value": {}}),
            _invocation_node("collect", "collect", {"collection": {"value": []}}),
            *_return_nodes(),
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
            *_return_edges("collect", "collection"),
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


def test_get_workflow_call_compatibility_allows_workflow_with_required_structured_exposed_input() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node("template", "prompt_template", {"style_preset": {}}),
            _invocation_node("collect", "collect", {"collection": {"value": []}}),
            *_return_nodes(),
        ],
        edges=[
            {
                "id": "edge-template-collect",
                "type": "default",
                "source": "template",
                "sourceHandle": "positive_prompt",
                "target": "collect",
                "targetHandle": "item",
            },
            *_return_edges("collect", "collection"),
        ],
    )
    workflow["exposedFields"] = [{"nodeId": "template", "fieldName": "style_preset"}]

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
