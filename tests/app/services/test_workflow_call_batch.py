from typing import Any

import pytest

from invokeai.app.services.session_processor.workflow_call_batch import (
    build_child_workflow_session_results,
    build_child_workflow_sessions,
)
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


def test_build_child_workflow_session_results_preserves_batch_field_values() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node(
                "batch",
                "integer_batch",
                {"integers": {"value": [2, 4]}, "batch_group_id": {"value": "None"}},
            ),
            _invocation_node("target", "integer", {"value": {"value": 0}}),
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
            }
        ],
    )

    child_results = build_child_workflow_session_results(
        parent_session=GraphExecutionState(graph=Graph()),
        workflow=workflow,
        workflow_inputs={},
        call_frame=_call_frame(),
        maximum_children=10,
    )

    assert [result.session.graph.nodes["target"].value for result in child_results] == [2, 4]
    assert [
        [(field_value.node_path, field_value.field_name, field_value.value) for field_value in result.field_values or []]
        for result in child_results
    ] == [[("target", "value", 2)], [("target", "value", 4)]]


def test_build_child_workflow_sessions_expands_direct_integer_batch_into_collection_input() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node(
                "batch",
                "integer_batch",
                {"integers": {"value": [2, 4, 6]}, "batch_group_id": {"value": "None"}},
            ),
            _invocation_node("return", "workflow_return", {"collection": {"value": []}}),
        ],
        edges=[
            {
                "id": "edge-batch-return",
                "type": "default",
                "source": "batch",
                "sourceHandle": "integers",
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
    assert [child_session.graph.nodes["return"].collection for child_session in child_sessions] == [[2], [4], [6]]


def test_build_child_workflow_sessions_rejects_inaccessible_image_generator_board() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node(
                "generator",
                "image_generator",
                {
                    "generator": {
                        "value": {
                            "type": "image_generator_images_from_board",
                            "board_id": "private-board",
                            "category": "images",
                        }
                    }
                },
            ),
            _invocation_node(
                "batch",
                "image_batch",
                {"images": {"value": []}, "batch_group_id": {"value": "None"}},
            ),
            _invocation_node("target", "img_paste", {"image": {"value": None}, "board": {"value": None}}),
            _invocation_node("return", "workflow_return", {"collection": {"value": []}}),
        ],
        edges=[
            {
                "id": "edge-generator-batch",
                "type": "default",
                "source": "generator",
                "sourceHandle": "images",
                "target": "batch",
                "targetHandle": "images",
            },
            {
                "id": "edge-batch-target",
                "type": "default",
                "source": "batch",
                "sourceHandle": "images",
                "target": "target",
                "targetHandle": "image",
            },
            {
                "id": "edge-target-return",
                "type": "default",
                "source": "target",
                "sourceHandle": "image",
                "target": "return",
                "targetHandle": "collection",
            },
        ],
    )

    services = type(
        "Services",
        (),
        {
            "board_images": type(
                "BoardImages",
                (),
                {
                    "get_all_board_image_names_for_board": staticmethod(
                        lambda board_id, categories, is_intermediate: ["img-a"]
                    )
                },
            )(),
            "board_records": type(
                "BoardRecords",
                (),
                {
                    "get": staticmethod(
                        lambda board_id: type(
                            "BoardRecord",
                            (),
                            {"board_id": board_id, "user_id": "owner-1", "board_visibility": "private"},
                        )()
                    )
                },
            )(),
            "users": type(
                "Users",
                (),
                {"get": staticmethod(lambda user_id: type("User", (), {"is_admin": False})())},
            )(),
        },
    )()

    with pytest.raises(UnsupportedWorkflowNodeError, match="does not have access to board 'private-board'"):
        build_child_workflow_sessions(
            parent_session=GraphExecutionState(graph=Graph()),
            workflow=workflow,
            workflow_inputs={},
            call_frame=_call_frame(),
            maximum_children=10,
            services=services,
            user_id="caller-1",
        )


def test_build_child_workflow_sessions_allows_shared_image_generator_board() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node(
                "generator",
                "image_generator",
                {
                    "generator": {
                        "value": {
                            "type": "image_generator_images_from_board",
                            "board_id": "shared-board",
                            "category": "images",
                        }
                    }
                },
            ),
            _invocation_node(
                "batch",
                "image_batch",
                {"images": {"value": []}, "batch_group_id": {"value": "None"}},
            ),
            _invocation_node("target", "img_paste", {"image": {"value": None}, "board": {"value": None}}),
            _invocation_node("return", "workflow_return", {"collection": {"value": []}}),
        ],
        edges=[
            {
                "id": "edge-generator-batch",
                "type": "default",
                "source": "generator",
                "sourceHandle": "images",
                "target": "batch",
                "targetHandle": "images",
            },
            {
                "id": "edge-batch-target",
                "type": "default",
                "source": "batch",
                "sourceHandle": "images",
                "target": "target",
                "targetHandle": "image",
            },
            {
                "id": "edge-target-return",
                "type": "default",
                "source": "target",
                "sourceHandle": "image",
                "target": "return",
                "targetHandle": "collection",
            },
        ],
    )

    services = type(
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
            "board_records": type(
                "BoardRecords",
                (),
                {
                    "get": staticmethod(
                        lambda board_id: type(
                            "BoardRecord",
                            (),
                            {"board_id": board_id, "user_id": "owner-1", "board_visibility": "shared"},
                        )()
                    )
                },
            )(),
            "users": type(
                "Users",
                (),
                {"get": staticmethod(lambda user_id: type("User", (), {"is_admin": False})())},
            )(),
        },
    )()

    child_sessions = build_child_workflow_sessions(
        parent_session=GraphExecutionState(graph=Graph()),
        workflow=workflow,
        workflow_inputs={},
        call_frame=_call_frame(),
        maximum_children=10,
        services=services,
        user_id="caller-1",
    )

    assert len(child_sessions) == 2


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


def test_build_child_workflow_sessions_rejects_non_generator_connected_batch_nodes() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node("generator", "integer", {"value": {"value": 7}}),
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

    with pytest.raises(UnsupportedWorkflowNodeError, match="connected batch child workflow inputs"):
        build_child_workflow_sessions(
            parent_session=GraphExecutionState(graph=Graph()),
            workflow=workflow,
            workflow_inputs={},
            call_frame=_call_frame(),
            maximum_children=10,
        )


def test_build_child_workflow_sessions_supports_generator_backed_batch_input_through_connector() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node(
                "generator",
                "integer_generator",
                {
                    "generator": {
                        "value": {
                            "type": "integer_generator_arithmetic_sequence",
                            "start": 1,
                            "step": 1,
                            "count": 3,
                        }
                    }
                },
            ),
            {
                "id": "connector",
                "type": "connector",
                "position": {"x": 0, "y": 0},
                "data": {"id": "connector", "type": "connector", "label": "Connector", "isOpen": True},
            },
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
                "id": "edge-generator-connector",
                "type": "default",
                "source": "generator",
                "sourceHandle": "integers",
                "target": "connector",
                "targetHandle": "in",
            },
            {
                "id": "edge-connector-batch",
                "type": "default",
                "source": "connector",
                "sourceHandle": "out",
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

    child_sessions = build_child_workflow_sessions(
        parent_session=GraphExecutionState(graph=Graph()),
        workflow=workflow,
        workflow_inputs={},
        call_frame=_call_frame(),
        maximum_children=10,
    )

    assert [child_session.graph.nodes["target"].value for child_session in child_sessions] == [1, 2, 3]


def test_build_child_workflow_sessions_rejects_hybrid_child_workflows_with_unrelated_generator_nodes() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node(
                "batch",
                "integer_batch",
                {"integers": {"value": [2, 4]}, "batch_group_id": {"value": "None"}},
            ),
            _invocation_node(
                "side-generator",
                "integer_generator",
                {
                    "generator": {
                        "value": {
                            "type": "integer_generator_arithmetic_sequence",
                            "start": 5,
                            "step": 1,
                            "count": 2,
                        }
                    }
                },
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

    with pytest.raises(
        UnsupportedWorkflowNodeError,
        match="mix supported batch nodes with unrelated generator nodes",
    ):
        build_child_workflow_sessions(
            parent_session=GraphExecutionState(graph=Graph()),
            workflow=workflow,
            workflow_inputs={},
            call_frame=_call_frame(),
            maximum_children=10,
        )


def test_build_child_workflow_sessions_supports_integer_generator() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node(
                "generator",
                "integer_generator",
                {
                    "generator": {
                        "value": {
                            "type": "integer_generator_arithmetic_sequence",
                            "start": 3,
                            "step": 2,
                            "count": 3,
                        }
                    }
                },
            ),
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

    child_sessions = build_child_workflow_sessions(
        parent_session=GraphExecutionState(graph=Graph()),
        workflow=workflow,
        workflow_inputs={},
        call_frame=_call_frame(),
        maximum_children=10,
    )

    assert [child_session.graph.nodes["target"].value for child_session in child_sessions] == [3, 5, 7]


def test_build_child_workflow_sessions_supports_float_generator() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node(
                "generator",
                "float_generator",
                {
                    "generator": {
                        "value": {
                            "type": "float_generator_linear_distribution",
                            "start": 0.5,
                            "end": 1.5,
                            "count": 3,
                        }
                    }
                },
            ),
            _invocation_node(
                "batch",
                "float_batch",
                {"floats": {"value": []}, "batch_group_id": {"value": "None"}},
            ),
            _invocation_node("target", "float", {"value": {"value": 0.0}}),
            _invocation_node("collect", "collect", {"collection": {"value": []}}),
            _invocation_node("return", "workflow_return", {"collection": {"value": []}}),
        ],
        edges=[
            {
                "id": "edge-generator-batch",
                "type": "default",
                "source": "generator",
                "sourceHandle": "floats",
                "target": "batch",
                "targetHandle": "floats",
            },
            {
                "id": "edge-batch-target",
                "type": "default",
                "source": "batch",
                "sourceHandle": "floats",
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

    child_sessions = build_child_workflow_sessions(
        parent_session=GraphExecutionState(graph=Graph()),
        workflow=workflow,
        workflow_inputs={},
        call_frame=_call_frame(),
        maximum_children=10,
    )

    assert [child_session.graph.nodes["target"].value for child_session in child_sessions] == [0.5, 1.0, 1.5]


def test_build_child_workflow_sessions_supports_string_generator() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node(
                "generator",
                "string_generator",
                {
                    "generator": {
                        "value": {
                            "type": "string_generator_parse_string",
                            "input": "alpha,beta,gamma",
                            "splitOn": ",",
                        }
                    }
                },
            ),
            _invocation_node(
                "batch",
                "string_batch",
                {"strings": {"value": []}, "batch_group_id": {"value": "None"}},
            ),
            _invocation_node("target", "string", {"value": {"value": ""}}),
            _invocation_node("collect", "collect", {"collection": {"value": []}}),
            _invocation_node("return", "workflow_return", {"collection": {"value": []}}),
        ],
        edges=[
            {
                "id": "edge-generator-batch",
                "type": "default",
                "source": "generator",
                "sourceHandle": "strings",
                "target": "batch",
                "targetHandle": "strings",
            },
            {
                "id": "edge-batch-target",
                "type": "default",
                "source": "batch",
                "sourceHandle": "strings",
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

    child_sessions = build_child_workflow_sessions(
        parent_session=GraphExecutionState(graph=Graph()),
        workflow=workflow,
        workflow_inputs={},
        call_frame=_call_frame(),
        maximum_children=10,
    )

    assert [child_session.graph.nodes["target"].value for child_session in child_sessions] == [
        "alpha",
        "beta",
        "gamma",
    ]


def test_build_child_workflow_sessions_supports_string_dynamic_prompts_generator() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node(
                "generator",
                "string_generator",
                {
                    "generator": {
                        "value": {
                            "type": "string_generator_dynamic_prompts_combinatorial",
                            "input": "a {red|blue} cube",
                            "maxPrompts": 10,
                        }
                    }
                },
            ),
            _invocation_node(
                "batch",
                "string_batch",
                {"strings": {"value": []}, "batch_group_id": {"value": "None"}},
            ),
            _invocation_node("target", "string", {"value": {"value": ""}}),
            _invocation_node("collect", "collect", {"collection": {"value": []}}),
            _invocation_node("return", "workflow_return", {"collection": {"value": []}}),
        ],
        edges=[
            {
                "id": "edge-generator-batch",
                "type": "default",
                "source": "generator",
                "sourceHandle": "strings",
                "target": "batch",
                "targetHandle": "strings",
            },
            {
                "id": "edge-batch-target",
                "type": "default",
                "source": "batch",
                "sourceHandle": "strings",
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

    child_sessions = build_child_workflow_sessions(
        parent_session=GraphExecutionState(graph=Graph()),
        workflow=workflow,
        workflow_inputs={},
        call_frame=_call_frame(),
        maximum_children=10,
    )

    assert [child_session.graph.nodes["target"].value for child_session in child_sessions] == [
        "a red cube",
        "a blue cube",
    ]


def test_build_child_workflow_sessions_supports_seeded_uniform_random_generators() -> None:
    integer_workflow = _workflow_dump(
        nodes=[
            _invocation_node(
                "generator",
                "integer_generator",
                {
                    "generator": {
                        "value": {
                            "type": "integer_generator_random_distribution_uniform",
                            "min": 1,
                            "max": 3,
                            "count": 3,
                            "seed": 4,
                        }
                    }
                },
            ),
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
    float_workflow = _workflow_dump(
        nodes=[
            _invocation_node(
                "generator",
                "float_generator",
                {
                    "generator": {
                        "value": {
                            "type": "float_generator_random_distribution_uniform",
                            "min": 0.0,
                            "max": 1.0,
                            "count": 2,
                            "seed": 4,
                        }
                    }
                },
            ),
            _invocation_node(
                "batch",
                "float_batch",
                {"floats": {"value": []}, "batch_group_id": {"value": "None"}},
            ),
            _invocation_node("target", "float", {"value": {"value": 0.0}}),
            _invocation_node("collect", "collect", {"collection": {"value": []}}),
            _invocation_node("return", "workflow_return", {"collection": {"value": []}}),
        ],
        edges=[
            {
                "id": "edge-generator-batch",
                "type": "default",
                "source": "generator",
                "sourceHandle": "floats",
                "target": "batch",
                "targetHandle": "floats",
            },
            {
                "id": "edge-batch-target",
                "type": "default",
                "source": "batch",
                "sourceHandle": "floats",
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

    integer_sessions = build_child_workflow_sessions(
        parent_session=GraphExecutionState(graph=Graph()),
        workflow=integer_workflow,
        workflow_inputs={},
        call_frame=_call_frame(),
        maximum_children=10,
    )
    float_sessions = build_child_workflow_sessions(
        parent_session=GraphExecutionState(graph=Graph()),
        workflow=float_workflow,
        workflow_inputs={},
        call_frame=_call_frame(),
        maximum_children=10,
    )

    assert [child_session.graph.nodes["target"].value for child_session in integer_sessions] == [1, 1, 2]
    assert [round(child_session.graph.nodes["target"].value, 6) for child_session in float_sessions] == [
        0.236048,
        0.103166,
    ]


def test_build_child_workflow_sessions_supports_image_generator_from_board() -> None:
    workflow = _workflow_dump(
        nodes=[
            _invocation_node(
                "generator",
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
            _invocation_node(
                "batch",
                "image_batch",
                {"images": {"value": []}, "batch_group_id": {"value": "None"}},
            ),
            _invocation_node("target", "image", {"image": {"value": None}}),
            _invocation_node("collect", "collect", {"collection": {"value": []}}),
            _invocation_node("return", "workflow_return", {"collection": {"value": []}}),
        ],
        edges=[
            {
                "id": "edge-generator-batch",
                "type": "default",
                "source": "generator",
                "sourceHandle": "images",
                "target": "batch",
                "targetHandle": "images",
            },
            {
                "id": "edge-batch-target",
                "type": "default",
                "source": "batch",
                "sourceHandle": "images",
                "target": "target",
                "targetHandle": "image",
            },
            {
                "id": "edge-collect-return",
                "type": "default",
                "source": "target",
                "sourceHandle": "image",
                "target": "collect",
                "targetHandle": "item",
            },
            {
                "id": "edge-target-return",
                "type": "default",
                "source": "collect",
                "sourceHandle": "collection",
                "target": "return",
                "targetHandle": "collection",
            },
        ],
    )

    services = type(
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
            )()
        },
    )()

    child_sessions = build_child_workflow_sessions(
        parent_session=GraphExecutionState(graph=Graph()),
        workflow=workflow,
        workflow_inputs={},
        call_frame=_call_frame(),
        maximum_children=10,
        services=services,
        user_id="user-1",
    )

    assert [child_session.graph.nodes["target"].image.image_name for child_session in child_sessions] == [
        "img-a",
        "img-b",
    ]
