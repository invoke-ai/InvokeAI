import pytest

from invokeai.app.services.shared.graph import Graph
from invokeai.app.services.shared.workflow_graph_builder import (
    UnsupportedWorkflowNodeError,
    build_graph_from_workflow,
)


def _build_workflow_node(
    node_id: str,
    invocation_type: str,
    inputs: dict[str, object],
    *,
    is_intermediate: bool = False,
    use_cache: bool = True,
):
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
            "isIntermediate": is_intermediate,
            "useCache": use_cache,
            "dynamicInputTemplates": {},
            "inputs": {name: {"value": value} for name, value in inputs.items()},
        },
    }


def _build_connector_node(node_id: str):
    return {
        "id": node_id,
        "type": "connector",
        "position": {"x": 0, "y": 0},
        "data": {
            "id": node_id,
            "type": "connector",
            "label": "Connector",
            "isOpen": True,
        },
    }


def _build_workflow(edges: list[dict], nodes: list[dict]):
    return {
        "name": "Child Workflow",
        "author": "Tester",
        "description": "",
        "version": "1.0.0",
        "contact": "",
        "tags": "",
        "notes": "",
        "exposedFields": [],
        "meta": {"version": "1.0.0", "category": "user"},
        "nodes": nodes,
        "edges": edges,
        "form": None,
    }


def test_build_graph_from_workflow_converts_invocation_nodes():
    workflow = _build_workflow(
        nodes=[
            _build_workflow_node("add-1", "add", {"a": 1, "b": 2}),
            _build_workflow_node("return-1", "workflow_return", {"collection": []}),
        ],
        edges=[],
    )

    graph = build_graph_from_workflow(workflow)

    assert isinstance(graph, Graph)
    assert set(graph.nodes.keys()) == {"add-1", "return-1"}
    assert graph.nodes["add-1"].get_type() == "add"
    assert graph.nodes["add-1"].a == 1
    assert graph.nodes["add-1"].b == 2
    assert graph.nodes["return-1"].get_type() == "workflow_return"


def test_build_graph_from_workflow_flattens_connector_edges():
    workflow = _build_workflow(
        nodes=[
            _build_workflow_node("add-1", "add", {"a": 1, "b": 2}),
            _build_connector_node("connector-1"),
            _build_workflow_node("add-2", "add", {"a": 999, "b": 3}),
            _build_workflow_node("return-1", "workflow_return", {"collection": []}),
        ],
        edges=[
            {
                "id": "edge-1",
                "type": "default",
                "source": "add-1",
                "sourceHandle": "value",
                "target": "connector-1",
                "targetHandle": "in",
            },
            {
                "id": "edge-2",
                "type": "default",
                "source": "connector-1",
                "sourceHandle": "out",
                "target": "add-2",
                "targetHandle": "a",
            },
            {
                "id": "edge-3",
                "type": "default",
                "source": "add-2",
                "sourceHandle": "value",
                "target": "return-1",
                "targetHandle": "collection",
            },
        ],
    )

    graph = build_graph_from_workflow(workflow)

    assert len(graph.edges) == 2
    first_edge, second_edge = graph.edges
    assert first_edge.source.node_id == "add-1"
    assert first_edge.source.field == "value"
    assert first_edge.destination.node_id == "add-2"
    assert first_edge.destination.field == "a"
    assert second_edge.source.node_id == "add-2"
    assert second_edge.source.field == "value"
    assert second_edge.destination.node_id == "return-1"
    assert second_edge.destination.field == "collection"
    assert graph.nodes["add-2"].a == 0
    assert graph.nodes["add-2"].b == 3
    assert graph.nodes["return-1"].collection == []


def test_build_graph_from_workflow_rejects_batch_special_nodes_with_clear_error():
    workflow = _build_workflow(
        nodes=[_build_workflow_node("image-batch-1", "image_batch", {"images": []})],
        edges=[],
    )

    with pytest.raises(UnsupportedWorkflowNodeError, match="call_saved_workflow does not yet support batch-special"):
        build_graph_from_workflow(workflow)


def test_build_graph_from_workflow_rejects_workflows_without_workflow_return():
    workflow = _build_workflow(
        nodes=[_build_workflow_node("add-1", "add", {"a": 1, "b": 2})],
        edges=[],
    )

    with pytest.raises(UnsupportedWorkflowNodeError, match="exactly one workflow_return"):
        build_graph_from_workflow(workflow)


def test_build_graph_from_workflow_rejects_workflows_with_multiple_workflow_return_nodes():
    workflow = _build_workflow(
        nodes=[
            _build_workflow_node("return-1", "workflow_return", {"collection": []}),
            _build_workflow_node("return-2", "workflow_return", {"collection": []}),
        ],
        edges=[],
    )

    with pytest.raises(UnsupportedWorkflowNodeError, match="exactly one workflow_return"):
        build_graph_from_workflow(workflow)
