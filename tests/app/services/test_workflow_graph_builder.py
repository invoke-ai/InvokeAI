import json
from pathlib import Path

import pytest

from invokeai.app.services.shared.graph import Graph, InvalidEdgeError
from invokeai.app.services.shared.workflow_graph_builder import (
    InvalidWorkflowInputError,
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


def _build_named_return_nodes():
    return [
        _build_workflow_node("return-value-1", "workflow_return_value", {"key": "result", "value": None}),
        _build_workflow_node("return-collect-1", "collect", {"collection": []}),
        _build_workflow_node("return-1", "workflow_return", {"values": []}),
    ]


def _build_named_return_edges(source: str, source_handle: str):
    return [
        {
            "id": "edge-return-value",
            "type": "default",
            "source": source,
            "sourceHandle": source_handle,
            "target": "return-value-1",
            "targetHandle": "value",
        },
        {
            "id": "edge-return-collect",
            "type": "default",
            "source": "return-value-1",
            "sourceHandle": "value",
            "target": "return-collect-1",
            "targetHandle": "item",
        },
        {
            "id": "edge-return-values",
            "type": "default",
            "source": "return-collect-1",
            "sourceHandle": "collection",
            "target": "return-1",
            "targetHandle": "values",
        },
    ]


FOR_LOOP_FIXTURE_DIR = (
    Path(__file__).parents[3]
    / "invokeai"
    / "frontend"
    / "webv2"
    / "src"
    / "features"
    / "workflow"
    / "core"
    / "fixtures"
)

FOR_LOOP_FIXTURES = sorted(FOR_LOOP_FIXTURE_DIR.glob("for-loop-*.json"))
# CI's python-tests change filter re-includes this directory by path; a silent empty glob would
# hide both a moved fixture directory and the resulting stale filter.
assert FOR_LOOP_FIXTURES, f"no for-loop fixtures found in {FOR_LOOP_FIXTURE_DIR}"


@pytest.mark.parametrize("fixture_path", FOR_LOOP_FIXTURES)
def test_for_loop_fixtures_have_matching_backend_validation(fixture_path: Path):
    fixture = json.loads(fixture_path.read_text())
    graph = build_graph_from_workflow(fixture)

    if fixture["fixture"] == "valid":
        graph.validate_self()
    else:
        with pytest.raises(InvalidEdgeError):
            graph.validate_self()


def test_build_graph_from_workflow_converts_invocation_nodes():
    workflow = _build_workflow(
        nodes=[
            _build_workflow_node("add-1", "add", {"a": 1, "b": 2}),
            _build_workflow_node("return-1", "workflow_return", {"values": []}),
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


def test_build_graph_from_workflow_preserves_loop_linkage_edges():
    workflow = _build_workflow(
        nodes=[
            _build_workflow_node("for-1", "for", {"collection": ["a"]}),
            _build_workflow_node("for-return-1", "for_return", {}),
            _build_workflow_node("workflow-return-1", "workflow_return", {"values": []}),
        ],
        edges=[
            {
                "id": "edge-for-body",
                "type": "default",
                "source": "for-1",
                "sourceHandle": "item",
                "target": "for-return-1",
                "targetHandle": "output",
            },
            {
                "id": "edge-for-linkage",
                "type": "loop_linkage",
                "source": "for-1",
                "sourceHandle": "loop_linkage",
                "target": "for-return-1",
                "targetHandle": "loop_linkage",
            },
        ],
    )

    graph = build_graph_from_workflow(workflow)

    assert [edge.type for edge in graph.edges] == ["default", "loop_linkage"]
    assert graph.edges[1].source.node_id == "for-1"
    assert graph.edges[1].destination.node_id == "for-return-1"


def test_build_graph_from_workflow_canonicalizes_connector_loop_linkage():
    workflow = _build_workflow(
        nodes=[
            _build_workflow_node("for-1", "for", {"collection": ["a"]}),
            _build_connector_node("connector-1"),
            _build_workflow_node("for-return-1", "for_return", {}),
            _build_workflow_node("workflow-return-1", "workflow_return", {"values": []}),
        ],
        edges=[
            {
                "id": "edge-for-body",
                "type": "default",
                "source": "for-1",
                "sourceHandle": "item",
                "target": "for-return-1",
                "targetHandle": "output",
            },
            {
                "id": "edge-linkage-input",
                "type": "default",
                "source": "for-1",
                "sourceHandle": "loop_linkage",
                "target": "connector-1",
                "targetHandle": "in",
            },
            {
                "id": "edge-linkage-output",
                "type": "default",
                "source": "connector-1",
                "sourceHandle": "out",
                "target": "for-return-1",
                "targetHandle": "loop_linkage",
            },
        ],
    )

    graph = build_graph_from_workflow(workflow)

    assert [edge.type for edge in graph.edges] == ["default", "loop_linkage"]
    assert graph.edges[1].source.node_id == "for-1"
    assert graph.edges[1].source.field == "loop_linkage"
    assert graph.edges[1].destination.node_id == "for-return-1"
    assert graph.edges[1].destination.field == "loop_linkage"


def test_build_graph_from_workflow_canonicalizes_chained_connector_loop_linkage():
    workflow = _build_workflow(
        nodes=[
            _build_workflow_node("for-1", "for", {"collection": ["a"]}),
            _build_connector_node("connector-1"),
            _build_connector_node("connector-2"),
            _build_workflow_node("for-return-1", "for_return", {}),
            _build_workflow_node("workflow-return-1", "workflow_return", {"values": []}),
        ],
        edges=[
            {
                "id": "edge-linkage-input",
                "type": "default",
                "source": "for-1",
                "sourceHandle": "loop_linkage",
                "target": "connector-1",
                "targetHandle": "in",
            },
            {
                "id": "edge-linkage-chain",
                "type": "default",
                "source": "connector-1",
                "sourceHandle": "out",
                "target": "connector-2",
                "targetHandle": "in",
            },
            {
                "id": "edge-linkage-output",
                "type": "default",
                "source": "connector-2",
                "sourceHandle": "out",
                "target": "for-return-1",
                "targetHandle": "loop_linkage",
            },
        ],
    )

    graph = build_graph_from_workflow(workflow)

    assert len(graph.edges) == 1
    assert graph.edges[0].type == "loop_linkage"
    assert graph.edges[0].source.node_id == "for-1"
    assert graph.edges[0].destination.node_id == "for-return-1"


def test_build_graph_from_workflow_rejects_branched_connector_loop_linkage():
    workflow = _build_workflow(
        nodes=[
            _build_workflow_node("for-1", "for", {"collection": ["a"]}),
            _build_connector_node("connector-1"),
            _build_workflow_node("for-return-1", "for_return", {}),
            _build_workflow_node("for-return-2", "for_return", {}),
            _build_workflow_node("workflow-return-1", "workflow_return", {"values": []}),
        ],
        edges=[
            {
                "id": "edge-linkage-input",
                "type": "default",
                "source": "for-1",
                "sourceHandle": "loop_linkage",
                "target": "connector-1",
                "targetHandle": "in",
            },
            {
                "id": "edge-linkage-output-1",
                "type": "default",
                "source": "connector-1",
                "sourceHandle": "out",
                "target": "for-return-1",
                "targetHandle": "loop_linkage",
            },
            {
                "id": "edge-linkage-output-2",
                "type": "default",
                "source": "connector-1",
                "sourceHandle": "out",
                "target": "for-return-2",
                "targetHandle": "loop_linkage",
            },
        ],
    )

    with pytest.raises(InvalidWorkflowInputError, match="loop_linkage connector path"):
        build_graph_from_workflow(workflow)


def test_build_graph_from_workflow_rejects_connector_loop_linkage_duplicate_ownership():
    workflow = _build_workflow(
        nodes=[
            _build_workflow_node("for-1", "for", {"collection": ["a"]}),
            _build_connector_node("connector-1"),
            _build_connector_node("connector-2"),
            _build_workflow_node("for-return-1", "for_return", {}),
            _build_workflow_node("workflow-return-1", "workflow_return", {"values": []}),
        ],
        edges=[
            {
                "id": "edge-linkage-input-1",
                "type": "default",
                "source": "for-1",
                "sourceHandle": "loop_linkage",
                "target": "connector-1",
                "targetHandle": "in",
            },
            {
                "id": "edge-linkage-output-1",
                "type": "default",
                "source": "connector-1",
                "sourceHandle": "out",
                "target": "for-return-1",
                "targetHandle": "loop_linkage",
            },
            {
                "id": "edge-linkage-input-2",
                "type": "default",
                "source": "for-1",
                "sourceHandle": "loop_linkage",
                "target": "connector-2",
                "targetHandle": "in",
            },
            {
                "id": "edge-linkage-output-2",
                "type": "default",
                "source": "connector-2",
                "sourceHandle": "out",
                "target": "for-return-1",
                "targetHandle": "loop_linkage",
            },
        ],
    )

    with pytest.raises(InvalidWorkflowInputError, match="already has a loop linkage"):
        build_graph_from_workflow(workflow)


def test_build_graph_from_workflow_rejects_loop_linkage_connector_ordinary_fanout():
    workflow = _build_workflow(
        nodes=[
            _build_workflow_node("for-1", "for", {"collection": ["a"]}),
            _build_connector_node("connector-1"),
            _build_workflow_node("add-1", "add", {"a": 1, "b": 2}),
            _build_workflow_node("workflow-return-1", "workflow_return", {"values": []}),
        ],
        edges=[
            {
                "id": "edge-linkage-input",
                "type": "default",
                "source": "for-1",
                "sourceHandle": "loop_linkage",
                "target": "connector-1",
                "targetHandle": "in",
            },
            {
                "id": "edge-ordinary-output",
                "type": "default",
                "source": "connector-1",
                "sourceHandle": "out",
                "target": "add-1",
                "targetHandle": "a",
            },
        ],
    )

    with pytest.raises(InvalidWorkflowInputError, match="loop_linkage connector path"):
        build_graph_from_workflow(workflow)


def test_build_graph_from_workflow_flattens_connector_edges():
    workflow = _build_workflow(
        nodes=[
            _build_workflow_node("add-1", "add", {"a": 1, "b": 2}),
            _build_connector_node("connector-1"),
            _build_workflow_node("add-2", "add", {"a": 999, "b": 3}),
            *_build_named_return_nodes(),
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
            *_build_named_return_edges("add-2", "value"),
        ],
    )

    graph = build_graph_from_workflow(workflow)

    assert len(graph.edges) == 4
    first_edge, second_edge, third_edge, fourth_edge = graph.edges
    assert first_edge.source.node_id == "add-1"
    assert first_edge.source.field == "value"
    assert first_edge.destination.node_id == "add-2"
    assert first_edge.destination.field == "a"
    assert second_edge.source.node_id == "add-2"
    assert second_edge.source.field == "value"
    assert second_edge.destination.node_id == "return-value-1"
    assert second_edge.destination.field == "value"
    assert third_edge.destination.node_id == "return-collect-1"
    assert third_edge.destination.field == "item"
    assert fourth_edge.destination.node_id == "return-1"
    assert fourth_edge.destination.field == "values"
    assert graph.nodes["add-2"].a == 0
    assert graph.nodes["add-2"].b == 3
    assert graph.nodes["return-1"].values == []


def test_build_graph_from_workflow_uses_defaults_for_inputs_without_saved_values():
    collect_node = _build_workflow_node("return-collect-1", "collect", {})
    collect_node["data"]["inputs"] = {
        "item": {"name": "item", "label": "", "description": ""},
        "collection": {"name": "collection", "label": "", "description": ""},
    }
    workflow = _build_workflow(
        nodes=[
            _build_workflow_node("return-value-1", "workflow_return_value", {"key": "result", "value": None}),
            collect_node,
            _build_workflow_node("return-1", "workflow_return", {"values": []}),
        ],
        edges=[
            {
                "id": "edge-return-collect",
                "type": "default",
                "source": "return-value-1",
                "sourceHandle": "value",
                "target": "return-collect-1",
                "targetHandle": "item",
            },
            {
                "id": "edge-return-values",
                "type": "default",
                "source": "return-collect-1",
                "sourceHandle": "collection",
                "target": "return-1",
                "targetHandle": "values",
            },
        ],
    )

    graph = build_graph_from_workflow(workflow)

    assert graph.nodes["return-collect-1"].collection == []


def test_build_graph_from_workflow_uses_default_for_legacy_auto_board_values():
    workflow = _build_workflow(
        nodes=[
            _build_workflow_node("image-1", "blank_image", {"board": "auto", "width": 64, "height": 64}),
            *_build_named_return_nodes(),
        ],
        edges=_build_named_return_edges("image-1", "image"),
    )

    graph = build_graph_from_workflow(workflow)

    assert graph.nodes["image-1"].board is None


def test_build_graph_from_workflow_uses_default_for_legacy_none_board_values():
    workflow = _build_workflow(
        nodes=[
            _build_workflow_node("image-1", "blank_image", {"board": "none", "width": 64, "height": 64}),
            *_build_named_return_nodes(),
        ],
        edges=_build_named_return_edges("image-1", "image"),
    )

    graph = build_graph_from_workflow(workflow)

    assert graph.nodes["image-1"].board is None


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
            _build_workflow_node("return-1", "workflow_return", {"values": []}),
            _build_workflow_node("return-2", "workflow_return", {"values": []}),
        ],
        edges=[],
    )

    with pytest.raises(UnsupportedWorkflowNodeError, match="exactly one workflow_return"):
        build_graph_from_workflow(workflow)
