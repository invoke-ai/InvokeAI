from collections.abc import Mapping, Sequence
from typing import Any

from invokeai.app.services.shared.graph import Edge, EdgeConnection, Graph

CONNECTOR_INPUT_HANDLE = "in"
CONNECTOR_OUTPUT_HANDLE = "out"


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_invocation_node(node: Any) -> bool:
    return _is_mapping(node) and node.get("type") == "invocation" and _is_mapping(node.get("data"))


def _is_connector_node(node: Any) -> bool:
    return _is_mapping(node) and node.get("type") == "connector"


def _get_default_edges(workflow_edges: Sequence[Any]) -> list[Mapping[str, Any]]:
    return [edge for edge in workflow_edges if _is_mapping(edge) and edge.get("type") == "default"]


def _get_connector_input_edge(
    connector_id: str, workflow_edges: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    return next(
        (
            edge
            for edge in workflow_edges
            if edge.get("target") == connector_id and edge.get("targetHandle") == CONNECTOR_INPUT_HANDLE
        ),
        None,
    )


def _resolve_connector_source(
    connector_id: str, workflow_nodes: dict[str, Mapping[str, Any]], workflow_edges: Sequence[Mapping[str, Any]]
) -> tuple[str, str] | None:
    visited: set[str] = set()

    def resolve(node_id: str) -> tuple[str, str] | None:
        if node_id in visited:
            return None
        visited.add(node_id)

        incoming_edge = _get_connector_input_edge(node_id, workflow_edges)
        if incoming_edge is None:
            return None

        source_id = incoming_edge.get("source")
        source_handle = incoming_edge.get("sourceHandle")
        if not isinstance(source_id, str) or not isinstance(source_handle, str):
            return None

        source_node = workflow_nodes.get(source_id)
        if source_node is None:
            return None

        if _is_invocation_node(source_node):
            return (source_id, source_handle)

        if _is_connector_node(source_node):
            return resolve(source_id)

        return None

    return resolve(connector_id)


def build_graph_from_workflow(workflow: Mapping[str, Any]) -> Graph:
    workflow_nodes_raw = workflow.get("nodes", [])
    workflow_edges_raw = workflow.get("edges", [])

    workflow_nodes = {
        node["id"]: node for node in workflow_nodes_raw if _is_mapping(node) and isinstance(node.get("id"), str)
    }
    default_edges = _get_default_edges(workflow_edges_raw if isinstance(workflow_edges_raw, Sequence) else [])

    parsed_nodes: dict[str, dict[str, Any]] = {}
    for node in workflow_nodes.values():
        if not _is_invocation_node(node):
            continue

        data = node["data"]
        node_id = data.get("id")
        node_type = data.get("type")
        if not isinstance(node_id, str) or not isinstance(node_type, str):
            continue

        graph_node: dict[str, Any] = {
            "id": node_id,
            "type": node_type,
            "use_cache": data.get("useCache", False),
            "is_intermediate": data.get("isIntermediate", False),
        }

        inputs = data.get("inputs", {})
        if _is_mapping(inputs):
            for field_name, field_value in inputs.items():
                if not isinstance(field_name, str) or not _is_mapping(field_value):
                    continue
                graph_node[field_name] = field_value.get("value")

        parsed_nodes[node_id] = graph_node

    parsed_edges: list[dict[str, dict[str, str]]] = []
    seen_edges: set[tuple[str, str, str, str]] = set()

    for edge in default_edges:
        source_id = edge.get("source")
        target_id = edge.get("target")
        source_handle = edge.get("sourceHandle")
        target_handle = edge.get("targetHandle")
        if not all(isinstance(v, str) for v in (source_id, target_id, source_handle, target_handle)):
            continue

        target_node = workflow_nodes.get(target_id)
        if not _is_invocation_node(target_node):
            continue

        source_node = workflow_nodes.get(source_id)
        resolved_source: tuple[str, str] | None = None
        if _is_invocation_node(source_node):
            resolved_source = (source_id, source_handle)
        elif _is_connector_node(source_node):
            resolved_source = _resolve_connector_source(source_id, workflow_nodes, default_edges)

        if resolved_source is None:
            continue

        resolved_source_id, resolved_source_handle = resolved_source
        edge_key = (resolved_source_id, resolved_source_handle, target_id, target_handle)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        parsed_edges.append(
            {
                "source": {
                    "node_id": resolved_source_id,
                    "field": resolved_source_handle,
                },
                "destination": {
                    "node_id": target_id,
                    "field": target_handle,
                },
            }
        )

    for edge in parsed_edges:
        destination_node_id = edge["destination"]["node_id"]
        destination_field = edge["destination"]["field"]
        parsed_nodes[destination_node_id].pop(destination_field, None)

    return Graph.model_validate(
        {
            "nodes": parsed_nodes,
            "edges": [
                Edge(
                    source=EdgeConnection(**edge["source"]),
                    destination=EdgeConnection(**edge["destination"]),
                )
                for edge in parsed_edges
            ],
        }
    )
