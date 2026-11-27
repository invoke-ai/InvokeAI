from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

from invokeai.app.invocations.baseinvocation import Classification, InvocationRegistry
from invokeai.app.invocations.call_saved_workflow import (
    CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX,
    parse_call_saved_workflow_dynamic_input,
)
from invokeai.app.invocations.loops import LOOP_LINKAGE_FIELD
from invokeai.app.services.shared.graph import Edge, EdgeConnection, Graph

CONNECTOR_INPUT_HANDLE = "in"
CONNECTOR_OUTPUT_HANDLE = "out"


class UnsupportedWorkflowNodeError(ValueError):
    pass


class InvalidWorkflowInputError(ValueError):
    pass


@dataclass(frozen=True)
class _ConnectorLoopLinkagePath:
    source_id: str
    target_id: str
    edge_path: tuple[Mapping[str, Any], ...]


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_invocation_node(node: Any) -> bool:
    return _is_mapping(node) and node.get("type") == "invocation" and _is_mapping(node.get("data"))


def _is_connector_node(node: Any) -> bool:
    return _is_mapping(node) and node.get("type") == "connector"


def _workflow_edge_key(edge: Mapping[str, Any]) -> tuple[str, str, str, str] | None:
    values = (edge.get("source"), edge.get("sourceHandle"), edge.get("target"), edge.get("targetHandle"))
    return values if all(isinstance(value, str) for value in values) else None


def _build_dynamic_input_name(node_id: str, field_name: str) -> str:
    return f"{CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX}{node_id}::{field_name}"


def _get_form_elements(workflow: Mapping[str, Any]) -> tuple[Mapping[str, Any], str | None]:
    form = workflow.get("form")
    if not _is_mapping(form):
        return {}, None

    elements = form.get("elements")
    root_element_id = form.get("rootElementId")
    if not _is_mapping(elements) or not isinstance(root_element_id, str):
        return {}, None

    return elements, root_element_id


def _collect_exposed_inputs_from_form(workflow: Mapping[str, Any]) -> set[str]:
    elements, root_element_id = _get_form_elements(workflow)
    if not elements or root_element_id is None:
        return set()

    exposed_inputs: set[str] = set()
    stack = [root_element_id]
    visited: set[str] = set()

    while stack:
        element_id = stack.pop()
        if element_id in visited:
            continue
        visited.add(element_id)

        element = elements.get(element_id)
        if not _is_mapping(element):
            continue

        if element.get("type") == "node-field":
            data = element.get("data")
            if _is_mapping(data):
                field_identifier = data.get("fieldIdentifier")
                if _is_mapping(field_identifier):
                    node_id = field_identifier.get("nodeId")
                    field_name = field_identifier.get("fieldName")
                    if isinstance(node_id, str) and isinstance(field_name, str):
                        exposed_inputs.add(_build_dynamic_input_name(node_id, field_name))

        data = element.get("data")
        if _is_mapping(data):
            children = data.get("children")
            if isinstance(children, Sequence):
                for child_id in reversed(children):
                    if isinstance(child_id, str):
                        stack.append(child_id)

    return exposed_inputs


def get_exposed_workflow_input_names(workflow: Mapping[str, Any]) -> set[str]:
    exposed_inputs = _collect_exposed_inputs_from_form(workflow)
    if exposed_inputs:
        return exposed_inputs

    workflow_exposed_fields = workflow.get("exposedFields", [])
    if not isinstance(workflow_exposed_fields, Sequence):
        return set()

    fallback_inputs: set[str] = set()
    for field in workflow_exposed_fields:
        if not _is_mapping(field):
            continue
        node_id = field.get("nodeId")
        field_name = field.get("fieldName")
        if isinstance(node_id, str) and isinstance(field_name, str):
            fallback_inputs.add(_build_dynamic_input_name(node_id, field_name))

    return fallback_inputs


def apply_workflow_inputs_to_workflow(workflow: MutableMapping[str, Any], workflow_inputs: Mapping[str, Any]) -> None:
    if not workflow_inputs:
        return

    allowed_inputs = get_exposed_workflow_input_names(workflow)
    for input_name, value in workflow_inputs.items():
        if input_name not in allowed_inputs:
            raise InvalidWorkflowInputError(
                f"call_saved_workflow input '{input_name}' is not exposed by the selected workflow"
            )

        node_id, field_name = parse_call_saved_workflow_dynamic_input(input_name)
        workflow_nodes = workflow.get("nodes", [])
        if not isinstance(workflow_nodes, list):
            raise InvalidWorkflowInputError(
                f"call_saved_workflow input '{input_name}' targets missing child workflow node '{node_id}'"
            )
        matching_node = next(
            (
                node
                for node in workflow_nodes
                if _is_mapping(node)
                and _is_mapping(node.get("data"))
                and node.get("id") == node_id
                and node["data"].get("id") == node_id
            ),
            None,
        )
        if matching_node is None:
            raise InvalidWorkflowInputError(
                f"call_saved_workflow input '{input_name}' targets missing child workflow node '{node_id}'"
            )
        matching_node_data = matching_node["data"]
        node_type = matching_node_data.get("type")
        if not isinstance(node_type, str):
            raise InvalidWorkflowInputError(
                f"call_saved_workflow input '{input_name}' targets missing child workflow node '{node_id}'"
            )
        invocation_class = InvocationRegistry.get_invocation_for_type(node_type)
        if invocation_class is None or field_name not in invocation_class.model_fields:
            raise InvalidWorkflowInputError(
                f"call_saved_workflow input '{input_name}' targets missing child workflow field '{field_name}'"
            )
        inputs = matching_node_data.setdefault("inputs", {})
        if not _is_mapping(inputs):
            raise InvalidWorkflowInputError(
                f"call_saved_workflow input '{input_name}' targets invalid child workflow inputs on '{node_id}'"
            )
        inputs[field_name] = {"value": value}


def apply_workflow_inputs_to_graph(
    graph: Graph, workflow: Mapping[str, Any], workflow_inputs: Mapping[str, Any]
) -> None:
    if not workflow_inputs:
        return

    mutable_workflow = dict(workflow)
    apply_workflow_inputs_to_workflow(mutable_workflow, workflow_inputs)
    for input_name, value in workflow_inputs.items():
        node_id, field_name = parse_call_saved_workflow_dynamic_input(input_name)
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        setattr(node, field_name, value)


def _raise_if_unsupported_invocation_type(node_type: str, node_id: str) -> None:
    invocation_class = InvocationRegistry.get_invocation_for_type(node_type)
    if invocation_class is None:
        return

    if (
        invocation_class.UIConfig.category == "batch"
        and invocation_class.UIConfig.classification == Classification.Special
        and not node_type.endswith("_generator")
    ):
        raise UnsupportedWorkflowNodeError(
            f"call_saved_workflow does not yet support batch-special child workflow nodes such as "
            f"'{node_type}' (node '{node_id}')"
        )


def _validate_callable_workflow_nodes(workflow_nodes: Sequence[Any]) -> None:
    workflow_return_node_ids: list[str] = []

    for node in workflow_nodes:
        if not _is_invocation_node(node):
            continue

        data = node["data"]
        node_id = data.get("id")
        node_type = data.get("type")
        if not isinstance(node_id, str) or not isinstance(node_type, str):
            continue

        _raise_if_unsupported_invocation_type(node_type, node_id)

        if node_type == "workflow_return":
            workflow_return_node_ids.append(node_id)

    if len(workflow_return_node_ids) != 1:
        raise UnsupportedWorkflowNodeError(
            "call_saved_workflow requires the selected workflow to contain exactly one workflow_return node"
        )


def _get_default_edges(workflow_edges: Sequence[Any]) -> list[Mapping[str, Any]]:
    return [edge for edge in workflow_edges if _is_mapping(edge) and edge.get("type") == "default"]


def _get_loop_linkage_edges(workflow_edges: Sequence[Any]) -> list[Mapping[str, Any]]:
    return [edge for edge in workflow_edges if _is_mapping(edge) and edge.get("type") == "loop_linkage"]


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


def _get_connector_input_edges(
    connector_id: str, workflow_edges: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    return [
        edge
        for edge in workflow_edges
        if edge.get("target") == connector_id
        and edge.get("targetHandle") == CONNECTOR_INPUT_HANDLE
        and isinstance(edge.get("sourceHandle"), str)
    ]


def _get_connector_output_edges(
    connector_id: str, workflow_edges: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    return [
        edge
        for edge in workflow_edges
        if edge.get("source") == connector_id
        and edge.get("sourceHandle") == CONNECTOR_OUTPUT_HANDLE
        and isinstance(edge.get("targetHandle"), str)
    ]


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


def _resolve_for_connector_loop_linkage_path(
    edge: Mapping[str, Any], workflow_nodes: dict[str, Mapping[str, Any]], workflow_edges: Sequence[Mapping[str, Any]]
) -> _ConnectorLoopLinkagePath | None:
    """Resolve one connector alias path from a For to its direct ForReturn association."""
    if edge.get("sourceHandle") != LOOP_LINKAGE_FIELD or edge.get("targetHandle") != CONNECTOR_INPUT_HANDLE:
        return None

    source_id = edge.get("source")
    connector_id = edge.get("target")
    source_node = workflow_nodes.get(source_id)
    if (
        not isinstance(source_id, str)
        or not _is_invocation_node(source_node)
        or source_node.get("data", {}).get("type") != "for"
    ):
        return None
    if not isinstance(connector_id, str) or not _is_connector_node(workflow_nodes.get(connector_id)):
        return None

    path_edges: list[Mapping[str, Any]] = [edge]
    visited_connectors: set[str] = set()
    current_connector_id = connector_id

    while True:
        if current_connector_id in visited_connectors:
            return None
        visited_connectors.add(current_connector_id)

        input_edges = _get_connector_input_edges(current_connector_id, workflow_edges)
        if len(input_edges) != 1 or input_edges[0] is not path_edges[-1]:
            return None

        output_edges = _get_connector_output_edges(current_connector_id, workflow_edges)
        if len(output_edges) != 1:
            return None
        output_edge = output_edges[0]
        path_edges.append(output_edge)

        target_id = output_edge.get("target")
        target_handle = output_edge.get("targetHandle")
        target_node = workflow_nodes.get(target_id)
        if _is_invocation_node(target_node):
            if target_node.get("data", {}).get("type") != "for_return" or target_handle != LOOP_LINKAGE_FIELD:
                return None
            return _ConnectorLoopLinkagePath(source_id=source_id, target_id=target_id, edge_path=tuple(path_edges))
        if (
            not isinstance(target_id, str)
            or not _is_connector_node(target_node)
            or target_handle != CONNECTOR_INPUT_HANDLE
        ):
            return None
        current_connector_id = target_id


def build_graph_from_workflow(workflow: Mapping[str, Any]) -> Graph:
    workflow_nodes_raw = workflow.get("nodes", [])
    workflow_edges_raw = workflow.get("edges", [])
    _validate_callable_workflow_nodes(workflow_nodes_raw if isinstance(workflow_nodes_raw, Sequence) else [])

    workflow_nodes = {
        node["id"]: node for node in workflow_nodes_raw if _is_mapping(node) and isinstance(node.get("id"), str)
    }
    workflow_edges = workflow_edges_raw if isinstance(workflow_edges_raw, Sequence) else []
    default_edges = _get_default_edges(workflow_edges)
    loop_linkage_edges = _get_loop_linkage_edges(workflow_edges)

    connector_loop_linkage_paths: list[_ConnectorLoopLinkagePath] = []
    connector_loop_linkage_edge_keys: set[tuple[str, str, str, str]] = set()
    linked_for_ids = {
        edge.get("source")
        for edge in loop_linkage_edges
        if isinstance(edge.get("source"), str) and edge.get("sourceHandle") == LOOP_LINKAGE_FIELD
    }
    linked_return_ids = {
        edge.get("target")
        for edge in loop_linkage_edges
        if isinstance(edge.get("target"), str) and edge.get("targetHandle") == LOOP_LINKAGE_FIELD
    }
    for edge in default_edges:
        source_id = edge.get("source")
        source_handle = edge.get("sourceHandle")
        target_id = edge.get("target")
        target_handle = edge.get("targetHandle")
        if not isinstance(source_id, str) or source_handle != LOOP_LINKAGE_FIELD:
            continue
        source_node = workflow_nodes.get(source_id)
        if not _is_invocation_node(source_node) or source_node.get("data", {}).get("type") != "for":
            continue
        target_node = workflow_nodes.get(target_id)
        if not _is_connector_node(target_node) or target_handle != CONNECTOR_INPUT_HANDLE:
            continue

        path = _resolve_for_connector_loop_linkage_path(edge, workflow_nodes, default_edges)
        if path is None:
            raise InvalidWorkflowInputError(
                "loop_linkage connector path must resolve to exactly one For and one ForReturn without branching"
            )
        if path.source_id in linked_for_ids or path.target_id in linked_return_ids:
            raise InvalidWorkflowInputError(
                "loop_linkage connector path targets a For or ForReturn that already has a loop linkage"
            )
        connector_loop_linkage_paths.append(path)
        connector_loop_linkage_edge_keys.update(
            edge_key for path_edge in path.edge_path if (edge_key := _workflow_edge_key(path_edge)) is not None
        )
        linked_for_ids.add(path.source_id)
        linked_return_ids.add(path.target_id)

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
                # Saved workflows may include input metadata for unfilled optional fields without a "value".
                # Omit those fields so invocation defaults are applied instead of forcing None.
                if "value" in field_value:
                    # The frontend board picker may persist sentinel strings; graph nodes model both
                    # automatic and no-board selection as the board field default, None.
                    if field_name == "board" and field_value["value"] in ("auto", "none"):
                        continue
                    graph_node[field_name] = field_value["value"]

        parsed_nodes[node_id] = graph_node

    parsed_edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str, str]] = set()

    for edge in default_edges:
        if _workflow_edge_key(edge) in connector_loop_linkage_edge_keys:
            continue
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
                "type": "default",
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

    for edge in loop_linkage_edges:
        source_id = edge.get("source")
        target_id = edge.get("target")
        source_handle = edge.get("sourceHandle")
        target_handle = edge.get("targetHandle")
        if not all(isinstance(v, str) for v in (source_id, target_id, source_handle, target_handle)):
            continue
        if not _is_invocation_node(workflow_nodes.get(source_id)) or not _is_invocation_node(
            workflow_nodes.get(target_id)
        ):
            continue
        parsed_edges.append(
            {
                "type": "loop_linkage",
                "source": {"node_id": source_id, "field": source_handle},
                "destination": {"node_id": target_id, "field": target_handle},
            }
        )

    for path in connector_loop_linkage_paths:
        parsed_edges.append(
            {
                "type": "loop_linkage",
                "source": {"node_id": path.source_id, "field": LOOP_LINKAGE_FIELD},
                "destination": {"node_id": path.target_id, "field": LOOP_LINKAGE_FIELD},
            }
        )

    for edge in parsed_edges:
        if edge["type"] == "default":
            destination_node_id = edge["destination"]["node_id"]
            destination_field = edge["destination"]["field"]
            parsed_nodes[destination_node_id].pop(destination_field, None)

    return Graph.model_validate(
        {
            "nodes": parsed_nodes,
            "edges": [
                Edge(
                    type=edge["type"],
                    source=EdgeConnection(**edge["source"]),
                    destination=EdgeConnection(**edge["destination"]),
                )
                for edge in parsed_edges
            ],
        }
    )
