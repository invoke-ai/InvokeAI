from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from invokeai.app.services.session_queue.session_queue_common import Batch, BatchDatum, create_session_nfv_tuples
from invokeai.app.services.shared.graph import GraphExecutionState, WorkflowCallFrame
from invokeai.app.services.shared.workflow_graph_builder import (
    UnsupportedWorkflowNodeError,
    apply_workflow_inputs_to_workflow,
    build_graph_from_workflow,
)

BATCH_FIELD_NAMES = {
    "image_batch": "images",
    "string_batch": "strings",
    "integer_batch": "integers",
    "float_batch": "floats",
}
SUPPORTED_BATCH_TYPES = set(BATCH_FIELD_NAMES)
SUPPORTED_BATCH_GROUP_IDS = {
    "None",
    "Group 1",
    "Group 2",
    "Group 3",
    "Group 4",
    "Group 5",
}


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_invocation_node(node: Any) -> bool:
    return _is_mapping(node) and node.get("type") == "invocation" and _is_mapping(node.get("data"))


def _is_connector_node(node: Any) -> bool:
    return _is_mapping(node) and node.get("type") == "connector"


def workflow_contains_supported_batch_nodes(workflow: Mapping[str, Any]) -> bool:
    workflow_nodes = workflow.get("nodes", [])
    if not isinstance(workflow_nodes, Sequence):
        return False
    return any(
        _is_invocation_node(node) and node["data"].get("type") in SUPPORTED_BATCH_TYPES for node in workflow_nodes
    )


def _get_workflow_nodes(workflow: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    workflow_nodes = workflow.get("nodes", [])
    if not isinstance(workflow_nodes, Sequence):
        return {}
    return {node["id"]: node for node in workflow_nodes if _is_mapping(node) and isinstance(node.get("id"), str)}


def _get_default_edges(workflow: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    workflow_edges = workflow.get("edges", [])
    if not isinstance(workflow_edges, Sequence):
        return []
    return [edge for edge in workflow_edges if _is_mapping(edge) and edge.get("type") == "default"]


def _build_child_graph_workflow(workflow: Mapping[str, Any]) -> dict[str, Any]:
    workflow_nodes = workflow.get("nodes", [])
    workflow_edges = workflow.get("edges", [])
    if not isinstance(workflow_nodes, list) or not isinstance(workflow_edges, list):
        raise UnsupportedWorkflowNodeError("call_saved_workflow child workflow is malformed")

    filtered_nodes = [
        node
        for node in workflow_nodes
        if not (_is_invocation_node(node) and node["data"].get("type") in SUPPORTED_BATCH_TYPES)
    ]
    filtered_node_ids = {node["id"] for node in filtered_nodes if _is_mapping(node) and isinstance(node.get("id"), str)}
    filtered_edges = [
        edge
        for edge in workflow_edges
        if _is_mapping(edge)
        and edge.get("type") == "default"
        and edge.get("source") in filtered_node_ids
        and edge.get("target") in filtered_node_ids
    ]
    return {**workflow, "nodes": filtered_nodes, "edges": filtered_edges}


def _get_batch_group_id(node_data: Mapping[str, Any]) -> str:
    inputs = node_data.get("inputs")
    if not _is_mapping(inputs):
        return "None"
    batch_group_input = inputs.get("batch_group_id")
    if not _is_mapping(batch_group_input):
        return "None"
    batch_group_id = batch_group_input.get("value")
    if not isinstance(batch_group_id, str):
        return "None"
    if batch_group_id not in SUPPORTED_BATCH_GROUP_IDS:
        raise UnsupportedWorkflowNodeError(f"Unsupported batch group id '{batch_group_id}' in called workflow")
    return batch_group_id


def _get_batch_items(node_data: Mapping[str, Any], field_name: str) -> list[Any]:
    inputs = node_data.get("inputs")
    if not _is_mapping(inputs):
        raise UnsupportedWorkflowNodeError("call_saved_workflow batch child workflow node inputs are malformed")
    batch_input = inputs.get(field_name)
    if not _is_mapping(batch_input):
        raise UnsupportedWorkflowNodeError(
            f"call_saved_workflow batch child workflow node is missing required '{field_name}' input"
        )
    batch_items = batch_input.get("value")
    if not isinstance(batch_items, list):
        raise UnsupportedWorkflowNodeError(
            f"call_saved_workflow batch child workflow node '{node_data.get('id')}' must provide a direct list for '{field_name}'"
        )
    if not batch_items:
        raise UnsupportedWorkflowNodeError(
            f"call_saved_workflow batch child workflow node '{node_data.get('id')}' must provide at least one batch item"
        )
    return batch_items


def _get_outgoing_default_edges(
    node_id: str, source_handle: str, workflow_edges: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    return [
        edge
        for edge in workflow_edges
        if edge.get("source") == node_id and edge.get("sourceHandle") == source_handle and edge.get("type") == "default"
    ]


def _resolve_connector_destinations(
    connector_id: str, workflow_nodes: Mapping[str, Mapping[str, Any]], workflow_edges: Sequence[Mapping[str, Any]]
) -> list[tuple[str, str]]:
    visited: set[str] = set()
    destinations: list[tuple[str, str]] = []
    stack = [connector_id]
    while stack:
        current_id = stack.pop()
        if current_id in visited:
            continue
        visited.add(current_id)
        outgoing_edges = _get_outgoing_default_edges(current_id, "out", workflow_edges)
        for edge in outgoing_edges:
            target_id = edge.get("target")
            target_handle = edge.get("targetHandle")
            if not isinstance(target_id, str) or not isinstance(target_handle, str):
                continue
            target_node = workflow_nodes.get(target_id)
            if target_node is None:
                continue
            if _is_invocation_node(target_node):
                destinations.append((target_id, target_handle))
            elif _is_connector_node(target_node):
                stack.append(target_id)
    return destinations


def _resolve_batch_destinations(
    node_id: str,
    source_handle: str,
    workflow_nodes: Mapping[str, Mapping[str, Any]],
    workflow_edges: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    destinations: list[tuple[str, str]] = []
    for edge in _get_outgoing_default_edges(node_id, source_handle, workflow_edges):
        target_id = edge.get("target")
        target_handle = edge.get("targetHandle")
        if not isinstance(target_id, str) or not isinstance(target_handle, str):
            continue
        target_node = workflow_nodes.get(target_id)
        if target_node is None:
            continue
        if _is_invocation_node(target_node):
            destinations.append((target_id, target_handle))
        elif _is_connector_node(target_node):
            destinations.extend(_resolve_connector_destinations(target_id, workflow_nodes, workflow_edges))
    return destinations


def _raise_if_generator_backed_batch(
    node_id: str,
    field_name: str,
    workflow_edges: Sequence[Mapping[str, Any]],
    workflow_nodes: Mapping[str, Mapping[str, Any]],
) -> None:
    incoming_edges = [
        edge
        for edge in workflow_edges
        if edge.get("target") == node_id and edge.get("targetHandle") == field_name and edge.get("type") == "default"
    ]
    if not incoming_edges:
        return
    generator_node_ids = {
        edge.get("source")
        for edge in incoming_edges
        if isinstance(edge.get("source"), str)
        and _is_invocation_node(workflow_nodes.get(edge.get("source")))
        and workflow_nodes[edge["source"]]["data"].get("type", "").endswith("_generator")
    }
    if generator_node_ids:
        raise UnsupportedWorkflowNodeError(
            "call_saved_workflow does not yet support generator-backed batch child workflow nodes"
        )
    raise UnsupportedWorkflowNodeError(
        f"call_saved_workflow does not yet support connected batch child workflow inputs on node '{node_id}'"
    )


def build_batch_child_workflow_sessions(
    *,
    parent_session: GraphExecutionState,
    workflow: Mapping[str, Any],
    workflow_inputs: Mapping[str, Any],
    call_frame: WorkflowCallFrame,
    maximum_children: int,
) -> list[GraphExecutionState]:
    mutable_workflow = copy.deepcopy(workflow)
    apply_workflow_inputs_to_workflow(mutable_workflow, workflow_inputs)

    workflow_nodes = _get_workflow_nodes(mutable_workflow)
    workflow_edges = _get_default_edges(mutable_workflow)

    batch_data_by_group: dict[str, list[BatchDatum]] = {}
    for node in workflow_nodes.values():
        if not _is_invocation_node(node):
            continue
        node_data = node["data"]
        node_id = node_data.get("id")
        node_type = node_data.get("type")
        if not isinstance(node_id, str) or not isinstance(node_type, str):
            continue
        if node_type.endswith("_generator"):
            raise UnsupportedWorkflowNodeError(
                "call_saved_workflow does not yet support generator-backed batch child workflow nodes"
            )
        if node_type not in SUPPORTED_BATCH_TYPES:
            continue

        field_name = BATCH_FIELD_NAMES[node_type]
        _raise_if_generator_backed_batch(node_id, field_name, workflow_edges, workflow_nodes)
        batch_items = _get_batch_items(node_data, field_name)
        batch_group_id = _get_batch_group_id(node_data)
        destinations = _resolve_batch_destinations(node_id, field_name, workflow_nodes, workflow_edges)
        if not destinations:
            raise UnsupportedWorkflowNodeError(
                f"call_saved_workflow batch child workflow node '{node_id}' is not connected to any invocation input"
            )
        group_batch_data = batch_data_by_group.setdefault(batch_group_id, [])
        for destination_node_id, destination_field in destinations:
            group_batch_data.append(
                BatchDatum(node_path=destination_node_id, field_name=destination_field, items=batch_items)
            )

    if not batch_data_by_group:
        raise UnsupportedWorkflowNodeError("call_saved_workflow batch child workflow contains no supported batch nodes")

    sanitized_workflow = _build_child_graph_workflow(mutable_workflow)
    child_graph = build_graph_from_workflow(sanitized_workflow)
    batch_data = [[datum] for datum in batch_data_by_group.pop("None", [])]
    batch_data.extend(batch_data_by_group.values())
    batch = Batch(graph=child_graph, data=batch_data)

    child_sessions: list[GraphExecutionState] = []
    for session_id, session_json, _field_values_json in create_session_nfv_tuples(batch, maximum_children):
        generated_session = GraphExecutionState.model_validate_json(session_json)
        child_session = parent_session.create_child_workflow_execution_state(generated_session.graph, call_frame)
        child_session.id = session_id
        child_sessions.append(child_session)
    return child_sessions


def build_child_workflow_sessions(
    *,
    parent_session: GraphExecutionState,
    workflow: Mapping[str, Any],
    workflow_inputs: Mapping[str, Any],
    call_frame: WorkflowCallFrame,
    maximum_children: int,
) -> list[GraphExecutionState]:
    if workflow_contains_supported_batch_nodes(workflow):
        return build_batch_child_workflow_sessions(
            parent_session=parent_session,
            workflow=workflow,
            workflow_inputs=workflow_inputs,
            call_frame=call_frame,
            maximum_children=maximum_children,
        )

    mutable_workflow = copy.deepcopy(workflow)
    apply_workflow_inputs_to_workflow(mutable_workflow, workflow_inputs)
    child_graph = build_graph_from_workflow(mutable_workflow)
    return [parent_session.create_child_workflow_execution_state(child_graph, call_frame)]
