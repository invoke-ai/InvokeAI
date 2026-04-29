from __future__ import annotations

import copy
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dynamicprompts.generators import CombinatorialPromptGenerator, RandomPromptGenerator

from invokeai.app.invocations.fields import ImageField
from invokeai.app.services.board_records.board_records_common import BoardRecordOrderBy, BoardVisibility
from invokeai.app.services.image_records.image_records_common import ASSETS_CATEGORIES, IMAGE_CATEGORIES
from invokeai.app.services.session_queue.session_queue_common import (
    Batch,
    BatchDatum,
    NodeFieldValue,
    TooManySessionsError,
    calc_session_count,
    create_session_nfv_tuples,
)
from invokeai.app.services.shared.graph import GraphExecutionState, WorkflowCallFrame
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
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
CONNECTOR_INPUT_HANDLE = "in"


@dataclass(frozen=True)
class WorkflowCallChildSessionResult:
    session: GraphExecutionState
    field_values: list[NodeFieldValue] | None = None


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
    connector_id: str, workflow_nodes: Mapping[str, Mapping[str, Any]], workflow_edges: Sequence[Mapping[str, Any]]
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


def _build_child_graph_workflow(workflow: Mapping[str, Any], used_generator_node_ids: set[str]) -> dict[str, Any]:
    workflow_nodes = workflow.get("nodes", [])
    workflow_edges = workflow.get("edges", [])
    if not isinstance(workflow_nodes, list) or not isinstance(workflow_edges, list):
        raise UnsupportedWorkflowNodeError("call_saved_workflow child workflow is malformed")

    filtered_nodes = [
        node
        for node in workflow_nodes
        if not (
            _is_invocation_node(node)
            and (
                node["data"].get("type") in SUPPORTED_BATCH_TYPES
                or (isinstance(node.get("id"), str) and node["id"] in used_generator_node_ids)
            )
        )
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


def _reject_unrelated_generator_nodes(workflow: Mapping[str, Any], used_generator_node_ids: set[str]) -> None:
    workflow_nodes = workflow.get("nodes", [])
    if not isinstance(workflow_nodes, list):
        raise UnsupportedWorkflowNodeError("call_saved_workflow child workflow is malformed")

    unrelated_generator_nodes: list[tuple[str, str]] = []
    for node in workflow_nodes:
        if not _is_invocation_node(node):
            continue

        node_data = node["data"]
        node_id = node_data.get("id")
        node_type = node_data.get("type")
        if not isinstance(node_id, str) or not isinstance(node_type, str):
            continue
        if node_type.endswith("_generator") and node_id not in used_generator_node_ids:
            unrelated_generator_nodes.append((node_type, node_id))

    if unrelated_generator_nodes:
        unsupported_nodes = ", ".join(
            f"'{node_type}' (node '{node_id}')" for node_type, node_id in unrelated_generator_nodes
        )
        raise UnsupportedWorkflowNodeError(
            "call_saved_workflow does not yet support child workflows that mix supported batch nodes with "
            f"unrelated generator nodes: {unsupported_nodes}"
        )


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
    return batch_items


def _parse_split_values(input_value: str, split_on: str) -> list[str]:
    if split_on == "":
        return [input_value]
    try:
        return input_value.split(json.loads(f'"{split_on}"'))
    except Exception:
        return input_value.split(split_on)


def _resolve_float_generator(value: Mapping[str, Any]) -> list[float]:
    generator_type = value.get("type")
    if generator_type == "float_generator_arithmetic_sequence":
        start = float(value.get("start", 0))
        step = float(value.get("step", 1))
        count = int(value.get("count", 10))
        if step == 0:
            return [start]
        return [start + i * step for i in range(count)]
    if generator_type == "float_generator_linear_distribution":
        start = float(value.get("start", 0))
        end = float(value.get("end", 1))
        count = int(value.get("count", 10))
        if count == 1:
            return [start]
        return [start + (end - start) * (i / (count - 1)) for i in range(count)]
    if generator_type == "float_generator_random_distribution_uniform":
        minimum = float(value.get("min", 0))
        maximum = float(value.get("max", 1))
        count = int(value.get("count", 10))
        if "values" in value and isinstance(value["values"], list):
            return [float(v) for v in value["values"]]
        rng = random.Random(value.get("seed"))
        return [rng.random() * (maximum - minimum) + minimum for _ in range(count)]
    if generator_type == "float_generator_parse_string":
        if "values" in value and isinstance(value["values"], list):
            return [float(v) for v in value["values"]]
        split_values = _parse_split_values(str(value.get("input", "")), str(value.get("splitOn", ",")))
        return [float(v.strip()) for v in split_values if v.strip()]
    raise UnsupportedWorkflowNodeError(f"Unsupported float generator type '{generator_type}'")


def _resolve_integer_generator(value: Mapping[str, Any]) -> list[int]:
    generator_type = value.get("type")
    if generator_type == "integer_generator_arithmetic_sequence":
        start = int(value.get("start", 0))
        step = int(value.get("step", 1))
        count = int(value.get("count", 10))
        if step == 0:
            return [start]
        return [start + i * step for i in range(count)]
    if generator_type == "integer_generator_linear_distribution":
        start = int(value.get("start", 0))
        end = int(value.get("end", 10))
        count = int(value.get("count", 10))
        if count == 1:
            return [start]
        return [start + round((end - start) * (i / (count - 1))) for i in range(count)]
    if generator_type == "integer_generator_random_distribution_uniform":
        minimum = int(value.get("min", 0))
        maximum = int(value.get("max", 10))
        count = int(value.get("count", 10))
        rng = random.Random(value.get("seed"))
        return [int(rng.random() * (maximum - minimum + 1)) + minimum for _ in range(count)]
    if generator_type == "integer_generator_parse_string":
        split_values = _parse_split_values(str(value.get("input", "")), str(value.get("splitOn", ",")))
        return [int(v.strip()) for v in split_values if v.strip()]
    raise UnsupportedWorkflowNodeError(f"Unsupported integer generator type '{generator_type}'")


def _resolve_string_generator(value: Mapping[str, Any]) -> list[str]:
    generator_type = value.get("type")
    if generator_type == "string_generator_parse_string":
        return [v for v in _parse_split_values(str(value.get("input", "")), str(value.get("splitOn", ","))) if v]
    if generator_type == "string_generator_dynamic_prompts_combinatorial":
        generator = CombinatorialPromptGenerator()
        return list(generator.generate(str(value.get("input", "")), max_prompts=int(value.get("maxPrompts", 10))))
    if generator_type == "string_generator_dynamic_prompts_random":
        seed = value.get("seed")
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        generator = RandomPromptGenerator(seed=int(seed))
        return list(generator.generate(str(value.get("input", "")), num_images=int(value.get("count", 10))))
    raise UnsupportedWorkflowNodeError(f"Unsupported string generator type '{generator_type}'")


def _assert_user_can_access_board(board_id: str, services: Any, user_id: str | None) -> None:
    if not user_id:
        return

    board_records = getattr(services, "board_records", None)
    if board_records is None or not hasattr(board_records, "get"):
        return

    users = getattr(services, "users", None)
    user = users.get(user_id) if users is not None and hasattr(users, "get") else None
    is_admin = bool(user and getattr(user, "is_admin", False))
    if is_admin:
        return

    try:
        board_record = board_records.get(board_id)
    except Exception as e:
        raise UnsupportedWorkflowNodeError(
            f"call_saved_workflow could not access board '{board_id}' for image generator expansion"
        ) from e

    if getattr(board_record, "user_id", None) == user_id:
        return

    board_visibility = getattr(board_record, "board_visibility", BoardVisibility.Private)
    if isinstance(board_visibility, str):
        try:
            board_visibility = BoardVisibility(board_visibility)
        except ValueError:
            board_visibility = BoardVisibility.Private
    if board_visibility in {BoardVisibility.Shared, BoardVisibility.Public}:
        return

    if hasattr(board_records, "get_all"):
        try:
            accessible_boards = board_records.get_all(
                user_id=user_id,
                is_admin=False,
                order_by=BoardRecordOrderBy.Name,
                direction=SQLiteDirection.Ascending,
                include_archived=True,
            )
        except Exception:
            accessible_boards = []
        if any(getattr(board, "board_id", None) == board_id for board in accessible_boards):
            return

    raise UnsupportedWorkflowNodeError(
        f"call_saved_workflow caller does not have access to board '{board_id}' for image generator expansion"
    )


def _resolve_image_generator(value: Mapping[str, Any], services: Any, user_id: str | None) -> list[ImageField]:
    generator_type = value.get("type")
    if generator_type != "image_generator_images_from_board":
        raise UnsupportedWorkflowNodeError(f"Unsupported image generator type '{generator_type}'")
    board_id = value.get("board_id")
    if not isinstance(board_id, str) or not board_id:
        return []
    _assert_user_can_access_board(board_id, services, user_id)
    category = value.get("category", "images")
    categories = IMAGE_CATEGORIES if category == "images" else ASSETS_CATEGORIES
    image_names = services.board_images.get_all_board_image_names_for_board(
        board_id=board_id,
        categories=categories,
        is_intermediate=False,
    )
    return [ImageField(image_name=image_name) for image_name in image_names]


def _resolve_generator_items(generator_node: Mapping[str, Any], services: Any, user_id: str | None) -> list[Any]:
    generator_node_data = generator_node["data"]
    node_type = generator_node_data.get("type")
    inputs = generator_node_data.get("inputs")
    if not isinstance(node_type, str) or not _is_mapping(inputs):
        raise UnsupportedWorkflowNodeError("call_saved_workflow generator node is malformed")
    generator_input = inputs.get("generator")
    if not _is_mapping(generator_input):
        raise UnsupportedWorkflowNodeError(
            f"call_saved_workflow generator node '{generator_node_data.get('id')}' is missing generator input"
        )
    generator_value = generator_input.get("value")
    if not _is_mapping(generator_value):
        raise UnsupportedWorkflowNodeError(
            f"call_saved_workflow generator node '{generator_node_data.get('id')}' has invalid generator value"
        )
    if node_type == "integer_generator":
        return _resolve_integer_generator(generator_value)
    if node_type == "float_generator":
        return _resolve_float_generator(generator_value)
    if node_type == "string_generator":
        return _resolve_string_generator(generator_value)
    if node_type == "image_generator":
        return _resolve_image_generator(generator_value, services, user_id)
    raise UnsupportedWorkflowNodeError(f"Unsupported generator node type '{node_type}'")


def _get_generator_placeholder_items(generator_node: Mapping[str, Any]) -> list[Any]:
    generator_node_data = generator_node["data"]
    node_type = generator_node_data.get("type")
    if node_type == "integer_generator":
        return [0]
    if node_type == "float_generator":
        return [0.0]
    if node_type == "string_generator":
        return [""]
    if node_type == "image_generator":
        return [ImageField(image_name="compatibility-placeholder")]
    raise UnsupportedWorkflowNodeError(f"Unsupported generator node type '{node_type}'")


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


def _normalize_batch_item_for_destination(destination_field: str, batch_items: list[Any]) -> list[Any]:
    if destination_field == "collection":
        return [[item] for item in batch_items]
    return batch_items


def _resolve_batch_items_from_inputs(
    node_id: str,
    field_name: str,
    workflow_edges: Sequence[Mapping[str, Any]],
    workflow_nodes: Mapping[str, Mapping[str, Any]],
) -> list[Any] | None:
    incoming_edges = [
        edge
        for edge in workflow_edges
        if edge.get("target") == node_id and edge.get("targetHandle") == field_name and edge.get("type") == "default"
    ]
    if not incoming_edges:
        return None
    incoming_source_ids = [edge.get("source") for edge in incoming_edges if isinstance(edge.get("source"), str)]
    if len(incoming_source_ids) != 1:
        raise UnsupportedWorkflowNodeError(
            f"call_saved_workflow does not yet support multiple connected batch inputs on node '{node_id}'"
        )
    source_id = incoming_source_ids[0]
    source_node = workflow_nodes.get(source_id)
    if _is_invocation_node(source_node) and source_node["data"].get("type", "").endswith("_generator"):
        return source_id
    if _is_connector_node(source_node):
        resolved_source = _resolve_connector_source(source_id, workflow_nodes, workflow_edges)
        if resolved_source is not None:
            resolved_source_id, _resolved_source_handle = resolved_source
            resolved_source_node = workflow_nodes.get(resolved_source_id)
            if _is_invocation_node(resolved_source_node) and resolved_source_node["data"].get("type", "").endswith(
                "_generator"
            ):
                return resolved_source_id
    raise UnsupportedWorkflowNodeError(
        f"call_saved_workflow does not yet support connected batch child workflow inputs on node '{node_id}'"
    )


def build_batch_child_workflow_session_results(
    *,
    parent_session: GraphExecutionState,
    workflow: Mapping[str, Any],
    workflow_inputs: Mapping[str, Any],
    call_frame: WorkflowCallFrame,
    maximum_children: int,
    services: Any = None,
    user_id: str | None = None,
    resolve_generator_items: bool = True,
) -> list[GraphExecutionState]:
    mutable_workflow = copy.deepcopy(workflow)
    apply_workflow_inputs_to_workflow(mutable_workflow, workflow_inputs)

    workflow_nodes = _get_workflow_nodes(mutable_workflow)
    workflow_edges = _get_default_edges(mutable_workflow)

    batch_data_by_group: dict[str, list[BatchDatum]] = {}
    used_generator_node_ids: set[str] = set()
    for node in workflow_nodes.values():
        if not _is_invocation_node(node):
            continue
        node_data = node["data"]
        node_id = node_data.get("id")
        node_type = node_data.get("type")
        if not isinstance(node_id, str) or not isinstance(node_type, str):
            continue
        if node_type.endswith("_generator"):
            continue
        if node_type not in SUPPORTED_BATCH_TYPES:
            continue

        field_name = BATCH_FIELD_NAMES[node_type]
        generator_source_id = _resolve_batch_items_from_inputs(node_id, field_name, workflow_edges, workflow_nodes)
        if generator_source_id is not None:
            generator_node = workflow_nodes.get(generator_source_id)
            if generator_node is None:
                raise UnsupportedWorkflowNodeError(
                    f"call_saved_workflow generator-backed batch child workflow is missing generator node '{generator_source_id}'"
                )
            generator_node_type = generator_node["data"].get("type") if _is_invocation_node(generator_node) else None
            if generator_node_type == "image_generator" and services is None and resolve_generator_items:
                raise UnsupportedWorkflowNodeError(
                    "call_saved_workflow image-generator-backed batch child workflows require runtime services"
                )
            batch_items = (
                _resolve_generator_items(generator_node, services, user_id)
                if resolve_generator_items
                else _get_generator_placeholder_items(generator_node)
            )
            used_generator_node_ids.add(generator_source_id)
            if not batch_items:
                raise UnsupportedWorkflowNodeError(
                    f"call_saved_workflow generator-backed batch child workflow node '{generator_source_id}' produced no batch items"
                )
        else:
            batch_items = _get_batch_items(node_data, field_name)
            if not batch_items:
                raise UnsupportedWorkflowNodeError(
                    f"call_saved_workflow batch child workflow node '{node_id}' must provide at least one batch item"
                )
        batch_group_id = _get_batch_group_id(node_data)
        destinations = _resolve_batch_destinations(node_id, field_name, workflow_nodes, workflow_edges)
        if not destinations:
            raise UnsupportedWorkflowNodeError(
                f"call_saved_workflow batch child workflow node '{node_id}' is not connected to any invocation input"
            )
        group_batch_data = batch_data_by_group.setdefault(batch_group_id, [])
        for destination_node_id, destination_field in destinations:
            group_batch_data.append(
                BatchDatum(
                    node_path=destination_node_id,
                    field_name=destination_field,
                    items=_normalize_batch_item_for_destination(destination_field, batch_items),
                )
            )

    if not batch_data_by_group:
        raise UnsupportedWorkflowNodeError("call_saved_workflow batch child workflow contains no supported batch nodes")

    _reject_unrelated_generator_nodes(mutable_workflow, used_generator_node_ids)
    sanitized_workflow = _build_child_graph_workflow(mutable_workflow, used_generator_node_ids)
    child_graph = build_graph_from_workflow(sanitized_workflow)
    batch_data = [[datum] for datum in batch_data_by_group.pop("None", [])]
    batch_data.extend(batch_data_by_group.values())
    batch = Batch(graph=child_graph, data=batch_data)
    if calc_session_count(batch) > maximum_children:
        raise TooManySessionsError("call_saved_workflow exceeds remaining queue capacity for child workflow executions")

    child_session_results: list[WorkflowCallChildSessionResult] = []
    for session_id, session_json, field_values_json in create_session_nfv_tuples(batch, maximum_children):
        generated_session = GraphExecutionState.model_validate_json(session_json)
        child_session = parent_session.create_child_workflow_execution_state(generated_session.graph, call_frame)
        child_session.id = session_id
        field_values = [NodeFieldValue.model_validate(field_value) for field_value in json.loads(field_values_json)]
        child_session_results.append(WorkflowCallChildSessionResult(session=child_session, field_values=field_values))
    return child_session_results


def build_batch_child_workflow_sessions(
    *,
    parent_session: GraphExecutionState,
    workflow: Mapping[str, Any],
    workflow_inputs: Mapping[str, Any],
    call_frame: WorkflowCallFrame,
    maximum_children: int,
    services: Any = None,
    user_id: str | None = None,
    resolve_generator_items: bool = True,
) -> list[GraphExecutionState]:
    return [
        child_result.session
        for child_result in build_batch_child_workflow_session_results(
            parent_session=parent_session,
            workflow=workflow,
            workflow_inputs=workflow_inputs,
            call_frame=call_frame,
            maximum_children=maximum_children,
            services=services,
            user_id=user_id,
            resolve_generator_items=resolve_generator_items,
        )
    ]


def build_child_workflow_session_results(
    *,
    parent_session: GraphExecutionState,
    workflow: Mapping[str, Any],
    workflow_inputs: Mapping[str, Any],
    call_frame: WorkflowCallFrame,
    maximum_children: int,
    services: Any = None,
    user_id: str | None = None,
    resolve_generator_items: bool = True,
) -> list[WorkflowCallChildSessionResult]:
    if workflow_contains_supported_batch_nodes(workflow):
        return build_batch_child_workflow_session_results(
            parent_session=parent_session,
            workflow=workflow,
            workflow_inputs=workflow_inputs,
            call_frame=call_frame,
            maximum_children=maximum_children,
            services=services,
            user_id=user_id,
            resolve_generator_items=resolve_generator_items,
        )

    mutable_workflow = copy.deepcopy(workflow)
    apply_workflow_inputs_to_workflow(mutable_workflow, workflow_inputs)
    child_graph = build_graph_from_workflow(mutable_workflow)
    child_session = parent_session.create_child_workflow_execution_state(child_graph, call_frame)
    return [WorkflowCallChildSessionResult(session=child_session)]


def build_child_workflow_sessions(
    *,
    parent_session: GraphExecutionState,
    workflow: Mapping[str, Any],
    workflow_inputs: Mapping[str, Any],
    call_frame: WorkflowCallFrame,
    maximum_children: int,
    services: Any = None,
    user_id: str | None = None,
    resolve_generator_items: bool = True,
) -> list[GraphExecutionState]:
    return [
        child_result.session
        for child_result in build_child_workflow_session_results(
            parent_session=parent_session,
            workflow=workflow,
            workflow_inputs=workflow_inputs,
            call_frame=call_frame,
            maximum_children=maximum_children,
            services=services,
            user_id=user_id,
            resolve_generator_items=resolve_generator_items,
        )
    ]
