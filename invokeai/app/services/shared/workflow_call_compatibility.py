from collections.abc import Mapping
from copy import deepcopy
from typing import Any, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from invokeai.app.invocations.baseinvocation import InvocationRegistry
from invokeai.app.invocations.call_saved_workflow import parse_call_saved_workflow_dynamic_input
from invokeai.app.invocations.fields import ImageField
from invokeai.app.services.session_processor.workflow_call_batch import build_child_workflow_sessions
from invokeai.app.services.shared.graph import Graph, GraphExecutionState, WorkflowCallFrame
from invokeai.app.services.shared.workflow_call_compatibility_common import (
    WorkflowCallCompatibility,
    WorkflowCallCompatibilityReason,
)
from invokeai.app.services.shared.workflow_graph_builder import (
    InvalidWorkflowInputError,
    UnsupportedWorkflowNodeError,
    get_exposed_workflow_input_names,
)


def _count_workflow_return_nodes(workflow: dict[str, Any]) -> int:
    workflow_return_count = 0
    for node in workflow.get("nodes", []):
        if not isinstance(node, dict) or node.get("type") != "invocation":
            continue
        data = node.get("data")
        if isinstance(data, dict) and data.get("type") == "workflow_return":
            workflow_return_count += 1
    return workflow_return_count


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _build_placeholder_model(annotation: type[BaseModel]) -> Any:
    values: dict[str, Any] = {}
    for field_name, field_info in annotation.model_fields.items():
        if field_info.default is not PydanticUndefined:
            values[field_name] = deepcopy(field_info.default)
            continue
        if field_info.default_factory is not None:
            values[field_name] = field_info.default_factory()
            continue
        placeholder = _get_placeholder_for_annotation(field_info.annotation)
        if placeholder is None:
            return None
        values[field_name] = placeholder
    return annotation.model_construct(**values)


def _get_placeholder_for_annotation(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is not None:
        if origin is list:
            return []
        if origin is dict:
            return {}
        if origin is tuple:
            return []
        if origin is set:
            return []
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if args:
            return _get_placeholder_for_annotation(args[0])
        return None

    if annotation is Any:
        return {}
    if annotation is str:
        return ""
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is bool:
        return False
    if annotation is ImageField:
        return ImageField(image_name="compatibility-placeholder")
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _build_placeholder_model(annotation)
    return None


def _build_compatibility_workflow_inputs(workflow: dict[str, Any]) -> dict[str, Any]:
    workflow_inputs: dict[str, Any] = {}
    workflow_nodes = workflow.get("nodes", [])
    if not isinstance(workflow_nodes, list):
        return workflow_inputs

    nodes_by_id = {
        node.get("id"): node
        for node in workflow_nodes
        if _is_mapping(node) and isinstance(node.get("id"), str) and _is_mapping(node.get("data"))
    }

    for input_name in get_exposed_workflow_input_names(workflow):
        node_id, field_name = parse_call_saved_workflow_dynamic_input(input_name)
        node = nodes_by_id.get(node_id)
        if not _is_mapping(node):
            continue
        node_data = node.get("data")
        if not _is_mapping(node_data):
            continue
        node_type = node_data.get("type")
        if not isinstance(node_type, str):
            continue
        invocation_class = InvocationRegistry.get_invocation_for_type(node_type)
        if invocation_class is None:
            continue
        field_info = invocation_class.model_fields.get(field_name)
        if field_info is None:
            continue
        if field_info.default is not PydanticUndefined:
            workflow_inputs[input_name] = deepcopy(field_info.default)
            continue
        if field_info.default_factory is not None:
            workflow_inputs[input_name] = field_info.default_factory()
            continue
        placeholder = _get_placeholder_for_annotation(field_info.annotation)
        if placeholder is not None:
            workflow_inputs[input_name] = placeholder

    return workflow_inputs


def get_workflow_call_compatibility(
    *,
    workflow: dict[str, Any],
    workflow_id: str,
    services: Any,
    user_id: str | None,
    maximum_children: int,
) -> WorkflowCallCompatibility:
    workflow_return_count = _count_workflow_return_nodes(workflow)
    if workflow_return_count == 0:
        return WorkflowCallCompatibility(
            is_callable=False,
            reason=WorkflowCallCompatibilityReason.MissingWorkflowReturn,
            message="The workflow must contain exactly one workflow_return node.",
        )
    if workflow_return_count > 1:
        return WorkflowCallCompatibility(
            is_callable=False,
            reason=WorkflowCallCompatibilityReason.MultipleWorkflowReturn,
            message="The workflow must not contain more than one workflow_return node.",
        )

    try:
        workflow_inputs = _build_compatibility_workflow_inputs(workflow)
        build_child_workflow_sessions(
            parent_session=GraphExecutionState(graph=Graph()),
            workflow=workflow,
            workflow_inputs=workflow_inputs,
            call_frame=WorkflowCallFrame(
                prepared_call_node_id="compatibility-call",
                source_call_node_id="compatibility-call",
                workflow_id=workflow_id,
                depth=1,
            ),
            maximum_children=maximum_children,
            services=services,
            user_id=user_id,
        )
    except InvalidWorkflowInputError as e:
        return WorkflowCallCompatibility(
            is_callable=False,
            reason=WorkflowCallCompatibilityReason.InvalidInputs,
            message=str(e),
        )
    except UnsupportedWorkflowNodeError as e:
        message = str(e)
        reason = WorkflowCallCompatibilityReason.UnsupportedNode
        if "connected batch child workflow inputs" in message:
            reason = WorkflowCallCompatibilityReason.UnsupportedBatchInput
        elif "exactly one workflow_return" in message:
            reason = WorkflowCallCompatibilityReason.MissingWorkflowReturn
        return WorkflowCallCompatibility(
            is_callable=False,
            reason=reason,
            message=message,
        )
    except ValueError as e:
        return WorkflowCallCompatibility(
            is_callable=False,
            reason=WorkflowCallCompatibilityReason.InvalidGraph,
            message=str(e),
        )
    except Exception as e:
        return WorkflowCallCompatibility(
            is_callable=False,
            reason=WorkflowCallCompatibilityReason.Unknown,
            message=str(e),
        )

    return WorkflowCallCompatibility(
        is_callable=True,
        reason=WorkflowCallCompatibilityReason.Ok,
    )
