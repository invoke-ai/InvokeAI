from typing import Any

from invokeai.app.services.session_processor.workflow_call_batch import build_child_workflow_sessions
from invokeai.app.services.shared.graph import Graph, GraphExecutionState, WorkflowCallFrame
from invokeai.app.services.shared.workflow_call_compatibility_common import (
    WorkflowCallCompatibility,
    WorkflowCallCompatibilityReason,
)
from invokeai.app.services.shared.workflow_graph_builder import InvalidWorkflowInputError, UnsupportedWorkflowNodeError


def _count_workflow_return_nodes(workflow: dict[str, Any]) -> int:
    workflow_return_count = 0
    for node in workflow.get("nodes", []):
        if not isinstance(node, dict) or node.get("type") != "invocation":
            continue
        data = node.get("data")
        if isinstance(data, dict) and data.get("type") == "workflow_return":
            workflow_return_count += 1
    return workflow_return_count


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
        build_child_workflow_sessions(
            parent_session=GraphExecutionState(graph=Graph()),
            workflow=workflow,
            workflow_inputs={},
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
